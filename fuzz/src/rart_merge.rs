#![no_main]

use std::collections::HashMap;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use rart::keys::array_key::ArrayKey;
use rart::tree::AdaptiveRadixTree;

/// Operations to build a tree. We use a simple Insert-only model since
/// merge correctness is independent of deletion history.
#[derive(Arbitrary, Debug)]
struct BuildAction {
    key: usize,
    val: usize,
}

/// Fuzz input for N-way merge testing.
/// Supports 2-5 trees to exercise both the 2-way fast path and N-way algorithm.
#[derive(Arbitrary, Debug)]
struct FuzzInput {
    tree_actions: Vec<Vec<BuildAction>>,
}

fuzz_target!(|input: FuzzInput| {
    // Limit to reasonable number of trees to avoid excessive runtime.
    let tree_actions: Vec<_> = input.tree_actions.into_iter().take(5).collect();
    if tree_actions.is_empty() {
        return;
    }

    // Build trees and track their state in HashMaps.
    let mut trees = Vec::new();
    let mut maps = Vec::new();

    for actions in &tree_actions {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, usize>::new();
        let mut map = HashMap::<usize, usize>::new();

        for action in actions {
            tree.insert(action.key, action.val);
            map.insert(action.key, action.val);
        }

        trees.push(tree);
        maps.push(map);
    }

    // Clone trees for fold comparison (since merge consumes them).
    let trees_for_fold: Vec<_> = tree_actions
        .iter()
        .map(|actions| {
            let mut tree = AdaptiveRadixTree::<ArrayKey<16>, usize>::new();
            for action in actions {
                tree.insert(action.key, action.val);
            }
            tree
        })
        .collect();

    // Compute expected result: later maps win on conflicts.
    let mut expected = HashMap::new();
    for map in &maps {
        for (k, v) in map {
            expected.insert(*k, *v);
        }
    }

    // Test merge_all.
    let merged_n_way = AdaptiveRadixTree::merge_all(trees);
    verify_merge(&merged_n_way, &expected, "merge_all");

    // Test fold approach and compare with merge_all.
    if !trees_for_fold.is_empty() {
        let merged_fold = trees_for_fold
            .into_iter()
            .reduce(|a, b| a.merge(b))
            .unwrap();
        verify_merge(&merged_fold, &expected, "fold");

        // Verify both approaches produce identical results.
        verify_identical(&merged_n_way, &merged_fold);
    }
});

/// Verify the merge result against the expected HashMap.
fn verify_merge(
    merged: &AdaptiveRadixTree<ArrayKey<16>, usize>,
    expected: &HashMap<usize, usize>,
    approach: &str,
) {
    // Check that all expected keys are present with correct values.
    for (k, expected_val) in expected {
        let merged_val = merged.get(k);
        assert_eq!(
            merged_val,
            Some(expected_val),
            "[{}] Key {} mismatch: expected {:?}, got {:?}",
            approach,
            k,
            expected_val,
            merged_val
        );
    }

    // Check that merged tree has exactly the expected number of entries.
    let merged_count = merged.iter().count();
    assert_eq!(
        merged_count,
        expected.len(),
        "[{}] Merged tree has {} entries, expected {}",
        approach,
        merged_count,
        expected.len()
    );

    // Verify iteration order is lexicographic (keys should be sorted).
    let merged_keys: Vec<_> = merged.iter().map(|(k, _)| k).collect();
    let mut sorted_keys = merged_keys.clone();
    sorted_keys.sort();
    assert_eq!(
        merged_keys, sorted_keys,
        "[{}] Merged tree iteration is not in lexicographic order",
        approach
    );
}

/// Verify that two merge results are identical.
fn verify_identical(
    n_way: &AdaptiveRadixTree<ArrayKey<16>, usize>,
    fold: &AdaptiveRadixTree<ArrayKey<16>, usize>,
) {
    let n_way_items: Vec<_> = n_way.iter().map(|(k, v)| (k.to_be_u64(), *v)).collect();
    let fold_items: Vec<_> = fold.iter().map(|(k, v)| (k.to_be_u64(), *v)).collect();

    assert_eq!(
        n_way_items, fold_items,
        "merge_all and fold produced different results"
    );
}
