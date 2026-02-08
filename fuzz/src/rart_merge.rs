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

/// Fuzz input: two sets of operations to build two separate trees.
#[derive(Arbitrary, Debug)]
struct FuzzInput {
    tree1_actions: Vec<BuildAction>,
    tree2_actions: Vec<BuildAction>,
}

fuzz_target!(|input: FuzzInput| {
    // Build tree1 and track its state in a HashMap.
    let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, usize>::new();
    let mut map1 = HashMap::<usize, usize>::new();

    for action in &input.tree1_actions {
        tree1.insert(action.key, action.val);
        map1.insert(action.key, action.val);
    }

    // Build tree2 and track its state in a HashMap.
    let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, usize>::new();
    let mut map2 = HashMap::<usize, usize>::new();

    for action in &input.tree2_actions {
        tree2.insert(action.key, action.val);
        map2.insert(action.key, action.val);
    }

    // Compute expected result: map2 values override map1 values (right-wins semantics).
    let mut expected = map1.clone();
    for (k, v) in &map2 {
        expected.insert(*k, *v);
    }

    // Perform the merge.
    let merged = merge_trees(tree1, tree2);

    // Verify the merge against the reference HashMaps.
    verify_merge(&merged, &map1, &map2, &expected);
});

/// Perform the actual merge operation.
#[allow(clippy::needless_pass_by_value)]
fn merge_trees(
    tree1: AdaptiveRadixTree<ArrayKey<16>, usize>,
    tree2: AdaptiveRadixTree<ArrayKey<16>, usize>,
) -> AdaptiveRadixTree<ArrayKey<16>, usize> {
    tree1.merge(tree2)
}

/// Verify the merge invariants against the reference HashMaps.
fn verify_merge(
    merged: &AdaptiveRadixTree<ArrayKey<16>, usize>,
    map1: &HashMap<usize, usize>,
    map2: &HashMap<usize, usize>,
    expected: &HashMap<usize, usize>,
) {
    // Verify the core merge invariant:
    // For all keys k: merged.get(k) == tree2.get(k).or(tree1.get(k))
    //
    // Since we consumed tree1 and tree2 in the merge, we use the HashMaps
    // as our source of truth for what values should be present.

    // Check that all expected keys are present with correct values.
    for (k, expected_val) in expected {
        let merged_val = merged.get(k);
        assert_eq!(
            merged_val,
            Some(expected_val),
            "Key {} mismatch: expected {:?}, got {:?}",
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
        "Merged tree has {} entries, expected {}",
        merged_count,
        expected.len()
    );

    // Verify iteration order is lexicographic (keys should be sorted).
    let merged_keys: Vec<_> = merged.iter().map(|(k, _)| k).collect();
    let mut sorted_keys = merged_keys.clone();
    sorted_keys.sort();
    assert_eq!(
        merged_keys, sorted_keys,
        "Merged tree iteration is not in lexicographic order"
    );

    // Verify the merge invariant directly: merged.get(k) == map2.get(k).or(map1.get(k))
    // for all keys in either tree.
    let all_keys: std::collections::HashSet<_> = map1.keys().chain(map2.keys()).copied().collect();
    for k in all_keys {
        let merged_val = merged.get(&k).copied();
        let expected_val = map2.get(&k).or_else(|| map1.get(&k)).copied();
        assert_eq!(
            merged_val, expected_val,
            "Merge invariant violated for key {}: merged={:?}, expected={:?}",
            k, merged_val, expected_val
        );
    }
}
