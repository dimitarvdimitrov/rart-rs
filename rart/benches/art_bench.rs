/// Overall simple performance bench for static # of keys in a few secnarios. Here to quickly test\
/// for regressions.
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::SliceRandom;
use rand::{Rng, rng};

use rart::keys::array_key::ArrayKey;

use rart::tree::AdaptiveRadixTree;

// Variations on the number of keys to insert into the tree for benchmarks that measure retrievals
const TREE_SIZES: [u64; 4] = [1 << 10, 1 << 12, 1 << 15, 1 << 17];

fn full_bench_profile() -> bool {
    std::env::var("RART_BENCH_FULL").as_deref() == Ok("1")
}

fn criterion_config() -> Criterion {
    if full_bench_profile() {
        Criterion::default()
    } else {
        Criterion::default()
            .sample_size(30)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2))
    }
}

pub fn rand_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand_insert");
    group.throughput(Throughput::Elements(1));

    let keys = gen_keys(3, 2, 3);
    let cached_keys = gen_cached_keys(3, 2, 3);

    group.bench_function("cached_keys", |b| {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
        let mut rng = rng();
        b.iter(|| {
            let key = &cached_keys[rng.random_range(0..cached_keys.len())];
            tree.insert_k(&key.0, key.1.clone());
        })
    });

    group.bench_function("uncached_keys", |b| {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
        let mut rng = rng();
        b.iter(|| {
            let key = &keys[rng.random_range(0..keys.len())];
            tree.insert(key, key.clone());
        })
    });

    group.finish();
}

pub fn rand_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("rand_remove");
    let keys = gen_keys(3, 2, 3);
    let cached_keys = gen_cached_keys(3, 2, 3);

    group.throughput(Throughput::Elements(1));
    group.bench_function("cached_keys", |b| {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
        let mut rng = rng();
        for key in &cached_keys {
            tree.insert_k(&key.0.clone(), key.1.clone());
        }
        b.iter(|| {
            let key = &cached_keys[rng.random_range(0..keys.len())];
            std::hint::black_box(tree.remove_k(&key.0));
        })
    });
    group.bench_function("uncached_keys", |b| {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
        let mut rng = rng();
        for key in &keys {
            tree.insert(key, key);
        }
        b.iter(|| {
            let key = &keys[rng.random_range(0..keys.len())];
            std::hint::black_box(tree.remove(key));
        })
    });

    group.finish();
}

pub fn rand_get(c: &mut Criterion) {
    for size in TREE_SIZES {
        c.bench_with_input(BenchmarkId::new("rand_get", size), &size, |b, size| {
            let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
            for i in 0..*size {
                tree.insert(i, i);
            }
            let mut rng = rng();
            b.iter(|| {
                let key = rng.random_range(0..*size);
                std::hint::black_box(tree.get(key));
            })
        });
    }
}

pub fn rand_get_str(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_get_str");
    let keys = gen_keys(3, 2, 3);
    let cached_keys = gen_cached_keys(3, 2, 3);
    group.throughput(Throughput::Elements(1));
    for size in TREE_SIZES {
        group.bench_with_input(BenchmarkId::new("cached_keys", size), &size, |b, _size| {
            let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
            for (i, key) in cached_keys.iter().enumerate() {
                tree.insert_k(&key.0, i);
            }
            let mut rng = rng();
            b.iter(|| {
                let key = &cached_keys[rng.random_range(0..keys.len())];
                std::hint::black_box(tree.get_k(&key.0));
            })
        });
    }

    for size in TREE_SIZES {
        group.bench_with_input(
            BenchmarkId::new("uncached_keys", size),
            &size,
            |b, _size| {
                let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
                for (i, key) in keys.iter().enumerate() {
                    tree.insert(key, i);
                }
                let mut rng = rng();
                b.iter(|| {
                    let key = &keys[rng.random_range(0..keys.len())];
                    std::hint::black_box(tree.get(key));
                })
            },
        );
    }

    group.finish();
}

pub fn seq_get(c: &mut Criterion) {
    for size in TREE_SIZES {
        c.bench_with_input(BenchmarkId::new("seq_get", size), &size, |b, size| {
            let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
            for i in 0..*size {
                tree.insert(i, i);
            }
            b.iter_custom(|iters| {
                let mut c = 0;
                let start = Instant::now();
                for _ in 0..iters {
                    if c == *size {
                        c = 0;
                    }
                    tree.get(c).unwrap();
                    c += 1;
                }
                start.elapsed()
            })
        });
    }
}

pub fn seq_insert(c: &mut Criterion) {
    c.bench_function("seq_insert", |b| {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
        let mut key = 0u64;
        b.iter(|| {
            tree.insert(key, key);
            key += 1;
        })
    });
}

pub fn seq_remove(c: &mut Criterion) {
    for size in TREE_SIZES {
        c.bench_with_input(BenchmarkId::new("seq_remove", size), &size, |b, size| {
            let mut tree = AdaptiveRadixTree::<ArrayKey<16>, _>::new();
            b.iter_custom(|iters| {
                for i in 0..*size {
                    tree.insert(i, i);
                }
                let mut start = Instant::now();
                let mut cumulative_time = Duration::new(0, 0);
                let mut c = 0;
                for _ in 0..iters {
                    if c == *size {
                        cumulative_time += start.elapsed();
                        c = 0;
                        for i in 0..*size {
                            tree.insert(i, i);
                        }
                        start = Instant::now();
                    }
                    tree.remove(c).unwrap();
                    c += 1;
                }
                cumulative_time += start.elapsed();
                cumulative_time
            })
        });
    }
}

fn gen_keys(l1_prefix: usize, l2_prefix: usize, suffix: usize) -> Vec<String> {
    let mut keys = Vec::new();
    let chars: Vec<char> = ('a'..='z').collect();
    for i in 0..chars.len() {
        let level1_prefix = chars[i].to_string().repeat(l1_prefix);
        for i in 0..chars.len() {
            let level2_prefix = chars[i].to_string().repeat(l2_prefix);
            let key_prefix = level1_prefix.clone() + &level2_prefix;
            for _ in 0..=u8::MAX {
                let suffix: String = (0..suffix)
                    .map(|_| chars[rng().random_range(0..chars.len())])
                    .collect();
                let k = key_prefix.clone() + &suffix;
                keys.push(k);
            }
        }
    }

    keys.shuffle(&mut rng());
    keys
}

fn gen_cached_keys(
    l1_prefix: usize,
    l2_prefix: usize,
    suffix: usize,
) -> Vec<(ArrayKey<16>, String)> {
    let mut keys = Vec::new();
    let chars: Vec<char> = ('a'..='z').collect();
    for i in 0..chars.len() {
        let level1_prefix = chars[i].to_string().repeat(l1_prefix);
        for i in 0..chars.len() {
            let level2_prefix = chars[i].to_string().repeat(l2_prefix);
            let key_prefix = level1_prefix.clone() + &level2_prefix;
            for _ in 0..=u8::MAX {
                let suffix: String = (0..suffix)
                    .map(|_| chars[rng().random_range(0..chars.len())])
                    .collect();
                let string = key_prefix.clone() + &suffix;
                let k = string.clone().into();
                keys.push((k, string));
            }
        }
    }

    keys.shuffle(&mut rng());
    keys
}

/// Merge two 10k-element trees with no overlapping keys.
pub fn bench_merge_disjoint_10k(c: &mut Criterion) {
    c.bench_function("merge_disjoint_10k", |b| {
        b.iter_batched(
            || {
                let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                // tree1 gets even keys, tree2 gets odd keys.
                for i in 0..10_000u64 {
                    tree1.insert(i * 2, i * 2);
                    tree2.insert(i * 2 + 1, i * 2 + 1);
                }
                (tree1, tree2)
            },
            |(tree1, tree2)| std::hint::black_box(tree1.merge(tree2)),
            criterion::BatchSize::SmallInput,
        )
    });
}

/// Merge two 10k-element trees with 50% key overlap.
pub fn bench_merge_50pct_overlap_10k(c: &mut Criterion) {
    c.bench_function("merge_50pct_overlap_10k", |b| {
        b.iter_batched(
            || {
                let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                // tree1 gets [0..10k), tree2 gets [5k..15k). Overlap is [5k..10k).
                for i in 0..10_000u64 {
                    tree1.insert(i, i);
                    tree2.insert(i + 5_000, i + 5_000);
                }
                (tree1, tree2)
            },
            |(tree1, tree2)| std::hint::black_box(tree1.merge(tree2)),
            criterion::BatchSize::SmallInput,
        )
    });
}

/// Merge two sequential ranges: [0..10k] with [10k..20k].
pub fn bench_merge_sequential_ranges(c: &mut Criterion) {
    c.bench_function("merge_sequential_ranges", |b| {
        b.iter_batched(
            || {
                let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                for i in 0..10_000u64 {
                    tree1.insert(i, i);
                    tree2.insert(i + 10_000, i + 10_000);
                }
                (tree1, tree2)
            },
            |(tree1, tree2)| std::hint::black_box(tree1.merge(tree2)),
            criterion::BatchSize::SmallInput,
        )
    });
}

/// Naive merge via iterators: iterate both trees and insert into a new tree.
/// This serves as a baseline to compare against the optimized merge().
pub fn bench_merge_naive_disjoint_10k(c: &mut Criterion) {
    c.bench_function("merge_naive_disjoint_10k", |b| {
        b.iter_batched(
            || {
                let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                // Same setup as disjoint: tree1 even, tree2 odd.
                for i in 0..10_000u64 {
                    tree1.insert(i * 2, i * 2);
                    tree2.insert(i * 2 + 1, i * 2 + 1);
                }
                (tree1, tree2)
            },
            |(tree1, tree2)| {
                let mut result = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                // Insert all from tree1, then all from tree2 (tree2 wins on conflict).
                for (k, v) in tree1.iter() {
                    result.insert_k(&k, *v);
                }
                for (k, v) in tree2.iter() {
                    result.insert_k(&k, *v);
                }
                std::hint::black_box(result)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

/// Naive merge via iterators with 50% overlap.
pub fn bench_merge_naive_50pct_overlap_10k(c: &mut Criterion) {
    c.bench_function("merge_naive_50pct_overlap_10k", |b| {
        b.iter_batched(
            || {
                let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                // Same setup as 50% overlap: tree1 [0..10k), tree2 [5k..15k).
                for i in 0..10_000u64 {
                    tree1.insert(i, i);
                    tree2.insert(i + 5_000, i + 5_000);
                }
                (tree1, tree2)
            },
            |(tree1, tree2)| {
                let mut result = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
                for (k, v) in tree1.iter() {
                    result.insert_k(&k, *v);
                }
                for (k, v) in tree2.iter() {
                    result.insert_k(&k, *v);
                }
                std::hint::black_box(result)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    name = rand_benches;
    config = criterion_config();
    targets = rand_get, rand_get_str, rand_insert, rand_remove
);
criterion_group!(
    name = seq_benches;
    config = criterion_config();
    targets = seq_get, seq_insert, seq_remove
);
criterion_group!(
    name = merge_benches;
    config = criterion_config();
    targets = bench_merge_disjoint_10k, bench_merge_50pct_overlap_10k, bench_merge_sequential_ranges, bench_merge_naive_disjoint_10k, bench_merge_naive_50pct_overlap_10k
);

// N-way merge benchmarks: 16 trees of 1000 values each

/// Build 16 trees with disjoint keys (tree i gets keys i, i+16, i+32, ...).
fn build_16_trees_disjoint() -> Vec<AdaptiveRadixTree<ArrayKey<16>, u64>> {
    (0..16)
        .map(|tree_idx| {
            let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
            for i in 0..1000u64 {
                let key = tree_idx as u64 + i * 16;
                tree.insert(key, key);
            }
            tree
        })
        .collect()
}

/// Build 16 trees with 50% overlap between consecutive trees.
/// Tree i gets keys [i*500 .. i*500 + 1000).
fn build_16_trees_50pct_overlap() -> Vec<AdaptiveRadixTree<ArrayKey<16>, u64>> {
    (0..16)
        .map(|tree_idx| {
            let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
            let base = tree_idx as u64 * 500;
            for i in 0..1000u64 {
                let key = base + i;
                tree.insert(key, key);
            }
            tree
        })
        .collect()
}

/// Build 16 trees with 100% overlap (all trees have the same keys 0..1000).
fn build_16_trees_full_overlap() -> Vec<AdaptiveRadixTree<ArrayKey<16>, u64>> {
    (0..16)
        .map(|tree_idx| {
            let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
            for i in 0..1000u64 {
                // Value encodes tree_idx so we can verify right-wins
                tree.insert(i, i + tree_idx as u64 * 10000);
            }
            tree
        })
        .collect()
}

/// Fold-based merge: trees.into_iter().reduce(|a, b| a.merge(b))
fn fold_merge(
    trees: Vec<AdaptiveRadixTree<ArrayKey<16>, u64>>,
) -> AdaptiveRadixTree<ArrayKey<16>, u64> {
    trees
        .into_iter()
        .reduce(|a, b| a.merge(b))
        .unwrap_or_else(AdaptiveRadixTree::new)
}

/// N-way merge: 16 disjoint trees via fold
pub fn bench_merge_16_fold_disjoint(c: &mut Criterion) {
    c.bench_function("merge_16_fold_disjoint", |b| {
        b.iter_batched(
            build_16_trees_disjoint,
            |trees| std::hint::black_box(fold_merge(trees)),
            criterion::BatchSize::SmallInput,
        )
    });
}

/// N-way merge: 16 disjoint trees via merge_all
pub fn bench_merge_16_nway_disjoint(c: &mut Criterion) {
    c.bench_function("merge_16_nway_disjoint", |b| {
        b.iter_batched(
            build_16_trees_disjoint,
            |trees| std::hint::black_box(AdaptiveRadixTree::merge_all(trees)),
            criterion::BatchSize::SmallInput,
        )
    });
}

/// N-way merge: 16 trees with 50% overlap via fold
pub fn bench_merge_16_fold_50pct_overlap(c: &mut Criterion) {
    c.bench_function("merge_16_fold_50pct_overlap", |b| {
        b.iter_batched(
            build_16_trees_50pct_overlap,
            |trees| std::hint::black_box(fold_merge(trees)),
            criterion::BatchSize::SmallInput,
        )
    });
}

/// N-way merge: 16 trees with 50% overlap via merge_all
pub fn bench_merge_16_nway_50pct_overlap(c: &mut Criterion) {
    c.bench_function("merge_16_nway_50pct_overlap", |b| {
        b.iter_batched(
            build_16_trees_50pct_overlap,
            |trees| std::hint::black_box(AdaptiveRadixTree::merge_all(trees)),
            criterion::BatchSize::SmallInput,
        )
    });
}

/// N-way merge: 16 trees with 100% overlap via fold
pub fn bench_merge_16_fold_full_overlap(c: &mut Criterion) {
    c.bench_function("merge_16_fold_full_overlap", |b| {
        b.iter_batched(
            build_16_trees_full_overlap,
            |trees| std::hint::black_box(fold_merge(trees)),
            criterion::BatchSize::SmallInput,
        )
    });
}

/// N-way merge: 16 trees with 100% overlap via merge_all
pub fn bench_merge_16_nway_full_overlap(c: &mut Criterion) {
    c.bench_function("merge_16_nway_full_overlap", |b| {
        b.iter_batched(
            build_16_trees_full_overlap,
            |trees| std::hint::black_box(AdaptiveRadixTree::merge_all(trees)),
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    name = merge_n_benches;
    config = criterion_config();
    targets = bench_merge_16_fold_disjoint, bench_merge_16_nway_disjoint,
              bench_merge_16_fold_50pct_overlap, bench_merge_16_nway_50pct_overlap,
              bench_merge_16_fold_full_overlap, bench_merge_16_nway_full_overlap
);

criterion_main!(seq_benches, rand_benches, merge_benches, merge_n_benches);
