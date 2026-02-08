//! Adaptive Radix Tree implementation.
//!
//! This module contains the main [`AdaptiveRadixTree`] implementation and related
//! functionality for the RART crate.

use std::cmp::{Ordering, min};
use std::collections::BTreeMap;
use std::ops::RangeBounds;

use smallvec::SmallVec;

use crate::iter::{Iter, ValuesIter};
use crate::keys::KeyTrait;
use crate::node::{Content, DefaultNode, Node};
use crate::partials::Partial;
use crate::range::Range;
use crate::stats::{TreeStats, TreeStatsTrait, update_tree_stats};

/// An Adaptive Radix Tree (ART) - a high-performance, memory-efficient trie data structure.
///
/// The Adaptive Radix Tree automatically adjusts its internal representation based on the
/// number of children at each node, providing excellent performance characteristics for
/// a wide range of workloads.
///
/// ## Features
///
/// - **Adaptive nodes**: Uses different node types (4, 16, 48, 256 children) based on density
/// - **Space efficient**: Compact representation that minimizes memory usage
/// - **Cache friendly**: Optimized memory layout for modern CPU architectures
/// - **Fast operations**: O(k) complexity for basic operations where k is the key length
/// - **Range queries**: Efficient iteration over key ranges with proper ordering
///
/// ## Type Parameters
///
/// - `KeyType`: The type of keys stored in the tree, must implement [`KeyTrait`]
/// - `ValueType`: The type of values associated with keys
///
/// ## Examples
///
/// Basic usage with string keys:
///
/// ```rust
/// use rart::{AdaptiveRadixTree, ArrayKey};
///
/// let mut tree = AdaptiveRadixTree::<ArrayKey<32>, String>::new();
///
/// // Insert some data
/// tree.insert("apple", "fruit".to_string());
/// tree.insert("application", "software".to_string());
///
/// // Query the tree
/// debug_assert_eq!(tree.get("apple"), Some(&"fruit".to_string()));
/// debug_assert_eq!(tree.get("orange"), None);
///
/// // Iterate over all entries
/// for (key, value) in tree.iter() {
///     println!("{:?} -> {}", key.as_ref(), value);
/// }
/// ```
///
/// Range queries:
///
/// ```rust
/// use rart::{AdaptiveRadixTree, ArrayKey};
///
/// let mut tree = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
/// tree.insert("apple", 1);
/// tree.insert("banana", 2);
/// tree.insert("cherry", 3);
///
/// // Get all keys starting with "a"
/// let start: ArrayKey<16> = "a".into();
/// let end: ArrayKey<16> = "b".into();
/// let a_keys: Vec<_> = tree.range(start..end).collect();
/// debug_assert_eq!(a_keys.len(), 1); // Just "apple"
/// ```
pub struct AdaptiveRadixTree<KeyType, ValueType>
where
    KeyType: KeyTrait,
{
    root: Option<DefaultNode<KeyType::PartialType, ValueType>>,
    _phantom: std::marker::PhantomData<KeyType>,
}

impl<KeyType: KeyTrait, ValueType> Default for AdaptiveRadixTree<KeyType, ValueType> {
    fn default() -> Self {
        Self::new()
    }
}

impl<KeyType, ValueType> AdaptiveRadixTree<KeyType, ValueType>
where
    KeyType: KeyTrait,
{
    /// Create a new empty Adaptive Radix Tree.
    pub fn new() -> Self {
        Self {
            root: None,
            _phantom: Default::default(),
        }
    }

    /// Create a new Adaptive Radix Tree with the given root node.
    /// This is primarily used for internal conversions.
    pub(crate) fn from_root(root: DefaultNode<KeyType::PartialType, ValueType>) -> Self {
        Self {
            root: Some(root),
            _phantom: Default::default(),
        }
    }

    /// Get a value by key (generic version).
    ///
    /// This method accepts any type that can be converted into the tree's key type.
    #[inline]
    pub fn get<Key>(&self, key: Key) -> Option<&ValueType>
    where
        Key: Into<KeyType>,
    {
        self.get_k(&key.into())
    }

    /// Get a value by key reference (direct version).
    ///
    /// This method works directly with key references for optimal performance.
    #[inline]
    pub fn get_k(&self, key: &KeyType) -> Option<&ValueType> {
        AdaptiveRadixTree::get_iterate(self.root.as_ref()?, key)
    }

    /// Get a mutable reference to a value by key (generic version).
    #[inline]
    pub fn get_mut<Key>(&mut self, key: Key) -> Option<&mut ValueType>
    where
        Key: Into<KeyType>,
    {
        self.get_mut_k(&key.into())
    }

    /// Get a mutable reference to a value by key reference (direct version).
    #[inline]
    pub fn get_mut_k(&mut self, key: &KeyType) -> Option<&mut ValueType> {
        AdaptiveRadixTree::get_iterate_mut(self.root.as_mut()?, key)
    }

    /// Return the deepest key/value pair whose key is a prefix of `key`.
    ///
    /// This differs from [`Self::get`] by allowing partial matches.
    #[inline]
    pub fn longest_prefix_match<Key>(&self, key: Key) -> Option<(KeyType, &ValueType)>
    where
        Key: Into<KeyType>,
    {
        self.longest_prefix_match_k(&key.into())
    }

    /// Return the deepest key/value pair whose key is a prefix of `key`.
    #[inline]
    pub fn longest_prefix_match_k(&self, key: &KeyType) -> Option<(KeyType, &ValueType)> {
        AdaptiveRadixTree::longest_prefix_match_iterate(self.root.as_ref()?, key)
    }

    /// Iterate over all entries whose keys start with `prefix`.
    #[inline]
    pub fn prefix_iter<Key>(
        &self,
        prefix: Key,
    ) -> Iter<'_, KeyType, KeyType::PartialType, ValueType>
    where
        Key: Into<KeyType>,
    {
        self.prefix_iter_k(&prefix.into())
    }

    /// Iterate over all entries whose keys start with `prefix`.
    pub fn prefix_iter_k(
        &self,
        prefix: &KeyType,
    ) -> Iter<'_, KeyType, KeyType::PartialType, ValueType> {
        let Some(root) = self.root.as_ref() else {
            return Iter::new(None);
        };
        let Some((subtree_root, subtree_root_key)) =
            AdaptiveRadixTree::find_prefix_subtree(root, prefix)
        else {
            return Iter::new(None);
        };
        Iter::new_with_prefix(Some(subtree_root), subtree_root_key)
    }

    /// Insert a key-value pair (generic version).
    ///
    /// Follows standard Rust container conventions by returning the old value
    /// when a key is replaced.
    ///
    /// # Returns
    ///
    /// - `Some(old_value)` if a previous value was replaced
    /// - `None` if this was a new key
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rart::{AdaptiveRadixTree, ArrayKey};
    ///
    /// let mut tree = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    ///
    /// // Insert new key returns None
    /// assert_eq!(tree.insert("key1", 100), None);
    ///
    /// // Insert same key returns old value
    /// assert_eq!(tree.insert("key1", 200), Some(100));
    /// assert_eq!(tree.get("key1"), Some(&200));
    /// ```
    #[inline]
    pub fn insert<KV>(&mut self, key: KV, value: ValueType) -> Option<ValueType>
    where
        KV: Into<KeyType>,
    {
        self.insert_k(&key.into(), value)
    }

    /// Insert a key-value pair using key reference (direct version).
    ///
    /// Follows standard Rust container conventions by returning the old value
    /// when a key is replaced.
    ///
    /// # Returns
    ///
    /// - `Some(old_value)` if a previous value was replaced
    /// - `None` if this was a new key
    #[inline]
    pub fn insert_k(&mut self, key: &KeyType, value: ValueType) -> Option<ValueType> {
        let Some(root) = &mut self.root else {
            self.root = Some(DefaultNode::new_leaf(key.to_partial(0), value));
            return None;
        };

        AdaptiveRadixTree::insert_recurse(root, key, value, 0)
    }

    /// Remove a key-value pair (generic version).
    ///
    /// Returns the removed value if the key existed.
    pub fn remove<KV>(&mut self, key: KV) -> Option<ValueType>
    where
        KV: Into<KeyType>,
    {
        self.remove_k(&key.into())
    }

    /// Remove a key-value pair using key reference (direct version).
    ///
    /// Returns the removed value if the key existed.
    pub fn remove_k(&mut self, key: &KeyType) -> Option<ValueType> {
        let root = self.root.as_mut()?;

        // Don't bother doing anything if there's no prefix match on the root at all.
        let prefix_common_match = root.prefix.prefix_length_key(key, 0);
        if prefix_common_match != root.prefix.len() {
            return None;
        }

        // Special case, if the root is a leaf and matches the key, we can just remove it
        // immediately. If it doesn't match our key, then we have nothing to do here anyways.
        if root.is_leaf() {
            // Move the value of the leaf in root. To do this, replace self.root  with None and
            // then unwrap the value out of the Option & Leaf.
            let stolen = self.root.take().unwrap();
            let leaf = match stolen.content {
                Content::Leaf(v) => v,
                _ => unreachable!(),
            };
            return Some(leaf);
        }

        let result = AdaptiveRadixTree::remove_recurse(root, key, prefix_common_match);

        // Prune root out if it's now empty.
        if root.is_inner() && root.num_children() == 0 {
            self.root = None;
        }
        result
    }

    /// Create an iterator over all key-value pairs in the tree.
    ///
    /// The iterator yields items in lexicographic order of the keys.
    pub fn iter(&self) -> Iter<'_, KeyType, KeyType::PartialType, ValueType> {
        Iter::new(self.root.as_ref())
    }

    /// Create an iterator over only the values in the tree.
    ///
    /// This iterator skips key reconstruction entirely and only yields values.
    /// It's more efficient when you don't need the keys.
    pub fn values_iter(&self) -> ValuesIter<'_, KeyType::PartialType, ValueType> {
        ValuesIter::new(self.root.as_ref())
    }

    /// Create an iterator over key-value pairs within a specified range.
    ///
    /// The range can be any type that implements `RangeBounds<KeyType>`.
    pub fn range<'a, R>(&'a self, range: R) -> Range<'a, KeyType, ValueType>
    where
        R: RangeBounds<KeyType> + 'a,
    {
        let Some(_) = &self.root else {
            return Range::empty();
        };

        let start_bound = range.start_bound().cloned();
        let end_bound = range.end_bound().cloned();

        // Use optimized O(log n) iteration for start bound
        match start_bound {
            std::collections::Bound::Unbounded => {
                // No start bound, use regular iterator
                let iter = self.iter();
                Range::for_iter(iter, end_bound)
            }
            _ => {
                // Use optimized start bound iteration
                let optimized_iter = Iter::new_with_start_bound(self.root.as_ref(), start_bound);
                Range::for_iter(optimized_iter, end_bound)
            }
        }
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Merge another tree into this one, consuming both and returning a new tree.
    ///
    /// When both trees contain the same key, the value from `other` is used
    /// (right-wins semantics, similar to `HashMap::extend`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rart::{AdaptiveRadixTree, ArrayKey};
    ///
    /// let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// tree1.insert("apple", 1);
    /// tree1.insert("banana", 2);
    ///
    /// let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// tree2.insert("apple", 10);  // Overwrites tree1's "apple"
    /// tree2.insert("cherry", 3);
    ///
    /// let merged = tree1.merge(tree2);
    ///
    /// assert_eq!(merged.get("apple"), Some(&10));  // From tree2
    /// assert_eq!(merged.get("banana"), Some(&2));  // From tree1
    /// assert_eq!(merged.get("cherry"), Some(&3));  // From tree2
    /// ```
    #[inline]
    pub fn merge(self, other: Self) -> Self {
        self.merge_with(other, |_left, right| right)
    }

    /// Merge two trees into one with custom conflict resolution, consuming both inputs.
    ///
    /// When both trees contain the same key, the provided resolver function is called
    /// with both values (left from `self`, right from `other`) and its return value
    /// is used.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rart::{AdaptiveRadixTree, ArrayKey};
    ///
    /// let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// tree1.insert("apple", 1);
    /// tree1.insert("banana", 2);
    ///
    /// let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// tree2.insert("apple", 10);
    /// tree2.insert("cherry", 3);
    ///
    /// // Sum conflicting values.
    /// let merged = tree1.merge_with(tree2, |left, right| left + right);
    ///
    /// assert_eq!(merged.get("apple"), Some(&11));  // 1 + 10
    /// assert_eq!(merged.get("banana"), Some(&2));  // From tree1
    /// assert_eq!(merged.get("cherry"), Some(&3));  // From tree2
    /// ```
    pub fn merge_with<F>(self, other: Self, mut f: F) -> Self
    where
        F: FnMut(ValueType, ValueType) -> ValueType,
    {
        match (self.root, other.root) {
            (None, None) => Self::new(),
            (Some(root), None) => Self::from_root(root),
            (None, Some(root)) => Self::from_root(root),
            (Some(root1), Some(root2)) => {
                Self::from_root(Self::merge_nodes_with(root1, root2, &mut f))
            }
        }
    }

    /// Merge multiple trees into one, consuming all inputs.
    ///
    /// Later trees in the iterator win on key conflicts (right-wins semantics).
    /// For N=2, this delegates to the optimized 2-way merge to avoid
    /// the overhead of the N-way machinery.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rart::{AdaptiveRadixTree, ArrayKey};
    ///
    /// let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// t1.insert("apple", 1);
    ///
    /// let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// t2.insert("apple", 2);  // Will overwrite t1's value
    /// t2.insert("banana", 3);
    ///
    /// let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// t3.insert("cherry", 4);
    ///
    /// let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3]);
    ///
    /// assert_eq!(merged.get("apple"), Some(&2));  // t2 wins
    /// assert_eq!(merged.get("banana"), Some(&3));
    /// assert_eq!(merged.get("cherry"), Some(&4));
    /// ```
    #[inline]
    pub fn merge_all(trees: impl IntoIterator<Item = Self>) -> Self {
        Self::merge_all_with(trees, |_left, right| right)
    }

    /// Merge multiple trees into one with custom conflict resolution, consuming all inputs.
    ///
    /// When multiple trees contain the same key, the provided resolver function is called
    /// repeatedly (fold-style) to combine values. The order in which values are folded is
    /// arbitrary, so the resolver should be commutative for deterministic results.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rart::{AdaptiveRadixTree, ArrayKey};
    ///
    /// let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// t1.insert("apple", 1);
    ///
    /// let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// t2.insert("apple", 2);
    /// t2.insert("banana", 3);
    ///
    /// let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
    /// t3.insert("apple", 4);
    /// t3.insert("cherry", 5);
    ///
    /// // Sum all values for conflicting keys.
    /// let merged = AdaptiveRadixTree::merge_all_with(vec![t1, t2, t3], |acc, v| acc + v);
    ///
    /// assert_eq!(merged.get("apple"), Some(&7));  // 1 + 2 + 4
    /// assert_eq!(merged.get("banana"), Some(&3));
    /// assert_eq!(merged.get("cherry"), Some(&5));
    /// ```
    pub fn merge_all_with<F>(trees: impl IntoIterator<Item = Self>, mut f: F) -> Self
    where
        F: FnMut(ValueType, ValueType) -> ValueType,
    {
        let trees: Vec<_> = trees.into_iter().collect();
        match trees.len() {
            0 => Self::new(),
            1 => trees.into_iter().next().unwrap(),
            2 => {
                // Use optimized 2-way merge for N=2.
                let mut iter = trees.into_iter();
                let a = iter.next().unwrap();
                let b = iter.next().unwrap();
                a.merge_with(b, f)
            }
            _ => {
                // Collect non-empty roots with their tree indices.
                let roots: SmallVec<[TaggedNode<KeyType::PartialType, ValueType>; 4]> = trees
                    .into_iter()
                    .enumerate()
                    .filter_map(|(idx, t)| {
                        t.root.map(|r| TaggedNode {
                            node: r,
                            tree_idx: idx,
                        })
                    })
                    .collect();

                if roots.is_empty() {
                    Self::new()
                } else if roots.len() == 1 {
                    Self::from_root(roots.into_iter().next().unwrap().node)
                } else {
                    Self::from_root(Self::merge_n_nodes_with(roots, &mut f))
                }
            }
        }
    }
}

impl<KeyType, ValueType> TreeStatsTrait for AdaptiveRadixTree<KeyType, ValueType>
where
    KeyType: KeyTrait,
{
    fn get_tree_stats(&self) -> TreeStats {
        let mut stats = TreeStats::default();

        if self.root.is_none() {
            return stats;
        }

        AdaptiveRadixTree::<KeyType, ValueType>::get_tree_stats_recurse(
            self.root.as_ref().unwrap(),
            &mut stats,
            1,
        );

        let total_inner_nodes = stats
            .node_stats
            .values()
            .map(|ns| ns.total_nodes)
            .sum::<usize>();
        let mut total_children = 0;
        let mut total_width = 0;
        for ns in stats.node_stats.values_mut() {
            total_children += ns.total_children;
            total_width += ns.width * ns.total_nodes;
            ns.density = ns.total_children as f64 / (ns.width * ns.total_nodes) as f64;
        }
        let total_density = total_children as f64 / total_width as f64;
        stats.num_inner_nodes = total_inner_nodes;
        stats.total_density = total_density;

        stats
    }
}

/// A node paired with its source tree index for right-wins semantics.
/// Higher tree_idx wins in conflicts (last tree in the input list).
struct TaggedNode<P: Partial, V> {
    node: DefaultNode<P, V>,
    tree_idx: usize,
}

// Internals implementation
impl<KeyType, ValueType> AdaptiveRadixTree<KeyType, ValueType>
where
    KeyType: KeyTrait,
{
    // ==================== N-Way Merge Helpers ====================

    /// Find the longest common prefix length among N nodes.
    ///
    /// Uses pairwise reduction: compare first node's prefix against all others,
    /// taking the minimum common length. This is O(N * L) where L is max prefix length.
    fn find_lcp_n(nodes: &[TaggedNode<KeyType::PartialType, ValueType>]) -> usize {
        debug_assert!(!nodes.is_empty(), "find_lcp_n called with empty nodes");

        if nodes.len() == 1 {
            return nodes[0].node.prefix.len();
        }

        let first_prefix = &nodes[0].node.prefix;
        let mut lcp_len = first_prefix.len();

        for tagged in nodes.iter().skip(1) {
            lcp_len = min(
                lcp_len,
                first_prefix.prefix_length_common(&tagged.node.prefix),
            );
            if lcp_len == 0 {
                break; // Early exit: no common prefix.
            }
        }

        lcp_len
    }

    /// Partition N nodes by their relationship to the LCP.
    ///
    /// Returns:
    /// - exact_matches: nodes whose prefix equals the LCP exactly (these are inner nodes)
    /// - extensions: nodes grouped by their divergence byte after the LCP
    /// - lcp: the longest common prefix itself
    fn partition_by_lcp(
        nodes: SmallVec<[TaggedNode<KeyType::PartialType, ValueType>; 4]>,
        lcp_len: usize,
    ) -> (
        SmallVec<[TaggedNode<KeyType::PartialType, ValueType>; 4]>,
        BTreeMap<u8, SmallVec<[TaggedNode<KeyType::PartialType, ValueType>; 4]>>,
        KeyType::PartialType,
    ) {
        // Extract LCP from the first node.
        let lcp = nodes[0].node.prefix.partial_before(lcp_len);

        let mut exact_matches: SmallVec<[TaggedNode<KeyType::PartialType, ValueType>; 4]> =
            SmallVec::new();
        let mut extensions: BTreeMap<
            u8,
            SmallVec<[TaggedNode<KeyType::PartialType, ValueType>; 4]>,
        > = BTreeMap::new();

        for tagged in nodes {
            let prefix_len = tagged.node.prefix.len();
            if prefix_len == lcp_len {
                // Exact match: prefix equals LCP.
                exact_matches.push(tagged);
            } else {
                // Extension: prefix extends beyond LCP.
                let diverge_byte = tagged.node.prefix.at(lcp_len);
                extensions.entry(diverge_byte).or_default().push(tagged);
            }
        }

        (exact_matches, extensions, lcp)
    }

    // ==================== N-Way Merge with Resolver ====================

    /// Recursively merge N nodes into a single node with a conflict resolver.
    fn merge_n_nodes_with<F>(
        nodes: SmallVec<[TaggedNode<KeyType::PartialType, ValueType>; 4]>,
        f: &mut F,
    ) -> DefaultNode<KeyType::PartialType, ValueType>
    where
        F: FnMut(ValueType, ValueType) -> ValueType,
    {
        debug_assert!(
            !nodes.is_empty(),
            "merge_n_nodes_with called with empty nodes"
        );

        // Base case: single node.
        if nodes.len() == 1 {
            return nodes.into_iter().next().unwrap().node;
        }

        // Find the longest common prefix among all nodes.
        let lcp_len = Self::find_lcp_n(&nodes);

        // Check if all nodes are leaves with identical prefixes (same key).
        // In this case, fold values using the resolver.
        let all_same_key_leaves = nodes
            .iter()
            .all(|tagged| tagged.node.is_leaf() && tagged.node.prefix.len() == lcp_len);
        if all_same_key_leaves {
            // Fold values in arbitrary order (no sorting by tree_idx).
            // Callers who need deterministic order should use commutative resolvers.
            let mut iter = nodes.into_iter();
            let first = iter.next().unwrap();
            let prefix = first.node.prefix.clone();
            let mut acc = first.node.into_value().unwrap();
            for tagged in iter {
                acc = f(acc, tagged.node.into_value().unwrap());
            }
            return DefaultNode::new_leaf(prefix, acc);
        }

        // Partition nodes by their relationship to the LCP.
        let (exact_matches, extensions, lcp) = Self::partition_by_lcp(nodes, lcp_len);

        // Build result node with prefix = LCP.
        let capacity = extensions.len() + exact_matches.len()/4;
        let mut result = DefaultNode::new_inner_with_capacity(lcp, capacity);

        // Track max tree_idx for each extension byte.
        let mut extension_tree_idx: BTreeMap<u8, usize> = BTreeMap::new();

        // Handle extension groups.
        for (diverge_byte, mut group) in extensions {
            let max_idx = group.iter().map(|t| t.tree_idx).max().unwrap();
            extension_tree_idx.insert(diverge_byte, max_idx);

            for tagged in &mut group {
                debug_assert!(
                    lcp_len < tagged.node.prefix.len(),
                    "extension node prefix should be longer than LCP"
                );
                tagged.node.prefix = tagged.node.prefix.partial_after(lcp_len);
            }

            let child = if group.len() == 1 {
                group.into_iter().next().unwrap().node
            } else {
                Self::merge_n_nodes_with(group, f)
            };
            result.add_child(diverge_byte, child);
        }

        // Handle exact matches.
        if !exact_matches.is_empty() {
            let merged_children = Self::merge_n_children_with(exact_matches, f);
            for (byte, child, exact_tree_idx) in merged_children {
                if let Some(existing) = result.delete_child(byte) {
                    let ext_tree_idx = extension_tree_idx[&byte];
                    let merged = Self::merge_n_nodes_with(
                        smallvec::smallvec![
                            TaggedNode {
                                node: existing,
                                tree_idx: ext_tree_idx,
                            },
                            TaggedNode {
                                node: child,
                                tree_idx: exact_tree_idx,
                            },
                        ],
                        f,
                    );
                    result.add_child(byte, merged);
                } else {
                    result.add_child(byte, child);
                }
            }
        }

        result
    }

    /// Merge children from N inner nodes with a conflict resolver.
    ///
    /// Uses streaming N-way merge-join over peekable iterators. All child
    /// iterators yield `(byte, child)` pairs in sorted byte order, so we can
    /// advance them in lockstep and collect children sharing the same byte.
    fn merge_n_children_with<F>(
        nodes: SmallVec<[TaggedNode<KeyType::PartialType, ValueType>; 4]>,
        f: &mut F,
    ) -> Vec<(u8, DefaultNode<KeyType::PartialType, ValueType>, usize)>
    where
        F: FnMut(ValueType, ValueType) -> ValueType,
    {
        // Pair each node's child iterator with its tree_idx.
        let mut iters: SmallVec<[_; 4]> = nodes
            .into_iter()
            .map(|tagged| (tagged.node.into_children().peekable(), tagged.tree_idx))
            .collect();

        let mut result = Vec::new();

        loop {
            // Find the minimum byte across all non-exhausted iterators.
            let min_byte = iters
                .iter_mut()
                .filter_map(|(iter, _)| iter.peek().map(|(b, _)| *b))
                .min();

            let Some(current_byte) = min_byte else {
                break;
            };

            // Collect all children at current_byte from any iterator that has it.
            let mut children: SmallVec<[TaggedNode<_, _>; 4]> = SmallVec::new();
            for (iter, tree_idx) in &mut iters {
                if iter.peek().map(|(b, _)| *b) == Some(current_byte) {
                    let (_, child) = iter.next().unwrap();
                    children.push(TaggedNode {
                        node: child,
                        tree_idx: *tree_idx,
                    });
                }
            }

            let max_tree_idx = children.iter().map(|t| t.tree_idx).max().unwrap();
            let merged_child = if children.len() == 1 {
                children.into_iter().next().unwrap().node
            } else {
                Self::merge_n_nodes_with(children, f)
            };

            result.push((current_byte, merged_child, max_tree_idx));
        }

        result
    }

    // ==================== 2-Way Merge Implementation ====================

    /// Merge two nodes whose prefixes diverge.
    /// Create a new parent node with the common prefix, and add both nodes as children.
    #[inline]
    fn merge_divergent(
        mut node1: DefaultNode<KeyType::PartialType, ValueType>,
        mut node2: DefaultNode<KeyType::PartialType, ValueType>,
        common_len: usize,
    ) -> DefaultNode<KeyType::PartialType, ValueType> {
        // Create new parent with the common prefix.
        let common_prefix = node1.prefix.partial_before(common_len);
        let mut result = DefaultNode::new_inner(common_prefix);

        // Get the diverging bytes and truncate both nodes' prefixes.
        let key1 = node1.prefix.at(common_len);
        let key2 = node2.prefix.at(common_len);
        node1.prefix = node1.prefix.partial_after(common_len);
        node2.prefix = node2.prefix.partial_after(common_len);

        // Add both as children. Since prefixes diverged, key1 != key2.
        debug_assert_ne!(key1, key2, "divergent prefixes should have different keys");
        result.add_child(key1, node1);
        result.add_child(key2, node2);

        result
    }

    // ==================== 2-Way Merge with Resolver ====================

    /// Recursively merge two nodes with a conflict resolver.
    #[inline]
    fn merge_nodes_with<F>(
        node1: DefaultNode<KeyType::PartialType, ValueType>,
        node2: DefaultNode<KeyType::PartialType, ValueType>,
        f: &mut F,
    ) -> DefaultNode<KeyType::PartialType, ValueType>
    where
        F: FnMut(ValueType, ValueType) -> ValueType,
    {
        let common_len = node1.prefix.prefix_length_common(&node2.prefix);
        let p1_len = node1.prefix.len();
        let p2_len = node2.prefix.len();

        // Case 1: Prefixes are identical.
        if common_len == p1_len && common_len == p2_len {
            return Self::merge_same_prefix_with(node1, node2, f);
        }

        // Case 2: node1's prefix is a prefix of node2's prefix.
        if common_len == p1_len && p1_len < p2_len {
            return Self::merge_node1_is_prefix_with(node1, node2, common_len, f);
        }

        // Case 3: node2's prefix is a prefix of node1's prefix.
        if common_len == p2_len && p2_len < p1_len {
            return Self::merge_node2_is_prefix_with(node1, node2, common_len, f);
        }

        // Case 4: Prefixes diverge - no conflict possible here.
        Self::merge_divergent(node1, node2, common_len)
    }

    /// Merge two nodes with identical prefixes using a conflict resolver.
    #[inline]
    fn merge_same_prefix_with<F>(
        node1: DefaultNode<KeyType::PartialType, ValueType>,
        node2: DefaultNode<KeyType::PartialType, ValueType>,
        f: &mut F,
    ) -> DefaultNode<KeyType::PartialType, ValueType>
    where
        F: FnMut(ValueType, ValueType) -> ValueType,
    {
        match (node1.is_leaf(), node2.is_leaf()) {
            // Both leaves with same key: use resolver.
            (true, true) => {
                let prefix = node2.prefix.clone();
                let v1 = node1.into_value().unwrap();
                let v2 = node2.into_value().unwrap();
                DefaultNode::new_leaf(prefix, f(v1, v2))
            }

            (true, false) | (false, true) => {
                unreachable!(
                    "leaf/inner with identical prefix should not occur due to null terminators"
                )
            }

            // Both inner nodes: merge children recursively using merge-join.
            (false, false) => {
                let prefix = node1.prefix.clone();
                let capacity = node1.num_children() + node2.num_children();
                let mut result = DefaultNode::new_inner_with_capacity(prefix, capacity);

                let mut iter1 = node1.into_children().peekable();
                let mut iter2 = node2.into_children().peekable();

                loop {
                    match (iter1.peek(), iter2.peek()) {
                        (None, None) => break,
                        (Some(_), None) => {
                            let (k, child) = iter1.next().unwrap();
                            result.add_child(k, child);
                        }
                        (None, Some(_)) => {
                            let (k, child) = iter2.next().unwrap();
                            result.add_child(k, child);
                        }
                        (Some((k1, _)), Some((k2, _))) => match k1.cmp(k2) {
                            Ordering::Less => {
                                let (k, child) = iter1.next().unwrap();
                                result.add_child(k, child);
                            }
                            Ordering::Greater => {
                                let (k, child) = iter2.next().unwrap();
                                result.add_child(k, child);
                            }
                            Ordering::Equal => {
                                let (k, child1) = iter1.next().unwrap();
                                let (_, child2) = iter2.next().unwrap();
                                result.add_child(k, Self::merge_nodes_with(child1, child2, f));
                            }
                        },
                    }
                }

                result
            }
        }
    }

    /// Merge when node1's prefix is a prefix of node2's prefix, with resolver.
    #[inline]
    fn merge_node1_is_prefix_with<F>(
        node1: DefaultNode<KeyType::PartialType, ValueType>,
        mut node2: DefaultNode<KeyType::PartialType, ValueType>,
        common_len: usize,
        f: &mut F,
    ) -> DefaultNode<KeyType::PartialType, ValueType>
    where
        F: FnMut(ValueType, ValueType) -> ValueType,
    {
        if node1.is_leaf() {
            unreachable!(
                "node1 is leaf but is prefix of node2: should not occur due to null terminators"
            )
        }

        let extra_key = node2.prefix.at(common_len);
        node2.prefix = node2.prefix.partial_after(common_len);

        let prefix = node1.prefix.clone();
        let capacity = node1.num_children() + 1;
        let mut result = DefaultNode::new_inner_with_capacity(prefix, capacity);

        let mut iter1 = node1.into_children().peekable();
        let mut extra = Some((extra_key, node2));

        loop {
            match (iter1.peek(), extra.as_ref()) {
                (None, None) => break,
                (Some(_), None) => {
                    let (k, child) = iter1.next().unwrap();
                    result.add_child(k, child);
                }
                (None, Some(_)) => {
                    let (k, child) = extra.take().unwrap();
                    result.add_child(k, child);
                }
                (Some((k1, _)), Some((ek, _))) => match k1.cmp(ek) {
                    Ordering::Less => {
                        let (k, child) = iter1.next().unwrap();
                        result.add_child(k, child);
                    }
                    Ordering::Greater => {
                        let (k, child) = extra.take().unwrap();
                        result.add_child(k, child);
                    }
                    Ordering::Equal => {
                        let (k, child1) = iter1.next().unwrap();
                        let (_, child2) = extra.take().unwrap();
                        result.add_child(k, Self::merge_nodes_with(child1, child2, f));
                    }
                },
            }
        }

        result
    }

    /// Merge when node2's prefix is a prefix of node1's prefix, with resolver.
    #[inline]
    fn merge_node2_is_prefix_with<F>(
        mut node1: DefaultNode<KeyType::PartialType, ValueType>,
        node2: DefaultNode<KeyType::PartialType, ValueType>,
        common_len: usize,
        f: &mut F,
    ) -> DefaultNode<KeyType::PartialType, ValueType>
    where
        F: FnMut(ValueType, ValueType) -> ValueType,
    {
        if node2.is_leaf() {
            unreachable!(
                "node2 is leaf but is prefix of node1: should not occur due to null terminators"
            )
        }

        let extra_key = node1.prefix.at(common_len);
        node1.prefix = node1.prefix.partial_after(common_len);

        let prefix = node2.prefix.clone();
        let capacity = node2.num_children() + 1;
        let mut result = DefaultNode::new_inner_with_capacity(prefix, capacity);

        let mut iter2 = node2.into_children().peekable();
        let mut extra = Some((extra_key, node1));

        loop {
            match (iter2.peek(), extra.as_ref()) {
                (None, None) => break,
                (Some(_), None) => {
                    let (k, child) = iter2.next().unwrap();
                    result.add_child(k, child);
                }
                (None, Some(_)) => {
                    let (k, child) = extra.take().unwrap();
                    result.add_child(k, child);
                }
                (Some((k2, _)), Some((ek, _))) => match k2.cmp(ek) {
                    Ordering::Less => {
                        let (k, child) = iter2.next().unwrap();
                        result.add_child(k, child);
                    }
                    Ordering::Greater => {
                        let (k, child) = extra.take().unwrap();
                        result.add_child(k, child);
                    }
                    Ordering::Equal => {
                        let (k, child2) = iter2.next().unwrap();
                        let (_, child1) = extra.take().unwrap();
                        result.add_child(k, Self::merge_nodes_with(child1, child2, f));
                    }
                },
            }
        }

        result
    }

    fn get_iterate<'a>(
        cur_node: &'a DefaultNode<KeyType::PartialType, ValueType>,
        key: &KeyType,
    ) -> Option<&'a ValueType> {
        let mut cur_node = cur_node;
        let mut depth = 0;
        loop {
            let prefix_common_match = cur_node.prefix.prefix_length_key(key, depth);
            if prefix_common_match != cur_node.prefix.len() {
                return None;
            }

            if cur_node.prefix.len() == key.length_at(depth) {
                return cur_node.value();
            }
            let k = key.at(depth + cur_node.prefix.len());
            depth += cur_node.prefix.len();
            cur_node = cur_node.seek_child(k)?
        }
    }

    fn longest_prefix_match_iterate<'a>(
        cur_node: &'a DefaultNode<KeyType::PartialType, ValueType>,
        key: &KeyType,
    ) -> Option<(KeyType, &'a ValueType)> {
        let mut cur_node = cur_node;
        let mut cur_key = KeyType::new_from_partial(&cur_node.prefix);
        let mut best_match = cur_node.value().map(|value| (cur_key.clone(), value));
        let mut depth = 0;

        loop {
            let prefix_common_match = cur_node.prefix.prefix_length_key(key, depth);
            if prefix_common_match != cur_node.prefix.len() {
                return best_match;
            }

            if let Some(value) = cur_node.value() {
                best_match = Some((cur_key.clone(), value));
            }

            if cur_node.prefix.len() == key.length_at(depth) {
                return best_match;
            }

            let k = key.at(depth + cur_node.prefix.len());
            depth += cur_node.prefix.len();

            let Some(child) = cur_node.seek_child(k) else {
                return best_match;
            };
            cur_node = child;
            cur_key = cur_key.extend_from_partial(&cur_node.prefix);
        }
    }

    fn find_prefix_subtree<'a>(
        cur_node: &'a DefaultNode<KeyType::PartialType, ValueType>,
        prefix: &KeyType,
    ) -> Option<(&'a DefaultNode<KeyType::PartialType, ValueType>, KeyType)> {
        let mut cur_node = cur_node;
        let mut cur_key = KeyType::new_from_partial(&cur_node.prefix);
        let mut depth = 0;

        loop {
            let prefix_common_match = cur_node.prefix.prefix_length_key(prefix, depth);
            if prefix_common_match != cur_node.prefix.len() {
                if prefix_common_match == prefix.length_at(depth) {
                    return Some((cur_node, cur_key));
                }
                return None;
            }

            if cur_node.prefix.len() == prefix.length_at(depth) {
                return Some((cur_node, cur_key));
            }

            let k = prefix.at(depth + cur_node.prefix.len());
            depth += cur_node.prefix.len();

            let child = cur_node.seek_child(k)?;
            cur_node = child;
            cur_key = cur_key.extend_from_partial(&cur_node.prefix);
        }
    }

    fn get_iterate_mut<'a>(
        cur_node: &'a mut DefaultNode<KeyType::PartialType, ValueType>,
        key: &KeyType,
    ) -> Option<&'a mut ValueType> {
        let mut cur_node = cur_node;
        let mut depth = 0;
        loop {
            let prefix_common_match = cur_node.prefix.prefix_length_key(key, depth);
            if prefix_common_match != cur_node.prefix.len() {
                return None;
            }

            if cur_node.prefix.len() == key.length_at(depth) {
                return cur_node.value_mut();
            }

            let k = key.at(depth + cur_node.prefix.len());
            depth += cur_node.prefix.len();
            cur_node = cur_node.seek_child_mut(k)?;
        }
    }

    fn insert_recurse(
        cur_node: &mut DefaultNode<KeyType::PartialType, ValueType>,
        key: &KeyType,
        value: ValueType,
        depth: usize,
    ) -> Option<ValueType> {
        let longest_common_prefix = cur_node.prefix.prefix_length_key(key, depth);

        let is_prefix_match =
            min(cur_node.prefix.len(), key.length_at(depth)) == longest_common_prefix;

        // Prefix fully covers this node.
        // Either sets the value or replaces the old value already here.
        if is_prefix_match
            && cur_node.prefix.len() == key.length_at(depth)
            && let Content::Leaf(v) = &mut cur_node.content
        {
            return Some(std::mem::replace(v, value));
        }

        // Prefix is part of the current node, but doesn't fully cover it.
        // We have to break this node up, creating a new parent node, and a sibling for our leaf.
        if !is_prefix_match {
            let new_prefix = cur_node.prefix.partial_after(longest_common_prefix);
            let old_node_prefix = std::mem::replace(&mut cur_node.prefix, new_prefix);

            // We will replace this leaf node with a new inner node. The new value will join the
            // current node as sibling, both a child of the new node.
            let n4 = DefaultNode::new_inner(old_node_prefix.partial_before(longest_common_prefix));

            let k1 = old_node_prefix.at(longest_common_prefix);
            let k2 = key.at(depth + longest_common_prefix);

            let replacement_current = std::mem::replace(cur_node, n4);

            // We've deferred creating the leaf til now so that we can take ownership over the
            // key after other things are done peering at it.
            let new_leaf =
                DefaultNode::new_leaf(key.to_partial(depth + longest_common_prefix), value);

            // Add the old leaf node as a child of the new inner node.
            cur_node.add_child(k1, replacement_current);
            cur_node.add_child(k2, new_leaf);

            return None;
        }

        // We must be an inner node, and either we need a new baby, or one of our children does, so
        // we'll hunt and see.
        let k = key.at(depth + longest_common_prefix);

        let Some(child) = cur_node.seek_child_mut(k) else {
            // We should not be a leaf at this point. If so, something bad has happened.
            debug_assert!(cur_node.is_inner());
            let new_leaf =
                DefaultNode::new_leaf(key.to_partial(depth + longest_common_prefix), value);
            cur_node.add_child(k, new_leaf);
            return None;
        };

        AdaptiveRadixTree::insert_recurse(child, key, value, depth + longest_common_prefix)
    }

    fn remove_recurse(
        parent_node: &mut DefaultNode<KeyType::PartialType, ValueType>,
        key: &KeyType,
        depth: usize,
    ) -> Option<ValueType> {
        // Seek the child that matches the key at this depth, which is the first character at the
        // depth we're at.
        let c = key.at(depth);
        let child_node = parent_node.seek_child_mut(c)?;

        let prefix_common_match = child_node.prefix.prefix_length_key(key, depth);
        if prefix_common_match != child_node.prefix.len() {
            return None;
        }

        // If the child is a leaf, and the prefix matches the key, we can remove it from this parent
        // node. If the prefix does not match, then we have nothing to do here.
        if child_node.is_leaf() {
            if child_node.prefix.len() != (key.length_at(depth)) {
                return None;
            }
            let node = parent_node.delete_child(c).unwrap();
            let v = match node.content {
                Content::Leaf(v) => v,
                _ => unreachable!(),
            };
            return Some(v);
        }

        // Otherwise, recurse down the branch in that direction.
        let result =
            AdaptiveRadixTree::remove_recurse(child_node, key, depth + child_node.prefix.len());

        // If after this our child we just recursed into no longer has children of its own, it can
        // be collapsed into us. In this way we can prune the tree as we go.
        if result.is_some() && child_node.is_inner() && child_node.num_children() == 0 {
            let prefix = child_node.prefix.clone();
            let deleted = parent_node.delete_child(c).unwrap();
            debug_assert_eq!(prefix.to_slice(), deleted.prefix.to_slice());
        }

        result
    }

    fn get_tree_stats_recurse(
        node: &DefaultNode<KeyType::PartialType, ValueType>,
        tree_stats: &mut TreeStats,
        height: usize,
    ) {
        if height > tree_stats.max_height {
            tree_stats.max_height = height;
        }
        if node.value().is_some() {
            tree_stats.num_values += 1;
        }
        match node.content {
            Content::Leaf(_) => {
                tree_stats.num_leaves += 1;
            }
            _ => {
                update_tree_stats(tree_stats, node);
            }
        }
        for (_k, child) in node.iter() {
            AdaptiveRadixTree::<KeyType, ValueType>::get_tree_stats_recurse(
                child,
                tree_stats,
                height + 1,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, btree_map};
    use std::fmt::Debug;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, rng};

    use crate::keys::KeyTrait;
    use crate::keys::array_key::ArrayKey;
    use crate::keys::vector_key::VectorKey;
    use crate::partials::array_partial::ArrPartial;
    use crate::stats::TreeStatsTrait;
    use crate::tree;
    use crate::tree::AdaptiveRadixTree;

    static PANIC_ON_FOUR_CMP: AtomicBool = AtomicBool::new(false);
    static PANIC_ON_BELOW_M_CMP: AtomicBool = AtomicBool::new(false);
    static PANIC_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Clone, Eq, PartialEq, Debug)]
    struct PanickyRangeKey(ArrayKey<16>);

    impl PanickyRangeKey {
        fn as_u64(&self) -> u64 {
            self.0.to_be_u64()
        }
    }

    impl AsRef<[u8]> for PanickyRangeKey {
        fn as_ref(&self) -> &[u8] {
            self.0.as_ref()
        }
    }

    impl PartialOrd for PanickyRangeKey {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for PanickyRangeKey {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            if PANIC_ON_FOUR_CMP.load(Ordering::Relaxed)
                && (self.as_u64() == 4 || other.as_u64() == 4)
            {
                panic!("range compared past first out-of-range key");
            }
            if PANIC_ON_BELOW_M_CMP.load(Ordering::Relaxed) {
                let lhs = self.as_ref().first().copied().unwrap_or_default();
                let rhs = other.as_ref().first().copied().unwrap_or_default();
                if lhs < b'm' || rhs < b'm' {
                    panic!("range start seek compared a key below start prefix");
                }
            }
            self.0.cmp(&other.0)
        }
    }

    impl From<u64> for PanickyRangeKey {
        fn from(value: u64) -> Self {
            Self(value.into())
        }
    }

    impl From<&str> for PanickyRangeKey {
        fn from(value: &str) -> Self {
            Self(value.into())
        }
    }

    impl From<PanickyRangeKey> for ArrPartial<16> {
        fn from(value: PanickyRangeKey) -> Self {
            value.0.to_partial(0)
        }
    }

    impl KeyTrait for PanickyRangeKey {
        type PartialType = ArrPartial<16>;
        const MAXIMUM_SIZE: Option<usize> = Some(16);

        fn new_from_slice(slice: &[u8]) -> Self {
            Self(ArrayKey::new_from_slice(slice))
        }

        fn new_from_partial(partial: &Self::PartialType) -> Self {
            Self(ArrayKey::new_from_partial(partial))
        }

        fn extend_from_partial(&self, partial: &Self::PartialType) -> Self {
            Self(self.0.extend_from_partial(partial))
        }

        fn truncate(&self, at_depth: usize) -> Self {
            Self(self.0.truncate(at_depth))
        }

        fn at(&self, pos: usize) -> u8 {
            self.0.at(pos)
        }

        fn length_at(&self, at_depth: usize) -> usize {
            self.0.length_at(at_depth)
        }

        fn to_partial(&self, at_depth: usize) -> Self::PartialType {
            self.0.to_partial(at_depth)
        }

        fn matches_slice(&self, slice: &[u8]) -> bool {
            self.0.matches_slice(slice)
        }
    }

    #[test]
    fn test_root_set_get() {
        let mut q = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let key: ArrayKey<16> = "abc".into();
        assert!(q.insert("abc", 1).is_none());
        assert_eq!(q.get_k(&key), Some(&1));
    }

    #[test]
    fn test_string_keys_get_set() {
        let mut q = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        q.insert("abcd", 1);
        q.insert("abc", 2);
        q.insert("abcde", 3);
        q.insert("xyz", 4);
        q.insert("xyz", 5);
        q.insert("axyz", 6);
        q.insert("1245zzz", 6);

        assert_eq!(*q.get("abcd").unwrap(), 1);
        assert_eq!(*q.get("abc").unwrap(), 2);
        assert_eq!(*q.get("abcde").unwrap(), 3);
        assert_eq!(*q.get("axyz").unwrap(), 6);
        assert_eq!(*q.get("xyz").unwrap(), 5);

        assert_eq!(q.remove("abcde"), Some(3));
        assert_eq!(q.get("abcde"), None);
        assert_eq!(*q.get("abc").unwrap(), 2);
        assert_eq!(*q.get("axyz").unwrap(), 6);
        assert_eq!(q.remove("abc"), Some(2));
        assert_eq!(q.get("abc"), None);
    }

    #[test]
    fn test_int_keys_get_set() {
        let mut q = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        q.insert_k(&500i32.into(), 3);
        assert_eq!(q.get_k(&500i32.into()), Some(&3));
        q.insert_k(&666i32.into(), 2);
        assert_eq!(q.get_k(&666i32.into()), Some(&2));
        q.insert_k(&1i32.into(), 1);
        assert_eq!(q.get_k(&1i32.into()), Some(&1));
    }

    fn gen_random_string_keys<const S: usize>(
        l1_prefix: usize,
        l2_prefix: usize,
        suffix: usize,
    ) -> Vec<(ArrayKey<S>, String)> {
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

    #[test]
    fn test_bulk_random_string_query() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, String>::new();
        let keys = gen_random_string_keys(3, 2, 3);
        let mut num_inserted = 0;
        for key in keys.iter() {
            if tree.insert_k(&key.0, key.1.clone()).is_none() {
                num_inserted += 1;
                assert!(tree.get_k(&key.0).is_some());
            }
        }
        let mut rng = rng();
        for _i in 0..5_000_000 {
            let entry = &keys[rng.random_range(0..keys.len())];
            let val = tree.get_k(&entry.0);
            debug_assert!(val.is_some());
            debug_assert_eq!(*val.unwrap(), entry.1);
        }

        let stats = tree.get_tree_stats();
        debug_assert_eq!(stats.num_values, num_inserted);
    }

    #[test]
    fn test_random_numeric_insert_get() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let count = 9_000_000;
        let mut rng = rng();
        let mut keys_inserted = vec![];
        for i in 0..count {
            let value = i;
            let rnd_key = rng.random_range(0..count);
            if tree.get(rnd_key).is_none() && tree.insert(rnd_key, value).is_none() {
                let result = tree.get(rnd_key);
                assert!(result.is_some());
                assert_eq!(*result.unwrap(), value);
                keys_inserted.push((rnd_key, value));
            }
        }

        let stats = tree.get_tree_stats();
        debug_assert_eq!(stats.num_values, keys_inserted.len());

        for (key, value) in &keys_inserted {
            let result = tree.get(key);
            debug_assert!(result.is_some(),);
            debug_assert_eq!(*result.unwrap(), *value,);
        }
    }

    #[test]
    fn test_iter() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let count = 100000;
        let mut rng = rng();
        let mut keys_inserted = BTreeSet::new();
        for i in 0..count {
            let _value = i;
            let rnd_val = rng.random_range(0..count);
            let rnd_key: ArrayKey<16> = rnd_val.into();
            if tree.get_k(&rnd_key).is_none() && tree.insert_k(&rnd_key, rnd_val).is_none() {
                let result = tree.get_k(&rnd_key);
                assert!(result.is_some());
                assert_eq!(*result.unwrap(), rnd_val);
                keys_inserted.insert((rnd_val, rnd_val));
            }
        }

        // Iteration of keys_inserted and tree should be *roughly* the same, but the iteration order
        // within a KeyedMapping is not guaranteed to be lexicographical, so we can't compare
        // directly.
        let mut tree_iter = tree.iter();
        let keys_inserted_iter = keys_inserted.iter();
        for btree_entry in keys_inserted_iter {
            let art_entry = tree_iter.next();
            debug_assert!(art_entry.is_some());
            let art_entry = art_entry.unwrap();
            debug_assert_eq!(*art_entry.1, btree_entry.1);
            let art_key = art_entry.0.to_be_u64();
            debug_assert_eq!(art_key, btree_entry.0);
        }
    }

    #[test]
    fn test_iter_one_regression() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        tree.insert(123, 456);
        let mut iter = tree.iter();
        let result = iter.next().expect("Expected an entry");
        assert_eq!(result.1, &456)
    }

    #[test]
    fn test_prefix_iter_returns_sorted_prefix_subset() {
        let mut tree = AdaptiveRadixTree::<VectorKey, i32>::new();
        tree.insert_k(&VectorKey::new_from_slice(b"alpha1"), 1);
        tree.insert_k(&VectorKey::new_from_slice(b"alpha2"), 2);
        tree.insert_k(&VectorKey::new_from_slice(b"alphabet"), 3);
        tree.insert_k(&VectorKey::new_from_slice(b"alpine"), 4);
        tree.insert_k(&VectorKey::new_from_slice(b"beta"), 5);

        let prefix = VectorKey::new_from_slice(b"alp");
        let got: Vec<(String, i32)> = tree
            .prefix_iter_k(&prefix)
            .map(|(k, v)| {
                (
                    String::from_utf8(k.as_ref().to_vec()).expect("key must be valid UTF-8"),
                    *v,
                )
            })
            .collect();

        assert_eq!(
            got,
            vec![
                ("alpha1".to_string(), 1),
                ("alpha2".to_string(), 2),
                ("alphabet".to_string(), 3),
                ("alpine".to_string(), 4),
            ]
        );
    }

    #[test]
    fn test_prefix_iter_no_match() {
        let mut tree = AdaptiveRadixTree::<VectorKey, i32>::new();
        tree.insert_k(&VectorKey::new_from_slice(b"alpha"), 1);
        tree.insert_k(&VectorKey::new_from_slice(b"beta"), 2);

        let prefix = VectorKey::new_from_slice(b"zzz");
        assert_eq!(tree.prefix_iter_k(&prefix).count(), 0);
    }

    #[test]
    fn test_prefix_iter_short_prefix_regression() {
        let mut tree = AdaptiveRadixTree::<VectorKey, i32>::new();
        tree.insert_k(&VectorKey::new_from_slice(&[0x01, 0x02, b'a']), 1);
        tree.insert_k(&VectorKey::new_from_slice(&[0x01, 0x02, b'b']), 2);
        tree.insert_k(&VectorKey::new_from_slice(&[0x01, 0x03, b'c']), 3);

        let prefix = VectorKey::new_from_slice(&[0x01, 0x02]);
        let got: Vec<i32> = tree.prefix_iter_k(&prefix).map(|(_, v)| *v).collect();
        assert_eq!(got, vec![1, 2]);
    }

    #[test]
    fn test_longest_prefix_match() {
        let mut tree = AdaptiveRadixTree::<VectorKey, i32>::new();
        tree.insert_k(&VectorKey::new_from_slice(b"cat"), 10);
        tree.insert_k(&VectorKey::new_from_slice(b"dog"), 20);

        let (matched_key, matched_value) = tree
            .longest_prefix_match(VectorKey::new_from_slice(b"catalog"))
            .expect("expected a prefix match");
        assert_eq!(matched_key.as_ref(), b"cat");
        assert_eq!(*matched_value, 10);

        let (matched_key, matched_value) = tree
            .longest_prefix_match(VectorKey::new_from_slice(b"dog"))
            .expect("expected exact match");
        assert_eq!(matched_key.as_ref(), b"dog");
        assert_eq!(*matched_value, 20);

        let (matched_key, matched_value) = tree
            .longest_prefix_match(VectorKey::new_from_slice(b"doge"))
            .expect("expected prefix match");
        assert_eq!(matched_key.as_ref(), b"dog");
        assert_eq!(*matched_value, 20);

        assert!(
            tree.longest_prefix_match(VectorKey::new_from_slice(b"do"))
                .is_none()
        );
        assert!(
            tree.longest_prefix_match(VectorKey::new_from_slice(b"zebra"))
                .is_none()
        );
    }

    #[test]
    // The following cases were found by fuzzing, and identified bugs in `remove`
    fn test_delete_regressions() {
        // DO_INSERT,12297829382473034287,72245244022401706
        // DO_INSERT,12297829382473034410,5425513372477729450
        // DO_DELETE,12297829382473056255,Some(5425513372477729450),None
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, usize>::new();
        assert!(
            tree.insert(12297829382473034287usize, 72245244022401706usize)
                .is_none()
        );
        assert!(
            tree.insert(12297829382473034410usize, 5425513372477729450usize)
                .is_none()
        );
        // assert!(tree.remove(&ArrayKey::new_from_unsigned(12297829382473056255usize)).is_none());

        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, usize>::new();
        // DO_INSERT,0,8101975729639522304
        // DO_INSERT,4934144,18374809624973934592
        // DO_DELETE,0,None,Some(8101975729639522304)
        assert!(tree.insert(0usize, 8101975729639522304usize).is_none());
        assert!(
            tree.insert(4934144usize, 18374809624973934592usize)
                .is_none()
        );
        assert_eq!(tree.get(0usize), Some(&8101975729639522304usize));
        assert_eq!(tree.remove(0usize), Some(8101975729639522304usize));
        assert_eq!(tree.get(4934144usize), Some(&18374809624973934592usize));

        // DO_INSERT,8102098874941833216,8101975729639522416
        // DO_INSERT,8102099357864587376,18374810107896688752
        // DO_DELETE,0,Some(8101975729639522416),None
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, usize>::new();
        assert!(
            tree.insert(8102098874941833216usize, 8101975729639522416usize)
                .is_none()
        );
        assert!(
            tree.insert(8102099357864587376usize, 18374810107896688752usize)
                .is_none()
        );
        assert_eq!(tree.get(0usize), None);
        assert_eq!(tree.remove(0usize), None);
    }

    #[test]
    fn test_insert_returns_replaced_value() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        // Insert new key should return None
        assert_eq!(tree.insert("key1", 100), None);
        assert_eq!(tree.get("key1"), Some(&100));

        // Insert same key should return previous value
        assert_eq!(tree.insert("key1", 200), Some(100));
        assert_eq!(tree.get("key1"), Some(&200));

        // Insert same key again should return current value
        assert_eq!(tree.insert("key1", 300), Some(200));
        assert_eq!(tree.get("key1"), Some(&300));

        // Insert different key should return None
        assert_eq!(tree.insert("key2", 400), None);
        assert_eq!(tree.get("key2"), Some(&400));

        // Original key should still have latest value
        assert_eq!(tree.get("key1"), Some(&300));
    }

    #[test]
    fn test_delete() {
        // Insert a bunch of random keys and values into both a btree and our tree, then iterate
        // over the btree and delete the keys from our tree. Then, iterate over our tree and make
        // sure it's empty.
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let mut btree = BTreeMap::new();
        let count = 5_000;
        let mut rng = rng();
        for i in 0..count {
            let _value = i;
            let rnd_val = rng.random_range(0..u64::MAX);
            let rnd_key: ArrayKey<16> = rnd_val.into();
            tree.insert_k(&rnd_key, rnd_val);
            btree.insert(rnd_val, rnd_val);
        }

        for (key, value) in btree.iter() {
            let key: ArrayKey<16> = (*key).into();
            let get_result = tree.get_k(&key);
            debug_assert_eq!(
                get_result.cloned(),
                Some(*value),
                "Key with prefix {:?} not found in tree; it should be",
                key.to_partial(0).to_slice()
            );
            let result = tree.remove_k(&key);
            debug_assert_eq!(result, Some(*value));
        }
    }
    // Compare the results of a range query on an AdaptiveRadixTree and a BTreeMap, because we can
    // safely assume the latter exhibits correct behavior.
    fn test_range_matches<'a, KeyType: KeyTrait, ValueType: PartialEq + Debug + 'a>(
        art_range: tree::Range<'a, KeyType, ValueType>,
        btree_range: btree_map::Range<'a, u64, ValueType>,
    ) {
        // collect both into vectors then compare
        let art_values = art_range.map(|(_, v)| v).collect::<Vec<_>>();
        let btree_values = btree_range.map(|(_, v)| v).collect::<Vec<_>>();
        debug_assert_eq!(art_values.len(), btree_values.len());
        debug_assert_eq!(art_values, btree_values);
    }

    #[test]
    fn test_range() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let count = 10000;
        let mut rng = rng();
        let mut keys_inserted = BTreeMap::new();
        for i in 0..count {
            let _value = i;
            let rnd_val = rng.random_range(0..count);
            let rnd_key: ArrayKey<16> = rnd_val.into();
            if tree.get_k(&rnd_key).is_none() && tree.insert_k(&rnd_key, rnd_val).is_none() {
                let result = tree.get_k(&rnd_key);
                assert!(result.is_some());
                assert_eq!(*result.unwrap(), rnd_val);
                keys_inserted.insert(rnd_val, rnd_val);
            }
        }

        // Test for range with unbounded start and exclusive end
        let end_key: ArrayKey<16> = 100u64.into();
        let t_r = tree.range(..end_key);
        let k_r = keys_inserted.range(..100);
        test_range_matches(t_r, k_r);

        // Test for range with unbounded start and inclusive end.
        let t_r = tree.range(..=end_key);
        let k_r = keys_inserted.range(..=100);
        test_range_matches(t_r, k_r);

        // Test for range with unbounded end and exclusive start
        let start_key: ArrayKey<16> = 100u64.into();
        let t_r = tree.range(start_key..);
        let k_r = keys_inserted.range(100..);
        test_range_matches(t_r, k_r);

        // Test for range with bounded start and end (exclusive)
        let end_key: ArrayKey<16> = 1000u64.into();
        let t_r = tree.range(start_key..end_key);
        let k_r = keys_inserted.range(100..1000);
        test_range_matches(t_r, k_r);

        // Test for range with bounded start and end (inclusive)
        let t_r = tree.range(start_key..=end_key);
        let k_r = keys_inserted.range(100..=1000);
        test_range_matches(t_r, k_r);
    }

    #[test]
    fn test_range_stops_after_first_out_of_bounds_regression() {
        let _guard = PANIC_TEST_LOCK.lock().unwrap();
        PANIC_ON_FOUR_CMP.store(false, Ordering::Relaxed);
        PANIC_ON_BELOW_M_CMP.store(false, Ordering::Relaxed);
        let mut tree = AdaptiveRadixTree::<PanickyRangeKey, u64>::new();
        for i in 0..=4u64 {
            let key: PanickyRangeKey = i.into();
            tree.insert_k(&key, i);
        }

        let end: PanickyRangeKey = 2u64.into();
        PANIC_ON_FOUR_CMP.store(true, Ordering::Relaxed);
        let results: Vec<u64> = tree.range(..=end).map(|(_, v)| *v).collect();
        PANIC_ON_FOUR_CMP.store(false, Ordering::Relaxed);
        PANIC_ON_BELOW_M_CMP.store(false, Ordering::Relaxed);

        assert_eq!(results, vec![0, 1, 2]);
    }

    #[test]
    fn test_range_start_seek_regression() {
        let _guard = PANIC_TEST_LOCK.lock().unwrap();
        PANIC_ON_FOUR_CMP.store(false, Ordering::Relaxed);
        PANIC_ON_BELOW_M_CMP.store(false, Ordering::Relaxed);
        let mut tree = AdaptiveRadixTree::<PanickyRangeKey, u64>::new();
        for (i, c) in ('a'..='z').enumerate() {
            let key: PanickyRangeKey = format!("{c}key").as_str().into();
            tree.insert_k(&key, i as u64);
        }

        let start: PanickyRangeKey = "m".into();
        PANIC_ON_BELOW_M_CMP.store(true, Ordering::Relaxed);
        let collected: Vec<u64> = tree.range(start..).map(|(_, v)| *v).collect();
        PANIC_ON_BELOW_M_CMP.store(false, Ordering::Relaxed);
        PANIC_ON_FOUR_CMP.store(false, Ordering::Relaxed);

        let expected: Vec<u64> = (12..=25).collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_range_start_sequence_matches_btreemap_seeded() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let mut btree = BTreeMap::new();
        let mut rng = StdRng::seed_from_u64(0x5eed_5eed_dead_beef);
        const COUNT: usize = 20_000;
        const SPACE: u64 = 80_000;

        for _ in 0..COUNT {
            let k = rng.random_range(0..SPACE);
            tree.insert_k(&ArrayKey::<16>::from(k), k * 3 + 7);
            btree.insert(k, k * 3 + 7);
        }

        let start_raw = rng.random_range(0..SPACE);
        let start_key: ArrayKey<16> = start_raw.into();

        let art_values: Vec<u64> = tree.range(start_key..).map(|(_, v)| *v).collect();
        let btree_values: Vec<u64> = btree.range(start_raw..).map(|(_, v)| *v).collect();

        assert_eq!(art_values, btree_values);
    }

    #[test]
    fn test_range_to_inclusive_fuzz_regression() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        tree.insert(248u64, 7_800_515_995_666_006_788);
        tree.insert(2_678_072_818_765_473_061u64, 2_387_225_703_656_202_751);
        tree.insert(16_100_209_717_274_439_535u64, 8_027_225_910_236_114_799);
        tree.insert(6_196_794_136_686_718_831u64, 18_446_744_073_709_514_607);
        tree.insert(12_219_677_559_081_489_409u64, 4_683_546_028_065_928_715);

        let end: ArrayKey<16> = 67_478_703_180u64.into();
        let got: Vec<u64> = tree.range(..=end).map(|(_, v)| *v).collect();

        assert_eq!(got, vec![7_800_515_995_666_006_788]);
    }

    #[test]
    fn test_range_from_fuzz_regression() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let mut btree = BTreeMap::<u64, u64>::new();

        let pairs = [
            (3_124_419_705_906_079_527u64, 3_110_813_966_761_929_515u64),
            (18_446_505_647_410_981_675u64, 23_171_125_240_484_607u64),
            (14_251_014_049_101_104_581u64, 18_446_743_327_766_348_229u64),
            (2_882_303_757_842_906_925u64, 71_779_585_756_702_509u64),
            (12_297_829_382_473_187_410u64, 682u64),
        ];

        for (k, v) in pairs {
            tree.insert(k, v);
            btree.insert(k, v);
        }

        let start_raw = 5_931_894_175_636_062_208u64;
        let start_key: ArrayKey<16> = start_raw.into();

        let art_values: Vec<u64> = tree.range(start_key..).map(|(_, v)| *v).collect();
        let btree_values: Vec<u64> = btree.range(start_raw..).map(|(_, v)| *v).collect();

        assert_eq!(art_values, btree_values);
    }

    // ==================== Merge Tests ====================

    #[test]
    fn test_merge_disjoint_keys() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 1);
        tree1.insert("banana", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("cherry", 3);
        tree2.insert("date", 4);

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("apple"), Some(&1));
        assert_eq!(merged.get("banana"), Some(&2));
        assert_eq!(merged.get("cherry"), Some(&3));
        assert_eq!(merged.get("date"), Some(&4));
        assert_eq!(merged.iter().count(), 4);
    }

    #[test]
    fn test_merge_identical_keys() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 1);
        tree1.insert("banana", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("apple", 10);
        tree2.insert("banana", 20);

        let merged = tree1.merge(tree2);

        // Right wins: tree2's values should be used.
        assert_eq!(merged.get("apple"), Some(&10));
        assert_eq!(merged.get("banana"), Some(&20));
        assert_eq!(merged.iter().count(), 2);
    }

    #[test]
    fn test_merge_partial_overlap() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 1);
        tree1.insert("banana", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("apple", 10); // Overwrites.
        tree2.insert("cherry", 3); // New key.

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("apple"), Some(&10)); // From tree2.
        assert_eq!(merged.get("banana"), Some(&2)); // From tree1.
        assert_eq!(merged.get("cherry"), Some(&3)); // From tree2.
        assert_eq!(merged.iter().count(), 3);
    }

    #[test]
    fn test_merge_empty_into_nonempty() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 1);
        tree1.insert("banana", 2);

        let tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("apple"), Some(&1));
        assert_eq!(merged.get("banana"), Some(&2));
        assert_eq!(merged.iter().count(), 2);
    }

    #[test]
    fn test_merge_nonempty_into_empty() {
        let tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("apple", 1);
        tree2.insert("banana", 2);

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("apple"), Some(&1));
        assert_eq!(merged.get("banana"), Some(&2));
        assert_eq!(merged.iter().count(), 2);
    }

    #[test]
    fn test_merge_both_empty() {
        let tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        let merged = tree1.merge(tree2);

        assert!(merged.is_empty());
        assert_eq!(merged.iter().count(), 0);
    }

    #[test]
    fn test_merge_single_key_each() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("a", 1);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("b", 2);

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("a"), Some(&1));
        assert_eq!(merged.get("b"), Some(&2));
        assert_eq!(merged.iter().count(), 2);
    }

    #[test]
    fn test_merge_prefix_keys() {
        // Keys like "app" and "apple" share a prefix but are different keys.
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("app", 1);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("apple", 2);

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("app"), Some(&1));
        assert_eq!(merged.get("apple"), Some(&2));
        assert_eq!(merged.iter().count(), 2);
    }

    #[test]
    fn test_merge_shared_prefix_different_values() {
        // Both trees have keys sharing the same prefix structure.
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 1);
        tree1.insert("application", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("apple", 10); // Overwrite.
        tree2.insert("apricot", 3); // New key with shared "ap" prefix.

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("apple"), Some(&10)); // tree2 wins.
        assert_eq!(merged.get("application"), Some(&2)); // From tree1.
        assert_eq!(merged.get("apricot"), Some(&3)); // From tree2.
        assert_eq!(merged.iter().count(), 3);
    }

    #[test]
    fn test_merge_deep_prefix_overlap() {
        // Keys with very long shared prefixes.
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();
        tree1.insert("aaaaaaaaaaaaaaa1", 1);
        tree1.insert("aaaaaaaaaaaaaaa2", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();
        tree2.insert("aaaaaaaaaaaaaaa1", 10); // Overwrite.
        tree2.insert("aaaaaaaaaaaaaaa3", 3); // New key with same prefix.

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("aaaaaaaaaaaaaaa1"), Some(&10));
        assert_eq!(merged.get("aaaaaaaaaaaaaaa2"), Some(&2));
        assert_eq!(merged.get("aaaaaaaaaaaaaaa3"), Some(&3));
        assert_eq!(merged.iter().count(), 3);
    }

    #[test]
    fn test_merge_vector_key_basic() {
        // Test with VectorKey to verify algorithm works with variable-length keys.
        let mut tree1 = AdaptiveRadixTree::<VectorKey, i32>::new();
        tree1.insert_k(&VectorKey::new_from_slice(b"alpha"), 1);
        tree1.insert_k(&VectorKey::new_from_slice(b"beta"), 2);

        let mut tree2 = AdaptiveRadixTree::<VectorKey, i32>::new();
        tree2.insert_k(&VectorKey::new_from_slice(b"alpha"), 10); // Overwrite.
        tree2.insert_k(&VectorKey::new_from_slice(b"gamma"), 3);

        let merged = tree1.merge(tree2);

        assert_eq!(
            merged.get_k(&VectorKey::new_from_slice(b"alpha")),
            Some(&10)
        );
        assert_eq!(merged.get_k(&VectorKey::new_from_slice(b"beta")), Some(&2));
        assert_eq!(merged.get_k(&VectorKey::new_from_slice(b"gamma")), Some(&3));
        assert_eq!(merged.iter().count(), 3);
    }

    #[test]
    fn test_merge_forces_node_growth() {
        // Insert enough keys in both trees to force node growth after merge.
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        // Insert keys with different first bytes to force node growth.
        for i in 0u8..10 {
            let key = format!("{}key", i as char);
            tree1.insert(key.as_str(), i as i32);
        }

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        for i in 10u8..20 {
            let key = format!("{}key", i as char);
            tree2.insert(key.as_str(), i as i32);
        }

        let merged = tree1.merge(tree2);
        assert_eq!(merged.iter().count(), 20);

        // Verify all keys are accessible.
        for i in 0u8..20 {
            let key = format!("{}key", i as char);
            assert_eq!(merged.get(key.as_str()), Some(&(i as i32)));
        }
    }

    #[test]
    fn test_merge_with_node4_and_node16() {
        // Create trees with different node densities.
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("a", 1);
        tree1.insert("b", 2);
        tree1.insert("c", 3);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        for c in 'd'..='p' {
            tree2.insert(c.to_string().as_str(), c as i32);
        }

        let merged = tree1.merge(tree2);
        assert_eq!(merged.iter().count(), 3 + 13); // a-c from tree1, d-p from tree2.
    }

    #[test]
    fn test_merge_to_node256() {
        // Create trees that when merged require Node256 (256 children).
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        for i in 0u8..128 {
            let key = [i, 0];
            tree1.insert_k(&ArrayKey::<16>::new_from_slice(&key), i as i32);
        }

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        for i in 128u8..=255 {
            let key = [i, 0];
            tree2.insert_k(&ArrayKey::<16>::new_from_slice(&key), i as i32);
        }

        let merged = tree1.merge(tree2);
        assert_eq!(merged.iter().count(), 256);

        // Verify all keys.
        for i in 0u8..=255 {
            let key = [i, 0];
            assert_eq!(
                merged.get_k(&ArrayKey::<16>::new_from_slice(&key)),
                Some(&(i as i32))
            );
        }
    }

    #[test]
    fn test_merge_iteration_order_is_lexicographic() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("zebra", 1);
        tree1.insert("apple", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("mango", 3);
        tree2.insert("banana", 4);

        let merged = tree1.merge(tree2);

        let keys: Vec<_> = merged.iter().map(|(k, _)| k).collect();
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();
        assert_eq!(keys, sorted_keys);
    }

    #[test]
    fn test_merge_numeric_keys() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert(100u64, 1);
        tree1.insert(200u64, 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert(100u64, 10); // Overwrite.
        tree2.insert(300u64, 3);

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get(100u64), Some(&10));
        assert_eq!(merged.get(200u64), Some(&2));
        assert_eq!(merged.get(300u64), Some(&3));
    }

    #[test]
    fn test_merge_seeded_random() {
        // Property-based test with seeded random data.
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        const COUNT: usize = 1000;
        const SPACE: u64 = 5000;

        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let mut map1 = BTreeMap::new();
        for _ in 0..COUNT {
            let k = rng.random_range(0..SPACE);
            let v = rng.random_range(0..u64::MAX);
            tree1.insert(k, v);
            map1.insert(k, v);
        }

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let mut map2 = BTreeMap::new();
        for _ in 0..COUNT {
            let k = rng.random_range(0..SPACE);
            let v = rng.random_range(0..u64::MAX);
            tree2.insert(k, v);
            map2.insert(k, v);
        }

        // Compute expected: map2 wins on conflicts.
        let mut expected = map1.clone();
        for (k, v) in &map2 {
            expected.insert(*k, *v);
        }

        let merged = tree1.merge(tree2);

        // Verify all expected keys are present with correct values.
        for (k, v) in &expected {
            assert_eq!(merged.get(*k), Some(v), "Key {} mismatch", k);
        }

        // Verify count matches.
        assert_eq!(merged.iter().count(), expected.len());

        // Verify iteration order.
        let merged_iter: Vec<_> = merged.iter().map(|(k, v)| (k.to_be_u64(), *v)).collect();
        let expected_iter: Vec<_> = expected.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(merged_iter, expected_iter);
    }

    #[test]
    fn test_merge_divergent_prefixes_at_root() {
        // Two trees whose roots have completely different prefixes.
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("aaaa", 1);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("zzzz", 2);

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("aaaa"), Some(&1));
        assert_eq!(merged.get("zzzz"), Some(&2));
        assert_eq!(merged.iter().count(), 2);
    }

    #[test]
    fn test_merge_one_prefix_of_other_at_root() {
        // tree1 has a key that's a prefix of tree2's root path.
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("abc", 1);
        tree1.insert("abcdef", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("abcd", 3);

        let merged = tree1.merge(tree2);

        assert_eq!(merged.get("abc"), Some(&1));
        assert_eq!(merged.get("abcd"), Some(&3));
        assert_eq!(merged.get("abcdef"), Some(&2));
        assert_eq!(merged.iter().count(), 3);
    }

    // ==================== N-Way Merge Tests ====================

    #[test]
    fn test_merge_all_empty() {
        let trees: Vec<AdaptiveRadixTree<ArrayKey<16>, i32>> = vec![];
        let merged = AdaptiveRadixTree::merge_all(trees);
        assert!(merged.is_empty());
        assert_eq!(merged.iter().count(), 0);
    }

    #[test]
    fn test_merge_all_single() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree.insert("key", 1);
        tree.insert("other", 2);

        let merged = AdaptiveRadixTree::merge_all(vec![tree]);

        assert_eq!(merged.get("key"), Some(&1));
        assert_eq!(merged.get("other"), Some(&2));
        assert_eq!(merged.iter().count(), 2);
    }

    #[test]
    fn test_merge_all_two_equals_pairwise() {
        // Property: merge_all([a, b]) == a.merge(b) for identical inputs.
        let mut a1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut b1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut a2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut b2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        // Populate identical trees.
        for (i, key) in ["apple", "banana", "cherry", "date"].iter().enumerate() {
            a1.insert(*key, i as i32);
            a2.insert(*key, i as i32);
        }
        for (i, key) in ["apple", "elderberry", "fig"].iter().enumerate() {
            b1.insert(*key, i as i32 + 100);
            b2.insert(*key, i as i32 + 100);
        }

        let n_way = AdaptiveRadixTree::merge_all(vec![a1, b1]);
        let pairwise = a2.merge(b2);

        // Compare iteration results.
        let n_way_items: Vec<_> = n_way.iter().map(|(k, v)| (k, *v)).collect();
        let pairwise_items: Vec<_> = pairwise.iter().map(|(k, v)| (k, *v)).collect();

        assert_eq!(n_way_items, pairwise_items);
    }

    #[test]
    fn test_merge_all_three_right_wins() {
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t1.insert("key", 1);
        t2.insert("key", 2);
        t3.insert("key", 3);

        let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3]);

        // Last tree wins.
        assert_eq!(merged.get("key"), Some(&3));
        assert_eq!(merged.iter().count(), 1);
    }

    #[test]
    fn test_merge_all_disjoint_prefixes() {
        // Trees with completely different prefixes should just combine.
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t1.insert("aaa", 1);
        t2.insert("bbb", 2);
        t3.insert("ccc", 3);

        let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3]);

        assert_eq!(merged.get("aaa"), Some(&1));
        assert_eq!(merged.get("bbb"), Some(&2));
        assert_eq!(merged.get("ccc"), Some(&3));
        assert_eq!(merged.iter().count(), 3);
    }

    #[test]
    fn test_merge_all_shared_prefix_different_children() {
        // All trees share "app" prefix but have different completions.
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t1.insert("apple", 1);
        t2.insert("application", 2);
        t3.insert("apricot", 3);

        let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3]);

        assert_eq!(merged.get("apple"), Some(&1));
        assert_eq!(merged.get("application"), Some(&2));
        assert_eq!(merged.get("apricot"), Some(&3));
        assert_eq!(merged.iter().count(), 3);
    }

    #[test]
    fn test_merge_all_deep_overlap_with_conflicts() {
        // Deep prefix overlap with conflicts at various levels.
        let mut t1 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();

        t1.insert("aaaaaaaaaaaa1", 1);
        t1.insert("aaaaaaaaaaaa2", 2);
        t2.insert("aaaaaaaaaaaa1", 10); // Conflict with t1.
        t2.insert("aaaaaaaaaaaa3", 3);
        t3.insert("aaaaaaaaaaaa2", 20); // Conflict with t1.
        t3.insert("aaaaaaaaaaaa4", 4);

        let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3]);

        // t2 wins for "...1", t3 wins for "...2".
        assert_eq!(merged.get("aaaaaaaaaaaa1"), Some(&10));
        assert_eq!(merged.get("aaaaaaaaaaaa2"), Some(&20));
        assert_eq!(merged.get("aaaaaaaaaaaa3"), Some(&3));
        assert_eq!(merged.get("aaaaaaaaaaaa4"), Some(&4));
        assert_eq!(merged.iter().count(), 4);
    }

    #[test]
    fn test_merge_all_with_empty_trees() {
        // Mix of empty and non-empty trees.
        let t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t4 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t2.insert("key2", 2);
        t4.insert("key4", 4);

        let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3, t4]);

        assert_eq!(merged.get("key2"), Some(&2));
        assert_eq!(merged.get("key4"), Some(&4));
        assert_eq!(merged.iter().count(), 2);
    }

    #[test]
    fn test_merge_all_four_trees_right_wins_chain() {
        // Test that rightmost value wins through the entire chain.
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t4 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        // All trees have same key with different values.
        t1.insert("shared", 1);
        t2.insert("shared", 2);
        t3.insert("shared", 3);
        t4.insert("shared", 4);

        // Each tree also has unique keys.
        t1.insert("unique1", 100);
        t2.insert("unique2", 200);
        t3.insert("unique3", 300);
        t4.insert("unique4", 400);

        let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3, t4]);

        // t4 wins for "shared".
        assert_eq!(merged.get("shared"), Some(&4));
        // All unique keys are present.
        assert_eq!(merged.get("unique1"), Some(&100));
        assert_eq!(merged.get("unique2"), Some(&200));
        assert_eq!(merged.get("unique3"), Some(&300));
        assert_eq!(merged.get("unique4"), Some(&400));
        assert_eq!(merged.iter().count(), 5);
    }

    #[test]
    fn test_merge_all_iteration_order_is_lexicographic() {
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t1.insert("zebra", 1);
        t2.insert("apple", 2);
        t3.insert("mango", 3);

        let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3]);

        let keys: Vec<_> = merged.iter().map(|(k, _)| k).collect();
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();
        assert_eq!(keys, sorted_keys);
    }

    #[test]
    fn test_merge_all_seeded_random() {
        // Property test: merge_all result matches fold approach.
        let mut rng = StdRng::seed_from_u64(0xCAFEBABE);
        const COUNT: usize = 200;
        const SPACE: u64 = 1000;
        const NUM_TREES: usize = 5;

        let mut trees = Vec::new();
        let mut maps = Vec::new();

        for _ in 0..NUM_TREES {
            let mut tree = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
            let mut map = BTreeMap::new();
            for _ in 0..COUNT {
                let k = rng.random_range(0..SPACE);
                let v = rng.random_range(0..u64::MAX);
                tree.insert(k, v);
                map.insert(k, v);
            }
            trees.push(tree);
            maps.push(map);
        }

        // Compute expected: later maps win on conflicts.
        let mut expected = BTreeMap::new();
        for map in &maps {
            for (k, v) in map {
                expected.insert(*k, *v);
            }
        }

        let merged = AdaptiveRadixTree::merge_all(trees);

        // Verify all expected keys are present with correct values.
        for (k, v) in &expected {
            assert_eq!(merged.get(*k), Some(v), "Key {} mismatch", k);
        }

        // Verify count matches.
        assert_eq!(merged.iter().count(), expected.len());

        // Verify iteration order.
        let merged_iter: Vec<_> = merged.iter().map(|(k, v)| (k.to_be_u64(), *v)).collect();
        let expected_iter: Vec<_> = expected.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(merged_iter, expected_iter);
    }

    #[test]
    fn test_merge_all_versus_fold() {
        // Property: merge_all should produce same result as fold for any number of trees.
        let mut rng = StdRng::seed_from_u64(0xDEADC0DE);
        const COUNT: usize = 100;
        const SPACE: u64 = 500;
        const NUM_TREES: usize = 4;

        // Build two identical sets of trees.
        let mut trees_for_merge_all = Vec::new();
        let mut trees_for_fold = Vec::new();

        for _ in 0..NUM_TREES {
            let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
            let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
            for _ in 0..COUNT {
                let k = rng.random_range(0..SPACE);
                let v = rng.random_range(0..u64::MAX);
                tree1.insert(k, v);
                tree2.insert(k, v);
            }
            trees_for_merge_all.push(tree1);
            trees_for_fold.push(tree2);
        }

        // Merge using merge_all.
        let merged_n_way = AdaptiveRadixTree::merge_all(trees_for_merge_all);

        // Merge using fold.
        let merged_fold = trees_for_fold
            .into_iter()
            .reduce(|a, b| a.merge(b))
            .unwrap();

        // Compare results.
        let n_way_items: Vec<_> = merged_n_way
            .iter()
            .map(|(k, v)| (k.to_be_u64(), *v))
            .collect();
        let fold_items: Vec<_> = merged_fold
            .iter()
            .map(|(k, v)| (k.to_be_u64(), *v))
            .collect();

        assert_eq!(n_way_items, fold_items);
    }

    #[test]
    fn test_merge_all_numeric_keys() {
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t1.insert(100u64, 1);
        t1.insert(200u64, 2);
        t2.insert(100u64, 10); // Overwrite.
        t2.insert(300u64, 3);
        t3.insert(200u64, 20); // Overwrite.
        t3.insert(400u64, 4);

        let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3]);

        assert_eq!(merged.get(100u64), Some(&10)); // t2 wins.
        assert_eq!(merged.get(200u64), Some(&20)); // t3 wins.
        assert_eq!(merged.get(300u64), Some(&3));
        assert_eq!(merged.get(400u64), Some(&4));
        assert_eq!(merged.iter().count(), 4);
    }

    #[test]
    fn test_merge_all_vector_key() {
        // Test with VectorKey to verify algorithm works with variable-length keys.
        let mut t1 = AdaptiveRadixTree::<VectorKey, i32>::new();
        let mut t2 = AdaptiveRadixTree::<VectorKey, i32>::new();
        let mut t3 = AdaptiveRadixTree::<VectorKey, i32>::new();

        t1.insert_k(&VectorKey::new_from_slice(b"alpha"), 1);
        t2.insert_k(&VectorKey::new_from_slice(b"alpha"), 10); // Overwrite.
        t2.insert_k(&VectorKey::new_from_slice(b"beta"), 2);
        t3.insert_k(&VectorKey::new_from_slice(b"gamma"), 3);

        let merged = AdaptiveRadixTree::merge_all(vec![t1, t2, t3]);

        assert_eq!(
            merged.get_k(&VectorKey::new_from_slice(b"alpha")),
            Some(&10)
        );
        assert_eq!(merged.get_k(&VectorKey::new_from_slice(b"beta")), Some(&2));
        assert_eq!(merged.get_k(&VectorKey::new_from_slice(b"gamma")), Some(&3));
        assert_eq!(merged.iter().count(), 3);
    }

    #[test]
    fn test_merge_all_extension_vs_exact_match_conflict() {
        // Regression test: when an extension child conflicts with an exact-match
        // child at the same byte, we must preserve the correct tree indices for
        // right-wins semantics.
        //
        // Tree 0: inner at prefix "a" with child 'b' -> leaf "c" (key "abc")
        // Tree 1: leaf at prefix "ab" (key "ab")
        //
        // After merge: both keys should exist, with Tree 1's "ab" taking precedence
        // if there were a conflict (but here they are distinct keys).
        let mut t0 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t0.insert("abc", 0);
        t1.insert("ab", 1);

        let merged = AdaptiveRadixTree::merge_all(vec![t0, t1]);

        // Both keys should exist after merge.
        assert_eq!(merged.get("abc"), Some(&0));
        assert_eq!(merged.get("ab"), Some(&1));
        assert_eq!(merged.iter().count(), 2);

        // Also test with the trees in opposite order to verify right-wins.
        let mut t0 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t0.insert("ab", 0);
        t1.insert("abc", 1);

        let merged = AdaptiveRadixTree::merge_all(vec![t0, t1]);

        assert_eq!(merged.get("ab"), Some(&0));
        assert_eq!(merged.get("abc"), Some(&1));
        assert_eq!(merged.iter().count(), 2);
    }

    #[test]
    fn test_merge_all_extension_vs_exact_match_same_key() {
        // Test conflict resolution when extension and exact-match have children leading
        // to the same key. Since merge order is non-deterministic, we verify that
        // one of the input values is chosen.
        //
        // Tree 0: inner at "a" with child 'b' -> inner with child 'c' -> leaf (key "abc")
        // Tree 1: leaf at "abc"
        let mut t0 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t0.insert("abc", 0);
        t1.insert("abc", 1);

        let merged = AdaptiveRadixTree::merge_all(vec![t0, t1]);

        // One of the values wins (order not guaranteed).
        let val = merged.get("abc").unwrap();
        assert!(*val == 0 || *val == 1);
        assert_eq!(merged.iter().count(), 1);

        // Three-way test with conflict at extension/exact boundary.
        let mut t0 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t0.insert("abc", 0);
        t0.insert("abd", 10);
        t1.insert("ab", 1); // Exact match at "ab", extension from t0 at "ab" + 'c'/'d'.
        t2.insert("abc", 2); // Conflicts with t0's "abc".

        let merged = AdaptiveRadixTree::merge_all(vec![t0, t1, t2]);

        assert_eq!(merged.get("ab"), Some(&1)); // Only t1 has "ab".
        let abc_val = merged.get("abc").unwrap();
        assert!(*abc_val == 0 || *abc_val == 2); // Either t0 or t2 wins.
        assert_eq!(merged.get("abd"), Some(&10)); // Only t0 has "abd".
        assert_eq!(merged.iter().count(), 3);
    }

    // ==================== merge_with Tests ====================

    #[test]
    fn test_merge_with_sum_values() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 1);
        tree1.insert("banana", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("apple", 10);
        tree2.insert("cherry", 3);

        let merged = tree1.merge_with(tree2, |left, right| left + right);

        assert_eq!(merged.get("apple"), Some(&11)); // 1 + 10
        assert_eq!(merged.get("banana"), Some(&2)); // From tree1
        assert_eq!(merged.get("cherry"), Some(&3)); // From tree2
        assert_eq!(merged.iter().count(), 3);
    }

    #[test]
    fn test_merge_with_max_values() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 100);
        tree1.insert("banana", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("apple", 10);
        tree2.insert("banana", 200);

        let merged = tree1.merge_with(tree2, |left, right| std::cmp::max(left, right));

        assert_eq!(merged.get("apple"), Some(&100)); // max(100, 10)
        assert_eq!(merged.get("banana"), Some(&200)); // max(2, 200)
    }

    #[test]
    fn test_merge_with_left_wins() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 1);
        tree1.insert("banana", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("apple", 10);
        tree2.insert("banana", 20);

        // left-wins: opposite of default behavior.
        let merged = tree1.merge_with(tree2, |left, _right| left);

        assert_eq!(merged.get("apple"), Some(&1)); // tree1's value
        assert_eq!(merged.get("banana"), Some(&2)); // tree1's value
    }

    #[test]
    fn test_merge_with_no_conflicts() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 1);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("banana", 2);

        let mut call_count = 0;
        let merged = tree1.merge_with(tree2, |left, right| {
            call_count += 1;
            left + right
        });

        // Resolver should never be called for disjoint keys.
        assert_eq!(call_count, 0);
        assert_eq!(merged.get("apple"), Some(&1));
        assert_eq!(merged.get("banana"), Some(&2));
    }

    #[test]
    fn test_merge_with_all_conflicts() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("a", 1);
        tree1.insert("b", 2);
        tree1.insert("c", 3);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("a", 10);
        tree2.insert("b", 20);
        tree2.insert("c", 30);

        let mut call_count = 0;
        let merged = tree1.merge_with(tree2, |left, right| {
            call_count += 1;
            left + right
        });

        assert_eq!(call_count, 3); // All three keys had conflicts.
        assert_eq!(merged.get("a"), Some(&11));
        assert_eq!(merged.get("b"), Some(&22));
        assert_eq!(merged.get("c"), Some(&33));
    }

    #[test]
    fn test_merge_with_empty_trees() {
        let tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        let merged = tree1.merge_with(tree2, |left, right| left + right);
        assert!(merged.is_empty());

        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("apple", 1);
        let tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        let merged = tree1.merge_with(tree2, |left, right| left + right);
        assert_eq!(merged.get("apple"), Some(&1));
    }

    #[test]
    fn test_merge_with_stateful_resolver() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree1.insert("a", 1);
        tree1.insert("b", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree2.insert("a", 10);
        tree2.insert("b", 20);
        tree2.insert("c", 30);

        // Stateful resolver that counts conflicts.
        let mut conflict_count = 0;
        let merged = tree1.merge_with(tree2, |left, right| {
            conflict_count += 1;
            left + right
        });

        assert_eq!(conflict_count, 2); // "a" and "b" had conflicts
        assert_eq!(merged.get("a"), Some(&11));
        assert_eq!(merged.get("b"), Some(&22));
        assert_eq!(merged.get("c"), Some(&30));
    }

    #[test]
    fn test_merge_with_deep_prefix_conflicts() {
        let mut tree1 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();
        tree1.insert("aaaaaaaaaaaaaaa1", 1);
        tree1.insert("aaaaaaaaaaaaaaa2", 2);

        let mut tree2 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();
        tree2.insert("aaaaaaaaaaaaaaa1", 10);
        tree2.insert("aaaaaaaaaaaaaaa3", 3);

        let merged = tree1.merge_with(tree2, |left, right| left + right);

        assert_eq!(merged.get("aaaaaaaaaaaaaaa1"), Some(&11)); // 1 + 10
        assert_eq!(merged.get("aaaaaaaaaaaaaaa2"), Some(&2));
        assert_eq!(merged.get("aaaaaaaaaaaaaaa3"), Some(&3));
    }

    #[test]
    fn test_merge_with_equals_merge_when_right_wins() {
        // Property: merge_with(|_, r| r) should equal merge().
        let mut rng = StdRng::seed_from_u64(0xCAFEBABE);
        const COUNT: usize = 500;
        const SPACE: u64 = 1000;

        let mut tree1 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let mut tree2 = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let mut tree1_clone = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();
        let mut tree2_clone = AdaptiveRadixTree::<ArrayKey<16>, u64>::new();

        for _ in 0..COUNT {
            let k = rng.random_range(0..SPACE);
            let v = rng.random_range(0..u64::MAX);
            tree1.insert(k, v);
            tree1_clone.insert(k, v);
        }
        for _ in 0..COUNT {
            let k = rng.random_range(0..SPACE);
            let v = rng.random_range(0..u64::MAX);
            tree2.insert(k, v);
            tree2_clone.insert(k, v);
        }

        let merged_with = tree1.merge_with(tree2, |_left, right| right);
        let merged = tree1_clone.merge(tree2_clone);

        let with_items: Vec<_> = merged_with.iter().map(|(k, v)| (k, *v)).collect();
        let orig_items: Vec<_> = merged.iter().map(|(k, v)| (k, *v)).collect();

        assert_eq!(with_items, orig_items);
    }

    // ==================== merge_all_with Tests ====================

    #[test]
    fn test_merge_all_with_sum_values() {
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t1.insert("key", 1);
        t2.insert("key", 2);
        t3.insert("key", 4);

        let merged = AdaptiveRadixTree::merge_all_with(vec![t1, t2, t3], |acc, v| acc + v);

        assert_eq!(merged.get("key"), Some(&7)); // 1 + 2 + 4
    }

    #[test]
    fn test_merge_all_with_commutative_resolver() {
        // With unordered fold, use a commutative resolver (sum).
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t1.insert("key", 1);
        t2.insert("key", 2);
        t3.insert("key", 3);

        let merged = AdaptiveRadixTree::merge_all_with(vec![t1, t2, t3], |acc, v| acc + v);

        // Sum is commutative, so order doesn't matter.
        assert_eq!(merged.get("key"), Some(&6));
    }

    #[test]
    fn test_merge_all_with_empty_trees_in_middle() {
        let t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t2.insert("key", 5);

        let merged = AdaptiveRadixTree::merge_all_with(vec![t1, t2, t3], |acc, v| acc + v);

        assert_eq!(merged.get("key"), Some(&5));
    }

    #[test]
    fn test_merge_all_with_single_tree() {
        let mut tree = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        tree.insert("key", 42);

        let mut call_count = 0;
        let merged = AdaptiveRadixTree::merge_all_with(vec![tree], |acc, v| {
            call_count += 1;
            acc + v
        });

        assert_eq!(call_count, 0); // No conflicts with single tree.
        assert_eq!(merged.get("key"), Some(&42));
    }

    #[test]
    fn test_merge_all_with_two_trees() {
        // N=2 should use 2-way fast path.
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t1.insert("key", 1);
        t2.insert("key", 2);

        let merged = AdaptiveRadixTree::merge_all_with(vec![t1, t2], |acc, v| acc * v);

        assert_eq!(merged.get("key"), Some(&2)); // 1 * 2
    }

    #[test]
    fn test_merge_all_with_deep_conflicts() {
        let mut t1 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<32>, i32>::new();

        t1.insert("verylongsharedprefix/a", 1);
        t2.insert("verylongsharedprefix/a", 10);
        t2.insert("verylongsharedprefix/b", 20);
        t3.insert("verylongsharedprefix/a", 100);

        let merged = AdaptiveRadixTree::merge_all_with(vec![t1, t2, t3], |acc, v| acc + v);

        assert_eq!(merged.get("verylongsharedprefix/a"), Some(&111)); // 1 + 10 + 100
        assert_eq!(merged.get("verylongsharedprefix/b"), Some(&20));
    }

    #[test]
    fn test_merge_all_with_stateful_resolver() {
        let mut t1 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t2 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();
        let mut t3 = AdaptiveRadixTree::<ArrayKey<16>, i32>::new();

        t1.insert("a", 1);
        t1.insert("b", 2);
        t2.insert("a", 10);
        t2.insert("c", 30);
        t3.insert("a", 100);
        t3.insert("b", 200);

        let mut conflict_count = 0;
        let merged = AdaptiveRadixTree::merge_all_with(vec![t1, t2, t3], |acc, v| {
            conflict_count += 1;
            acc + v
        });

        // "a" has 3 values -> 2 resolver calls, "b" has 2 values -> 1 resolver call
        assert_eq!(conflict_count, 3);
        assert_eq!(merged.get("a"), Some(&111)); // 1 + 10 + 100
        assert_eq!(merged.get("b"), Some(&202)); // 2 + 200
        assert_eq!(merged.get("c"), Some(&30));
    }

    #[test]
    fn test_merge_all_with_equals_merge_all_when_right_wins() {
        // Property: merge_all_with(|_, r| r) should equal merge_all().
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        const TREE_COUNT: usize = 4;
        const KEYS_PER_TREE: usize = 200;
        const SPACE: u64 = 500;

        let mut trees_with: Vec<AdaptiveRadixTree<ArrayKey<16>, u64>> = Vec::new();
        let mut trees_orig: Vec<AdaptiveRadixTree<ArrayKey<16>, u64>> = Vec::new();

        for _ in 0..TREE_COUNT {
            let mut tree_with = AdaptiveRadixTree::new();
            let mut tree_orig = AdaptiveRadixTree::new();
            for _ in 0..KEYS_PER_TREE {
                let k = rng.random_range(0..SPACE);
                let v = rng.random_range(0..u64::MAX);
                tree_with.insert(k, v);
                tree_orig.insert(k, v);
            }
            trees_with.push(tree_with);
            trees_orig.push(tree_orig);
        }

        let merged_with = AdaptiveRadixTree::merge_all_with(trees_with, |_left, right| right);
        let merged_orig = AdaptiveRadixTree::merge_all(trees_orig);

        let with_items: Vec<_> = merged_with.iter().map(|(k, v)| (k, *v)).collect();
        let orig_items: Vec<_> = merged_orig.iter().map(|(k, v)| (k, *v)).collect();

        assert_eq!(with_items, orig_items);
    }
}
