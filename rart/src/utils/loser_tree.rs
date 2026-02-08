//! A loser tree for efficient N-way merge operations.
//!
//! A loser tree (also known as a tournament tree) is a binary tree structure
//! optimized for repeatedly finding the minimum among N participants that
//! provide new values. After initialization, each "get next minimum" operation
//! updates exactly log2(N) nodes along a single path from leaf to root.
//!
//! ## When to use a loser tree vs BinaryHeap
//!
//! - **Loser tree**: Best for N-way merge where you repeatedly take the minimum
//!   and immediately replace it with a new value from the same source. The
//!   replay operation is O(log N) with excellent cache locality.
//!
//! - **BinaryHeap**: Better for general priority queue operations where you
//!   don't always replace the minimum, or where insertions/deletions are
//!   interleaved unpredictably.
//!
//! ## Complexity
//!
//! - Initialization: O(N log N) comparisons
//! - `peek_min()`: O(1)
//! - `winner_idx()`: O(1)
//! - `replace_winner()`: O(log N) comparisons
//!
//! ## Tree Layout
//!
//! For N leaves, we use a virtual complete binary tree of size 2N:
//! - Leaves occupy virtual positions N..2N
//! - Internal nodes occupy virtual positions 1..N
//! - Position 0 stores the overall winner
//!
//! The `losers` array has N entries where:
//! - `losers[0]` = index of overall winner (leaf index 0..N)
//! - `losers[1..N]` = losers at each internal node

/// Entry in the loser tree, tracking the source iterator index.
#[derive(Debug, Clone)]
pub struct Entry<T> {
    /// The current value from this iterator (None if exhausted).
    pub value: Option<T>,
    /// Index of the source iterator.
    pub source_idx: usize,
}

impl<T> Entry<T> {
    /// Create a new entry with the given value and source index.
    pub fn new(value: Option<T>, source_idx: usize) -> Self {
        Self { value, source_idx }
    }
}

/// A loser tree for efficient N-way merge operations.
///
/// The tree maintains N participants, each with a current value. After finding
/// the minimum, the caller replaces the winner's value and the tree replays
/// the tournament in O(log N) time.
///
/// # Example
///
/// ```ignore
/// use rart::utils::loser_tree::{LoserTree, Entry};
///
/// // Three sorted iterators to merge.
/// let entries = vec![
///     Entry::new(Some(1), 0),
///     Entry::new(Some(3), 1),
///     Entry::new(Some(2), 2),
/// ];
///
/// let mut tree = LoserTree::new(entries);
///
/// // Winner is participant 0 with value 1.
/// assert_eq!(tree.peek_min().unwrap().value, Some(1));
/// assert_eq!(tree.winner_idx(), 0);
///
/// // Replace with next value from that iterator.
/// tree.replace_winner(Some(4));
///
/// // Now participant 2 with value 2 is the winner.
/// assert_eq!(tree.peek_min().unwrap().value, Some(2));
/// ```
pub struct LoserTree<T> {
    /// Internal nodes store the loser of each match (indices into leaves).
    /// Index 0 holds the overall winner, indices 1..k hold losers at internal nodes,
    /// where k is the tree capacity (power of 2 >= n).
    losers: Vec<usize>,
    /// Leaves store the current value and source index for each participant.
    /// Padded to capacity k with dummy entries that always lose.
    leaves: Vec<Entry<T>>,
    /// Tree capacity (smallest power of 2 >= n).
    k: usize,
}

impl<T: Ord> LoserTree<T> {
    /// Build a new loser tree from the given entries.
    ///
    /// Each entry represents a participant with an initial value. `None` values
    /// sort after all `Some` values, representing exhausted iterators.
    ///
    /// # Panics
    ///
    /// Panics if `entries` is empty.
    pub fn new(mut entries: Vec<Entry<T>>) -> Self {
        assert!(!entries.is_empty(), "LoserTree requires at least one entry");

        let n = entries.len();
        // Round up to next power of 2.
        let k = n.next_power_of_two();

        // Pad with dummy entries that always lose (None values).
        // These have out-of-bounds source_idx but will never win since None always loses.
        for i in n..k {
            entries.push(Entry::new(None, i));
        }

        let mut tree = Self {
            losers: vec![0; k],
            leaves: entries,
            k,
        };

        tree.build();
        tree
    }

    /// Build the tournament tree using bottom-up pairwise construction.
    ///
    /// With k leaves (power of 2), we have k-1 internal nodes at positions 1..k,
    /// plus position 0 for the overall winner. Leaves are at virtual positions k..2k.
    fn build(&mut self) {
        // Process level by level from leaves to root.
        // Level 0: leaves at virtual positions k..2k
        // Level 1: internal nodes at positions k/2..k
        // ...and so on up to the root.

        // Start with all leaves as competitors.
        let mut current_level: Vec<usize> = (0..self.k).collect();
        let mut level_start = self.k; // Virtual position of first element at this level.

        while current_level.len() > 1 {
            let mut next_level = Vec::with_capacity(current_level.len() / 2);

            for i in (0..current_level.len()).step_by(2) {
                let left = current_level[i];
                let right = current_level[i + 1];

                // Parent virtual position.
                let left_pos = level_start + i;
                let parent_pos = left_pos / 2;

                // Compete: smaller wins.
                let (winner, loser) = if self.is_less(left, right) {
                    (left, right)
                } else {
                    (right, left)
                };

                // Store loser at this internal node (position 1..k).
                self.losers[parent_pos] = loser;
                next_level.push(winner);
            }

            level_start /= 2;
            current_level = next_level;
        }

        // The final winner.
        self.losers[0] = current_level[0];
    }

    /// Compute the parent (internal node index) for a given virtual tree position.
    ///
    /// Leaves are at virtual positions n..2n, internal nodes at 1..n.
    /// The parent of virtual position p is p/2.
    #[inline]
    fn parent(&self, virtual_pos: usize) -> usize {
        virtual_pos / 2
    }

    /// Replay the tournament from the given leaf to the root.
    ///
    /// Starting from the leaf's position in the virtual tree, we walk up to
    /// the root. At each internal node, compare the advancing winner against
    /// the stored loser. The new loser is stored at the node, the winner advances.
    fn replay(&mut self, leaf_idx: usize) {
        // Convert leaf index (0..k) to virtual tree position (k..2k).
        let mut pos = leaf_idx + self.k;
        let mut winner = leaf_idx;

        // Walk from leaf to root.
        while pos > 1 {
            let parent = self.parent(pos);

            // Internal node 'parent' stores the loser from a previous match.
            // Compare our advancing winner against that loser.
            let opponent = self.losers[parent];

            if self.is_less(opponent, winner) {
                // Opponent beats our winner: opponent advances, our winner
                // becomes the new loser at this node.
                self.losers[parent] = winner;
                winner = opponent;
            } else {
                // Our winner beats the opponent: we advance, opponent stays
                // as the loser at this node.
                self.losers[parent] = opponent;
            }

            pos = parent;
        }

        // Store the overall winner at position 0.
        self.losers[0] = winner;
    }

    /// Compare two leaves: returns true if leaf `a` should beat leaf `b`.
    ///
    /// A leaf beats another if its value is smaller, or on tie, if its index
    /// is smaller (leftmost wins).
    #[inline]
    fn is_less(&self, a: usize, b: usize) -> bool {
        let val_a = &self.leaves[a].value;
        let val_b = &self.leaves[b].value;

        match (val_a, val_b) {
            (Some(va), Some(vb)) => {
                // Smaller value wins. Ties go to lower index.
                va < vb || (va == vb && a < b)
            }
            (Some(_), None) => true,  // Some beats None.
            (None, Some(_)) => false, // None loses to Some.
            (None, None) => a < b,    // Both exhausted, lower index wins.
        }
    }

    /// Return a reference to the current winner's entry.
    ///
    /// Returns `None` only if all participants are exhausted.
    #[inline]
    pub fn peek_min(&self) -> Option<&Entry<T>> {
        let winner = self.losers[0];
        let entry = &self.leaves[winner];

        // If the winner has no value, all participants are exhausted.
        if entry.value.is_some() {
            Some(entry)
        } else {
            None
        }
    }

    /// Return the source index of the current winner.
    #[inline]
    pub fn winner_idx(&self) -> usize {
        self.leaves[self.losers[0]].source_idx
    }

    /// Replace the winner's value and replay the tournament.
    ///
    /// This is the key operation for N-way merge: after consuming the winner's
    /// current value, provide the next value from that source (or `None` if
    /// exhausted) and the tree updates in O(log N).
    #[inline]
    pub fn replace_winner(&mut self, new_value: Option<T>) {
        let winner_leaf = self.losers[0];
        self.leaves[winner_leaf].value = new_value;
        self.replay(winner_leaf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_element() {
        let entries = vec![Entry::new(Some(42), 0)];
        let mut tree = LoserTree::new(entries);

        assert_eq!(tree.peek_min().unwrap().value, Some(42));
        assert_eq!(tree.winner_idx(), 0);

        tree.replace_winner(Some(100));
        assert_eq!(tree.peek_min().unwrap().value, Some(100));

        tree.replace_winner(None);
        assert!(tree.peek_min().is_none());
    }

    #[test]
    fn test_two_elements() {
        let entries = vec![Entry::new(Some(5), 0), Entry::new(Some(3), 1)];
        let mut tree = LoserTree::new(entries);

        // 3 < 5, so participant 1 wins.
        assert_eq!(tree.peek_min().unwrap().value, Some(3));
        assert_eq!(tree.winner_idx(), 1);

        // Replace with 7.
        tree.replace_winner(Some(7));

        // Now 5 < 7, participant 0 wins.
        assert_eq!(tree.peek_min().unwrap().value, Some(5));
        assert_eq!(tree.winner_idx(), 0);
    }

    #[test]
    fn test_four_elements() {
        let entries = vec![
            Entry::new(Some(4), 0),
            Entry::new(Some(2), 1),
            Entry::new(Some(5), 2),
            Entry::new(Some(1), 3),
        ];
        let mut tree = LoserTree::new(entries);

        // 1 is minimum.
        assert_eq!(tree.peek_min().unwrap().value, Some(1));
        assert_eq!(tree.winner_idx(), 3);

        tree.replace_winner(Some(3));
        // Now 2 is minimum.
        assert_eq!(tree.peek_min().unwrap().value, Some(2));
        assert_eq!(tree.winner_idx(), 1);
    }

    #[test]
    fn test_three_elements_non_power_of_two() {
        let entries = vec![
            Entry::new(Some(10), 0),
            Entry::new(Some(5), 1),
            Entry::new(Some(8), 2),
        ];
        let mut tree = LoserTree::new(entries);

        assert_eq!(tree.peek_min().unwrap().value, Some(5));
        assert_eq!(tree.winner_idx(), 1);

        tree.replace_winner(Some(12));
        assert_eq!(tree.peek_min().unwrap().value, Some(8));
        assert_eq!(tree.winner_idx(), 2);

        tree.replace_winner(Some(9));
        assert_eq!(tree.peek_min().unwrap().value, Some(9));
        assert_eq!(tree.winner_idx(), 2);

        tree.replace_winner(Some(11));
        assert_eq!(tree.peek_min().unwrap().value, Some(10));
        assert_eq!(tree.winner_idx(), 0);
    }

    #[test]
    fn test_five_elements() {
        let entries = vec![
            Entry::new(Some(5), 0),
            Entry::new(Some(3), 1),
            Entry::new(Some(7), 2),
            Entry::new(Some(1), 3),
            Entry::new(Some(4), 4),
        ];
        let tree = LoserTree::new(entries);

        // 1 is minimum.
        assert_eq!(tree.peek_min().unwrap().value, Some(1));
        assert_eq!(tree.winner_idx(), 3);
    }

    #[test]
    fn test_exhausted_iterator_loses() {
        let entries = vec![
            Entry::new(Some(5), 0),
            Entry::new(None, 1), // Already exhausted.
            Entry::new(Some(3), 2),
        ];
        let mut tree = LoserTree::new(entries);

        // 3 should win (None loses).
        assert_eq!(tree.peek_min().unwrap().value, Some(3));
        assert_eq!(tree.winner_idx(), 2);

        tree.replace_winner(None);
        // Now 5 should win.
        assert_eq!(tree.peek_min().unwrap().value, Some(5));
        assert_eq!(tree.winner_idx(), 0);

        tree.replace_winner(None);
        // All exhausted.
        assert!(tree.peek_min().is_none());
    }

    #[test]
    fn test_all_same_value_leftmost_wins() {
        let entries = vec![
            Entry::new(Some(5), 0),
            Entry::new(Some(5), 1),
            Entry::new(Some(5), 2),
            Entry::new(Some(5), 3),
        ];
        let tree = LoserTree::new(entries);

        // Leftmost (index 0) should win on ties.
        assert_eq!(tree.peek_min().unwrap().value, Some(5));
        assert_eq!(tree.winner_idx(), 0);
    }

    #[test]
    fn test_all_exhausted() {
        let entries = vec![
            Entry::new(None::<i32>, 0),
            Entry::new(None, 1),
            Entry::new(None, 2),
        ];
        let tree = LoserTree::new(entries);

        assert!(tree.peek_min().is_none());
    }

    #[test]
    fn test_sorted_output() {
        // Simulate merging three sorted sequences.
        let mut seqs = vec![
            vec![1, 4, 7].into_iter().peekable(),
            vec![2, 5, 8].into_iter().peekable(),
            vec![3, 6, 9].into_iter().peekable(),
        ];

        let entries: Vec<_> = seqs
            .iter_mut()
            .enumerate()
            .map(|(idx, iter)| Entry::new(iter.next(), idx))
            .collect();

        let mut tree = LoserTree::new(entries);
        let mut result = Vec::new();

        while let Some(entry) = tree.peek_min() {
            let val = entry.value.unwrap();
            let idx = tree.winner_idx();
            result.push(val);

            let next = seqs[idx].next();
            tree.replace_winner(next);
        }

        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_equivalent_to_sorted_merge() {
        // Verify the loser tree produces the same output as a naive sorted merge.
        let sequences = vec![
            vec![1, 5, 9, 13],
            vec![2, 6, 10, 14],
            vec![3, 7, 11, 15],
            vec![4, 8, 12, 16],
        ];

        // Naive merge: concat and sort.
        let mut expected: Vec<_> = sequences.iter().flatten().copied().collect();
        expected.sort();

        // Loser tree merge.
        let mut seqs: Vec<_> = sequences
            .into_iter()
            .map(|s| s.into_iter().peekable())
            .collect();
        let entries: Vec<_> = seqs
            .iter_mut()
            .enumerate()
            .map(|(idx, iter)| Entry::new(iter.next(), idx))
            .collect();

        let mut tree = LoserTree::new(entries);
        let mut result = Vec::new();

        while let Some(entry) = tree.peek_min() {
            let val = entry.value.unwrap();
            let idx = tree.winner_idx();
            result.push(val);
            tree.replace_winner(seqs[idx].next());
        }

        assert_eq!(result, expected);
    }

    #[test]
    fn test_alternating_winners() {
        // Two sequences that alternate.
        let mut seqs = vec![
            vec![1, 3, 5, 7].into_iter().peekable(),
            vec![2, 4, 6, 8].into_iter().peekable(),
        ];

        let entries: Vec<_> = seqs
            .iter_mut()
            .enumerate()
            .map(|(idx, iter)| Entry::new(iter.next(), idx))
            .collect();

        let mut tree = LoserTree::new(entries);
        let mut winners = Vec::new();

        while tree.peek_min().is_some() {
            let idx = tree.winner_idx();
            winners.push(idx);
            tree.replace_winner(seqs[idx].next());
        }

        // Should alternate: 0, 1, 0, 1, 0, 1, 0, 1.
        assert_eq!(winners, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_large_tree() {
        // Test with 16 participants.
        let n = 16;
        let entries: Vec<_> = (0..n)
            .map(|i| Entry::new(Some((n - i) as i32), i))
            .collect();

        let mut tree = LoserTree::new(entries);

        // Minimum should be 1, from index n-1.
        assert_eq!(tree.peek_min().unwrap().value, Some(1));
        assert_eq!(tree.winner_idx(), n - 1);

        // Exhaust all participants.
        for expected in 1..=n as i32 {
            assert_eq!(tree.peek_min().unwrap().value, Some(expected));
            tree.replace_winner(None);
        }

        assert!(tree.peek_min().is_none());
    }
}
