//! Core vector database — the central struct callers interact with.
//!
//! # Architecture
//! ```text
//!   VectorDB
//!   └── HashMap<VectorId, VectorRecord>   ← O(1) insert / delete / lookup
//!       └── VectorRecord { id, embedding, metadata }
//! ```
//!
//! Search is a full linear scan: O(n × d) where *n* = number of records and
//! *d* = embedding dimension. This is intentional — the goal is clarity.
//! For larger corpora, the scan can be replaced with an HNSW or IVF index
//! without changing the public API.

use std::collections::HashMap;

use crate::types::{SearchResult, VectorId, VectorRecord};
use crate::vector::cosine_similarity;

// ─────────────────────────────────────────────────────────────────────────────
// VectorDB
// ─────────────────────────────────────────────────────────────────────────────

/// An in-memory vector database.
///
/// # Example
/// ```rust
/// use simple_vector_db::db::VectorDB;
///
/// let mut db = VectorDB::new();
///
/// db.insert("rust".to_string(), vec![0.9, 0.1, 0.0], Some("Rust language".to_string()));
/// db.insert("ml".to_string(),   vec![0.1, 0.2, 0.9], Some("Machine learning".to_string()));
///
/// let results = db.search(&[0.1, 0.2, 0.85], 1);
/// assert_eq!(results[0].record.id, "ml");
/// ```
pub struct VectorDB {
    store: HashMap<VectorId, VectorRecord>,
}

impl VectorDB {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create an empty [`VectorDB`].
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    /// Pre-allocate space for `capacity` records.
    ///
    /// Use this when the approximate corpus size is known upfront to avoid
    /// HashMap rehashing during bulk inserts.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            store: HashMap::with_capacity(capacity),
        }
    }

    // ── Mutating operations ───────────────────────────────────────────────────

    /// Insert (or overwrite) a record identified by `id`.
    ///
    /// If a record with the same `id` already exists it is **replaced**,
    /// which lets callers re-embed documents without first deleting them.
    ///
    /// # Arguments
    /// * `id`        — unique key (any non-empty string works: UUID, slug, path)
    /// * `embedding` — the vector to store; length must be consistent across
    ///                 all records used in a single `search` call
    /// * `metadata`  — optional human-readable payload stored alongside the vector
    ///
    /// # Returns
    /// `true` if a previous record was overwritten, `false` if this was a fresh insert.
    pub fn insert(&mut self, id: VectorId, embedding: Vec<f32>, metadata: Option<String>) -> bool {
        let record = VectorRecord::new(id.clone(), embedding, metadata);
        self.store.insert(id, record).is_some() // `insert` returns the old value if any
    }

    /// Delete the record with the given `id`.
    ///
    /// # Returns
    /// `true` if the record existed and was removed, `false` if no record
    /// had that id (no-op).
    pub fn delete(&mut self, id: &str) -> bool {
        self.store.remove(id).is_some()
    }

    /// Remove **all** records from the database.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Search for the `top_k` most similar vectors to `query`.
    ///
    /// Similarity is measured with **cosine similarity**. Results are returned
    /// sorted by descending score (most similar first).
    ///
    /// Returns fewer than `top_k` items when the database has fewer records.
    /// Returns an empty `Vec` if the database is empty or `top_k == 0`.
    ///
    /// # Arguments
    /// * `query`  — the embedding to match against; must have the same
    ///              dimension as the stored embeddings
    /// * `top_k`  — maximum number of results to return
    ///
    /// # Panics
    /// Panics (via [`cosine_similarity`]) if `query.len()` differs from the
    /// stored embedding length of any record.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if top_k == 0 || self.store.is_empty() {
            return Vec::new();
        }

        // Score every record
        let mut scored: Vec<SearchResult> = self
            .store
            .values()
            .map(|record| {
                let score = cosine_similarity(query, &record.embedding);
                SearchResult::new(record.clone(), score)
            })
            .collect();

        // Sort descending by score.
        // `partial_cmp` can return `None` for NaN, which we treat as equal.
        scored.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return at most top_k results
        scored.truncate(top_k);
        scored
    }

    /// Look up a single record by exact `id`.
    ///
    /// Returns `None` if no record with that id exists.
    pub fn get(&self, id: &str) -> Option<&VectorRecord> {
        self.store.get(id)
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    /// Return the number of stored records.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return `true` if no records are stored.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Return an iterator over all stored record ids.
    pub fn ids(&self) -> impl Iterator<Item = &VectorId> {
        self.store.keys()
    }
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── insert ────────────────────────────────────────────────────────────────

    #[test]
    fn insert_increases_len() {
        let mut db = VectorDB::new();
        assert_eq!(db.len(), 0);
        db.insert("a".to_string(), vec![1.0, 0.0], None);
        assert_eq!(db.len(), 1);
        db.insert("b".to_string(), vec![0.0, 1.0], None);
        assert_eq!(db.len(), 2);
    }

    #[test]
    fn insert_returns_false_on_fresh() {
        let mut db = VectorDB::new();
        let was_overwrite = db.insert("a".to_string(), vec![1.0], None);
        assert!(!was_overwrite);
    }

    #[test]
    fn insert_returns_true_on_overwrite() {
        let mut db = VectorDB::new();
        db.insert("a".to_string(), vec![1.0], None);
        let was_overwrite = db.insert("a".to_string(), vec![2.0], None);
        assert!(was_overwrite);
    }

    #[test]
    fn insert_overwrites_without_growing() {
        let mut db = VectorDB::new();
        db.insert("a".to_string(), vec![1.0, 0.0], None);
        db.insert("a".to_string(), vec![0.0, 1.0], Some("updated".to_string()));
        assert_eq!(db.len(), 1);
        assert_eq!(db.get("a").unwrap().metadata.as_deref(), Some("updated"));
    }

    // ── delete ────────────────────────────────────────────────────────────────

    #[test]
    fn delete_existing_returns_true() {
        let mut db = VectorDB::new();
        db.insert("a".to_string(), vec![1.0], None);
        assert!(db.delete("a"));
    }

    #[test]
    fn delete_missing_returns_false() {
        let mut db = VectorDB::new();
        assert!(!db.delete("ghost"));
    }

    #[test]
    fn delete_reduces_len() {
        let mut db = VectorDB::new();
        db.insert("a".to_string(), vec![1.0], None);
        db.insert("b".to_string(), vec![2.0], None);
        db.delete("a");
        assert_eq!(db.len(), 1);
        assert!(db.get("a").is_none());
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn clear_empties_db() {
        let mut db = VectorDB::new();
        db.insert("a".to_string(), vec![1.0], None);
        db.insert("b".to_string(), vec![2.0], None);
        db.clear();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    // ── search ────────────────────────────────────────────────────────────────

    #[test]
    fn search_empty_db_returns_empty() {
        let db = VectorDB::new();
        assert!(db.search(&[1.0, 0.0], 5).is_empty());
    }

    #[test]
    fn search_top_k_zero_returns_empty() {
        let mut db = VectorDB::new();
        db.insert("a".to_string(), vec![1.0, 0.0], None);
        assert!(db.search(&[1.0, 0.0], 0).is_empty());
    }

    #[test]
    fn search_k_larger_than_db_returns_all() {
        let mut db = VectorDB::new();
        db.insert("a".to_string(), vec![1.0, 0.0], None);
        db.insert("b".to_string(), vec![0.0, 1.0], None);
        let results = db.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 2); // only 2 docs exist
    }

    #[test]
    fn search_respects_top_k() {
        let mut db = VectorDB::new();
        for i in 0..5 {
            db.insert(format!("doc{i}"), vec![i as f32, 0.0], None);
        }
        let results = db.search(&[1.0, 0.0], 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_most_similar_is_first() {
        let mut db = VectorDB::new();
        // "close" is almost identical to the query; "far" is orthogonal
        db.insert("close".to_string(), vec![0.99, 0.01], None);
        db.insert("far".to_string(),   vec![0.0,  1.0 ], None);

        let results = db.search(&[1.0, 0.0], 2);
        assert_eq!(results[0].record.id, "close");
        assert_eq!(results[1].record.id, "far");
    }

    #[test]
    fn search_scores_descending() {
        let mut db = VectorDB::new();
        db.insert("a".to_string(), vec![1.0, 0.0, 0.0], None);
        db.insert("b".to_string(), vec![0.7, 0.7, 0.0], None);
        db.insert("c".to_string(), vec![0.0, 0.0, 1.0], None);

        let results = db.search(&[1.0, 0.0, 0.0], 3);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score, "results not sorted descending");
        }
    }

    #[test]
    fn search_identical_query_scores_one() {
        let mut db = VectorDB::new();
        let v = vec![0.6, 0.8]; // already unit-length (3-4-5 triple / 5)
        db.insert("exact".to_string(), v.clone(), None);

        let results = db.search(&v, 1);
        assert!(approx_eq(results[0].score, 1.0), "got {}", results[0].score);
    }

    // ── get ───────────────────────────────────────────────────────────────────

    #[test]
    fn get_existing_record() {
        let mut db = VectorDB::new();
        db.insert("z".to_string(), vec![1.0, 2.0], Some("meta".to_string()));
        let rec = db.get("z").expect("should exist");
        assert_eq!(rec.id, "z");
        assert_eq!(rec.metadata.as_deref(), Some("meta"));
    }

    #[test]
    fn get_missing_returns_none() {
        let db = VectorDB::new();
        assert!(db.get("missing").is_none());
    }
}