//! Core vector database — insert, search (top-*k*), and delete operations.

use std::collections::HashMap;

use crate::types::{SearchResult, VectorId, VectorRecord};
use crate::vector::cosine_similarity;

/// An in-memory vector database backed by a [`HashMap`].
///
/// # Example
/// ```rust
/// use VectorDB_with_Rust::db::VectorDB;
///
/// let mut db = VectorDB::new();
/// db.insert("doc1".to_string(), vec![0.1, 0.9], None);
/// let results = db.search(&[0.2, 0.8], 1);
/// assert_eq!(results[0].record.id, "doc1");
/// ```
pub struct VectorDB {
    /// Internal storage: id → record.
    store: HashMap<VectorId, VectorRecord>,
}

impl VectorDB {
    /// Create an empty [`VectorDB`].
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    /// Insert (or overwrite) a vector with the given `id`.
    ///
    /// # Arguments
    /// * `id`        — unique identifier for the record
    /// * `embedding` — the raw embedding values
    /// * `metadata`  — optional free-form metadata string
    pub fn insert(&mut self, id: VectorId, embedding: Vec<f32>, metadata: Option<String>) {
        // TODO: implement
        todo!("insert")
    }

    /// Search for the `top_k` most similar vectors to `query`.
    ///
    /// Returns results sorted by descending cosine similarity.
    /// Returns fewer than `top_k` results if the database has fewer entries.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        // TODO: implement
        // Hint: score every record with cosine_similarity, collect, sort desc, take top_k
        todo!("search")
    }

    /// Delete the record with the given `id`.
    ///
    /// Returns `true` if the record existed, `false` otherwise.
    pub fn delete(&mut self, id: &str) -> bool {
        // TODO: implement
        todo!("delete")
    }

    /// Return the number of stored vectors.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return `true` if the database contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_len() {
        // TODO: add assertions
    }

    #[test]
    fn test_delete_existing() {
        // TODO: add assertions
    }

    #[test]
    fn test_search_returns_top_k() {
        // TODO: add assertions
    }

    #[test]
    fn test_search_ordering() {
        // Closest vector should be first
        // TODO: add assertions
    }
}
