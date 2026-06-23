//! Shared data types used throughout the vector database.
//!
//! Dependency order: no internal deps — imported by every other module.

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// VectorId
// ─────────────────────────────────────────────────────────────────────────────

/// Unique identifier for a stored vector.
///
/// Using an owned `String` keeps the API ergonomic: callers can pass
/// `"my-doc".to_string()`, a UUID string, a file path, etc.
pub type VectorId = String;

// ─────────────────────────────────────────────────────────────────────────────
// VectorRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry stored inside [`crate::db::VectorDB`].
///
/// Holds an id, the raw embedding, and an optional metadata payload.
/// Metadata is intentionally stringly-typed so this crate stays dependency-free
/// (no `serde`, no JSON). Callers may serialise richer data themselves and store
/// the result as the metadata string.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorRecord {
    /// Unique identifier for this record.
    pub id: VectorId,

    /// The raw embedding values (f32 for memory efficiency).
    pub embedding: Vec<f32>,

    /// Optional free-form metadata — e.g. the original text, a URL, a tag list.
    pub metadata: Option<String>,
}

impl VectorRecord {
    /// Create a new [`VectorRecord`].
    ///
    /// # Arguments
    /// * `id`        — unique key used to address this record
    /// * `embedding` — the vector to store
    /// * `metadata`  — optional human-readable payload
    pub fn new(id: VectorId, embedding: Vec<f32>, metadata: Option<String>) -> Self {
        Self { id, embedding, metadata }
    }

    /// Return the number of dimensions in this vector.
    pub fn dim(&self) -> usize {
        self.embedding.len()
    }
}

impl fmt::Display for VectorRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VectorRecord {{ id: {:?}, dim: {}, metadata: {:?} }}",
            self.id,
            self.dim(),
            self.metadata,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SearchResult
// ─────────────────────────────────────────────────────────────────────────────

/// One result returned by [`crate::db::VectorDB::search`].
///
/// Results are ranked by `score` in **descending** order (highest = most similar).
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matching record (id, embedding, metadata).
    pub record: VectorRecord,

    /// Cosine similarity score in the range [−1.0, 1.0].
    ///
    /// - `1.0`  → perfectly identical direction
    /// - `0.0`  → orthogonal (no similarity)
    /// - `−1.0` → diametrically opposite
    pub score: f32,
}

impl SearchResult {
    /// Wrap a `record` + `score` pair.
    pub fn new(record: VectorRecord, score: f32) -> Self {
        Self { record, score }
    }
}

impl fmt::Display for SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SearchResult {{ id: {:?}, score: {:.4} }}",
            self.record.id, self.score
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_record_dim() {
        let r = VectorRecord::new("a".to_string(), vec![1.0, 2.0, 3.0], None);
        assert_eq!(r.dim(), 3);
    }

    #[test]
    fn vector_record_display() {
        let r = VectorRecord::new("doc1".to_string(), vec![0.1, 0.2], Some("hello".to_string()));
        let s = format!("{}", r);
        assert!(s.contains("doc1"));
        assert!(s.contains("dim: 2"));
    }

    #[test]
    fn search_result_display() {
        let r  = VectorRecord::new("x".to_string(), vec![1.0], None);
        let sr = SearchResult::new(r, 0.9876);
        let s  = format!("{}", sr);
        assert!(s.contains("0.9876"));
    }
}