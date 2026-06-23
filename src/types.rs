//! Shared data types used throughout the vector database.

/// Unique identifier for a stored vector (owned `String` for simplicity).
pub type VectorId = String;

/// A single record stored inside [`crate::db::VectorDB`].
#[derive(Debug, Clone)]
pub struct VectorRecord {
    /// Unique identifier for this record.
    pub id: VectorId,

    /// The raw embedding values.
    pub embedding: Vec<f32>,

    /// Optional free-form metadata (e.g. the original text, a URL, tags …).
    pub metadata: Option<String>,
}

impl VectorRecord {
    /// Create a new [`VectorRecord`].
    pub fn new(id: VectorId, embedding: Vec<f32>, metadata: Option<String>) -> Self {
        Self { id, embedding, metadata }
    }
}

/// A single result returned by [`crate::db::VectorDB::search`].
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matching record.
    pub record: VectorRecord,

    /// Cosine similarity score in the range [−1, 1] (higher = more similar).
    pub score: f32,
}

impl SearchResult {
    pub fn new(record: VectorRecord, score: f32) -> Self {
        Self { record, score }
    }
}
