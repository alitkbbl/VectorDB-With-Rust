use std::fmt;

/// Errors that can occur during vector database operations.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorDbError {
    /// A query or inserted vector did not match the database dimension.
    DimensionMismatch { expected: usize, got: usize },
    /// A vector with the given id was not found.
    NotFound(u64),
    /// An operation required a non-empty vector but received an empty one.
    EmptyVector,
    /// A search was attempted on an empty database.
    EmptyDatabase,
    /// A vector with a zero magnitude cannot be normalized or compared with cosine.
    ZeroNorm,
}

impl fmt::Display for VectorDbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorDbError::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected}, got {got}"
            ),
            VectorDbError::NotFound(id) => write!(f, "vector with id {id} not found"),
            VectorDbError::EmptyVector => write!(f, "vector data is empty"),
            VectorDbError::EmptyDatabase => write!(f, "cannot search an empty database"),
            VectorDbError::ZeroNorm => write!(f, "vector has zero magnitude"),
        }
    }
}

impl std::error::Error for VectorDbError {}

/// Convenience result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, VectorDbError>;
