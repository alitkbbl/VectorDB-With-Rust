//! Low-level vector mathematics.
//!
//! All functions operate on `&[f32]` slices so they work with any backing storage.

use std::panic;

/// Compute the dot product of two equal-length vectors.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    // TODO: implement
    todo!("dot_product")
}

/// Compute the Euclidean magnitude (L2 norm) of a vector.
pub fn magnitude(v: &[f32]) -> f32 {
    // TODO: implement
    todo!("magnitude")
}

/// Compute the cosine similarity between two vectors.
///
/// Returns a value in [−1, 1]:
/// - `1.0`  → identical direction
/// - `0.0`  → orthogonal
/// - `−1.0` → opposite direction
///
/// Returns `0.0` when either vector is the zero vector.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // TODO: implement
    todo!("cosine_similarity")
}

/// Return a unit-length copy of `v` (L2 normalised).
///
/// Returns the zero vector unchanged.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    // TODO: implement
    todo!("normalize")
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_basic() {
        // TODO: add assertions
    }

    #[test]
    fn test_magnitude_basic() {
        // TODO: add assertions
    }

    #[test]
    fn test_cosine_identical_vectors() {
        // Identical vectors should have similarity ≈ 1.0
        // TODO: add assertions
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        // Orthogonal vectors should have similarity ≈ 0.0
        // TODO: add assertions
    }

    #[test]
    fn test_normalize_unit_length() {
        // Normalised vector should have magnitude ≈ 1.0
        // TODO: add assertions
    }
}
