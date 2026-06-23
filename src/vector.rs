use crate::error::{Result, VectorDbError};

/// A stored vector: a unique id, an optional human-readable label, and the raw data.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub id: u64,
    pub label: String,
    pub data: Vec<f32>,
}

impl Vector {
    /// Create a new vector. Returns an error if `data` is empty.
    pub fn new(id: u64, label: impl Into<String>, data: Vec<f32>) -> Result<Self> {
        if data.is_empty() {
            return Err(VectorDbError::EmptyVector);
        }
        Ok(Self {
            id,
            label: label.into(),
            data,
        })
    }

    /// Number of dimensions of the vector.
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// The Euclidean (L2) magnitude of the vector.
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Return a unit-length copy of this vector.
    /// Errors if the vector has zero magnitude.
    pub fn normalized(&self) -> Result<Vec<f32>> {
        let n = self.norm();
        if n == 0.0 {
            return Err(VectorDbError::ZeroNorm);
        }
        Ok(self.data.iter().map(|x| x / n).collect())
    }
}
