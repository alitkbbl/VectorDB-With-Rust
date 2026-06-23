//! # VectorDB-with-Rust
//!
//! A lightweight, **zero-dependency** in-memory vector database for semantic
//! similarity search, implemented entirely in safe Rust.
//!
//! ## Quick start
//!
//! ```rust
//! use VectorDB_with_Rust::db::VectorDB;
//!
//! let mut db = VectorDB::new();
//!
//! // Store embeddings (in practice: output of a sentence-transformer model)
//! db.insert("rust-book".to_string(),   vec![0.9, 0.1, 0.0], Some("The Rust Programming Language".to_string()));
//! db.insert("ml-crash".to_string(),    vec![0.1, 0.9, 0.2], Some("Machine Learning Crash Course".to_string()));
//! db.insert("deep-learn".to_string(),  vec![0.0, 0.8, 0.9], Some("Deep Learning".to_string()));
//!
//! // Query: find the 2 most semantically similar documents
//! let query  = vec![0.05, 0.85, 0.85]; // "neural networks and deep learning"
//! let top2   = db.search(&query, 2);
//!
//! println!("Most similar: {}", top2[0].record.metadata.as_deref().unwrap());
//! ```
//!
//! ## Modules
//!
//! | Module         | Contents |
//! |----------------|----------|
//! | [`types`]      | `VectorId`, `VectorRecord`, `SearchResult` |
//! | [`vector`]     | `dot_product`, `magnitude`, `cosine_similarity`, `normalize` |
//! | [`db`]         | `VectorDB` — insert / search / delete / get |

pub mod db;
pub mod types;
pub mod vector;
