//! # VectorDB-with-Rust
//!
//! A lightweight in-memory vector database for semantic similarity search.
//!
//! ## Modules
//! - [`types`]  — shared data types (`VectorRecord`, `SearchResult`, `VectorId`)
//! - [`vector`] — vector math and cosine similarity
//! - [`db`]     — the `VectorDB` struct (insert / search / delete)

pub mod db;
pub mod types;
pub mod vector;
