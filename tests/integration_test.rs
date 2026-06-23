//! Integration tests for `VectorDB-with-Rust`.
//!
//! These tests exercise the public API end-to-end (insert → search → delete).

use VectorDB_with_Rust::db::VectorDB;

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

#[test]
fn insert_increases_len() {
    // TODO: implement
}

#[test]
fn insert_overwrites_existing_id() {
    // TODO: implement
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[test]
fn search_returns_correct_top_k_count() {
    // TODO: implement
}

#[test]
fn search_most_similar_is_first() {
    // TODO: implement
}

#[test]
fn search_empty_db_returns_empty_vec() {
    let db = VectorDB::new();
    let results = db.search(&[1.0, 0.0], 3);
    assert!(results.is_empty());
}

#[test]
fn search_k_larger_than_db_returns_all() {
    // TODO: implement
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

#[test]
fn delete_existing_returns_true() {
    // TODO: implement
}

#[test]
fn delete_nonexistent_returns_false() {
    let mut db = VectorDB::new();
    assert!(!db.delete("ghost"));
}

#[test]
fn delete_reduces_len() {
    // TODO: implement
}
