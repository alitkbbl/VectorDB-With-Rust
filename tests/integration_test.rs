//! Integration tests — exercise the public API end-to-end.
//!
//! These tests import `VectorDB_with_Rust` exactly as an external crate would,
//! validating the full stack: types → vector math → database operations.

use VectorDB_with_Rust::db::VectorDB;
use VectorDB_with_Rust::vector::{cosine_similarity, magnitude, normalize};

const EPS: f32 = 1e-5;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a small three-document database used by several tests below.
///
/// ```text
/// "lang-rust"  → [0.9, 0.1, 0.0]   (programming direction)
/// "lang-ml"    → [0.1, 0.2, 0.9]   (ML direction)
/// "lang-web"   → [0.0, 0.1, 0.1]   (web / networking direction)
/// ```
fn three_doc_db() -> VectorDB {
    let mut db = VectorDB::new();
    db.insert("lang-rust".to_string(), vec![0.9, 0.1, 0.0], Some("Rust".to_string()));
    db.insert("lang-ml".to_string(),   vec![0.1, 0.2, 0.9], Some("Machine learning".to_string()));
    db.insert("lang-web".to_string(),  vec![0.0, 0.1, 0.1], Some("Web protocols".to_string()));
    db
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert
// ─────────────────────────────────────────────────────────────────────────────

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
fn insert_fresh_returns_false() {
    let mut db = VectorDB::new();
    assert!(!db.insert("a".to_string(), vec![1.0], None));
}

#[test]
fn insert_overwrite_returns_true() {
    let mut db = VectorDB::new();
    db.insert("a".to_string(), vec![1.0], None);
    assert!(db.insert("a".to_string(), vec![2.0], None));
}

#[test]
fn insert_overwrite_does_not_grow_db() {
    let mut db = VectorDB::new();
    db.insert("x".to_string(), vec![1.0, 0.0], None);
    db.insert("x".to_string(), vec![0.0, 1.0], Some("new".to_string()));
    assert_eq!(db.len(), 1);
}

#[test]
fn insert_overwrite_updates_embedding() {
    let mut db = VectorDB::new();
    db.insert("x".to_string(), vec![1.0, 0.0], None);
    db.insert("x".to_string(), vec![0.0, 1.0], None);
    let rec = db.get("x").unwrap();
    assert!(approx_eq(rec.embedding[0], 0.0));
    assert!(approx_eq(rec.embedding[1], 1.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Delete
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn delete_existing_returns_true() {
    let mut db = three_doc_db();
    assert!(db.delete("lang-rust"));
}

#[test]
fn delete_nonexistent_returns_false() {
    let mut db = VectorDB::new();
    assert!(!db.delete("ghost"));
}

#[test]
fn delete_reduces_len_by_one() {
    let mut db = three_doc_db();
    let before = db.len();
    db.delete("lang-ml");
    assert_eq!(db.len(), before - 1);
}

#[test]
fn deleted_record_not_retrievable() {
    let mut db = three_doc_db();
    db.delete("lang-rust");
    assert!(db.get("lang-rust").is_none());
}

#[test]
fn deleted_record_not_in_search_results() {
    let mut db = three_doc_db();
    db.delete("lang-rust");

    let results = db.search(&[0.9, 0.1, 0.0], 5);
    let found = results.iter().any(|r| r.record.id == "lang-rust");
    assert!(!found, "deleted record appeared in search results");
}

// ─────────────────────────────────────────────────────────────────────────────
// Search
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn search_empty_db_returns_empty() {
    let db = VectorDB::new();
    assert!(db.search(&[1.0, 0.0], 3).is_empty());
}

#[test]
fn search_top_k_zero_returns_empty() {
    let db = three_doc_db();
    assert!(db.search(&[1.0, 0.0, 0.0], 0).is_empty());
}

#[test]
fn search_k_larger_than_db_returns_all() {
    let db = three_doc_db(); // 3 docs
    let results = db.search(&[1.0, 0.0, 0.0], 100);
    assert_eq!(results.len(), 3);
}

#[test]
fn search_respects_top_k_limit() {
    let db = three_doc_db();
    let results = db.search(&[1.0, 0.0, 0.0], 2);
    assert_eq!(results.len(), 2);
}

#[test]
fn search_results_sorted_descending() {
    let db = three_doc_db();
    let results = db.search(&[0.9, 0.1, 0.0], 3);
    for pair in results.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "results not sorted: {} < {}",
            pair[0].score,
            pair[1].score
        );
    }
}

#[test]
fn search_most_similar_doc_is_first() {
    let db = three_doc_db();
    // Query is essentially "lang-rust"
    let results = db.search(&[0.9, 0.1, 0.0], 3);
    assert_eq!(results[0].record.id, "lang-rust");
}

#[test]
fn search_exact_match_scores_one() {
    let mut db = VectorDB::new();
    let v = vec![0.6_f32, 0.8]; // unit vector (3-4-5 / 5)
    db.insert("unit".to_string(), v.clone(), None);

    let results = db.search(&v, 1);
    assert!(
        approx_eq(results[0].score, 1.0),
        "exact match score was {}",
        results[0].score
    );
}

#[test]
fn search_orthogonal_vector_scores_zero() {
    let mut db = VectorDB::new();
    db.insert("a".to_string(), vec![1.0, 0.0], None);

    let results = db.search(&[0.0, 1.0], 1); // orthogonal query
    assert!(
        approx_eq(results[0].score, 0.0),
        "orthogonal score was {}",
        results[0].score
    );
}

#[test]
fn search_scores_stay_in_valid_range() {
    let db = three_doc_db();
    let results = db.search(&[0.5, 0.5, 0.5], 3);
    for r in &results {
        assert!(
            r.score >= -1.0 && r.score <= 1.0,
            "score {} out of [-1, 1]",
            r.score
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector math (public API)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn public_cosine_similarity_accessible() {
    let a = vec![1.0_f32, 0.0];
    let b = vec![1.0_f32, 0.0];
    assert!(approx_eq(cosine_similarity(&a, &b), 1.0));
}

#[test]
fn public_normalize_produces_unit_vector() {
    let v = normalize(&[3.0_f32, 4.0]);
    assert!(approx_eq(magnitude(&v), 1.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Full workflow: insert → search → delete → search
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn full_workflow() {
    let mut db = VectorDB::new();

    // 1. Insert
    db.insert("cat".to_string(),  vec![0.9, 0.1], Some("A small furry animal".to_string()));
    db.insert("dog".to_string(),  vec![0.8, 0.2], Some("A loyal companion".to_string()));
    db.insert("fish".to_string(), vec![0.1, 0.9], Some("An aquatic animal".to_string()));
    assert_eq!(db.len(), 3);

    // 2. Search — expect "cat" and "dog" near [0.9, 0.1]
    let r = db.search(&[0.9, 0.1], 2);
    assert_eq!(r.len(), 2);
    let top_ids: Vec<&str> = r.iter().map(|x| x.record.id.as_str()).collect();
    assert!(top_ids.contains(&"cat"));
    assert!(top_ids.contains(&"dog"));
    assert!(!top_ids.contains(&"fish"));

    // 3. Delete "cat"
    assert!(db.delete("cat"));
    assert_eq!(db.len(), 2);

    // 4. Search again — "dog" should now be top
    let r2 = db.search(&[0.9, 0.1], 2);
    assert_eq!(r2[0].record.id, "dog");
}