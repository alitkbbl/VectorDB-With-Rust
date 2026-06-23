//! Entry point — a small self-contained demo of the vector database.
//!
//! Run with:  cargo run

use VectorDB_with_Rust::db::VectorDB;

fn main() {
    println!("=== Simple Vector DB Demo ===\n");

    let mut db = VectorDB::new();

    // ── Insert some toy 3-D embeddings ──────────────────────────────────────
    // (In a real application these would come from a model like sentence-transformers)
    let documents = vec![
        ("rust-lang",   vec![0.9_f32, 0.1, 0.05], "Rust programming language"),
        ("python-lang", vec![0.1_f32, 0.9, 0.05], "Python programming language"),
        ("ml-basics",   vec![0.05_f32, 0.6, 0.8], "Introduction to machine learning"),
        ("deep-learn",  vec![0.02_f32, 0.5, 0.9], "Deep learning with neural networks"),
        ("databases",   vec![0.7_f32, 0.2, 0.1],  "Relational databases and SQL"),
    ];

    for (id, emb, meta) in documents {
        db.insert(id.to_string(), emb, Some(meta.to_string()));
    }

    println!("Stored {} documents.\n", db.len());

    // ── Query: something close to the ML / deep-learning cluster ────────────
    let query = vec![0.03_f32, 0.55, 0.85];
    println!("Query vector: {:?}", query);
    println!("Top-3 results:\n");

    let results = db.search(&query, 3);
    for (rank, r) in results.iter().enumerate() {
        println!(
            "  {}. [score: {:.4}]  id={:<12}  meta={}",
            rank + 1,
            r.score,
            r.record.id,
            r.record.metadata.as_deref().unwrap_or("—"),
        );
    }

    // ── Delete a record ──────────────────────────────────────────────────────
    println!("\nDeleting 'databases'…");
    let removed = db.delete("databases");
    println!("  removed: {}  |  remaining: {}", removed, db.len());
}
