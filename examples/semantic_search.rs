//! Example: semantic search over a small toy corpus.
//!
//! In a real application the embeddings would come from a model
//! (e.g. sentence-transformers via a Python bridge or an ONNX runtime).
//! Here we hand-craft 8-dimensional embeddings that cluster by topic.
//!
//! Run with:  cargo run --example semantic_search

use VectorDB_with_Rust::db::VectorDB;

fn main() {
    let mut db = VectorDB::new();

    // ── Corpus (id, 8-D embedding, original text) ───────────────────────────
    // Dimensions (rough groupings):
    //   [0-1] programming  [2-3] data/ML  [4-5] systems  [6-7] web
    let corpus: Vec<(&str, Vec<f32>, &str)> = vec![
        ("t1", vec![0.9, 0.8, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1], "Rust ownership and borrowing"),
        ("t2", vec![0.8, 0.7, 0.1, 0.1, 0.3, 0.1, 0.2, 0.1], "Python decorators and generators"),
        ("t3", vec![0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1], "Machine learning with neural networks"),
        ("t4", vec![0.1, 0.1, 0.8, 0.9, 0.1, 0.2, 0.1, 0.1], "Deep learning and transformers"),
        ("t5", vec![0.2, 0.1, 0.1, 0.1, 0.9, 0.8, 0.1, 0.1], "Operating system kernel design"),
        ("t6", vec![0.1, 0.2, 0.1, 0.2, 0.8, 0.9, 0.1, 0.1], "Memory management and virtual memory"),
        ("t7", vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.8], "REST APIs and HTTP protocols"),
        ("t8", vec![0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.9], "WebAssembly and browser performance"),
    ];

    for (id, emb, text) in corpus {
        db.insert(id.to_string(), emb, Some(text.to_string()));
    }

    println!("Corpus loaded: {} documents\n", db.len());

    // ── Queries ──────────────────────────────────────────────────────────────
    let queries: Vec<(&str, Vec<f32>)> = vec![
        ("machine learning", vec![0.1, 0.1, 0.9, 0.85, 0.1, 0.1, 0.1, 0.1]),
        ("systems programming", vec![0.85, 0.75, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1]),
        ("web development", vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.85]),
    ];

    for (label, q) in &queries {
        println!("Query: \"{}\"\"", label);
        let results = db.search(q, 3);
        for (i, r) in results.iter().enumerate() {
            println!(
                "  {}. score={:.4}  \"{}\"\"",
                i + 1,
                r.score,
                r.record.metadata.as_deref().unwrap_or("—"),
            );
        }
        println!();
    }
}
