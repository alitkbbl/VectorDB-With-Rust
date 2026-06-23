//! Demo binary — exercises the full VectorDB API end-to-end.
//!
//! Run with:  cargo run

use VectorDB_with_Rust::db::VectorDB;

fn main() {
    println!("╔══════════════════════════════════════╗");
    println!("║           Vector DB — Rust           ║");
    println!("╚══════════════════════════════════════╝\n");

    // ── 1. Build the database ─────────────────────────────────────────────────
    //
    // In a real application the vectors would come from an embedding model such
    // as `sentence-transformers`.  Here we use hand-crafted 6-D vectors whose
    // values cluster by topic so the similarity results are intuitive:
    //
    //  dim 0-1 → programming languages
    //  dim 2-3 → machine learning / AI
    //  dim 4-5 → web / networking
    //
    let mut db = VectorDB::with_capacity(8);

    let corpus: &[(&str, [f32; 6], &str)] = &[
        ("rust",        [0.95, 0.80, 0.05, 0.05, 0.10, 0.05], "Rust systems programming language"),
        ("python",      [0.80, 0.70, 0.20, 0.15, 0.10, 0.10], "Python programming language"),
        ("ml-intro",    [0.10, 0.10, 0.90, 0.85, 0.05, 0.05], "Introduction to machine learning"),
        ("deep-learn",  [0.05, 0.10, 0.85, 0.95, 0.05, 0.05], "Deep learning and neural networks"),
        ("transformers",[0.10, 0.15, 0.75, 0.90, 0.10, 0.10], "Transformer architecture and attention"),
        ("http",        [0.10, 0.05, 0.05, 0.05, 0.90, 0.85], "HTTP protocol and REST APIs"),
        ("websockets",  [0.10, 0.10, 0.10, 0.05, 0.80, 0.90], "WebSockets and real-time communication"),
        ("databases",   [0.30, 0.20, 0.20, 0.15, 0.40, 0.35], "Relational databases and SQL"),
    ];

    for (id, emb, meta) in corpus {
        db.insert(id.to_string(), emb.to_vec(), Some(meta.to_string()));
    }

    println!("✓ Stored {} documents\n", db.len());

    // ── 2. Run semantic queries ───────────────────────────────────────────────

    let queries: &[(&str, [f32; 6])] = &[
        ("machine learning",       [0.05, 0.05, 0.90, 0.90, 0.05, 0.05]),
        ("systems programming",    [0.95, 0.85, 0.05, 0.05, 0.10, 0.05]),
        ("real-time web protocols",[0.05, 0.05, 0.05, 0.05, 0.85, 0.95]),
    ];

    for (label, q) in queries {
        println!("┌─ Query: \"{label}\"");
        let results = db.search(q, 3);
        for (rank, r) in results.iter().enumerate() {
            println!(
                "│  {}. score={:.4}  [{}]  {}",
                rank + 1,
                r.score,
                r.record.id,
                r.record.metadata.as_deref().unwrap_or("—"),
            );
        }
        println!("└─\n");
    }

    // ── 3. Direct lookup ──────────────────────────────────────────────────────
    println!("── Direct lookup: \"rust\" ─────────────────");
    match db.get("rust") {
        Some(rec) => println!("  Found: {rec}"),
        None      => println!("  Not found"),
    }
    println!();

    // ── 4. Delete a record ────────────────────────────────────────────────────
    println!("── Delete \"databases\" ────────────────────");
    let removed = db.delete("databases");
    println!("  removed={removed}  remaining={}", db.len());
    println!("  lookup after delete: {:?}\n", db.get("databases"));

    // ── 5. Overwrite (re-embed) ───────────────────────────────────────────────
    println!("── Overwrite \"rust\" with updated embedding ");
    let overwritten = db.insert(
        "rust".to_string(),
        vec![0.97, 0.82, 0.03, 0.03, 0.08, 0.04],
        Some("Rust — memory-safe systems language (updated)".to_string()),
    );
    println!("  was_overwrite={overwritten}  len still={}", db.len());
    println!("  new meta: {:?}", db.get("rust").unwrap().metadata);
}