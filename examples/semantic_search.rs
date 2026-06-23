//! Example: semantic search over a realistic mini-corpus.
//!
//! Demonstrates the full workflow:
//!   1. Encode documents as embeddings (hand-crafted here; replace with a
//!      real model such as `sentence-transformers` in production)
//!   2. Index all embeddings into `VectorDB`
//!   3. Issue several queries and display ranked results
//!   4. Show how normalising query vectors affects scores
//!
//! Run with:  cargo run --example semantic_search

use simple_vector_db::db::VectorDB;
use simple_vector_db::vector::normalize;

// ─────────────────────────────────────────────────────────────────────────────
// "Embeddings" — 8-dimensional, hand-crafted to cluster by topic
//
//  Dimension map
//  ─────────────────────────────────────────────────────
//  0-1  → systems / low-level programming
//  2-3  → high-level / scripting languages
//  4-5  → machine learning / AI
//  6-7  → web / networking
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║     Semantic Search Demo — simple-vector-db  ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // ── Step 1: define the corpus ────────────────────────────────────────────
    #[rustfmt::skip]
    let corpus: Vec<(&str, Vec<f32>, &str)> = vec![
        // id           embedding (8-D)                                   original text
        ("doc-01", vec![0.90, 0.85, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05], "Rust memory safety and ownership model"),
        ("doc-02", vec![0.80, 0.75, 0.10, 0.10, 0.05, 0.05, 0.10, 0.05], "C++ performance and zero-cost abstractions"),
        ("doc-03", vec![0.10, 0.05, 0.85, 0.90, 0.10, 0.05, 0.10, 0.05], "Python data science and pandas"),
        ("doc-04", vec![0.05, 0.05, 0.80, 0.85, 0.05, 0.05, 0.10, 0.10], "JavaScript and TypeScript for scripting"),
        ("doc-05", vec![0.10, 0.10, 0.10, 0.15, 0.90, 0.85, 0.05, 0.05], "Training deep neural networks with backprop"),
        ("doc-06", vec![0.05, 0.05, 0.15, 0.20, 0.85, 0.90, 0.05, 0.05], "Transformer architecture and self-attention"),
        ("doc-07", vec![0.05, 0.05, 0.20, 0.20, 0.80, 0.80, 0.05, 0.05], "Gradient descent and loss functions"),
        ("doc-08", vec![0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.90, 0.85], "HTTP/2 protocol and REST API design"),
        ("doc-09", vec![0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.85, 0.80], "WebSockets and real-time communication"),
        ("doc-10", vec![0.10, 0.05, 0.05, 0.05, 0.10, 0.05, 0.80, 0.75], "TCP/IP networking fundamentals"),
    ];

    // ── Step 2: build the index ──────────────────────────────────────────────
    let mut db = VectorDB::with_capacity(corpus.len());

    for (id, emb, text) in &corpus {
        db.insert(id.to_string(), emb.clone(), Some(text.to_string()));
    }

    println!("Indexed {} documents.\n", db.len());

    // ── Step 3: run semantic queries ─────────────────────────────────────────
    //
    // Each query vector lives in the same 8-D space as the corpus.
    // A real system would pass the query text through the same embedding model
    // used to embed the corpus.

    #[rustfmt::skip]
    let queries: Vec<(&str, Vec<f32>)> = vec![
        // label                             query vector (raw, not normalised)
        ("machine learning / neural nets",   vec![0.05, 0.05, 0.10, 0.15, 0.90, 0.88, 0.05, 0.05]),
        ("low-level systems programming",    vec![0.92, 0.88, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
        ("web APIs and network protocols",   vec![0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.88, 0.92]),
        ("scripting and interpreted langs",  vec![0.05, 0.05, 0.88, 0.90, 0.05, 0.05, 0.05, 0.05]),
    ];

    let top_k = 3;

    for (label, raw_query) in &queries {
        // Normalise the query so scores are purely directional
        let query = normalize(raw_query);

        println!("┌─ Query: \"{label}\"");

        let results = db.search(&query, top_k);
        for (i, r) in results.iter().enumerate() {
            let text = r.record.metadata.as_deref().unwrap_or("—");
            println!("│  {}. [{:.4}]  {}", i + 1, r.score, text);
        }
        println!("└─\n");
    }

    // ── Step 4: demonstrate normalised vs raw query ───────────────────────────
    println!("─── Score comparison: raw vs normalised query ───────────────\n");

    let raw   = vec![0.05_f32, 0.05, 0.10, 0.15, 0.90, 0.88, 0.05, 0.05];
    let normd = normalize(&raw);

    println!("Raw query magnitude  : {:.4}", simple_vector_db::vector::magnitude(&raw));
    println!("Normd query magnitude: {:.4}", simple_vector_db::vector::magnitude(&normd));
    println!();

    let r_raw   = db.search(&raw, 3);
    let r_normd = db.search(&normd, 3);

    println!("{:<35}  {:>10}  {:>10}", "doc", "raw score", "normd score");
    println!("{}", "─".repeat(60));
    for (rr, rn) in r_raw.iter().zip(r_normd.iter()) {
        let text = rr.record.metadata.as_deref().unwrap_or("—");
        // Truncate long text for display
        let short: String = text.chars().take(33).collect();
        println!("{:<35}  {:>10.4}  {:>10.4}", short, rr.score, rn.score);
    }

    println!("\nNote: ranking is identical; only the scale differs slightly");
    println!("because cosine similarity is scale-invariant by design.\n");

    // ── Step 5: simulate adding a new document ────────────────────────────────
    println!("─── Inserting a new document mid-session ─────────────────────\n");

    db.insert(
        "doc-11".to_string(),
        vec![0.50, 0.45, 0.05, 0.05, 0.88, 0.82, 0.05, 0.05],
        Some("High-performance ML inference in Rust".to_string()),
    );
    println!("Corpus now has {} documents.\n", db.len());

    // Re-run the ML query — new document should appear
    let ml_query = normalize(&vec![0.05, 0.05, 0.10, 0.15, 0.90, 0.88, 0.05, 0.05]);
    println!("Re-running ML query after insert:");
    for (i, r) in db.search(&ml_query, 4).iter().enumerate() {
        let text = r.record.metadata.as_deref().unwrap_or("—");
        let marker = if r.record.id == "doc-11" { " ◀ new" } else { "" };
        println!("  {}. [{:.4}]  {}{}", i + 1, r.score, text, marker);
    }
}