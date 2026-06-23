# Simple Vector Database in Rust

A lightweight, **zero-dependency** vector database for storing embeddings and
performing efficient semantic similarity search.

## Features

- In-memory vector storage (HashMap-backed)
- Cosine similarity implemented from scratch
- Top-*k* nearest-neighbor retrieval
- Simple API: `insert`, `search`, `delete`

## Project Layout

```
VectorDB-with-Rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs          # Public API surface
│   ├── main.rs         # CLI demo
│   ├── db.rs           # VectorDB — insert / search / delete
│   ├── vector.rs       # Vector math — cosine similarity
│   └── types.rs        # VectorRecord, SearchResult, VectorId
├── examples/
│   └── semantic_search.rs
└── tests/
    └── integration_test.rs
```

## Quick Start

```bash
cargo run                    # run the built-in demo
cargo test                   # run all tests
cargo run --example semantic_search
```

## API Overview

```rust
use VectorDB_with_Rust::db::VectorDB;

let mut db = VectorDB::new();

// Insert a named embedding
db.insert("doc1".to_string(), vec![0.1, 0.8, 0.3], None);

// Search for the top-3 most similar vectors
let results = db.search(&[0.15, 0.75, 0.25], 3);
for r in results {
    println!("{} — score: {:.4}", r.id, r.score);
}
```
