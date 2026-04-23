"""
Day 4: Vector index comparison.

This script compares:
  - flat exact vector search
  - clustered approximate search (IVF-style idea)
  - metadata filtering

Run:
  python 02_code_example.py
"""

from __future__ import annotations

from pathlib import Path

from search_utils import (
    ClusteredVectorIndex,
    FlatVectorIndex,
    HashingEmbeddings,
    SearchDocument,
    maybe_build_gemini_model,
    print_results,
    timed_embed_documents,
    load_env_near,
)


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    load_env_near(BASE_DIR)

    documents = [
        SearchDocument("d1", "Python is a versatile programming language for backend services, data pipelines, and AI work.", {"lang": "python", "type": "general"}),
        SearchDocument("d2", "JavaScript powers interactive web applications in the browser.", {"lang": "javascript", "type": "web"}),
        SearchDocument("d3", "Rust provides memory safety and strong systems programming ergonomics.", {"lang": "rust", "type": "systems"}),
        SearchDocument("d4", "Go is often used for concurrent network services and backend systems.", {"lang": "go", "type": "systems"}),
        SearchDocument("d5", "TypeScript adds types and tooling support to JavaScript projects.", {"lang": "typescript", "type": "web"}),
        SearchDocument("d6", "PostgreSQL with pgvector lets teams add vector search inside SQL workflows.", {"lang": "sql", "type": "database"}),
    ]

    model = maybe_build_gemini_model() or HashingEmbeddings(dimensions=256)
    texts = [document.text for document in documents]
    vectors, latency_ms = timed_embed_documents(model, texts)

    flat_index = FlatVectorIndex(documents, vectors)
    clustered_index = ClusteredVectorIndex.build(documents, vectors, n_lists=3)

    query = "best typed language for browser web app development"
    query_vector = model.embed_query(query)

    print("=" * 72)
    print("DAY 4: VECTOR INDEX COMPARISON")
    print("=" * 72)
    print(f"\nEmbedding model: {model.name}")
    print(f"Index build embedding latency: {latency_ms:.1f} ms")
    print(f"Query: {query}")

    flat_results = flat_index.search(query_vector, k=3)
    print_results("Flat exact search", flat_results)

    approx_results = clustered_index.search(query_vector, k=3, n_probe=1)
    print_results("Clustered approximate search (1 probe)", approx_results)

    filtered_results = flat_index.search(
        query_vector,
        k=3,
        filters={"type": "systems"},
    )
    print_results("Metadata filtered search (systems only)", filtered_results)

    flat_index.save(BASE_DIR / ".search_cache" / "flat_index.json")
    reloaded_index = FlatVectorIndex.load(BASE_DIR / ".search_cache" / "flat_index.json")
    persisted_results = reloaded_index.search(model.embed_query("sql vector search"), k=2)
    print_results("Reloaded persistent index", persisted_results)

    print("\nTakeaway:")
    print("  Flat search is exact. Clustered search is faster in spirit but can miss some results.")
    print("  Metadata filtering narrows the search space before ranking.")


if __name__ == "__main__":
    main()
