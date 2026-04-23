"""
Day 4 Mini Project: Hybrid Search Engine.

This project lets you compare:
  - semantic-only retrieval
  - keyword-only retrieval (BM25)
  - hybrid retrieval with reciprocal rank fusion
  - metadata filtering by category, author, year, or level

Run:
  python 03_mini_project.py
"""

from __future__ import annotations

from pathlib import Path

from search_utils import (
    BM25Index,
    ClusteredVectorIndex,
    FlatVectorIndex,
    HashingEmbeddings,
    load_documents_from_csv,
    load_env_near,
    maybe_build_gemini_model,
    print_results,
    reciprocal_rank_fusion,
    timed_embed_documents,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sample_data" / "hybrid_docs.csv"
CACHE_PATH = BASE_DIR / ".search_cache" / "hybrid_flat_index.json"


def run_search(mode, query, filters, model, flat_index, clustered_index, bm25_index):
    if mode == "semantic":
        query_vector = model.embed_query(query)
        return flat_index.search(query_vector, k=5, filters=filters)

    if mode == "keyword":
        return bm25_index.search(query, k=5, filters=filters)

    if mode == "hybrid":
        query_vector = model.embed_query(query)
        semantic_results = clustered_index.search(query_vector, k=5, filters=filters, n_probe=2)
        keyword_results = bm25_index.search(query, k=5, filters=filters)
        return reciprocal_rank_fusion([semantic_results, keyword_results])[:5]

    raise ValueError(f"Unsupported mode: {mode}")


def print_mode_results(mode, query, filters, results):
    label = f"{mode.upper()} | query='{query}'"
    if filters:
        label += f" | filters={filters}"
    print_results(label, results)


def interactive_loop(model, flat_index, clustered_index, bm25_index):
    print("\nInteractive mode. Modes: semantic, keyword, hybrid. Type 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if not query:
            continue

        mode = input("Mode [semantic/keyword/hybrid]: ").strip().lower() or "hybrid"
        category = input("Optional category filter: ").strip()
        author = input("Optional author filter: ").strip()
        year = input("Optional year filter: ").strip()
        level = input("Optional level filter: ").strip()

        filters = {}
        if category:
            filters["category"] = category
        if author:
            filters["author"] = author
        if year:
            filters["year"] = year
        if level:
            filters["level"] = level

        results = run_search(mode, query, filters or None, model, flat_index, clustered_index, bm25_index)
        print_mode_results(mode, query, filters or None, results)


def main() -> None:
    load_env_near(BASE_DIR)

    documents = load_documents_from_csv(DATA_PATH)
    model = maybe_build_gemini_model() or HashingEmbeddings(dimensions=256)

    print("=" * 84)
    print("DAY 4 MINI PROJECT: HYBRID SEARCH ENGINE")
    print("=" * 84)
    print(f"Documents loaded: {len(documents)}")
    print(f"Semantic model: {model.name}")

    texts = [document.text for document in documents]
    vectors, latency_ms = timed_embed_documents(model, texts)
    print(f"Embedding build latency: {latency_ms:.1f} ms")

    flat_index = FlatVectorIndex(documents, vectors)
    clustered_index = ClusteredVectorIndex.build(documents, vectors, n_lists=4)
    bm25_index = BM25Index(documents)

    flat_index.save(CACHE_PATH)
    reloaded_flat = FlatVectorIndex.load(CACHE_PATH)
    persisted = reloaded_flat.search(model.embed_query("vector database scaling"), k=2)
    print_results("Reloaded saved index", persisted)

    demo_cases = [
        ("semantic", "how do i scale vector search in production", None),
        ("keyword", "bm25 keyword ranking", None),
        ("hybrid", "metadata filters for finance reports", {"category": "finance"}),
        ("hybrid", "web retrieval for api docs", {"category": "docs", "level": "intermediate"}),
    ]

    for mode, query, filters in demo_cases:
        results = run_search(mode, query, filters, model, flat_index, clustered_index, bm25_index)
        print_mode_results(mode, query, filters, results)

    print("\nModes summary:")
    print("  semantic = vector similarity only")
    print("  keyword  = BM25 only")
    print("  hybrid   = clustered semantic + BM25 merged with RRF")

    interactive_loop(model, flat_index, clustered_index, bm25_index)


if __name__ == "__main__":
    main()
