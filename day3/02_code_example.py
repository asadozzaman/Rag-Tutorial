"""
Day 3: Embedding similarity comparison.

This script compares:
  - a keyword-overlap baseline
  - a hashing baseline
  - Gemini embeddings, if GEMINI_API_KEY is available

Run:
  python 02_code_example.py
"""

from __future__ import annotations

from pathlib import Path

from embedding_utils import (
    HashingEmbeddings,
    KeywordOverlapEmbeddings,
    build_vocabulary,
    cosine_similarity,
    load_env_near,
    maybe_build_gemini_model,
    timed_embed_documents,
)


BASE_DIR = Path(__file__).resolve().parent


def rank_documents(query_vector: list[float], doc_vectors: list[list[float]], docs: list[str]):
    scored = []
    for doc, vector in zip(docs, doc_vectors):
        scored.append((cosine_similarity(query_vector, vector), doc))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def print_rankings(model, query: str, docs: list[str]) -> None:
    all_texts = [query, *docs]
    result = timed_embed_documents(model, all_texts)
    query_vector = result.vectors[0]
    doc_vectors = result.vectors[1:]
    rankings = rank_documents(query_vector, doc_vectors, docs)

    print(f"\n=== {model.name} ===")
    print(f"Dimensions: {len(query_vector)}")
    print(f"Embedding latency: {result.latency_ms:.1f} ms")
    for score, doc in rankings:
        print(f"  {score: .4f} | {doc}")


def main() -> None:
    load_env_near(BASE_DIR)

    query = "How can I build a RAG pipeline?"
    docs = [
        "Step-by-step guide to retrieval augmented generation",
        "Building RAG systems with chunking embeddings and vector search",
        "Best pizza restaurants in New York City",
        "Introduction to neural network training loops",
        "How to prepare strong coffee at home",
    ]

    vocabulary = build_vocabulary([query, *docs])
    models = [
        KeywordOverlapEmbeddings(vocabulary=vocabulary),
        HashingEmbeddings(dimensions=256),
    ]

    gemini = maybe_build_gemini_model()
    if gemini:
        models.append(gemini)
    else:
        print("[INFO] No GEMINI_API_KEY found, skipping Gemini embedding model.")

    print("=" * 66)
    print("DAY 3: EMBEDDING MODEL SIMILARITY COMPARISON")
    print("=" * 66)
    print(f"\nQuery: {query}")

    for model in models:
        print_rankings(model, query, docs)

    print("\nTakeaway:")
    print("  The best embedding model should rank the RAG-related documents highest.")
    print("  Local baselines are useful for learning, but real semantic models usually win.")


if __name__ == "__main__":
    main()
