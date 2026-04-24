"""
Day 5: Advanced retrieval and reranking demo.

This script shows:
  - semantic retrieval
  - BM25 retrieval
  - weighted ensemble
  - MMR
  - self-query parsing
  - parent-child aggregation
  - pairwise reranking

Run:
  python 02_code_example.py
"""

from __future__ import annotations

from pathlib import Path

from retrieval_utils import (
    BM25Retriever,
    HashingEmbeddings,
    PairwiseReranker,
    RetrievalDocument,
    SemanticRetriever,
    aggregate_to_parents,
    load_env_near,
    maybe_build_gemini_model,
    mmr_select,
    parse_self_query,
    print_results,
    timed_embed_documents,
    weighted_ensemble,
)


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    load_env_near(BASE_DIR)

    documents = [
        RetrievalDocument("c1", "p1", "Ensemble Retrieval Basics", "https://example.com/p1", "Ensemble retrieval combines BM25 and vector search so exact terms and semantic meaning both contribute.", {"category": "retrieval", "author": "samira", "year": "2025", "level": "intermediate"}),
        RetrievalDocument("c2", "p1", "Ensemble Retrieval Basics", "https://example.com/p1", "Weighted fusion helps balance semantic recall with keyword precision in production RAG pipelines.", {"category": "retrieval", "author": "samira", "year": "2025", "level": "intermediate"}),
        RetrievalDocument("c3", "p2", "MMR for Diverse Results", "https://example.com/p2", "MMR reduces redundancy by balancing relevance against similarity to already selected chunks.", {"category": "retrieval", "author": "nora", "year": "2024", "level": "advanced"}),
        RetrievalDocument("c4", "p3", "Parent Child Retrieval", "https://example.com/p3", "Parent-child retrieval embeds small children but returns larger parent sections for richer context.", {"category": "retrieval", "author": "omar", "year": "2025", "level": "advanced"}),
        RetrievalDocument("c5", "p4", "Self Query Filters", "https://example.com/p4", "Self-query retrieval separates semantic intent from structured filters such as year author and level.", {"category": "search", "author": "nora", "year": "2025", "level": "intermediate"}),
        RetrievalDocument("c6", "p5", "Reranking Precision", "https://example.com/p5", "Cross-encoder reranking scores query document pairs jointly and usually improves final precision.", {"category": "reranking", "author": "liam", "year": "2025", "level": "advanced"}),
    ]

    model = HashingEmbeddings(dimensions=256)
    vectors, latency_ms = timed_embed_documents(model, [doc.title + " " + doc.text for doc in documents])
    vector_lookup = {doc.doc_id: vector for doc, vector in zip(documents, vectors)}

    semantic = SemanticRetriever(documents, vectors)
    bm25 = BM25Retriever(documents)
    reranker = PairwiseReranker(model)

    raw_query = "advanced Nora MMR retrieval article from 2024"
    spec = parse_self_query(raw_query)
    query_vector = model.embed_query(spec.query)

    semantic_results = semantic.search(query_vector, k=4, filters=spec.filters)
    bm25_results = bm25.search(spec.query, k=4, filters=spec.filters)
    ensemble_results = weighted_ensemble([(semantic_results, 0.6), (bm25_results, 0.4)])[:5]
    mmr_results = mmr_select(query_vector, ensemble_results, vector_lookup, k=3, lambda_mult=0.6)
    parent_results = aggregate_to_parents(ensemble_results)
    reranked_results = reranker.rerank(spec.query, ensemble_results, top_k=3)

    print("=" * 78)
    print("DAY 5: ADVANCED RETRIEVAL AND RERANKING")
    print("=" * 78)
    print(f"\nEmbedding model: {model.name}")
    print(f"Embedding build latency: {latency_ms:.1f} ms")
    print(f"Raw query: {raw_query}")
    print(f"Parsed semantic query: {spec.query}")
    print(f"Parsed filters: {spec.filters}")

    print_results("Semantic retriever", semantic_results)
    print_results("BM25 retriever", bm25_results)
    print_results("Ensemble results", ensemble_results)
    print_results("MMR diverse results", mmr_results)
    print_results("Parent-child aggregated parents", parent_results)
    print_results("Reranked final results", reranked_results)

    print("\nTakeaway:")
    print("  Retrieve broadly with more than one retriever, diversify if needed, then rerank down.")


if __name__ == "__main__":
    main()
