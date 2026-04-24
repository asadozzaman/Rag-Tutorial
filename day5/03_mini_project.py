"""
Day 5 Mini Project: Smart document search with reranking.

This project:
  - loads technical article chunks
  - retrieves with semantic + BM25 ensemble
  - applies parent-child aggregation
  - reranks the candidate set
  - measures A/B performance before vs after reranking

Run:
  python 03_mini_project.py
"""

from __future__ import annotations

from pathlib import Path

from retrieval_utils import (
    BM25Retriever,
    HashingEmbeddings,
    PairwiseReranker,
    SemanticRetriever,
    aggregate_to_parents,
    load_documents,
    load_env_near,
    load_eval_queries,
    maybe_build_gemini_model,
    parse_self_query,
    print_results,
    reciprocal_rank,
    timed_embed_documents,
    weighted_ensemble,
)


BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = BASE_DIR / "sample_data" / "blog_chunks.csv"
QUERIES_PATH = BASE_DIR / "sample_data" / "eval_queries.csv"


def find_rank(results, expected_parent_id):
    for index, result in enumerate(results, start=1):
        if result.parent_id == expected_parent_id:
            return index
    return None


def evaluate(queries, model, semantic, bm25, reranker):
    before_hits = 0
    after_hits = 0
    before_mrr = 0.0
    after_mrr = 0.0

    print("\nA/B evaluation on labeled queries:")
    for item in queries:
        spec = parse_self_query(item["query"])
        query_vector = model.embed_query(spec.query)

        # Baseline: semantic-only top-3 parents.
        semantic_results = semantic.search(query_vector, k=12, filters=spec.filters)
        keyword_results = bm25.search(spec.query, k=12, filters=spec.filters)
        ensemble_results = weighted_ensemble([(semantic_results, 0.6), (keyword_results, 0.4)])[:10]

        before_parents = aggregate_to_parents(semantic_results[:3])[:3]
        after_rerank = reranker.rerank(spec.query, ensemble_results, top_k=10)
        after_parents = aggregate_to_parents(after_rerank)[:3]

        expected = item["expected_parent_id"]
        before_rank = find_rank(before_parents, expected)
        after_rank = find_rank(after_parents, expected)

        before_hits += 1 if before_rank is not None else 0
        after_hits += 1 if after_rank is not None else 0
        before_mrr += reciprocal_rank(before_rank)
        after_mrr += reciprocal_rank(after_rank)

        print(f"  {item['query_id']}: {item['query']}")
        print(f"    expected={expected} before_rank={before_rank} after_rank={after_rank}")

    total = len(queries)
    return {
        "before_hit3": before_hits / total,
        "after_hit3": after_hits / total,
        "before_mrr": before_mrr / total,
        "after_mrr": after_mrr / total,
    }


def interactive_loop(model, semantic, bm25, reranker):
    print("\nInteractive mode. Type 'quit' to exit.")
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

        spec = parse_self_query(query)
        query_vector = model.embed_query(spec.query)
        semantic_results = semantic.search(query_vector, k=8, filters=spec.filters)
        keyword_results = bm25.search(spec.query, k=8, filters=spec.filters)
        ensemble_results = weighted_ensemble([(semantic_results, 0.6), (keyword_results, 0.4)])[:8]
        reranked = reranker.rerank(spec.query, ensemble_results, top_k=5)
        parents = aggregate_to_parents(reranked)[:3]

        print(f"Parsed semantic query: {spec.query}")
        print(f"Parsed filters: {spec.filters}")
        print_results("Top reranked children", reranked)
        print_results("Top parent results", parents)


def main() -> None:
    load_env_near(BASE_DIR)

    documents = load_documents(DOCS_PATH)
    eval_queries = load_eval_queries(QUERIES_PATH)
    model = HashingEmbeddings(dimensions=256)

    print("=" * 84)
    print("DAY 5 MINI PROJECT: SMART DOCUMENT SEARCH WITH RERANKING")
    print("=" * 84)
    print(f"Chunks loaded: {len(documents)}")
    print(f"Evaluation queries: {len(eval_queries)}")
    print(f"Embedding model: {model.name}")

    vectors, latency_ms = timed_embed_documents(
        model,
        [doc.title + " " + doc.text for doc in documents],
    )
    print(f"Embedding build latency: {latency_ms:.1f} ms")

    semantic = SemanticRetriever(documents, vectors)
    bm25 = BM25Retriever(documents)
    reranker = PairwiseReranker(model)

    demo_query = "advanced Nora search article from 2025 about filters"
    spec = parse_self_query(demo_query)
    query_vector = model.embed_query(spec.query)
    semantic_results = semantic.search(query_vector, k=6, filters=spec.filters)
    keyword_results = bm25.search(spec.query, k=6, filters=spec.filters)
    ensemble_results = weighted_ensemble([(semantic_results, 0.6), (keyword_results, 0.4)])[:6]
    reranked_results = reranker.rerank(spec.query, ensemble_results, top_k=5)
    parent_results = aggregate_to_parents(reranked_results)[:3]

    print_results("Semantic retrieval", semantic_results)
    print_results("Keyword retrieval", keyword_results)
    print_results("Ensemble retrieval", ensemble_results)
    print_results("Reranked child results", reranked_results)
    print_results("Parent results", parent_results)

    metrics = evaluate(eval_queries, model, semantic, bm25, reranker)
    print("\nA/B summary:")
    print(f"  Baseline semantic Hit@3: {metrics['before_hit3']:.2f}")
    print(f"  Ensemble+rerank Hit@3:   {metrics['after_hit3']:.2f}")
    print(f"  Baseline semantic MRR:   {metrics['before_mrr']:.2f}")
    print(f"  Ensemble+rerank MRR:     {metrics['after_mrr']:.2f}")

    if metrics["after_mrr"] > metrics["before_mrr"]:
        print("  Reranking improved ranking quality on this dataset.")
    else:
        print("  Reranking matched or underperformed baseline; inspect candidate quality and labels.")

    interactive_loop(model, semantic, bm25, reranker)


if __name__ == "__main__":
    main()
