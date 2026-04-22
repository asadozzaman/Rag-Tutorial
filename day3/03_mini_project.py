"""
Day 3 Mini Project: Embedding Benchmark Suite.

This benchmark:
  - loads query-document relevance pairs
  - evaluates multiple embedding models
  - computes Recall@K, MRR, latency, and vector dimensions
  - prints a recommendation based on retrieval quality

Run:
  python 03_mini_project.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
DATA_PATH = BASE_DIR / "sample_data" / "embedding_eval.csv"


@dataclass
class EvalQuery:
    query_id: str
    query: str
    relevant_doc_ids: set[str]


@dataclass
class EvalDocument:
    doc_id: str
    text: str
    topic: str


@dataclass
class ModelResult:
    model_name: str
    dimensions: int
    recall_at_1: float
    recall_at_3: float
    mrr: float
    latency_ms: float


def load_eval_dataset(path: Path) -> tuple[list[EvalQuery], list[EvalDocument]]:
    queries: dict[str, EvalQuery] = {}
    docs: dict[str, EvalDocument] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query_id = row["query_id"]
            doc_id = row["doc_id"]

            if query_id not in queries:
                queries[query_id] = EvalQuery(
                    query_id=query_id,
                    query=row["query"],
                    relevant_doc_ids=set(),
                )

            if row["is_relevant"].strip() == "1":
                queries[query_id].relevant_doc_ids.add(doc_id)

            docs[doc_id] = EvalDocument(
                doc_id=doc_id,
                text=row["doc_text"],
                topic=row["topic"],
            )

    return list(queries.values()), list(docs.values())


def rank_docs(query_vector: list[float], doc_vectors: list[list[float]], docs: list[EvalDocument]):
    scored = []
    for doc, doc_vector in zip(docs, doc_vectors):
        scored.append((doc.doc_id, cosine_similarity(query_vector, doc_vector), doc.text))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def evaluate_model(model, queries: list[EvalQuery], docs: list[EvalDocument]) -> ModelResult:
    all_texts = [doc.text for doc in docs] + [query.query for query in queries]
    timed = timed_embed_documents(model, all_texts)

    doc_vectors = timed.vectors[: len(docs)]
    query_vectors = timed.vectors[len(docs) :]
    dimensions = len(timed.vectors[0])

    recall_1_hits = 0
    recall_3_hits = 0
    reciprocal_ranks = []

    print(f"\nTop results for {model.name}:")
    for query, query_vector in zip(queries, query_vectors):
        rankings = rank_docs(query_vector, doc_vectors, docs)
        ranked_doc_ids = [doc_id for doc_id, _, _ in rankings]

        if any(doc_id in query.relevant_doc_ids for doc_id in ranked_doc_ids[:1]):
            recall_1_hits += 1
        if any(doc_id in query.relevant_doc_ids for doc_id in ranked_doc_ids[:3]):
            recall_3_hits += 1

        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in query.relevant_doc_ids:
                reciprocal_rank = 1 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

        top_doc_id, top_score, top_text = rankings[0]
        print(f"  {query.query_id}: {query.query}")
        print(f"    top1={top_doc_id} score={top_score:.4f} text={top_text[:80]}...")

    total = len(queries)
    return ModelResult(
        model_name=model.name,
        dimensions=dimensions,
        recall_at_1=recall_1_hits / total,
        recall_at_3=recall_3_hits / total,
        mrr=sum(reciprocal_ranks) / total,
        latency_ms=timed.latency_ms,
    )


def print_results(results: list[ModelResult]) -> None:
    print("\n" + "=" * 86)
    print("EMBEDDING BENCHMARK RESULTS")
    print("=" * 86)
    print(
        f"{'Model':<28} {'Dims':>8} {'R@1':>8} {'R@3':>8} "
        f"{'MRR':>8} {'Latency ms':>12}"
    )
    print("-" * 86)

    for result in results:
        print(
            f"{result.model_name:<28} {result.dimensions:>8} "
            f"{result.recall_at_1:>8.2f} {result.recall_at_3:>8.2f} "
            f"{result.mrr:>8.2f} {result.latency_ms:>12.1f}"
        )


def recommend(results: list[ModelResult]) -> ModelResult:
    return sorted(
        results,
        key=lambda result: (
            result.recall_at_3,
            result.mrr,
            result.recall_at_1,
            -result.latency_ms,
        ),
        reverse=True,
    )[0]


def main() -> None:
    load_env_near(BASE_DIR)

    queries, docs = load_eval_dataset(DATA_PATH)
    all_texts = [doc.text for doc in docs] + [query.query for query in queries]
    vocabulary = build_vocabulary(all_texts, max_terms=120)

    models = [
        KeywordOverlapEmbeddings(vocabulary=vocabulary),
        HashingEmbeddings(dimensions=256),
    ]

    gemini = maybe_build_gemini_model()
    if gemini:
        models.append(gemini)
    else:
        print("[INFO] No GEMINI_API_KEY found, skipping Gemini embedding model.")

    print("=" * 86)
    print("DAY 3 MINI PROJECT: EMBEDDING BENCHMARK SUITE")
    print("=" * 86)
    print(f"Queries: {len(queries)}")
    print(f"Documents: {len(docs)}")
    print(f"Models: {', '.join(model.name for model in models)}")

    results = [evaluate_model(model, queries, docs) for model in models]
    print_results(results)

    winner = recommend(results)
    print("\nRecommendation:")
    print(
        f"  Start with {winner.model_name}. It has the strongest balance of "
        f"Recall@3={winner.recall_at_3:.2f}, MRR={winner.mrr:.2f}, "
        f"and latency={winner.latency_ms:.1f} ms on this dataset."
    )

    print("\nProduction reminder:")
    print("  If you change embedding models, rebuild the whole vector index.")


if __name__ == "__main__":
    main()
