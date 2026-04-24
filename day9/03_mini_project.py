from __future__ import annotations

from pathlib import Path

from query_transform_utils import (
    ask_clarifying_question,
    benchmark_pipeline,
    classify_query,
    load_benchmark,
    load_corpus,
    route_query,
    transformed_retrieve,
)


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "sample_data" / "corpus.csv"
BENCHMARK_PATH = ROOT / "sample_data" / "benchmark_queries.csv"


def main() -> None:
    docs = load_corpus(CORPUS_PATH)
    cases = load_benchmark(BENCHMARK_PATH)
    results = benchmark_pipeline(cases, docs, k=3)

    print("=" * 102)
    print("DAY 9 MINI PROJECT: INTELLIGENT QUERY PIPELINE")
    print("=" * 102)
    print(f"Loaded corpus documents: {len(docs)}")
    print(f"Loaded benchmark queries: {len(cases)}")
    print()

    print("Pipeline behavior examples")
    print("-" * 102)
    sample_queries = [
        "How do I reset my password?",
        "Compare IVF and HNSW for speed and memory tradeoffs",
        "Help, it is broken",
    ]
    for query in sample_queries:
        mode = classify_query(query)
        route = route_query(query)
        print(f"Query: {query}")
        print(f"  Mode:  {mode}")
        print(f"  Route: {route}")
        if mode == "ambiguous":
            print(f"  Action: {ask_clarifying_question(query, route)}")
        else:
            docs_found, _ = transformed_retrieve(query, docs, k=2)
            print(f"  Action: retrieve {', '.join(doc.doc_id for doc, _ in docs_found)}")
        print()

    print("Benchmark summary")
    print("-" * 102)
    print(f"Mode accuracy:          {results['mode_accuracy']:.2%}")
    print(f"Route accuracy:         {results['route_accuracy']:.2%}")
    print(f"Plain Recall@3:         {results['plain']['recall_at_3']:.2f}")
    print(f"Plain MRR:              {results['plain']['mrr']:.2f}")
    print(f"Transformed Recall@3:   {results['transformed']['recall_at_3']:.2f}")
    print(f"Transformed MRR:        {results['transformed']['mrr']:.2f}")
    print()

    print("Queries with biggest gain")
    print("-" * 102)
    gains = sorted(
        results["details"],
        key=lambda item: item["transformed"]["mrr"] - item["plain"]["mrr"],
        reverse=True,
    )
    for item in gains[:5]:
        gain = item["transformed"]["mrr"] - item["plain"]["mrr"]
        print(f"Query: {item['query']}")
        print(f"  Mode / Route: {item['mode']} / {item['route']}")
        print(f"  Plain IDs:    {', '.join(item['plain_ids']) or 'none'}")
        print(f"  Better IDs:   {', '.join(item['transformed_ids']) or 'none'}")
        print(f"  MRR gain:     {gain:.2f}")
        print()


if __name__ == "__main__":
    main()
