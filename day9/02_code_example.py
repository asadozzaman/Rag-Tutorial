from __future__ import annotations

from pathlib import Path

from query_transform_utils import (
    load_corpus,
    multi_query_variations,
    retrieve,
    route_query,
    step_back_query,
    transformed_retrieve,
)


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "sample_data" / "corpus.csv"


def main() -> None:
    docs = load_corpus(CORPUS_PATH)
    query = "Why is my vector search slow?"

    print("=" * 90)
    print("DAY 9: QUERY TRANSFORMATION TECHNIQUES")
    print("=" * 90)
    print(f"Query: {query}")
    print(f"Route: {route_query(query)}")
    print()

    print("Multi-query variations")
    print("-" * 90)
    for item in multi_query_variations(query):
        print(f"- {item}")
    print()

    print("Step-back query")
    print("-" * 90)
    print(step_back_query(query))
    print()

    print("Plain retrieval")
    print("-" * 90)
    for doc, score in retrieve(query, docs, k=3):
        print(f"- {doc.doc_id} | {doc.title} | score={score:.2f}")
    print()

    transformed, transforms = transformed_retrieve(query, docs, k=4)
    print("HyDE hypothesis")
    print("-" * 90)
    print(transforms["hyde"])
    print()

    print("Transformed retrieval")
    print("-" * 90)
    for doc, score in transformed:
        print(f"- {doc.doc_id} | {doc.title} | score={score:.2f}")


if __name__ == "__main__":
    main()
