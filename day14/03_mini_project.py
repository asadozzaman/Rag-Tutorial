from __future__ import annotations

from pathlib import Path

from advanced_rag_utils import (
    benchmark_crag,
    build_graph_index,
    graph_query,
    load_fallback_snippets,
    load_knowledge_docs,
)


ROOT = Path(__file__).resolve().parent
DOC_PATH = ROOT / "sample_data" / "knowledge_docs.csv"
FALLBACK_PATH = ROOT / "sample_data" / "fallback_snippets.csv"
BENCHMARK_PATH = ROOT / "sample_data" / "benchmark_queries.csv"


def load_benchmark(path: Path) -> list[dict[str, str]]:
    items = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        query, expected_action = line.split(",", 1)
        items.append({"query": query.strip(), "expected_action": expected_action.strip()})
    return items


def main() -> None:
    docs = load_knowledge_docs(DOC_PATH)
    fallback = load_fallback_snippets(FALLBACK_PATH)
    graph = build_graph_index(docs)
    benchmark = load_benchmark(BENCHMARK_PATH)
    report = benchmark_crag(benchmark, docs, fallback)

    print("=" * 112)
    print("DAY 14 MINI PROJECT: SELF-CORRECTING RAG SYSTEM")
    print("=" * 112)
    print(f"Knowledge docs: {len(docs)}")
    print(f"Fallback snippets: {len(fallback)}")
    print()

    print("GraphRAG demo")
    print("-" * 112)
    graph_result = graph_query("How are LangChain and FAISS related?", docs, graph)
    print(f"Answer: {graph_result['answer']}")
    print(f"Sources: {', '.join(graph_result['sources'])}")
    print()

    print("CRAG benchmark")
    print("-" * 112)
    print(f"Routing accuracy: {report['accuracy']:.2%}")
    print(f"Correction rate: {report['correction_rate']:.2%}")
    print()

    print("Per-query actions")
    print("-" * 112)
    for item in report["results"]:
        print(f"Query: {item['query']}")
        print(f"  Expected: {item['expected_action']}")
        print(f"  Actual:   {item['actual_action']}")
        print(f"  Matched:  {item['matched']}")
        print(f"  Answer:   {item['answer']}")
        print()


if __name__ == "__main__":
    main()
