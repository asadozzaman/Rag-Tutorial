from __future__ import annotations

from pathlib import Path

from advanced_rag_utils import (
    build_graph_index,
    build_raptor_summaries,
    corrective_rag,
    load_fallback_snippets,
    load_knowledge_docs,
)


ROOT = Path(__file__).resolve().parent
DOC_PATH = ROOT / "sample_data" / "knowledge_docs.csv"
FALLBACK_PATH = ROOT / "sample_data" / "fallback_snippets.csv"


def print_crag_case(query: str, docs, snippets) -> None:
    result = corrective_rag(query, docs, snippets)
    print(f"Query: {query}")
    for doc, score, quality in result["evaluations"]:
        print(f"  [{quality.value:>9}] score={score:.1f} | {doc.title}")
    if result["action"] == "refine_and_reretrieve":
        print(f"  Action: refine -> {result['refined_query']}")
    elif result["action"] == "fallback_search":
        print("  Action: fallback search")
    else:
        print("  Action: use correct docs")
    print(f"  Answer: {result['answer']}")
    print()


def main() -> None:
    docs = load_knowledge_docs(DOC_PATH)
    fallback = load_fallback_snippets(FALLBACK_PATH)
    graph = build_graph_index(docs)
    summaries = build_raptor_summaries(docs)

    print("=" * 104)
    print("DAY 14: ADVANCED RAG PATTERNS")
    print("=" * 104)
    print("GraphRAG snapshot")
    print("-" * 104)
    print(f"Graph nodes: {len(graph)}")
    print(f"LangChain neighbors: {', '.join(sorted(graph.get('LangChain', [])))}")
    print()

    print("RAPTOR snapshot")
    print("-" * 104)
    print(f"Leaf nodes: {len(summaries['leaves'])}")
    print(f"Topic summaries: {len(summaries['topic_level'])}")
    print(f"Root summary title: {summaries['root'][0]['title']}")
    print()

    print("CRAG examples")
    print("-" * 104)
    print_crag_case("What is LangChain used for in LLM apps?", docs, fallback)
    print_crag_case("How do self-healing retrieval systems work?", docs, fallback)
    print_crag_case("What is the weather today?", docs, fallback)


if __name__ == "__main__":
    main()
