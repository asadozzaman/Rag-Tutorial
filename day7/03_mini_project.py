"""
Day 7 Mini Project: Multi-index RAG router.

This project:
  - builds three separate indices
  - creates query engines for each
  - routes queries automatically
  - measures routing accuracy on diverse examples

Run:
  python 03_mini_project.py
"""

from __future__ import annotations

from pathlib import Path

from mini_llamaindex import (
    QueryEngineTool,
    RouterQueryEngine,
    SentenceSplitter,
    SimpleDirectoryReader,
    ToolMetadata,
    VectorStoreIndex,
    load_router_queries,
)


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "sample_data" / "docs"
QUERIES_PATH = BASE_DIR / "sample_data" / "router_queries.csv"


def build_tools():
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    parser = SentenceSplitter(chunk_size=240, chunk_overlap=30)

    technical_docs = [doc for doc in documents if doc.metadata["category"] == "technical"]
    faq_docs = [doc for doc in documents if doc.metadata["category"] == "faq"]
    api_docs = [doc for doc in documents if doc.metadata["category"] == "api"]

    technical_engine = VectorStoreIndex.from_documents(technical_docs, node_parser=parser).as_query_engine(
        similarity_top_k=2,
        response_mode="compact",
    )
    faq_engine = VectorStoreIndex.from_documents(faq_docs, node_parser=parser).as_query_engine(
        similarity_top_k=2,
        response_mode="compact",
    )
    api_engine = VectorStoreIndex.from_documents(api_docs, node_parser=parser).as_query_engine(
        similarity_top_k=2,
        response_mode="compact",
    )

    tools = [
        QueryEngineTool(
            query_engine=technical_engine,
            metadata=ToolMetadata(
                name="technical_docs",
                description="Architecture, indexing, retrieval, embeddings, and implementation concepts.",
            ),
        ),
        QueryEngineTool(
            query_engine=faq_engine,
            metadata=ToolMetadata(
                name="faq_support",
                description="Support help, billing questions, password reset, account issues, and user-facing troubleshooting.",
            ),
        ),
        QueryEngineTool(
            query_engine=api_engine,
            metadata=ToolMetadata(
                name="api_reference",
                description="Endpoints, request parameters, authentication headers, SDK behavior, and response formats.",
            ),
        ),
    ]
    return tools


def main() -> None:
    tools = build_tools()
    router = RouterQueryEngine(tools)
    test_queries = load_router_queries(QUERIES_PATH)

    print("=" * 86)
    print("DAY 7 MINI PROJECT: MULTI-INDEX RAG ROUTER")
    print("=" * 86)
    print(f"Tools available: {', '.join(tool.metadata.name for tool in tools)}")

    correct = 0
    for row in test_queries:
        tool, response = router.query(row["query"])
        is_correct = tool.metadata.name == row["expected_route"]
        correct += 1 if is_correct else 0
        print(f"\nQuery: {row['query']}")
        print(f"Expected route: {row['expected_route']}")
        print(f"Chosen route:   {tool.metadata.name}")
        print(f"Route correct:  {is_correct}")
        print(f"Answer: {response.answer}")
        print("Sources:")
        for node in response.source_nodes:
            print(f"  {node.metadata.get('title')} | {node.metadata.get('source')}")

    accuracy = correct / max(len(test_queries), 1)
    print(f"\nRouting accuracy: {accuracy:.2%} ({correct}/{len(test_queries)})")


if __name__ == "__main__":
    main()
