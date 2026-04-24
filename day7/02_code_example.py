"""
Day 7: LlamaIndex-style RAG pipeline demo.

This script demonstrates:
  - document loading
  - node parsing
  - vector index creation
  - persistence and reload
  - query engine usage
  - sub-question decomposition

Run:
  python 02_code_example.py
"""

from __future__ import annotations

from pathlib import Path

from mini_llamaindex import (
    HashingEmbedModel,
    QueryEngineTool,
    RouterQueryEngine,
    SentenceSplitter,
    SimpleDirectoryReader,
    StorageContext,
    SubQuestionQueryEngine,
    SummaryIndex,
    ToolMetadata,
    VectorStoreIndex,
    load_index_from_storage,
)


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "sample_data" / "docs"
STORAGE_DIR = BASE_DIR / "storage" / "day7_basic"


def main() -> None:
    documents = [
        doc
        for doc in SimpleDirectoryReader(DOCS_DIR).load_data()
        if doc.metadata["category"] == "technical"
    ]
    parser = SentenceSplitter(chunk_size=260, chunk_overlap=40)
    embed_model = HashingEmbedModel(dimensions=256)

    vector_index = VectorStoreIndex.from_documents(
        documents,
        node_parser=parser,
        embed_model=embed_model,
    )
    summary_index = SummaryIndex.from_documents(documents)

    vector_index.persist(STORAGE_DIR)
    reloaded_index = load_index_from_storage(StorageContext.from_defaults(STORAGE_DIR))

    query_engine = reloaded_index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
    )
    summary_engine = summary_index.as_query_engine(
        similarity_top_k=2,
        response_mode="tree_summarize",
    )

    query = "What are the key concepts in RAG?"
    response = query_engine.query(query)

    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="rag_docs",
            description="Technical documentation about RAG systems, vector search, and retrieval.",
        ),
    )
    sub_question_engine = SubQuestionQueryEngine.from_defaults([tool])
    complex_response = sub_question_engine.query(
        "Compare vector search and keyword search for RAG retrieval"
    )

    print("=" * 78)
    print("DAY 7: LLAMAINDEX-STYLE RAG PIPELINE")
    print("=" * 78)
    print(f"\nLoaded documents: {len(documents)}")
    print(f"Persisted index: {STORAGE_DIR}")

    print(f"\nBasic query: {query}")
    print(response.answer)
    print("Sources:")
    for node in response.source_nodes:
        print(f"  {node.metadata.get('title')} | chunk {node.metadata.get('chunk')}")

    print("\nSummary engine output:")
    print(summary_engine.query("Give me a high-level overview of the docs").answer)

    print("\nSub-question engine output:")
    print(complex_response.answer)


if __name__ == "__main__":
    main()
