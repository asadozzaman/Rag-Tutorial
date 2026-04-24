from __future__ import annotations

from pathlib import Path

from capstone_rag import CapstoneRAG


ROOT = Path(__file__).resolve().parent
DOCS_PATH = ROOT / "sample_data" / "docs.csv"
EVAL_PATH = ROOT / "sample_data" / "eval_queries.csv"


def main() -> None:
    rag = CapstoneRAG()
    n_chunks = rag.ingest_csv(DOCS_PATH)
    print("=" * 110)
    print("DAY 15: CAPSTONE RAG CORE MODULE")
    print("=" * 110)
    print(f"Ingested chunks: {n_chunks}")

    first = __import__("asyncio").run(rag.query("How can I improve RAG precision?", session_id="demo"))
    second = __import__("asyncio").run(rag.query("How can I improve RAG precision?", session_id="demo"))
    eval_report = rag.evaluate(EVAL_PATH)

    print()
    print("First query")
    print("-" * 110)
    print(f"Answer: {first.answer}")
    print(f"Confidence: {first.confidence}")
    print(f"Latency: {first.latency_ms:.2f}ms")
    print(f"Cached: {first.cached} ({first.cache_type})")
    print(f"Citations: {len(first.citations)}")

    print()
    print("Second query")
    print("-" * 110)
    print(f"Answer: {second.answer}")
    print(f"Confidence: {second.confidence}")
    print(f"Latency: {second.latency_ms:.2f}ms")
    print(f"Cached: {second.cached} ({second.cache_type})")

    print()
    print("Metrics")
    print("-" * 110)
    print(rag.metrics_snapshot())

    print()
    print("Evaluation")
    print("-" * 110)
    print(eval_report)


if __name__ == "__main__":
    main()
