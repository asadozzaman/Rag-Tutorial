from __future__ import annotations

from fastapi.testclient import TestClient

from day15_app import app


def main() -> None:
    print("=" * 116)
    print("DAY 15 MINI PROJECT: ENTERPRISE DOCUMENT INTELLIGENCE PLATFORM")
    print("=" * 116)

    with TestClient(app) as client:
        upload = client.post(
            "/upload",
            json={
                "filename": "ops_notes.md",
                "content": "Circuit breakers help a production RAG service fail safely during model outages.\nSmaller models can be used on low-risk traffic to reduce cost.",
                "doc_type": "markdown",
            },
        )

        q1 = client.post(
            "/query",
            json={"question": "Tell me about semantic caching", "session_id": "team"},
        )
        q2 = client.post(
            "/query",
            json={"question": "How does it help latency?", "session_id": "team"},
        )
        q3 = client.post(
            "/query",
            json={"question": "How can I improve RAG precision?", "session_id": "team"},
        )
        q4 = client.post(
            "/query",
            json={"question": "How can I improve RAG precision?", "session_id": "team"},
        )
        stream = client.get(
            "/query/stream",
            params={"question": "How does CRAG help when retrieval is weak?", "session_id": "team"},
        )
        evaluate = client.get("/evaluate")
        metrics = client.get("/metrics")
        health = client.get("/health")

        print("Upload")
        print("-" * 116)
        print(upload.json())
        print()

        print("Conversation query 1")
        print("-" * 116)
        print(q1.json())
        print()

        print("Conversation query 2 (history-aware)")
        print("-" * 116)
        print(q2.json())
        print()

        print("Precision query")
        print("-" * 116)
        print(q3.json())
        print()

        print("Repeated precision query (cache demo)")
        print("-" * 116)
        print(q4.json())
        print()

        print("Streaming preview")
        print("-" * 116)
        print("\n".join(stream.text.splitlines()[:6]))
        print()

        print("Evaluation")
        print("-" * 116)
        print(evaluate.json())
        print()

        print("Metrics")
        print("-" * 116)
        print(metrics.json())
        print()

        print("Health")
        print("-" * 116)
        print(health.json())


if __name__ == "__main__":
    main()
