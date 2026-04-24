from __future__ import annotations

from fastapi.testclient import TestClient

from day13_app import app


def main() -> None:
    print("=" * 112)
    print("DAY 13 MINI PROJECT: PRODUCTION-READY RAG SERVICE")
    print("=" * 112)

    with TestClient(app) as client:
        upload = client.post(
            "/upload",
            json={
                "filename": "ops_notes.txt",
                "content": "Circuit breakers protect the system during provider outages.\nAsync processing improves throughput under load.",
            },
        )

        exact = client.post(
            "/query",
            json={"query": "Why do we use circuit breakers in production RAG?", "user_id": "bob", "top_k": 2},
        )
        semantic = client.post(
            "/query",
            json={"query": "Why do we use circuit breakers during provider outages?", "user_id": "bob", "top_k": 2},
        )

        stream = client.get(
            "/query/stream",
            params={"query": "Why does caching reduce latency?", "user_id": "carol", "top_k": 2},
        )

        health = client.get("/health")

        print("Upload")
        print("-" * 112)
        print(upload.json())
        print()

        print("Exact or fresh query")
        print("-" * 112)
        print(exact.json())
        print()

        print("Near-duplicate semantic cache query")
        print("-" * 112)
        print(semantic.json())
        print()

        print("Streaming response preview")
        print("-" * 112)
        preview = "\n".join(stream.text.splitlines()[:6])
        print(preview)
        print()

        print("Health")
        print("-" * 112)
        print(health.json())


if __name__ == "__main__":
    main()
