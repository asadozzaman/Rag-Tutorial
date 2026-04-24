from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from day13_app import app


ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("=" * 102)
    print("DAY 13: PRODUCTION RAG API")
    print("=" * 102)

    with TestClient(app) as client:
        first = client.post(
            "/query",
            json={"query": "How does caching reduce latency?", "user_id": "alice", "top_k": 2},
        )
        second = client.post(
            "/query",
            json={"query": "How does caching reduce latency?", "user_id": "alice", "top_k": 2},
        )
        health = client.get("/health")

        print("First query")
        print("-" * 102)
        print(first.json())
        print()

        print("Second query (should hit cache)")
        print("-" * 102)
        print(second.json())
        print()

        print("Health")
        print("-" * 102)
        print(health.json())


if __name__ == "__main__":
    main()
