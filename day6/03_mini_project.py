"""
Day 6 Mini Project: RAG chatbot with graceful failure handling.

This project:
  - retrieves top-3 context docs
  - checks if the context is relevant enough
  - answers with inline citations
  - handles contradictions
  - rewrites follow-up questions using memory

Run:
  python 03_mini_project.py
"""

from __future__ import annotations

from pathlib import Path

from prompt_utils import (
    ChatMemory,
    build_prompt_package,
    compose_query_with_history,
    docs_are_relevant,
    generate_structured_answer,
    load_context_docs,
    retrieve_top_docs,
)


BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = BASE_DIR / "sample_data" / "support_docs.csv"


def print_chat_turn(user_query, rewritten_query, prompt_package, response):
    print(f"\nUser query: {user_query}")
    if rewritten_query != user_query:
        print(f"Rewritten with memory: {rewritten_query}")
    print("\nPrompt package:")
    print(f"SYSTEM: {prompt_package['system']}")
    print(f"HUMAN:\n{prompt_package['human']}")
    print("\nAnswer:")
    print(response.answer)
    print(f"Confidence: {response.confidence}")
    print(f"Sources: {response.sources}")
    print(f"Reasoning: {response.reasoning}")


def run_turn(user_query, docs, memory):
    rewritten_query = compose_query_with_history(user_query, memory.recent())
    scored_docs = retrieve_top_docs(rewritten_query, docs, k=3)
    prompt_package = build_prompt_package(rewritten_query, scored_docs)
    response = generate_structured_answer(rewritten_query, scored_docs, threshold=0.18)
    memory.add_turn(user_query, response)
    print_chat_turn(user_query, rewritten_query, prompt_package, response)


def main() -> None:
    docs = load_context_docs(DOCS_PATH)
    memory = ChatMemory()

    print("=" * 84)
    print("DAY 6 MINI PROJECT: RAG CHATBOT WITH GRACEFUL FAILURE HANDLING")
    print("=" * 84)
    print(f"Support docs loaded: {len(docs)}")

    demo_queries = [
        "What is the refund policy for digital products?",
        "What about defective downloads?",
        "Are subscriptions always final sale?",
        "Do you offer offline retail pickup in Brazil?",
    ]

    for query in demo_queries:
        run_turn(query, docs, memory)
        print("-" * 84)

    print("\nInteractive mode. Type 'quit' to exit.")
    while True:
        try:
            user_query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if not user_query:
            continue

        run_turn(user_query, docs, memory)


if __name__ == "__main__":
    main()
