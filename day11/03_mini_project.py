from __future__ import annotations

from pathlib import Path

from memory_rag_utils import ConversationMemory, load_docs, run_turn


ROOT = Path(__file__).resolve().parent
DOC_PATH = ROOT / "sample_data" / "framework_docs.csv"


def main() -> None:
    docs = load_docs(DOC_PATH)
    memory = ConversationMemory(max_turns=3)

    first_session = [
        "Tell me about Python",
        "What about its web frameworks?",
        "Which one is best for APIs?",
        "Does it support async?",
    ]

    second_session = [
        "Tell me about JavaScript",
        "What about its frontend frameworks?",
    ]

    print("=" * 110)
    print("DAY 11 MINI PROJECT: DOCUMENT CHAT ASSISTANT")
    print("=" * 110)

    print("Session 1")
    print("-" * 110)
    for query in first_session:
        result = run_turn(query, memory, docs)
        print(f"User: {result['original_query']}")
        print(f"Standalone: {result['standalone_query']}")
        print(f"Answer: {result['answer']}")
        print(f"Source count: {len(result['sources'])}")
        if result["memory_summary"]:
            print(f"Memory summary: {result['memory_summary']}")
        print()

    print("Clear history")
    print("-" * 110)
    memory.clear()
    print("History reset complete.")
    print()

    print("Session 2")
    print("-" * 110)
    for query in second_session:
        result = run_turn(query, memory, docs)
        print(f"User: {result['original_query']}")
        print(f"Standalone: {result['standalone_query']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {', '.join(source.doc_id for source in result['sources'])}")
        print()


if __name__ == "__main__":
    main()
