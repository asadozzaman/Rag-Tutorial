from __future__ import annotations

from pathlib import Path

from memory_rag_utils import ConversationMemory, load_docs, run_turn


ROOT = Path(__file__).resolve().parent
DOC_PATH = ROOT / "sample_data" / "framework_docs.csv"


def main() -> None:
    docs = load_docs(DOC_PATH)
    memory = ConversationMemory(max_turns=4)
    conversation = [
        "Tell me about Python",
        "What web frameworks does it have?",
        "Which one is best for APIs?",
        "Does it support async?",
    ]

    print("=" * 98)
    print("DAY 11: CONVERSATIONAL RAG WITH MEMORY")
    print("=" * 98)

    for query in conversation:
        result = run_turn(query, memory, docs)
        print(f"User: {result['original_query']}")
        print(f"Standalone query: {result['standalone_query']}")
        print(f"Bot: {result['answer']}")
        print("Sources:")
        for source in result["sources"]:
            print(f"  - {source.title} ({source.doc_id})")
        print()


if __name__ == "__main__":
    main()
