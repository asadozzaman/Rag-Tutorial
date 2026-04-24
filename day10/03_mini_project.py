from __future__ import annotations

from pathlib import Path

from agent_utils import load_kb, load_web, run_agent


ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "sample_data" / "company_kb.csv"
WEB_PATH = ROOT / "sample_data" / "web_snippets.csv"


def main() -> None:
    kb = load_kb(KB_PATH)
    web = load_web(WEB_PATH)
    queries = [
        "What was our Q1 revenue?",
        "If Q1 revenue grows at the same 15% rate, what will Q2 revenue be?",
        "How does our churn rate compare to the industry average?",
        "What is today's date?",
    ]

    print("=" * 106)
    print("DAY 10 MINI PROJECT: RESEARCH ASSISTANT AGENT")
    print("=" * 106)
    print(f"Knowledge base docs: {len(kb)}")
    print(f"External snippets:   {len(web)}")
    print()

    for query in queries:
        result = run_agent(query, kb, web)
        print(f"Query: {query}")
        print(f"Tool plan: {', '.join(result['tools'])}")
        for index, call in enumerate(result["calls"], start=1):
            print(f"  Step {index}: {call.tool}")
            print(f"    Observation: {call.output}")
        print(f"Answer: {result['answer']}")
        print()


if __name__ == "__main__":
    main()
