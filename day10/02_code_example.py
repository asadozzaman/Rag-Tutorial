from __future__ import annotations

from pathlib import Path

from agent_utils import load_kb, load_web, run_agent


ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "sample_data" / "company_kb.csv"
WEB_PATH = ROOT / "sample_data" / "web_snippets.csv"


def print_run(query: str) -> None:
    result = run_agent(query, load_kb(KB_PATH), load_web(WEB_PATH))
    print("=" * 94)
    print(f"Query: {query}")
    print("Chosen tools:", ", ".join(result["tools"]))
    print("-" * 94)
    for call in result["calls"]:
        print(f"TOOL: {call.tool}")
        print(f"INPUT: {call.input}")
        print(f"OUTPUT: {call.output}")
        print()
    print("Final answer:")
    print(result["answer"])


def main() -> None:
    print("=" * 94)
    print("DAY 10: AGENTIC RAG WITH TOOLS")
    print("=" * 94)
    print_run("What was our Q1 revenue?")
    print_run("If Q1 revenue grows at the same 15% rate, what will Q2 revenue be?")


if __name__ == "__main__":
    main()
