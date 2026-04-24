from __future__ import annotations

from pathlib import Path

from structured_rag_utils import answer_query, build_sales_db, load_table_rows, load_text_chunks


ROOT = Path(__file__).resolve().parent
TEXT_PATH = ROOT / "sample_data" / "report_chunks.csv"
TABLE_PATH = ROOT / "sample_data" / "sales_rows.csv"
DB_PATH = ROOT / "hybrid_report.db"


def main() -> None:
    text_chunks = load_text_chunks(TEXT_PATH)
    rows = load_table_rows(TABLE_PATH)
    build_sales_db(rows, DB_PATH)

    questions = [
        "Summarize the North region demand outlook.",
        "What is the total revenue by product?",
        "What was Widget A's Q1 revenue and what does the report say about North region demand?",
    ]

    print("=" * 110)
    print("DAY 12 MINI PROJECT: HYBRID DOCUMENT Q&A SYSTEM")
    print("=" * 110)
    print(f"Text chunks loaded: {len(text_chunks)}")
    print(f"Structured rows loaded: {len(rows)}")
    print()

    for question in questions:
        result = answer_query(question, text_chunks, DB_PATH)
        print(f"Question: {question}")
        print(f"Route: {result['route']}")
        if "sql" in result:
            print(f"SQL: {result['sql']}")
        if "sources" in result:
            print(f"Sources: {', '.join(result['sources'])}")
        print(f"Answer: {result['answer']}")
        print()


if __name__ == "__main__":
    main()
