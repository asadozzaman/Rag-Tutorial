from __future__ import annotations

from pathlib import Path

from structured_rag_utils import (
    build_sales_db,
    execute_generated_code,
    format_sql_result,
    generate_dataframe_code,
    generate_sql,
    load_table_rows,
    run_sql,
)


ROOT = Path(__file__).resolve().parent
TABLE_PATH = ROOT / "sample_data" / "sales_rows.csv"
DB_PATH = ROOT / "sales.db"


def main() -> None:
    rows = load_table_rows(TABLE_PATH)
    build_sales_db(rows, DB_PATH)

    questions = [
        "What is the total revenue by product?",
        "Which region had the highest revenue in Q1?",
        "Show quarter-over-quarter growth for Widget A in North",
    ]

    print("=" * 96)
    print("DAY 12: TEXT-TO-SQL AND TABLE Q&A")
    print("=" * 96)
    for question in questions:
        sql = generate_sql(question)
        result = run_sql(DB_PATH, sql)
        print(f"Q: {question}")
        print(f"SQL: {sql}")
        print(f"Result: {format_sql_result(question, result)}")
        print()

    inventory_rows = [
        {"product": "Widget A", "price": 29.99, "stock": 500, "rating": 4.5},
        {"product": "Widget B", "price": 49.99, "stock": 200, "rating": 4.8},
        {"product": "Widget C", "price": 19.99, "stock": 1000, "rating": 4.2},
    ]
    dataframe_question = "What is the total value of inventory (price x stock) per product?"
    code = generate_dataframe_code(dataframe_question)
    answer = execute_generated_code(code, inventory_rows)
    print("DataFrame-style Q&A")
    print("-" * 96)
    print(f"Question: {dataframe_question}")
    print(f"Generated code: {code}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
