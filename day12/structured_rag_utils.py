from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}


@dataclass
class TextChunk:
    chunk_id: str
    title: str
    text: str


@dataclass
class DataRow:
    product: str
    region: str
    revenue: float
    quarter: str
    year: int


def tokenize(text: str) -> list[str]:
    cleaned = []
    current = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        else:
            if current:
                cleaned.append("".join(current))
                current = []
    if current:
        cleaned.append("".join(current))
    return cleaned


def content_terms(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def load_text_chunks(csv_path: str | Path) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            chunks.append(
                TextChunk(
                    chunk_id=row["chunk_id"].strip(),
                    title=row["title"].strip(),
                    text=row["text"].strip(),
                )
            )
    return chunks


def load_table_rows(csv_path: str | Path) -> list[DataRow]:
    rows: list[DataRow] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                DataRow(
                    product=row["product"].strip(),
                    region=row["region"].strip(),
                    revenue=float(row["revenue"]),
                    quarter=row["quarter"].strip(),
                    year=int(row["year"]),
                )
            )
    return rows


def build_sales_db(rows: list[DataRow], db_path: str | Path) -> Path:
    db_file = Path(db_path)
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS sales")
    cursor.execute(
        """
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product TEXT NOT NULL,
            region TEXT NOT NULL,
            revenue REAL NOT NULL,
            quarter TEXT NOT NULL,
            year INTEGER NOT NULL
        )
        """
    )
    cursor.executemany(
        "INSERT INTO sales (product, region, revenue, quarter, year) VALUES (?, ?, ?, ?, ?)",
        [(row.product, row.region, row.revenue, row.quarter, row.year) for row in rows],
    )
    connection.commit()
    connection.close()
    return db_file


def route_query(query: str) -> str:
    lowered = query.lower()
    structured_markers = {
        "revenue",
        "quarter",
        "growth",
        "total",
        "highest",
        "inventory",
        "sql",
        "table",
    }
    text_markers = {
        "strategy",
        "risk",
        "demand",
        "outlook",
        "report",
        "narrative",
        "summary",
        "explain",
    }

    has_structured = any(marker in lowered for marker in structured_markers)
    if "widget" in lowered and "revenue" in lowered:
        has_structured = True
    has_text = any(marker in lowered for marker in text_markers)

    if has_structured and has_text:
        return "hybrid"
    if has_structured:
        return "structured"
    return "text"


def retrieve_text(query: str, chunks: list[TextChunk], k: int = 2) -> list[tuple[TextChunk, float]]:
    query_terms = content_terms(query)
    ranked: list[tuple[TextChunk, float]] = []
    for chunk in chunks:
        doc_terms = content_terms(f"{chunk.title} {chunk.text}")
        overlap = len(query_terms & doc_terms)
        phrase_bonus = 1.0 if any(term in chunk.text.lower() for term in query_terms) else 0.0
        score = overlap + phrase_bonus
        if score > 0:
            ranked.append((chunk, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def generate_sql(question: str) -> str:
    lowered = question.lower()

    if "total revenue by product" in lowered:
        return (
            "SELECT product, ROUND(SUM(revenue), 2) AS total_revenue "
            "FROM sales GROUP BY product ORDER BY total_revenue DESC;"
        )
    if "highest revenue in q1" in lowered:
        return (
            "SELECT region, ROUND(SUM(revenue), 2) AS total_revenue "
            "FROM sales WHERE quarter = 'Q1' GROUP BY region "
            "ORDER BY total_revenue DESC LIMIT 1;"
        )
    if "widget a" in lowered and "q1" in lowered and "revenue" in lowered:
        return (
            "SELECT product, region, revenue, quarter, year "
            "FROM sales WHERE product = 'Widget A' AND quarter = 'Q1' "
            "ORDER BY revenue DESC;"
        )
    if "growth" in lowered and "widget a" in lowered and "north" in lowered:
        return (
            "SELECT quarter, revenue FROM sales "
            "WHERE product = 'Widget A' AND region = 'North' "
            "ORDER BY year, quarter;"
        )
    if "widget b" in lowered and "south" in lowered:
        return (
            "SELECT product, region, revenue, quarter, year FROM sales "
            "WHERE product = 'Widget B' AND region = 'South' ORDER BY year, quarter;"
        )
    return (
        "SELECT product, region, revenue, quarter, year "
        "FROM sales ORDER BY year, quarter, product LIMIT 5;"
    )


def validate_sql(sql: str) -> None:
    lowered = sql.lower()
    banned = ["drop ", "delete ", "update ", "insert ", "alter "]
    if any(item in lowered for item in banned):
        raise ValueError("Unsafe SQL detected")
    if not lowered.strip().startswith("select"):
        raise ValueError("Only SELECT statements are allowed")


def run_sql(db_path: str | Path, sql: str) -> list[tuple]:
    validate_sql(sql)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    rows = cursor.execute(sql).fetchall()
    connection.close()
    return rows


def generate_dataframe_code(question: str) -> str:
    lowered = question.lower()
    if "total value of inventory" in lowered:
        return "result = {row['product']: round(row['price'] * row['stock'], 2) for row in rows}"
    if "highest rated product" in lowered:
        return "result = max(rows, key=lambda row: row['rating'])['product']"
    return "result = rows[:3]"


def execute_generated_code(code: str, rows: list[dict]) -> object:
    local_vars = {"rows": rows, "result": None, "round": round, "max": max}
    exec(code, {"__builtins__": {}}, local_vars)
    return local_vars["result"]


def format_sql_result(question: str, rows: list[tuple]) -> str:
    lowered = question.lower()
    if not rows:
        return "No structured rows matched the question."
    if "total revenue by product" in lowered:
        return "; ".join(f"{product}: ${total:,.0f}" for product, total in rows)
    if "highest revenue in q1" in lowered:
        region, total = rows[0]
        return f"{region} had the highest Q1 revenue at ${total:,.0f}."
    if "growth" in lowered and len(rows) >= 2:
        first_quarter, first_revenue = rows[0]
        second_quarter, second_revenue = rows[-1]
        growth = second_revenue - first_revenue
        pct = (growth / first_revenue) * 100 if first_revenue else 0.0
        return f"{first_quarter} to {second_quarter} growth was ${growth:,.0f} ({pct:.1f}%)."
    if "widget a" in lowered and "q1" in lowered:
        return "; ".join(
            f"{product} in {region} generated ${revenue:,.0f} in {quarter} {year}"
            for product, region, revenue, quarter, year in rows
        )
    return str(rows)


def answer_text_query(query: str, chunks: list[TextChunk]) -> dict[str, object]:
    results = retrieve_text(query, chunks, k=2)
    answer = " ".join(chunk.text for chunk, _ in results) if results else "No relevant report text found."
    return {"route": "text", "answer": answer, "sources": [chunk.title for chunk, _ in results]}


def answer_structured_query(query: str, db_path: str | Path) -> dict[str, object]:
    sql = generate_sql(query)
    rows = run_sql(db_path, sql)
    return {
        "route": "structured",
        "sql": sql,
        "rows": rows,
        "answer": format_sql_result(query, rows),
    }


def answer_hybrid_query(query: str, chunks: list[TextChunk], db_path: str | Path) -> dict[str, object]:
    text_part = answer_text_query(query, chunks)
    structured_part = answer_structured_query(query, db_path)
    combined = f"{structured_part['answer']} Report context: {text_part['answer']}"
    return {
        "route": "hybrid",
        "sql": structured_part["sql"],
        "rows": structured_part["rows"],
        "sources": text_part["sources"],
        "answer": combined,
    }


def answer_query(query: str, chunks: list[TextChunk], db_path: str | Path) -> dict[str, object]:
    route = route_query(query)
    if route == "structured":
        return answer_structured_query(query, db_path)
    if route == "hybrid":
        return answer_hybrid_query(query, chunks, db_path)
    return answer_text_query(query, chunks)
