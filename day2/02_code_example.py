"""
Day 2: Comparing document loading and chunking strategies.

This script shows:
  1. Loading text from Markdown, TXT, CSV, and the web
  2. Comparing fixed-size, recursive, and Markdown-aware chunking
  3. Inspecting chunk quality instead of guessing

Run:
  python 02_code_example.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from chunking_utils import (
    fixed_size_chunks,
    markdown_aware_chunks,
    recursive_text_chunks,
)


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_csv_as_text(path: Path) -> str:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(" | ".join(f"{key}: {value}" for key, value in row.items()))
    return "\n".join(rows)


def load_web_text(url: str) -> str:
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "day2-rag-learning/1.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.select("p")
        if p.get_text(" ", strip=True)
    ]
    return "\n\n".join(paragraphs[:12])


def describe_chunks(name: str, chunks: list[str]) -> None:
    sizes = [len(chunk) for chunk in chunks]
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Chunks: {len(chunks)}")
    print(f"Average size: {sum(sizes) // max(len(sizes), 1)} chars")
    print(f"Smallest: {min(sizes)} chars")
    print(f"Largest: {max(sizes)} chars")


def preview_boundaries(name: str, chunks: list[str], limit: int = 2) -> None:
    print(f"\nBoundary preview: {name}")
    for index, chunk in enumerate(chunks[:limit], start=1):
        cleaned = chunk.replace("\n", " ")
        print(f"  Chunk {index}: {cleaned[:180]}...")


def main() -> None:
    markdown_text = load_text_file(SAMPLE_DIR / "retrieval_notes.md")
    plain_text = load_text_file(SAMPLE_DIR / "architecture.txt")
    csv_text = load_csv_as_text(SAMPLE_DIR / "faq.csv")
    web_text = load_web_text("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

    combined_text = "\n\n".join(
        [
            "# Markdown sample\n" + markdown_text,
            "# Text sample\n" + plain_text,
            "# CSV sample\n" + csv_text,
            "# Web sample\n" + web_text,
        ]
    )

    fixed_chunks = fixed_size_chunks(combined_text, chunk_size=500, chunk_overlap=50)
    recursive_chunks = recursive_text_chunks(
        combined_text,
        chunk_size=500,
        chunk_overlap=50,
    )
    markdown_chunks = [
        chunk["content"]
        for chunk in markdown_aware_chunks(
            markdown_text,
            chunk_size=450,
            chunk_overlap=60,
        )
    ]

    print("=" * 64)
    print("DAY 2: DOCUMENT LOADING AND CHUNKING COMPARISON")
    print("=" * 64)
    print("\nLoaded sources:")
    print(f"  Markdown chars: {len(markdown_text)}")
    print(f"  Text chars:     {len(plain_text)}")
    print(f"  CSV chars:      {len(csv_text)}")
    print(f"  Web chars:      {len(web_text)}")

    describe_chunks("Fixed-size chunks", fixed_chunks)
    describe_chunks("Recursive chunks", recursive_chunks)
    describe_chunks("Markdown-aware chunks", markdown_chunks)

    preview_boundaries("Fixed-size", fixed_chunks)
    preview_boundaries("Recursive", recursive_chunks)
    preview_boundaries("Markdown-aware", markdown_chunks)

    print("\nObservations:")
    print("  1. Fixed-size chunking is simple, but often cuts in awkward places.")
    print("  2. Recursive chunking usually preserves sentence boundaries better.")
    print("  3. Markdown-aware chunking keeps section meaning, which helps retrieval.")


if __name__ == "__main__":
    main()
