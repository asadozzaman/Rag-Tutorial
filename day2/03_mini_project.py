"""
Day 2 Mini Project: Multi-format document ingester.

What it does:
  - Loads TXT, Markdown, CSV, PDF, and web URLs
  - Chooses a chunking strategy based on source type
  - Preserves metadata on every chunk
  - Builds one FAISS index across all sources
  - Lets you query the combined knowledge base
  - Reports chunk quality statistics

Run:
  python 03_mini_project.py
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from chunking_utils import markdown_aware_chunks, recursive_text_chunks

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"
INDEX_DIR = BASE_DIR / "faiss_multi_format_index"

for env_path in [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
    BASE_DIR.parent / "day1" / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv(find_dotenv())


@dataclass
class ChunkStats:
    total_chunks: int
    average_size: int
    min_size: int
    max_size: int
    too_short: int
    too_long: int


def configure_gemini_api() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY in your environment or .env file."
        )
    return api_key


def load_web_document(url: str) -> list[Document]:
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
    text = "\n\n".join(paragraphs)
    return [
        Document(
            page_content=text,
            metadata={"source": url, "source_type": "web", "url": url},
        )
    ]


def load_csv_documents(path: Path) -> list[Document]:
    docs = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=1):
            page_content = " | ".join(f"{key}: {value}" for key, value in row.items())
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": str(path),
                        "source_type": "csv",
                        "row": row_number,
                    },
                )
            )
    return docs


def load_markdown_documents(path: Path) -> list[Document]:
    markdown_text = path.read_text(encoding="utf-8")
    docs = []
    for chunk in markdown_aware_chunks(markdown_text, chunk_size=450, chunk_overlap=60):
        metadata = {
            "source": str(path),
            "source_type": "markdown",
            **chunk["metadata"],
        }
        docs.append(Document(page_content=chunk["content"], metadata=metadata))
    return docs


def load_text_documents(path: Path) -> list[Document]:
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path), "source_type": "text"},
        )
    ]


def load_pdf_documents(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "source_type": "pdf",
                        "page": page_number,
                    },
                )
            )
    return docs


def split_document(doc: Document) -> list[Document]:
    source_type = doc.metadata.get("source_type", "text")

    if source_type == "csv":
        return [doc]

    if source_type == "markdown":
        return [doc]

    chunks = []
    for chunk_text in recursive_text_chunks(
        doc.page_content,
        chunk_size=500,
        chunk_overlap=50,
    ):
        chunks.append(Document(page_content=chunk_text, metadata=dict(doc.metadata)))
    return chunks


def analyze_chunks(chunks: list[Document]) -> ChunkStats:
    sizes = [len(chunk.page_content) for chunk in chunks]
    return ChunkStats(
        total_chunks=len(chunks),
        average_size=sum(sizes) // max(len(sizes), 1),
        min_size=min(sizes),
        max_size=max(sizes),
        too_short=sum(size < 50 for size in sizes),
        too_long=sum(size > 800 for size in sizes),
    )


def format_sources(docs: Iterable[Document]) -> str:
    lines = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        source_type = doc.metadata.get("source_type", "unknown")
        section = doc.metadata.get("section")
        page = doc.metadata.get("page")
        location = []

        if section:
            location.append(f"section={section}")
        if page is not None:
            location.append(f"page={page}")

        suffix = f" ({', '.join(location)})" if location else ""
        lines.append(f"[Source {index}] {source_type} | {source}{suffix}")
    return "\n".join(lines)


def build_answer(llm, query: str, retrieved_docs: list[Document]) -> str:
    context = "\n\n".join(
        f"[Source {index}] {doc.page_content}"
        for index, doc in enumerate(retrieved_docs, start=1)
    )
    prompt = f"""You are a careful RAG assistant.

Answer using only the retrieved context below.
If the answer is not present, say that clearly.
Include citations like [Source 1].

Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def gather_default_inputs() -> list[str]:
    return [
        str(SAMPLE_DIR / "architecture.txt"),
        str(SAMPLE_DIR / "retrieval_notes.md"),
        str(SAMPLE_DIR / "faq.csv"),
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    ]


def load_source(path_or_url: str) -> list[Document]:
    if path_or_url.startswith(("http://", "https://")):
        return load_web_document(path_or_url)

    path = Path(path_or_url)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_text_documents(path)
    if suffix == ".md":
        return load_markdown_documents(path)
    if suffix == ".csv":
        return load_csv_documents(path)
    if suffix == ".pdf":
        return load_pdf_documents(path)

    raise ValueError(f"Unsupported source type: {path}")


def print_chunk_breakdown(chunks: list[Document]) -> None:
    counts = {}
    for chunk in chunks:
        source_type = chunk.metadata.get("source_type", "unknown")
        counts[source_type] = counts.get(source_type, 0) + 1

    print("\nChunk breakdown by source type:")
    for source_type, count in sorted(counts.items()):
        print(f"  {source_type:<10} {count}")


def main() -> None:
    api_key = configure_gemini_api()
    os.environ.setdefault("USER_AGENT", "day2-rag-learning/1.0")

    print("=" * 66)
    print("DAY 2 MINI PROJECT: MULTI-FORMAT DOCUMENT INGESTER")
    print("=" * 66)

    inputs = gather_default_inputs()
    print("\nLoading sources:")
    raw_docs = []
    for item in inputs:
        docs = load_source(item)
        raw_docs.extend(docs)
        print(f"  Loaded {len(docs)} document(s) from: {item}")

    chunks = []
    for doc in raw_docs:
        chunks.extend(split_document(doc))

    stats = analyze_chunks(chunks)
    print(f"\nCreated {stats.total_chunks} chunks")
    print(f"Average chunk size: {stats.average_size} chars")
    print(f"Smallest chunk: {stats.min_size} chars")
    print(f"Largest chunk: {stats.max_size} chars")
    print(f"Too short (<50 chars): {stats.too_short}")
    print(f"Too long (>800 chars): {stats.too_long}")
    print_chunk_breakdown(chunks)

    print("\nCreating embeddings with models/gemini-embedding-001...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=api_key,
    )

    if INDEX_DIR.exists():
        vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"Loaded existing index from: {INDEX_DIR}")
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(INDEX_DIR))
        print(f"Saved new FAISS index to: {INDEX_DIR}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=api_key,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    demo_queries = [
        "Why is recursive chunking usually a strong default?",
        "What metadata should I preserve for RAG chunks?",
        "Why can bad chunking reduce retrieval quality?",
    ]

    print("\n" + "=" * 66)
    print("DEMO QUERIES")
    print("=" * 66)

    for query in demo_queries:
        retrieved_docs = retriever.invoke(query)
        answer = build_answer(llm, query, retrieved_docs)

        print(f"\nQuestion: {query}")
        print(f"Answer: {answer}")
        print("Sources:")
        print(format_sources(retrieved_docs))
        print("-" * 66)

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

        retrieved_docs = retriever.invoke(user_query)
        answer = build_answer(llm, user_query, retrieved_docs)
        print(f"Bot: {answer}")
        print("Sources:")
        print(format_sources(retrieved_docs))


if __name__ == "__main__":
    main()
