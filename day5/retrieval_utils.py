"""
Lightweight retrieval and reranking utilities for Day 5.

This module demonstrates:
  - semantic retrieval
  - BM25 keyword retrieval
  - ensemble fusion
  - MMR diversity selection
  - self-query filter parsing
  - parent-child retrieval
  - pairwise reranking
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(20\d{2})\b")
FIELD_PATTERN = re.compile(r"\b(category|author|year|level):([a-z0-9_-]+)\b", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "show",
    "the",
    "to",
    "what",
    "which",
    "with",
}


@dataclass
class RetrievalDocument:
    doc_id: str
    parent_id: str
    title: str
    url: str
    text: str
    metadata: dict


@dataclass
class RetrievalResult:
    doc_id: str
    parent_id: str
    score: float
    title: str
    text: str
    url: str
    metadata: dict
    source: str


@dataclass
class QuerySpec:
    query: str
    filters: dict


def load_env_near(base_dir: Path) -> None:
    for env_path in [
        base_dir / ".env",
        base_dir.parent / ".env",
        base_dir.parent / "day4" / ".env",
        base_dir.parent / "day3" / ".env",
        base_dir.parent / "day2" / ".env",
        base_dir.parent / "day1" / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            return
    load_dotenv(find_dotenv())


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensionality.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class EmbeddingModel:
    name = "base"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashingEmbeddings(EmbeddingModel):
    name = "hashing-semantic-baseline"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % self.dimensions
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                vector[bucket] += sign
            vectors.append(l2_normalize(vector))
        return vectors


class GeminiEmbeddings(EmbeddingModel):
    name = "gemini-embedding-001"

    def __init__(self, api_key: str):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self.model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            api_key=api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.embed_documents(texts)
        return [l2_normalize([float(value) for value in vector]) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.embed_query(text)
        return l2_normalize([float(value) for value in vector])


def maybe_build_gemini_model() -> GeminiEmbeddings | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        return GeminiEmbeddings(api_key=api_key)
    except Exception as exc:
        print(f"[WARN] Gemini embeddings unavailable: {exc}")
        return None


def timed_embed_documents(model: EmbeddingModel, texts: list[str]) -> tuple[list[list[float]], float]:
    start = time.perf_counter()
    vectors = model.embed_documents(texts)
    latency_ms = (time.perf_counter() - start) * 1000
    return vectors, latency_ms


def metadata_matches(metadata: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if str(metadata.get(key)).lower() != str(expected).lower():
            return False
    return True


def load_documents(path: Path) -> list[RetrievalDocument]:
    documents = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata = {
                "category": row["category"],
                "author": row["author"],
                "year": row["year"],
                "level": row["level"],
            }
            documents.append(
                RetrievalDocument(
                    doc_id=row["doc_id"],
                    parent_id=row["parent_id"],
                    title=row["title"],
                    url=row["url"],
                    text=row["text"],
                    metadata=metadata,
                )
            )
    return documents


class SemanticRetriever:
    def __init__(self, documents: list[RetrievalDocument], vectors: list[list[float]]):
        self.documents = documents
        self.vectors = vectors

    def search(self, query_vector: list[float], k: int = 5, filters: dict | None = None) -> list[RetrievalResult]:
        results = []
        for document, vector in zip(self.documents, self.vectors):
            if not metadata_matches(document.metadata, filters):
                continue
            score = cosine_similarity(query_vector, vector)
            results.append(
                RetrievalResult(
                    doc_id=document.doc_id,
                    parent_id=document.parent_id,
                    score=score,
                    title=document.title,
                    text=document.text,
                    url=document.url,
                    metadata=document.metadata,
                    source="semantic",
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]


class BM25Retriever:
    def __init__(self, documents: list[RetrievalDocument], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document.text) + tokenize(document.title) for document in documents]
        self.avg_len = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        self.doc_freqs = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def _idf(self, term: str) -> float:
        n_docs = len(self.documents)
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> list[RetrievalResult]:
        terms = tokenize(query)
        results = []
        for document, tokens in zip(self.documents, self.doc_tokens):
            if not metadata_matches(document.metadata, filters):
                continue
            doc_len = len(tokens)
            score = 0.0
            for term in terms:
                tf = tokens.count(term)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1))
                score += self._idf(term) * numerator / denominator
            results.append(
                RetrievalResult(
                    doc_id=document.doc_id,
                    parent_id=document.parent_id,
                    score=score,
                    title=document.title,
                    text=document.text,
                    url=document.url,
                    metadata=document.metadata,
                    source="bm25",
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]


def weighted_ensemble(result_sets: list[tuple[list[RetrievalResult], float]]) -> list[RetrievalResult]:
    merged = {}
    for results, weight in result_sets:
        if not results:
            continue
        max_score = max(result.score for result in results) or 1.0
        for result in results:
            if result.doc_id not in merged:
                merged[result.doc_id] = RetrievalResult(
                    doc_id=result.doc_id,
                    parent_id=result.parent_id,
                    score=0.0,
                    title=result.title,
                    text=result.text,
                    url=result.url,
                    metadata=result.metadata,
                    source="ensemble",
                )
            merged[result.doc_id].score += weight * (result.score / max_score)
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def mmr_select(
    query_vector: list[float],
    candidate_results: list[RetrievalResult],
    vector_lookup: dict[str, list[float]],
    k: int = 3,
    lambda_mult: float = 0.5,
) -> list[RetrievalResult]:
    selected: list[RetrievalResult] = []
    remaining = candidate_results[:]

    while remaining and len(selected) < k:
        best_result = None
        best_score = float("-inf")
        for candidate in remaining:
            relevance = candidate.score
            diversity_penalty = 0.0
            candidate_vector = vector_lookup[candidate.doc_id]
            if selected:
                diversity_penalty = max(
                    cosine_similarity(candidate_vector, vector_lookup[item.doc_id])
                    for item in selected
                )
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_result = candidate
        selected.append(best_result)
        remaining = [item for item in remaining if item.doc_id != best_result.doc_id]

    return selected


def parse_self_query(query: str) -> QuerySpec:
    filters = {}

    for match in FIELD_PATTERN.finditer(query):
        filters[match.group(1).lower()] = match.group(2).lower()

    year_match = YEAR_RE.search(query)
    if year_match:
        filters.setdefault("year", year_match.group(1))

    lowered = query.lower()
    for level in ["beginner", "intermediate", "advanced"]:
        if level in lowered:
            filters.setdefault("level", level)
    for author in ["samira", "nora", "omar", "liam"]:
        if author in lowered:
            filters.setdefault("author", author)

    # Keep natural-language filter words in the semantic query so retrieval
    # still sees the topical hints. Only strip explicit field syntax.
    cleaned = FIELD_PATTERN.sub("", query)
    cleaned = " ".join(cleaned.split()).strip() or query

    return QuerySpec(query=cleaned, filters=filters)


def aggregate_to_parents(results: list[RetrievalResult]) -> list[RetrievalResult]:
    parents = {}
    for result in results:
        current = parents.get(result.parent_id)
        if current is None or result.score > current.score:
            parents[result.parent_id] = RetrievalResult(
                doc_id=result.parent_id,
                parent_id=result.parent_id,
                score=result.score,
                title=result.title,
                text=result.text,
                url=result.url,
                metadata=result.metadata,
                source="parent-child",
            )
    return sorted(parents.values(), key=lambda item: item.score, reverse=True)


class PairwiseReranker:
    """Scores query-document pairs jointly using lexical and semantic cues.

    This is a lightweight stand-in for a full cross-encoder. It lets the Day 5
    project demonstrate the reranking workflow without heavyweight model
    dependencies.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self._embedding_cache: dict[str, list[float]] = {}
        self._fallback_model: EmbeddingModel | None = None

    def _embed_text(self, text: str) -> list[float]:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached

        try:
            vector = self.embedding_model.embed_query(text)
        except Exception:
            if self._fallback_model is None:
                self._fallback_model = HashingEmbeddings(dimensions=256)
            vector = self._fallback_model.embed_query(text)

        self._embedding_cache[text] = vector
        return vector

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 3) -> list[RetrievalResult]:
        query_tokens = set(tokenize(query))
        query_vector = self._embed_text(query)
        reranked = []

        for result in results:
            doc_tokens = set(tokenize(result.title + " " + result.text))
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            semantic = cosine_similarity(query_vector, self._embed_text(result.title + " " + result.text))
            title_boost = 0.15 if query_tokens & set(tokenize(result.title)) else 0.0
            metadata_boost = 0.08 if any(value in result.text.lower() for value in result.metadata.values()) else 0.0
            final_score = 0.55 * semantic + 0.30 * overlap + title_boost + metadata_boost

            reranked.append(
                RetrievalResult(
                    doc_id=result.doc_id,
                    parent_id=result.parent_id,
                    score=final_score,
                    title=result.title,
                    text=result.text,
                    url=result.url,
                    metadata=result.metadata,
                    source="reranker",
                )
            )

        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]


def print_results(title: str, results: list[RetrievalResult]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for result in results:
        preview = result.text[:85].replace("\n", " ")
        print(
            f"  {result.score: .4f} | {result.doc_id} | {result.metadata} | "
            f"{preview}..."
        )


def load_eval_queries(path: Path) -> list[dict]:
    queries = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            queries.append(row)
    return queries


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1 / rank
