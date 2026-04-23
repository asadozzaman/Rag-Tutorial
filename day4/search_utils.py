"""
Lightweight search and indexing utilities for Day 4.

This module is intentionally dependency-light so the indexing concepts are easy
to inspect:
  - flat exact vector search
  - clustered approximate search (IVF-style idea)
  - metadata filtering
  - BM25 keyword search
  - reciprocal rank fusion (RRF)
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
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
    "that",
    "the",
    "to",
    "what",
    "with",
}


@dataclass
class SearchDocument:
    doc_id: str
    text: str
    metadata: dict


@dataclass
class SearchResult:
    doc_id: str
    score: float
    text: str
    metadata: dict
    source: str


def load_env_near(base_dir: Path) -> None:
    for env_path in [
        base_dir / ".env",
        base_dir.parent / ".env",
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


class FlatVectorIndex:
    def __init__(self, documents: list[SearchDocument], vectors: list[list[float]]):
        self.documents = documents
        self.vectors = vectors

    def search(
        self,
        query_vector: list[float],
        k: int = 3,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        results = []
        for document, vector in zip(self.documents, self.vectors):
            if not metadata_matches(document.metadata, filters):
                continue
            score = cosine_similarity(query_vector, vector)
            results.append(
                SearchResult(
                    doc_id=document.doc_id,
                    score=score,
                    text=document.text,
                    metadata=document.metadata,
                    source="semantic-flat",
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]

    def save(self, path: Path) -> None:
        payload = {
            "documents": [asdict(document) for document in self.documents],
            "vectors": self.vectors,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FlatVectorIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = [SearchDocument(**item) for item in payload["documents"]]
        return cls(documents=documents, vectors=payload["vectors"])


class ClusteredVectorIndex:
    """A tiny IVF-style index.

    We choose a few centroids, assign documents to the closest centroid, and at
    query time search only the nearest centroid lists. This is approximate, not
    exact, which makes it a good teaching tool for IVF concepts.
    """

    def __init__(
        self,
        documents: list[SearchDocument],
        vectors: list[list[float]],
        centroids: list[list[float]],
        assignments: list[int],
    ):
        self.documents = documents
        self.vectors = vectors
        self.centroids = centroids
        self.assignments = assignments

    @classmethod
    def build(
        cls,
        documents: list[SearchDocument],
        vectors: list[list[float]],
        n_lists: int = 3,
    ) -> "ClusteredVectorIndex":
        n_lists = max(1, min(n_lists, len(vectors)))
        centroids = [list(vector) for vector in vectors[:n_lists]]
        assignments = [0] * len(vectors)

        for _ in range(3):
            for index, vector in enumerate(vectors):
                scores = [cosine_similarity(vector, centroid) for centroid in centroids]
                assignments[index] = max(range(len(scores)), key=lambda idx: scores[idx])

            for cluster_id in range(n_lists):
                members = [vectors[i] for i, assigned in enumerate(assignments) if assigned == cluster_id]
                if not members:
                    continue
                dims = len(members[0])
                mean = [0.0] * dims
                for member in members:
                    for dim, value in enumerate(member):
                        mean[dim] += value
                centroids[cluster_id] = l2_normalize([value / len(members) for value in mean])

        return cls(documents=documents, vectors=vectors, centroids=centroids, assignments=assignments)

    def search(
        self,
        query_vector: list[float],
        k: int = 3,
        filters: dict | None = None,
        n_probe: int = 1,
    ) -> list[SearchResult]:
        centroid_scores = [
            (cluster_id, cosine_similarity(query_vector, centroid))
            for cluster_id, centroid in enumerate(self.centroids)
        ]
        chosen_clusters = {
            cluster_id for cluster_id, _ in sorted(centroid_scores, key=lambda item: item[1], reverse=True)[:n_probe]
        }

        results = []
        for document, vector, cluster_id in zip(self.documents, self.vectors, self.assignments):
            if cluster_id not in chosen_clusters:
                continue
            if not metadata_matches(document.metadata, filters):
                continue
            score = cosine_similarity(query_vector, vector)
            results.append(
                SearchResult(
                    doc_id=document.doc_id,
                    score=score,
                    text=document.text,
                    metadata=document.metadata,
                    source="semantic-clustered",
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]


class BM25Index:
    def __init__(self, documents: list[SearchDocument], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document.text) for document in documents]
        self.avg_len = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        self.doc_freqs = {}

        for tokens in self.doc_tokens:
            seen = set(tokens)
            for token in seen:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def _idf(self, term: str) -> float:
        n_docs = len(self.documents)
        doc_freq = self.doc_freqs.get(term, 0)
        return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, k: int = 3, filters: dict | None = None) -> list[SearchResult]:
        query_terms = tokenize(query)
        results = []

        for document, tokens in zip(self.documents, self.doc_tokens):
            if not metadata_matches(document.metadata, filters):
                continue

            score = 0.0
            doc_len = len(tokens)
            for term in query_terms:
                tf = tokens.count(term)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1))
                score += self._idf(term) * numerator / denominator

            results.append(
                SearchResult(
                    doc_id=document.doc_id,
                    score=score,
                    text=document.text,
                    metadata=document.metadata,
                    source="bm25-keyword",
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:k]


def reciprocal_rank_fusion(result_sets: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    fused = {}
    for result_set in result_sets:
        for rank, result in enumerate(result_set, start=1):
            if result.doc_id not in fused:
                fused[result.doc_id] = SearchResult(
                    doc_id=result.doc_id,
                    score=0.0,
                    text=result.text,
                    metadata=result.metadata,
                    source="hybrid-rrf",
                )
            fused[result.doc_id].score += 1 / (k + rank)

    return sorted(fused.values(), key=lambda item: item.score, reverse=True)


def load_documents_from_csv(path: Path) -> list[SearchDocument]:
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
                SearchDocument(
                    doc_id=row["doc_id"],
                    text=row["text"],
                    metadata=metadata,
                )
            )
    return documents


def print_results(title: str, results: list[SearchResult]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for result in results:
        preview = result.text[:90].replace("\n", " ")
        print(
            f"  {result.score: .4f} | {result.doc_id} | {result.metadata} | "
            f"{preview}..."
        )
