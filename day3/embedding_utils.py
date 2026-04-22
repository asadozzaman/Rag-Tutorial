"""
Small embedding utilities for Day 3.

The goal is learning and benchmarking, so this file includes:
  - cosine similarity
  - normalization helpers
  - simple local embedding baselines
  - optional Gemini embedding wrapper
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
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
    "before",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "why",
    "with",
}


def load_env_near(base_dir: Path) -> None:
    for env_path in [
        base_dir / ".env",
        base_dir.parent / ".env",
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
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


class EmbeddingModel:
    name = "base"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def dimensions(self) -> int | str:
        sample = self.embed_query("dimension check")
        return len(sample)


class KeywordOverlapEmbeddings(EmbeddingModel):
    """Simple vocabulary vector baseline.

    This is not a production embedding model. It helps show the gap between
    lexical matching and semantic embedding models.
    """

    name = "keyword-overlap-baseline"

    def __init__(self, vocabulary: list[str]):
        self.vocabulary = vocabulary

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            token_counts = {}
            for token in tokenize(text):
                token_counts[token] = token_counts.get(token, 0) + 1
            vector = [float(token_counts.get(token, 0)) for token in self.vocabulary]
            vectors.append(l2_normalize(vector))
        return vectors


class HashingEmbeddings(EmbeddingModel):
    """Deterministic local embedding baseline using token hashing."""

    name = "hashing-baseline"

    def __init__(self, dimensions: int = 256):
        self._dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self._dimensions
            for token in tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % self._dimensions
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                vector[bucket] += sign
            vectors.append(l2_normalize(vector))
        return vectors

    @property
    def dimensions(self) -> int:
        return self._dimensions


class GeminiEmbeddings(EmbeddingModel):
    name = "gemini-embedding-001"

    def __init__(self, api_key: str):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self.model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            api_key=api_key,
        )
        self._dimensions: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.embed_documents(texts)
        if vectors and self._dimensions is None:
            self._dimensions = len(vectors[0])
        return [l2_normalize([float(value) for value in vector]) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.embed_query(text)
        if self._dimensions is None:
            self._dimensions = len(vector)
        return l2_normalize([float(value) for value in vector])

    @property
    def dimensions(self) -> int | str:
        return self._dimensions or "unknown until first embedding"


@dataclass
class TimedEmbeddingResult:
    vectors: list[list[float]]
    latency_ms: float


def timed_embed_documents(model: EmbeddingModel, texts: list[str]) -> TimedEmbeddingResult:
    start = time.perf_counter()
    vectors = model.embed_documents(texts)
    latency_ms = (time.perf_counter() - start) * 1000
    return TimedEmbeddingResult(vectors=vectors, latency_ms=latency_ms)


def build_vocabulary(texts: list[str], max_terms: int = 80) -> list[str]:
    counts = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_terms]]


def maybe_build_gemini_model() -> GeminiEmbeddings | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        return GeminiEmbeddings(api_key=api_key)
    except Exception as exc:
        print(f"[WARN] Gemini embeddings unavailable: {exc}")
        return None
