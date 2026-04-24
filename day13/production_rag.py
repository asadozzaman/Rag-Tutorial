from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
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
    "when",
    "with",
}


@dataclass
class DocumentChunk:
    chunk_id: str
    title: str
    text: str
    source: str


@dataclass
class QueryResult:
    answer: str
    sources: list[str]
    latency_ms: float
    cached: bool
    cache_type: str
    token_count: int
    trace_id: str


@dataclass
class CacheEntry:
    query: str
    result: QueryResult
    created_at: float


@dataclass
class RequestLog:
    query: str
    user_id: str
    latency_ms: float
    cache_hit: bool
    cache_type: str
    token_count: int
    trace_id: str


def tokenize(text: str) -> list[str]:
    current = []
    tokens = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def content_terms(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def jaccard_similarity(left: str, right: str) -> float:
    left_terms = content_terms(left)
    right_terms = content_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def get_cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode("utf-8")).hexdigest()


class ProductionRAGService:
    def __init__(self, docs: list[DocumentChunk], rate_limit_per_minute: int = 10) -> None:
        self.docs = docs
        self.rate_limit_per_minute = rate_limit_per_minute
        self.exact_cache: dict[str, CacheEntry] = {}
        self.semantic_cache: list[CacheEntry] = []
        self.request_windows: dict[str, deque[float]] = {}
        self.logs: list[RequestLog] = []

    @classmethod
    def from_sample_file(cls, csv_path: str | Path) -> "ProductionRAGService":
        docs = []
        path = Path(csv_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            chunk_id, title, text, source = line.split(",", 3)
            docs.append(
                DocumentChunk(
                    chunk_id=chunk_id.strip(),
                    title=title.strip(),
                    text=text.strip().strip('"'),
                    source=source.strip(),
                )
            )
        return cls(docs)

    def _check_rate_limit(self, user_id: str) -> None:
        now = time.time()
        window = self.request_windows.setdefault(user_id, deque())
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.rate_limit_per_minute:
            raise ValueError(f"Rate limit exceeded for user '{user_id}'. Try again in a minute.")
        window.append(now)

    def _find_cached(self, query: str) -> tuple[QueryResult | None, str]:
        exact_key = get_cache_key(query)
        if exact_key in self.exact_cache:
            cached = self.exact_cache[exact_key].result
            return cached, "exact"

        best_entry: CacheEntry | None = None
        best_similarity = 0.0
        for entry in self.semantic_cache:
            similarity = jaccard_similarity(query, entry.query)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry
        if best_entry and best_similarity >= 0.45:
            return best_entry.result, "semantic"
        return None, "miss"

    def _retrieve(self, query: str, top_k: int) -> list[DocumentChunk]:
        query_terms = content_terms(query)
        scored: list[tuple[DocumentChunk, float]] = []
        for doc in self.docs:
            doc_terms = content_terms(f"{doc.title} {doc.text}")
            overlap = len(query_terms & doc_terms)
            phrase_bonus = 1.0 if any(term in doc.text.lower() for term in query_terms) else 0.0
            score = overlap + phrase_bonus
            if score > 0:
                scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    def _generate_answer(self, query: str, docs: list[DocumentChunk]) -> str:
        lowered = query.lower()
        combined = " ".join(doc.text for doc in docs)
        if "caching" in lowered:
            return "Caching reduces latency and API costs significantly. In production, pair exact cache keys with near-duplicate semantic cache checks."
        if "fastapi" in lowered:
            return "FastAPI is a strong choice for production RAG APIs because it supports async handlers, validation, and streaming-friendly responses."
        if "rate" in lowered and "limit" in lowered:
            return "Rate limiting protects the service from burst traffic and keeps latency predictable under load."
        if "observability" in lowered or "trace" in lowered:
            return "Production RAG needs trace logging for retrieval, generation, latency, cache hits, and failure paths."
        if docs:
            return " ".join(doc.text for doc in docs)
        return f"No strong retrieval hit for '{query}'. In production, this should trigger a fallback or safe failure message."

    def _cache_result(self, query: str, result: QueryResult) -> None:
        entry = CacheEntry(query=query, result=result, created_at=time.time())
        self.exact_cache[get_cache_key(query)] = entry
        self.semantic_cache.append(entry)
        self.semantic_cache = self.semantic_cache[-50:]

    def _log_request(self, query: str, user_id: str, result: QueryResult) -> None:
        self.logs.append(
            RequestLog(
                query=query,
                user_id=user_id,
                latency_ms=result.latency_ms,
                cache_hit=result.cached,
                cache_type=result.cache_type,
                token_count=result.token_count,
                trace_id=result.trace_id,
            )
        )

    async def query(self, query: str, user_id: str = "demo", top_k: int = 3, use_cache: bool = True) -> QueryResult:
        start = time.perf_counter()
        trace_id = hashlib.md5(f"{user_id}:{query}:{time.time()}".encode("utf-8")).hexdigest()[:10]
        self._check_rate_limit(user_id)

        if use_cache:
            cached, cache_type = self._find_cached(query)
            if cached:
                result = QueryResult(
                    answer=cached.answer,
                    sources=cached.sources,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    cached=True,
                    cache_type=cache_type,
                    token_count=len(tokenize(cached.answer)),
                    trace_id=trace_id,
                )
                self._log_request(query, user_id, result)
                return result

        await asyncio.sleep(0.01)
        docs = self._retrieve(query, top_k=top_k)
        answer = self._generate_answer(query, docs)
        result = QueryResult(
            answer=answer,
            sources=[f"{doc.title} [{doc.source}]" for doc in docs],
            latency_ms=(time.perf_counter() - start) * 1000,
            cached=False,
            cache_type="miss",
            token_count=len(tokenize(answer)),
            trace_id=trace_id,
        )
        if use_cache:
            self._cache_result(query, result)
        self._log_request(query, user_id, result)
        return result

    async def stream_query(self, query: str, user_id: str = "demo", top_k: int = 3, use_cache: bool = True):
        result = await self.query(query=query, user_id=user_id, top_k=top_k, use_cache=use_cache)
        for token in result.answer.split():
            payload = {"trace_id": result.trace_id, "token": token}
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.001)
        yield f"data: {{'done': True, 'cached': {str(result.cached).lower()}}}\n\n"

    def upload_text(self, filename: str, content: str) -> int:
        paragraphs = [part.strip() for part in content.splitlines() if part.strip()]
        count = 0
        for idx, paragraph in enumerate(paragraphs, start=1):
            self.docs.append(
                DocumentChunk(
                    chunk_id=f"upload_{int(time.time())}_{idx}",
                    title=f"{filename} chunk {idx}",
                    text=paragraph,
                    source=filename,
                )
            )
            count += 1
        return count

    def health(self) -> dict[str, object]:
        total_requests = len(self.logs)
        cache_hits = sum(1 for log in self.logs if log.cache_hit)
        avg_latency = (
            sum(log.latency_ms for log in self.logs) / total_requests
            if total_requests
            else 0.0
        )
        return {
            "status": "healthy",
            "documents": len(self.docs),
            "cache_size": len(self.exact_cache),
            "cache_hit_rate": round(cache_hits / total_requests, 2) if total_requests else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
            "total_requests": total_requests,
        }
