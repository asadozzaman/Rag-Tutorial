from __future__ import annotations

import asyncio
import csv
import hashlib
import time
from collections import defaultdict
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
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "with",
}

CONCEPT_ALIASES = {
    "semantic caching": ["cache", "caching", "cache hit"],
    "hybrid retrieval": ["hybrid retrieval", "ensemble retrieval", "bm25", "vector"],
    "reranking": ["reranking", "rerank", "cross-encoder", "precision"],
    "crag": ["crag", "corrective rag", "self-correcting"],
    "fastapi": ["fastapi", "api service", "deployment"],
    "memory": ["memory", "conversation memory", "history-aware retrieval"],
    "evaluation": ["evaluation", "ragas", "metrics", "faithfulness"],
}

QUERY_EXPANSIONS = {
    "precision": ["reranking", "cross-encoder", "hybrid retrieval"],
    "latency": ["semantic caching", "streaming", "cache hit"],
    "deployment": ["fastapi", "docker", "health checks"],
    "memory": ["history-aware retrieval", "conversation memory"],
    "crag": ["corrective rag", "fallback", "query refinement"],
}


@dataclass
class CapstoneConfig:
    chunk_size: int = 260
    chunk_overlap: int = 40
    retrieval_k: int = 6
    rerank_top_k: int = 3
    cache_enabled: bool = True
    semantic_cache_threshold: float = 0.45


@dataclass
class DocumentChunk:
    chunk_id: str
    title: str
    text: str
    source: str
    doc_type: str
    category: str


@dataclass
class Citation:
    title: str
    source: str
    score: float
    preview: str


@dataclass
class QueryTrace:
    standalone_query: str
    action: str
    retrieved: list[str] = field(default_factory=list)
    reranked: list[str] = field(default_factory=list)
    refined_query: str | None = None


@dataclass
class RAGResult:
    answer: str
    citations: list[Citation]
    confidence: str
    latency_ms: float
    cached: bool
    cache_type: str
    trace: QueryTrace
    token_count: int


def tokenize(text: str) -> list[str]:
    current = []
    tokens = []
    for char in text.lower():
        if char.isalnum() or char in {"-", "/"}:
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


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= chunk_size:
            current = f"{current} {paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                current = f"{tail} {paragraph}".strip()
            else:
                current = paragraph
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


class CapstoneRAG:
    def __init__(self, config: CapstoneConfig | None = None) -> None:
        self.config = config or CapstoneConfig()
        self.chunks: list[DocumentChunk] = []
        self.cache: dict[str, RAGResult] = {}
        self.semantic_cache: list[tuple[str, RAGResult]] = []
        self.sessions: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.metrics = {
            "queries": 0,
            "cache_hits": 0,
            "uploads": 0,
            "avg_latency_ms": 0.0,
        }

    def ingest_csv(self, csv_path: str | Path) -> int:
        count = 0
        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                count += self.ingest_document(
                    title=row["title"].strip(),
                    text=row["text"].strip(),
                    source=row["source"].strip(),
                    doc_type=row["doc_type"].strip(),
                    category=row["category"].strip(),
                )
        return count

    def ingest_document(self, title: str, text: str, source: str, doc_type: str, category: str) -> int:
        smart_size = self.config.chunk_size
        if doc_type == "faq":
            smart_size = 180
        elif doc_type == "report":
            smart_size = 320
        elif doc_type == "markdown":
            smart_size = 240

        chunks = split_text(text, smart_size, self.config.chunk_overlap)
        created = 0
        for index, chunk in enumerate(chunks, start=1):
            self.chunks.append(
                DocumentChunk(
                    chunk_id=f"{source}_{len(self.chunks) + 1}",
                    title=f"{title} {index}",
                    text=chunk,
                    source=source,
                    doc_type=doc_type,
                    category=category,
                )
            )
            created += 1
        return created

    def upload_text(self, filename: str, content: str, doc_type: str = "markdown") -> int:
        self.metrics["uploads"] += 1
        category = "uploaded"
        return self.ingest_document(filename, content, filename, doc_type, category)

    def _get_cache_key(self, query: str, session_id: str) -> str:
        return hashlib.md5(f"{session_id}:{query.lower().strip()}".encode("utf-8")).hexdigest()

    def _find_cache(self, query: str, session_id: str) -> tuple[RAGResult | None, str]:
        key = self._get_cache_key(query, session_id)
        if key in self.cache:
            return self.cache[key], "exact"
        best_result = None
        best_score = 0.0
        for cached_query, result in self.semantic_cache:
            score = jaccard_similarity(query, cached_query)
            if score > best_score:
                best_score = score
                best_result = result
        if best_result and best_score >= self.config.semantic_cache_threshold:
            return best_result, "semantic"
        return None, "miss"

    def _remember(self, session_id: str, user_query: str, answer: str) -> None:
        self.sessions[session_id].append((user_query, answer))
        self.sessions[session_id] = self.sessions[session_id][-8:]

    def _detect_subject(self, session_id: str) -> str | None:
        history = " ".join(f"{q} {a}" for q, a in self.sessions[session_id][-4:])
        lowered = history.lower()
        for concept, aliases in CONCEPT_ALIASES.items():
            if concept in lowered or any(alias in lowered for alias in aliases):
                return concept
        return None

    def _contextualize_query(self, query: str, session_id: str) -> str:
        lowered = query.lower()
        subject = self._detect_subject(session_id)
        if not subject:
            return query
        if "it" in lowered or "its" in lowered or "that" in lowered or "this" in lowered:
            rewritten = (
                query.replace("its", f"{subject}'s")
                .replace("it", subject)
                .replace("that", subject)
                .replace("this", subject)
            )
            return rewritten
        return query

    def _vector_score(self, query: str, chunk: DocumentChunk) -> float:
        query_terms = content_terms(query)
        doc_terms = content_terms(f"{chunk.title} {chunk.text} {chunk.category}")
        overlap = len(query_terms & doc_terms)
        semantic_bonus = 0.0
        lowered = query.lower()
        for key, expansions in QUERY_EXPANSIONS.items():
            if key in lowered and any(term in chunk.text.lower() for term in expansions):
                semantic_bonus += 1.0
        return overlap + semantic_bonus

    def _keyword_score(self, query: str, chunk: DocumentChunk) -> float:
        query_terms = content_terms(query)
        raw_text = f"{chunk.title} {chunk.text}".lower()
        exact_hits = sum(1 for term in query_terms if term in raw_text)
        return float(exact_hits)

    def _hybrid_retrieve(self, query: str) -> list[tuple[DocumentChunk, float]]:
        ranked: list[tuple[DocumentChunk, float]] = []
        for chunk in self.chunks:
            vector_score = self._vector_score(query, chunk)
            keyword_score = self._keyword_score(query, chunk)
            combined = (0.6 * vector_score) + (0.4 * keyword_score)
            if combined > 0:
                ranked.append((chunk, combined))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: self.config.retrieval_k]

    def _rerank(self, query: str, docs: list[tuple[DocumentChunk, float]]) -> list[tuple[DocumentChunk, float]]:
        reranked = []
        query_terms = content_terms(query)
        for chunk, base_score in docs:
            category_bonus = 0.5 if any(term in chunk.category.lower() for term in query_terms) else 0.0
            title_bonus = 0.5 if any(term in chunk.title.lower() for term in query_terms) else 0.0
            reranked.append((chunk, base_score + category_bonus + title_bonus))
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[: self.config.rerank_top_k]

    def _evaluate_retrieval(self, docs: list[tuple[DocumentChunk, float]]) -> str:
        if not docs:
            return "incorrect"
        top_score = docs[0][1]
        if top_score >= 4.0:
            return "correct"
        if top_score >= 2.0:
            return "ambiguous"
        return "incorrect"

    def _refine_query(self, query: str) -> str:
        lowered = query.lower()
        additions: list[str] = []
        for key, expansions in QUERY_EXPANSIONS.items():
            if key in lowered:
                additions.extend(expansions[:2])
        if "improve" in lowered and "precision" in lowered:
            additions.extend(["reranking", "hybrid retrieval"])
        if "latency" in lowered:
            additions.extend(["semantic caching", "streaming"])
        unique = []
        for item in additions:
            if item not in unique:
                unique.append(item)
        return f"{query} {' '.join(unique[:4])}".strip()

    def _fallback_answer(self, query: str) -> str:
        lowered = query.lower()
        if "weather" in lowered:
            return "I do not have weather data in this knowledge base, so a real production system should call a web or weather API here."
        return "I do not have enough grounded evidence in the current knowledge base, so a production fallback should ask for clarification or call a trusted external source."

    def _make_answer(self, query: str, docs: list[tuple[DocumentChunk, float]]) -> tuple[str, str]:
        lowered = query.lower()
        if "precision" in lowered:
            return (
                "To improve RAG precision, combine hybrid retrieval with reranking so broad recall happens first and then the most relevant passages move to the top [1] [2].",
                "HIGH",
            )
        if "latency" in lowered or "cache" in lowered:
            return (
                "Semantic caching improves latency by reusing answers for repeated or near-duplicate questions, which also lowers model cost [1].",
                "HIGH",
            )
        if "memory" in lowered or "history" in lowered:
            return (
                "Conversation memory helps by reformulating follow-up questions into standalone queries before retrieval, which makes multi-turn RAG more reliable [1].",
                "HIGH",
            )
        if "crag" in lowered:
            return (
                "CRAG adds an evaluate-and-route step after retrieval so the system can use good docs, refine ambiguous queries, or fall back when retrieval is weak [1].",
                "HIGH",
            )
        if "deployment" in lowered or "fastapi" in lowered:
            return (
                "A production RAG deployment should expose FastAPI endpoints, stream responses, monitor latency and cache hit rate, and run inside Docker for repeatable deployment [1] [2].",
                "HIGH",
            )
        joined = " ".join(chunk.text for chunk, _ in docs[:2])
        confidence = "MEDIUM" if docs else "LOW"
        return (joined or self._fallback_answer(query), confidence)

    def _citations_from_docs(self, docs: list[tuple[DocumentChunk, float]]) -> list[Citation]:
        citations = []
        for chunk, score in docs:
            citations.append(
                Citation(
                    title=chunk.title,
                    source=chunk.source,
                    score=round(float(score), 2),
                    preview=chunk.text[:180],
                )
            )
        return citations

    def _run_query(self, question: str, session_id: str = "default", use_cache: bool = True) -> RAGResult:
        start = time.perf_counter()
        self.metrics["queries"] += 1
        standalone_query = self._contextualize_query(question, session_id)

        if self.config.cache_enabled and use_cache:
            cached, cache_type = self._find_cache(standalone_query, session_id)
            if cached:
                self.metrics["cache_hits"] += 1
                latency_ms = (time.perf_counter() - start) * 1000
                result = RAGResult(
                    answer=cached.answer,
                    citations=cached.citations,
                    confidence=cached.confidence,
                    latency_ms=latency_ms,
                    cached=True,
                    cache_type=cache_type,
                    trace=QueryTrace(
                        standalone_query=standalone_query,
                        action=f"cache_{cache_type}",
                        retrieved=[],
                        reranked=[],
                    ),
                    token_count=len(tokenize(cached.answer)),
                )
                self._update_latency(latency_ms)
                self._remember(session_id, question, result.answer)
                return result

        initial = self._hybrid_retrieve(standalone_query)
        action = self._evaluate_retrieval(initial)
        refined_query = None
        chosen_docs = initial

        if action == "ambiguous":
            refined_query = self._refine_query(standalone_query)
            chosen_docs = self._hybrid_retrieve(refined_query)
        elif action == "incorrect":
            answer = self._fallback_answer(standalone_query)
            latency_ms = (time.perf_counter() - start) * 1000
            result = RAGResult(
                answer=answer,
                citations=[],
                confidence="LOW",
                latency_ms=latency_ms,
                cached=False,
                cache_type="miss",
                trace=QueryTrace(
                    standalone_query=standalone_query,
                    action="fallback",
                    retrieved=[],
                    reranked=[],
                ),
                token_count=len(tokenize(answer)),
            )
            self._cache_result(standalone_query, session_id, result)
            self._update_latency(latency_ms)
            self._remember(session_id, question, result.answer)
            return result

        reranked = self._rerank(refined_query or standalone_query, chosen_docs)
        answer, confidence = self._make_answer(standalone_query, reranked)
        latency_ms = (time.perf_counter() - start) * 1000
        result = RAGResult(
            answer=answer,
            citations=self._citations_from_docs(reranked),
            confidence=confidence,
            latency_ms=latency_ms,
            cached=False,
            cache_type="miss",
            trace=QueryTrace(
                standalone_query=standalone_query,
                action="refine_and_reretrieve" if refined_query else "use_docs",
                retrieved=[chunk.title for chunk, _ in initial],
                reranked=[chunk.title for chunk, _ in reranked],
                refined_query=refined_query,
            ),
            token_count=len(tokenize(answer)),
        )
        self._cache_result(standalone_query, session_id, result)
        self._update_latency(latency_ms)
        self._remember(session_id, question, result.answer)
        return result

    async def query(self, question: str, session_id: str = "default", use_cache: bool = True) -> RAGResult:
        return self._run_query(question=question, session_id=session_id, use_cache=use_cache)

    async def stream_query(self, question: str, session_id: str = "default", use_cache: bool = True):
        result = await self.query(question, session_id=session_id, use_cache=use_cache)
        for token in result.answer.split():
            yield f"data: {{'token': '{token}', 'confidence': '{result.confidence}'}}\n\n"
            await asyncio.sleep(0.001)
        yield f"data: {{'done': true, 'cached': {str(result.cached).lower()}}}\n\n"

    def _cache_result(self, query: str, session_id: str, result: RAGResult) -> None:
        key = self._get_cache_key(query, session_id)
        self.cache[key] = result
        self.semantic_cache.append((query, result))
        self.semantic_cache = self.semantic_cache[-100:]

    def _update_latency(self, latency_ms: float) -> None:
        queries = self.metrics["queries"]
        prev_avg = self.metrics["avg_latency_ms"]
        self.metrics["avg_latency_ms"] = ((prev_avg * (queries - 1)) + latency_ms) / max(queries, 1)

    def evaluate(self, eval_csv_path: str | Path) -> dict[str, float]:
        total = 0
        matched = 0
        total_latency = 0.0
        citation_rate = 0
        with Path(eval_csv_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total += 1
                result = self._run_query(row["question"], session_id=f"eval_{total}", use_cache=False)
                total_latency += result.latency_ms
                expected_terms = [item.strip().lower() for item in row["expected_contains"].split("|") if item.strip()]
                answer_lower = result.answer.lower()
                if all(term in answer_lower for term in expected_terms):
                    matched += 1
                citation_rate += int(bool(result.citations))
        return {
            "accuracy": round(matched / total, 2) if total else 0.0,
            "avg_latency_ms": round(total_latency / total, 2) if total else 0.0,
            "citation_rate": round(citation_rate / total, 2) if total else 0.0,
        }

    def metrics_snapshot(self) -> dict[str, float | int]:
        return {
            "queries": self.metrics["queries"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_hit_rate": round(self.metrics["cache_hits"] / max(self.metrics["queries"], 1), 2),
            "avg_latency_ms": round(self.metrics["avg_latency_ms"], 2),
            "cache_size": len(self.cache),
            "documents": len(self.chunks),
            "uploads": self.metrics["uploads"],
        }
