"""
Prompt and response helpers for Day 6.

This module is intentionally lightweight and deterministic so the prompt
engineering lessons remain easy to run locally:
  - prompt construction
  - structured JSON-style output
  - relevance threshold handling
  - contradiction detection
  - follow-up query composition
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
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
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "with",
}
FOLLOW_UP_PREFIXES = ("and ", "what about", "how about", "does that", "can i", "what if")


@dataclass
class ContextDoc:
    doc_id: str
    title: str
    text: str
    source: str
    topic: str


@dataclass
class RAGResponse:
    answer: str
    confidence: str
    sources: list[str]
    reasoning: str
    used_context: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def load_context_docs(path: Path) -> list[ContextDoc]:
    docs = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            docs.append(
                ContextDoc(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    text=row["text"],
                    source=row["source"],
                    topic=row["topic"],
                )
            )
    return docs


def retrieve_top_docs(query: str, docs: list[ContextDoc], k: int = 3) -> list[tuple[ContextDoc, float]]:
    scored = []
    for doc in docs:
        score = overlap_score(query, f"{doc.title} {doc.text}")
        scored.append((doc, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]


def docs_are_relevant(scored_docs: list[tuple[ContextDoc, float]], threshold: float = 0.18) -> bool:
    if not scored_docs:
        return False
    return scored_docs[0][1] >= threshold


def detect_conflict(query: str, docs: list[ContextDoc]) -> bool:
    combined = " ".join(doc.text.lower() for doc in docs)
    if "subscription" in query.lower():
        return "final sale" in combined and "14 days" in combined
    if "refund" in query.lower():
        return "non-refundable" in combined and "full refunds are guaranteed" in combined
    return False


def compose_query_with_history(user_query: str, history: list[dict]) -> str:
    lowered = user_query.strip().lower()
    if not history or not lowered.startswith(FOLLOW_UP_PREFIXES):
        return user_query

    previous_user = history[-1]["user"]
    return f"{previous_user} Follow-up: {user_query}"


def build_prompt_package(question: str, scored_docs: list[tuple[ContextDoc, float]]) -> dict:
    system_prompt = (
        "You are a precise research assistant. Answer using ONLY the provided context. "
        "If the context is insufficient, say so clearly. Cite sources inline like [Source 1]. "
        "If sources conflict, mention the conflict."
    )
    context_blocks = []
    for index, (doc, score) in enumerate(scored_docs, start=1):
        context_blocks.append(
            f"[Source {index}] {doc.title} | {doc.source} | score={score:.2f}\n{doc.text}"
        )
    human_prompt = f"CONTEXT:\n" + "\n\n".join(context_blocks) + f"\n\nQUESTION: {question}"
    return {"system": system_prompt, "human": human_prompt}


def summarize_reasoning(question: str, scored_docs: list[tuple[ContextDoc, float]], conflict: bool, relevant: bool) -> str:
    if not relevant:
        return "No retrieved document passed the relevance threshold, so the system should avoid overclaiming."

    source_titles = ", ".join(doc.title for doc, _ in scored_docs)
    if conflict:
        return f"Relevant context was found in {source_titles}, but the sources contain conflicting guidance that must be acknowledged."
    return f"Relevant context was found in {source_titles}, so the answer can be grounded in retrieved evidence."


def _extract_relevant_lines(question: str, docs: list[ContextDoc]) -> list[str]:
    question_tokens = set(tokenize(question))
    lines = []
    for index, doc in enumerate(docs, start=1):
        sentences = re.split(r"(?<=[.!?])\s+", doc.text.strip())
        matched = False
        for sentence in sentences:
            if question_tokens & set(tokenize(sentence)):
                lines.append(f"{sentence.strip()} [Source {index}]")
                matched = True
                break
        if not matched and sentences:
            lines.append(f"{sentences[0].strip()} [Source {index}]")
    return lines


def generate_structured_answer(question: str, scored_docs: list[tuple[ContextDoc, float]], threshold: float = 0.18) -> RAGResponse:
    relevant = docs_are_relevant(scored_docs, threshold=threshold)
    docs = [doc for doc, _ in scored_docs]
    conflict = detect_conflict(question, docs)

    if not relevant:
        return RAGResponse(
            answer="I don't have specific information about that in the retrieved context, but I can answer once better evidence is found.",
            confidence="LOW",
            sources=[],
            reasoning=summarize_reasoning(question, scored_docs, conflict=False, relevant=False),
            used_context=False,
        )

    lines = _extract_relevant_lines(question, docs)
    answer_parts = []

    if conflict:
        answer_parts.append("The retrieved sources do not fully agree, so the answer needs caution.")

    answer_parts.extend(lines[:2])
    answer = " ".join(answer_parts)
    confidence = "MEDIUM" if conflict else ("HIGH" if scored_docs[0][1] >= 0.35 else "MEDIUM")

    return RAGResponse(
        answer=answer,
        confidence=confidence,
        sources=[doc.source for doc in docs],
        reasoning=summarize_reasoning(question, scored_docs, conflict=conflict, relevant=True),
        used_context=True,
    )


class ChatMemory:
    def __init__(self):
        self.turns: list[dict] = []

    def add_turn(self, user: str, response: RAGResponse) -> None:
        self.turns.append({"user": user, "answer": response.answer})

    def recent(self, n: int = 3) -> list[dict]:
        return self.turns[-n:]
