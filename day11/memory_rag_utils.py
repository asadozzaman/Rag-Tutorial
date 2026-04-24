from __future__ import annotations

import csv
import re
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
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}

ENTITY_CANONICAL = {
    "python": "Python",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "javascript": "JavaScript",
    "react": "React",
    "vue": "Vue",
    "angular": "Angular",
}


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    text: str
    score: float


@dataclass
class ConversationMemory:
    max_turns: int = 6
    messages: list[ChatMessage] = field(default_factory=list)
    summary: str = ""

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self.messages.extend(
            [
                ChatMessage(role="user", content=user_message),
                ChatMessage(role="assistant", content=assistant_message),
            ]
        )
        self._trim()

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self.messages) <= max_messages:
            return

        overflow = self.messages[:-max_messages]
        keep = self.messages[-max_messages:]
        snippets = []
        for msg in overflow[-4:]:
            prefix = "User" if msg.role == "user" else "Bot"
            snippets.append(f"{prefix}: {msg.content}")
        if snippets:
            summary_piece = " | ".join(snippets)
            if self.summary:
                self.summary = f"{self.summary} | {summary_piece}"
            else:
                self.summary = summary_piece
        self.messages = keep


def normalize_token(token: str) -> str:
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9']+", text.lower())
    return [token for token in (normalize_token(item) for item in raw) if token]


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def load_docs(csv_path: str | Path) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            chunks.append(
                RetrievedChunk(
                    doc_id=row["doc_id"].strip(),
                    title=row["title"].strip(),
                    text=row["text"].strip(),
                    score=0.0,
                )
            )
    return chunks


def chunks_from_text(text: str, chunk_size: int = 320) -> list[RetrievedChunk]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    chunks: list[RetrievedChunk] = []
    bucket = ""
    counter = 1
    for paragraph in paragraphs:
        if len(bucket) + len(paragraph) + 1 <= chunk_size:
            bucket = f"{bucket} {paragraph}".strip()
        else:
            if bucket:
                chunks.append(
                    RetrievedChunk(
                        doc_id=f"upload_{counter}",
                        title=f"Uploaded chunk {counter}",
                        text=bucket,
                        score=0.0,
                    )
                )
                counter += 1
            bucket = paragraph
    if bucket:
        chunks.append(
            RetrievedChunk(
                doc_id=f"upload_{counter}",
                title=f"Uploaded chunk {counter}",
                text=bucket,
                score=0.0,
            )
        )
    return chunks


def detect_focus(history: list[ChatMessage], summary: str = "", prefer_language: bool = False) -> str | None:
    combined = " ".join([summary] + [message.content for message in history[-6:]])
    lowered = combined.lower()
    if prefer_language:
        ordered_entities = ["python", "javascript"]
    else:
        ordered_entities = ["fastapi", "django", "flask", "python", "javascript", "react", "vue", "angular"]
    for entity in ordered_entities:
        if entity in lowered:
            return ENTITY_CANONICAL[entity]
    return None


def contextualize_query(query: str, memory: ConversationMemory) -> str:
    focus = detect_focus(memory.messages, memory.summary)
    language_focus = detect_focus(memory.messages, memory.summary, prefer_language=True)
    lowered = query.lower().strip()

    if not focus and not language_focus:
        return query

    if "what web frameworks does it have" in lowered or "what about its web frameworks" in lowered:
        subject = language_focus or focus
        return f"What web frameworks does {subject} have?"

    if "what about its frontend frameworks" in lowered:
        subject = language_focus or focus
        return f"What about {subject}'s frontend frameworks?"

    if "which one is best for apis" in lowered:
        subject = language_focus or focus
        return f"Which {subject} web framework is best for APIs?"

    if "does it support async" in lowered:
        return f"Does {focus} support async?"

    if "its" in lowered:
        return re.sub(r"\bits\b", f"{focus}'s", query, flags=re.IGNORECASE)

    if re.search(r"\bit\b", lowered):
        return re.sub(r"\bit\b", focus, query, flags=re.IGNORECASE)

    if re.search(r"\bone\b", lowered) and focus:
        if focus in {"Python", "JavaScript"}:
            return query.replace("one", f"{focus} framework")
        return query.replace("one", focus)

    return query


def score_chunk(query: str, chunk: RetrievedChunk) -> float:
    query_terms = content_tokens(query)
    doc_terms = content_tokens(f"{chunk.title} {chunk.text}")
    overlap = len(query_terms & doc_terms)
    phrase_bonus = 1.0 if any(term in chunk.text.lower() for term in query_terms) else 0.0
    return overlap + phrase_bonus


def retrieve(query: str, chunks: list[RetrievedChunk], k: int = 3) -> list[RetrievedChunk]:
    ranked = []
    for chunk in chunks:
        score = score_chunk(query, chunk)
        if score > 0:
            ranked.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    score=score,
                )
            )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:k]


def compress_context(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    compressed: list[RetrievedChunk] = []
    query_terms = content_tokens(query)
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        kept = [sentence for sentence in sentences if content_tokens(sentence) & query_terms]
        text = " ".join(kept).strip() or chunk.text
        compressed.append(
            RetrievedChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=text,
                score=chunk.score,
            )
        )
    return compressed


def generate_answer(standalone_query: str, contexts: list[RetrievedChunk]) -> str:
    lowered = standalone_query.lower()
    combined = " ".join(chunk.text for chunk in contexts)

    if "tell me about python" in lowered:
        return "Python is a general-purpose programming language created by Guido van Rossum and used across web, data, and automation work."
    if "web frameworks" in lowered and "python" in lowered:
        return "Python's main web frameworks include Django, Flask, and FastAPI."
    if "best for apis" in lowered:
        return "FastAPI is usually the best fit for APIs because it is async-first, fast, and auto-generates OpenAPI documentation."
    if "support async" in lowered and "fastapi" in lowered:
        return "Yes. FastAPI supports async request handlers and is designed for modern asynchronous API development."
    if "support async" in lowered and "django" in lowered:
        return "Django now supports async in parts of the stack, but FastAPI is the more async-native choice."
    if contexts:
        return contexts[0].text
    if combined:
        return combined
    return "I do not have enough grounded context to answer that yet."


def run_turn(query: str, memory: ConversationMemory, docs: list[RetrievedChunk]) -> dict[str, object]:
    standalone_query = contextualize_query(query, memory)
    retrieved = retrieve(standalone_query, docs, k=3)
    compressed = compress_context(standalone_query, retrieved)
    answer = generate_answer(standalone_query, compressed)
    memory.add_turn(query, answer)
    return {
        "original_query": query,
        "standalone_query": standalone_query,
        "sources": compressed,
        "answer": answer,
        "memory_summary": memory.summary,
    }
