"""
Lightweight LlamaIndex-style utilities for Day 7.

This module models the core ideas behind LlamaIndex while staying fully local:
  - document loading
  - node parsing
  - vector-like indexing
  - summary indexing
  - query engines
  - sub-question decomposition
  - routing across multiple indices
  - persistence and reload
"""

from __future__ import annotations

import csv
import json
import math
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
    "the",
    "to",
    "what",
    "with",
}


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict


@dataclass
class Node:
    node_id: str
    doc_id: str
    text: str
    metadata: dict


@dataclass
class Response:
    answer: str
    source_nodes: list[Node]


@dataclass
class ToolMetadata:
    name: str
    description: str


@dataclass
class QueryEngineTool:
    query_engine: "QueryEngine"
    metadata: ToolMetadata


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must share dimensionality.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class HashingEmbedModel:
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            bucket = sum(ord(ch) for ch in token) % self.dimensions
            sign = 1.0 if len(token) % 2 == 0 else -1.0
            vector[bucket] += sign
        return l2_normalize(vector)


class SentenceSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = []
        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if self.chunk_overlap and chunks:
                overlap = current[-self.chunk_overlap :] if current else ""
                current = (overlap + " " + sentence).strip()
            else:
                current = sentence
        if current:
            chunks.append(current)
        return chunks

    def get_nodes_from_documents(self, documents: list[Document]) -> list[Node]:
        nodes = []
        for document in documents:
            for index, chunk in enumerate(self.split_text(document.text), start=1):
                node_id = f"{document.doc_id}_n{index}"
                metadata = {**document.metadata, "title": document.title, "chunk": index}
                nodes.append(Node(node_id=node_id, doc_id=document.doc_id, text=chunk, metadata=metadata))
        return nodes


class SimpleDirectoryReader:
    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)

    def load_data(self) -> list[Document]:
        documents = []
        for path in sorted(self.input_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            stem = path.stem
            parts = stem.split("_", 1)
            category = parts[0]
            title = parts[1].replace("_", " ").title() if len(parts) > 1 else stem.title()
            documents.append(
                Document(
                    doc_id=stem,
                    title=title,
                    text=text,
                    metadata={"source": str(path), "category": category},
                )
            )
        return documents


class StorageContext:
    def __init__(self, persist_dir: str | Path):
        self.persist_dir = Path(persist_dir)

    @classmethod
    def from_defaults(cls, persist_dir: str | Path) -> "StorageContext":
        return cls(persist_dir)


class BaseIndex:
    index_type = "base"

    def __init__(self, documents: list[Document], nodes: list[Node], embed_model: HashingEmbedModel | None = None):
        self.documents = documents
        self.nodes = nodes
        self.embed_model = embed_model or HashingEmbedModel()

    def as_query_engine(self, similarity_top_k: int = 3, response_mode: str = "compact") -> "QueryEngine":
        return QueryEngine(self, similarity_top_k=similarity_top_k, response_mode=response_mode)

    def persist(self, persist_dir: str | Path) -> None:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_type": self.index_type,
            "documents": [asdict(doc) for doc in self.documents],
            "nodes": [asdict(node) for node in self.nodes],
        }
        (persist_path / "index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class VectorStoreIndex(BaseIndex):
    index_type = "vector"

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        node_parser: SentenceSplitter | None = None,
        embed_model: HashingEmbedModel | None = None,
    ) -> "VectorStoreIndex":
        node_parser = node_parser or SentenceSplitter()
        nodes = node_parser.get_nodes_from_documents(documents)
        return cls(documents=documents, nodes=nodes, embed_model=embed_model)

    def retrieve(self, query: str, top_k: int = 3) -> list[Node]:
        query_vector = self.embed_model.embed(query)
        scored = []
        for node in self.nodes:
            score = cosine_similarity(query_vector, self.embed_model.embed(node.text))
            scored.append((score, node))
        return [node for _, node in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]


class SummaryIndex(BaseIndex):
    index_type = "summary"

    @classmethod
    def from_documents(cls, documents: list[Document]) -> "SummaryIndex":
        nodes = []
        for document in documents:
            first_sentence = re.split(r"(?<=[.!?])\s+", document.text.strip())[0]
            nodes.append(
                Node(
                    node_id=f"{document.doc_id}_summary",
                    doc_id=document.doc_id,
                    text=first_sentence,
                    metadata={**document.metadata, "title": document.title, "summary": True},
                )
            )
        return cls(documents=documents, nodes=nodes)

    def retrieve(self, query: str, top_k: int = 3) -> list[Node]:
        query_tokens = set(tokenize(query))
        scored = []
        for node in self.nodes:
            score = len(query_tokens & set(tokenize(node.text + " " + node.metadata.get("title", ""))))
            scored.append((score, node))
        return [node for _, node in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]


def load_index_from_storage(storage_context: StorageContext) -> BaseIndex:
    payload = json.loads((storage_context.persist_dir / "index.json").read_text(encoding="utf-8"))
    documents = [Document(**item) for item in payload["documents"]]
    nodes = [Node(**item) for item in payload["nodes"]]
    if payload["index_type"] == "vector":
        return VectorStoreIndex(documents=documents, nodes=nodes)
    if payload["index_type"] == "summary":
        return SummaryIndex(documents=documents, nodes=nodes)
    raise ValueError(f"Unsupported index type: {payload['index_type']}")


class QueryEngine:
    def __init__(self, index: BaseIndex, similarity_top_k: int = 3, response_mode: str = "compact"):
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.response_mode = response_mode

    def _synthesize(self, nodes: list[Node]) -> str:
        if not nodes:
            return "No relevant information found."
        if self.response_mode == "compact":
            return " ".join(f"{node.text} [{node.metadata.get('title', node.doc_id)}]" for node in nodes[: self.similarity_top_k])
        if self.response_mode == "refine":
            answer = nodes[0].text
            for node in nodes[1 : self.similarity_top_k]:
                answer += f" Then refine with: {node.text}"
            return answer
        if self.response_mode == "tree_summarize":
            bullets = "\n".join(f"- {node.text}" for node in nodes[: self.similarity_top_k])
            return f"Summary:\n{bullets}"
        return " ".join(node.text for node in nodes[: self.similarity_top_k])

    def query(self, question: str) -> Response:
        nodes = self.index.retrieve(question, top_k=self.similarity_top_k)
        return Response(answer=self._synthesize(nodes), source_nodes=nodes)


class SubQuestionQueryEngine:
    def __init__(self, query_engine_tools: list[QueryEngineTool]):
        self.query_engine_tools = query_engine_tools

    @classmethod
    def from_defaults(cls, query_engine_tools: list[QueryEngineTool]) -> "SubQuestionQueryEngine":
        return cls(query_engine_tools=query_engine_tools)

    def _decompose(self, query: str) -> list[str]:
        lowered = query.lower()
        if "compare" in lowered and " and " in lowered:
            parts = query.split("Compare", 1)[-1].strip()
            concepts = [item.strip(" ?") for item in parts.split(" and ") if item.strip()]
            if len(concepts) >= 2:
                return [f"What is {concepts[0]}?", f"What is {concepts[1]}?", query]
        return [query]

    def query(self, query: str) -> Response:
        subquestions = self._decompose(query)
        collected_nodes = []
        parts = []
        for subquestion in subquestions:
            tool = self.query_engine_tools[0]
            response = tool.query_engine.query(subquestion)
            collected_nodes.extend(response.source_nodes)
            parts.append(f"Sub-question: {subquestion}\n{response.answer}")
        return Response(answer="\n\n".join(parts), source_nodes=collected_nodes[:6])


class RouterQueryEngine:
    def __init__(self, query_engine_tools: list[QueryEngineTool]):
        self.query_engine_tools = query_engine_tools

    def _keyword_boost(self, query: str, tool: QueryEngineTool) -> int:
        lowered = query.lower()
        name = tool.metadata.name
        boosts = {
            "technical_docs": [
                "rag",
                "node",
                "nodes",
                "vector search",
                "keyword search",
                "query engine",
                "semantic",
                "architecture",
                "retrieval",
            ],
            "faq_support": [
                "password",
                "refund",
                "invoice",
                "billing",
                "support",
                "account",
                "email",
                "purchase",
                "defective",
            ],
            "api_reference": [
                "/v1/",
                "endpoint",
                "api",
                "authorization",
                "json",
                "request body",
                "streaming",
                "chat/completions",
                "embeddings",
                "header",
            ],
        }
        return sum(2 for phrase in boosts.get(name, []) if phrase in lowered)

    def _score_tool(self, query: str, tool: QueryEngineTool) -> int:
        query_tokens = set(tokenize(query))
        tool_tokens = set(tokenize(tool.metadata.name + " " + tool.metadata.description))
        return len(query_tokens & tool_tokens) + self._keyword_boost(query, tool)

    def route(self, query: str) -> QueryEngineTool:
        return max(self.query_engine_tools, key=lambda tool: self._score_tool(query, tool))

    def query(self, query: str) -> tuple[QueryEngineTool, Response]:
        tool = self.route(query)
        return tool, tool.query_engine.query(query)


def load_router_queries(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows
