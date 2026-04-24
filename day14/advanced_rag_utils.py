from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
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
    "why",
    "with",
}

SYNONYMS = {
    "weather": ["forecast", "temperature", "rain"],
    "transformers": ["self-attention", "sequence modeling", "nlp"],
    "langchain": ["llm framework", "agents", "chains"],
    "faiss": ["vector search", "similarity search", "index"],
    "graphrag": ["knowledge graph", "entity graph", "relationship retrieval"],
    "raptor": ["hierarchical summary", "multi-level summary", "tree retrieval"],
    "crag": ["corrective rag", "evaluate retrieval", "self-correcting"],
    "self-healing": ["crag", "corrective rag", "fallback search"],
}


class RetrievalQuality(Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class KnowledgeDoc:
    doc_id: str
    title: str
    text: str
    topic: str
    entities: list[str]


@dataclass
class FallbackSnippet:
    topic: str
    text: str


def tokenize(text: str) -> list[str]:
    current = []
    tokens = []
    for char in text.lower():
        if char.isalnum() or char == "-":
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


def load_knowledge_docs(csv_path: str | Path) -> list[KnowledgeDoc]:
    docs: list[KnowledgeDoc] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            docs.append(
                KnowledgeDoc(
                    doc_id=row["doc_id"].strip(),
                    title=row["title"].strip(),
                    text=row["text"].strip(),
                    topic=row["topic"].strip(),
                    entities=[item.strip() for item in row["entities"].split("|") if item.strip()],
                )
            )
    return docs


def load_fallback_snippets(csv_path: str | Path) -> list[FallbackSnippet]:
    snippets: list[FallbackSnippet] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            snippets.append(FallbackSnippet(topic=row["topic"].strip(), text=row["text"].strip()))
    return snippets


def build_graph_index(docs: list[KnowledgeDoc]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        for left in doc.entities:
            for right in doc.entities:
                if left != right:
                    graph[left].add(right)
    return graph


def graph_query(query: str, docs: list[KnowledgeDoc], graph: dict[str, set[str]]) -> dict[str, object]:
    lowered = query.lower()
    mentioned = [entity for entity in graph if entity.lower() in lowered]
    related = []
    if len(mentioned) >= 2:
        left, right = mentioned[:2]
        related = [doc for doc in docs if left in doc.entities and right in doc.entities]
        relation = f"{left} is connected to {right} through {len(related)} supporting document(s)."
    elif mentioned:
        entity = mentioned[0]
        neighbors = sorted(graph.get(entity, set()))
        relation = f"{entity} is directly connected to: {', '.join(neighbors) or 'no neighbors'}."
        related = [doc for doc in docs if entity in doc.entities][:2]
    else:
        relation = "No graph entities matched the query."
    return {"answer": relation, "sources": [doc.title for doc in related]}


def build_raptor_summaries(docs: list[KnowledgeDoc]) -> dict[str, list[dict[str, str]]]:
    by_topic: dict[str, list[KnowledgeDoc]] = defaultdict(list)
    for doc in docs:
        by_topic[doc.topic].append(doc)

    leaves = [{"title": doc.title, "summary": doc.text} for doc in docs]
    topic_level = []
    for topic, topic_docs in by_topic.items():
        summary = " ".join(doc.text for doc in topic_docs)
        topic_level.append({"title": f"{topic} cluster", "summary": summary})

    root_summary = {
        "title": "root summary",
        "summary": " ".join(item["summary"] for item in topic_level),
    }
    return {"leaves": leaves, "topic_level": topic_level, "root": [root_summary]}


def retrieve(query: str, docs: list[KnowledgeDoc], k: int = 3) -> list[tuple[KnowledgeDoc, float]]:
    query_terms = content_terms(query)
    ranked: list[tuple[KnowledgeDoc, float]] = []
    for doc in docs:
        doc_terms = content_terms(f"{doc.title} {doc.text}")
        overlap = len(query_terms & doc_terms)
        entity_bonus = sum(1 for entity in doc.entities if entity.lower() in query.lower())
        score = overlap + entity_bonus
        if score > 0:
            ranked.append((doc, float(score)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def evaluate_retrieval(query: str, doc: KnowledgeDoc) -> RetrievalQuality:
    query_terms = content_terms(query)
    doc_terms = content_terms(f"{doc.title} {doc.text}")
    overlap = len(query_terms & doc_terms)
    entity_match = any(entity.lower() in query.lower() for entity in doc.entities)
    if overlap >= 3:
        return RetrievalQuality.CORRECT
    if overlap >= 2 and entity_match:
        return RetrievalQuality.CORRECT
    if overlap == 2 or entity_match:
        return RetrievalQuality.AMBIGUOUS
    return RetrievalQuality.INCORRECT


def refine_query(query: str) -> str:
    tokens = tokenize(query)
    additions = []
    for token in tokens:
        additions.extend(SYNONYMS.get(token, [])[:2])

    if "transformers" in tokens:
        additions.extend(["self-attention", "NLP"])
    if "self-healing" in tokens:
        additions.extend(["CRAG", "corrective rag", "fallback search"])
    if "rag" in tokens and "hallucination" in tokens:
        additions.extend(["grounding", "retrieval"])
    if "crag" in tokens:
        additions.extend(["evaluate retrieval", "fallback search"])

    unique_additions = []
    for item in additions:
        if item not in unique_additions:
            unique_additions.append(item)
    return f"{query} {' '.join(unique_additions[:4])}".strip()


def fallback_search(query: str, snippets: list[FallbackSnippet]) -> list[FallbackSnippet]:
    query_terms = content_terms(query)
    ranked: list[tuple[FallbackSnippet, float]] = []
    for snippet in snippets:
        snippet_terms = content_terms(f"{snippet.topic} {snippet.text}")
        overlap = len(query_terms & snippet_terms)
        if overlap > 0:
            ranked.append((snippet, float(overlap)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [snippet for snippet, _ in ranked[:2]]


def generate_answer(query: str, context: str) -> str:
    lowered = query.lower()
    if "langchain" in lowered:
        return "LangChain is used to build LLM applications with chains, agents, retrieval, and orchestration patterns."
    if "weather" in lowered:
        return context or "No reliable weather context was available."
    if "transformers" in lowered:
        return "Transformers use self-attention to model relationships across tokens in a sequence."
    if "graphrag" in lowered:
        return "GraphRAG combines knowledge-graph traversal with retrieval so relationship questions can use explicit entity links."
    if "raptor" in lowered:
        return "RAPTOR builds hierarchical summaries so long documents can be retrieved at different abstraction levels."
    return context or "No grounded answer could be generated."


def corrective_rag(query: str, docs: list[KnowledgeDoc], fallback_snippets: list[FallbackSnippet]) -> dict[str, object]:
    results = retrieve(query, docs, k=3)
    evaluations = []
    for doc, score in results:
        quality = evaluate_retrieval(query, doc)
        evaluations.append((doc, score, quality))

    correct_docs = [doc for doc, _, quality in evaluations if quality == RetrievalQuality.CORRECT]
    ambiguous_docs = [doc for doc, _, quality in evaluations if quality == RetrievalQuality.AMBIGUOUS]

    if correct_docs:
        context = " ".join(doc.text for doc in correct_docs)
        answer = generate_answer(query, context)
        return {
            "query": query,
            "action": "use_docs",
            "answer": answer,
            "evaluations": evaluations,
            "context": context,
        }

    if ambiguous_docs:
        refined = refine_query(query)
        refined_results = retrieve(refined, docs, k=2)
        context = " ".join(doc.text for doc, _ in refined_results)
        answer = generate_answer(query, context)
        return {
            "query": query,
            "action": "refine_and_reretrieve",
            "refined_query": refined,
            "answer": answer,
            "evaluations": evaluations,
            "context": context,
            "refined_results": refined_results,
        }

    snippets = fallback_search(query, fallback_snippets)
    context = " ".join(snippet.text for snippet in snippets)
    answer = generate_answer(query, context)
    return {
        "query": query,
        "action": "fallback_search",
        "answer": answer,
        "evaluations": evaluations,
        "context": context,
        "fallback_snippets": snippets,
    }


def benchmark_crag(queries: list[dict[str, str]], docs: list[KnowledgeDoc], snippets: list[FallbackSnippet]) -> dict[str, object]:
    results = []
    corrections = 0
    for item in queries:
        run = corrective_rag(item["query"], docs, snippets)
        corrections += int(run["action"] != "use_docs")
        results.append(
            {
                "query": item["query"],
                "expected_action": item["expected_action"],
                "actual_action": run["action"],
                "matched": item["expected_action"] == run["action"],
                "answer": run["answer"],
            }
        )

    accuracy = sum(1 for item in results if item["matched"]) / len(results) if results else 0.0
    correction_rate = corrections / len(results) if results else 0.0
    return {
        "accuracy": accuracy,
        "correction_rate": correction_rate,
        "results": results,
    }
