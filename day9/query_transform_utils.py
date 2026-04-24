from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "why",
    "with",
}

SYNONYMS = {
    "slow": ["latency", "performance", "response time"],
    "app": ["application", "system", "service"],
    "vector": ["embedding", "semantic"],
    "search": ["retrieval", "lookup"],
    "password": ["login", "account access"],
    "reset": ["recover", "change"],
    "api": ["endpoint", "sdk", "integration"],
    "error": ["bug", "failure", "issue"],
    "memory": ["ram", "resource"],
    "recall": ["coverage", "missed results"],
    "clustered": ["ivf", "inverted file"],
    "graph": ["hnsw", "graph index"],
    "ram": ["memory", "resource"],
    "hypothetical": ["hyde", "synthetic answer"],
    "vague": ["low recall", "weak query"],
    "ranker": ["reranking", "rerank"],
    "evidence": ["source", "citation", "context"],
}

ROUTE_KEYWORDS = {
    "TECHNICAL": {"api", "endpoint", "sdk", "faiss", "hnsw", "ivf", "code", "index", "parameter"},
    "CONCEPTUAL": {"difference", "concept", "theory", "architecture", "rerank", "chunk", "hyde", "step"},
    "TROUBLESHOOTING": {"slow", "error", "bug", "fail", "memory", "latency", "wrong", "issue", "broken", "missing", "citation", "source"},
    "ACCOUNT": {"password", "billing", "refund", "workspace", "plan", "account", "pricing", "sso"},
}

COMPLEX_MARKERS = {
    "compare",
    "best",
    "tradeoff",
    "trade-off",
    "versus",
    "vs",
    "combine",
    "design",
    "which index",
    "what should i tune",
    "how can",
}
AMBIGUOUS_PATTERNS = [
    re.compile(r"^(help|fix|what now)\b", re.IGNORECASE),
    re.compile(r"\b(it|this|that)\b", re.IGNORECASE),
]


@dataclass
class Document:
    doc_id: str
    route: str
    title: str
    text: str


@dataclass
class QueryCase:
    query: str
    expected_mode: str
    expected_route: str
    relevant_ids: list[str]


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
    raw_tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [token for token in (normalize_token(item) for item in raw_tokens) if token]


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def load_corpus(csv_path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            docs.append(
                Document(
                    doc_id=row["doc_id"].strip(),
                    route=row["route"].strip(),
                    title=row["title"].strip(),
                    text=row["text"].strip(),
                )
            )
    return docs


def load_benchmark(csv_path: str | Path) -> list[QueryCase]:
    cases: list[QueryCase] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cases.append(
                QueryCase(
                    query=row["query"].strip(),
                    expected_mode=row["expected_mode"].strip(),
                    expected_route=row["expected_route"].strip(),
                    relevant_ids=[item.strip() for item in row["relevant_ids"].split("|") if item.strip()],
                )
            )
    return cases


def route_query(query: str) -> str:
    query_lower = query.lower()
    tokens = content_tokens(query)
    if any(term in query_lower for term in ("password", "billing", "refund", "workspace", "plan", "pricing", "sso")):
        return "ACCOUNT"
    if any(term in query_lower for term in ("api", "endpoint", "sdk", "faiss", "hnsw", "ivf", "index", "clustered", "graph-based", "graph", "ram")):
        return "TECHNICAL"
    if any(term in query_lower for term in ("slow", "error", "bug", "memory", "latency", "broken", "missing", "citation", "source", "wrong")):
        return "TROUBLESHOOTING"
    if any(phrase in query_lower for phrase in ("what is", "explain", "difference", "why does", "how can hyde")):
        return "CONCEPTUAL"
    scores = {
        route: len(tokens & keywords)
        for route, keywords in ROUTE_KEYWORDS.items()
    }
    if max(scores.values(), default=0) == 0:
        return "CONCEPTUAL"
    return max(scores, key=scores.get)


def classify_query(query: str) -> str:
    query_lower = query.lower().strip()
    tokens = content_tokens(query)
    if len(tokens) < 3:
        return "ambiguous"
    if "broken" in tokens or query_lower.startswith("help"):
        return "ambiguous"
    if any(pattern.search(query_lower) for pattern in AMBIGUOUS_PATTERNS) and "?" not in query_lower:
        return "ambiguous"
    if any(marker in query_lower for marker in COMPLEX_MARKERS):
        return "complex"
    if " and " in query_lower and route_query(query) == "TROUBLESHOOTING":
        return "complex"
    if query_lower.count(" and ") >= 1 or query_lower.count(" vs ") >= 1:
        return "complex"
    return "simple"


def ask_clarifying_question(query: str, route: str) -> str:
    prompts = {
        "TECHNICAL": "Are you asking about an API endpoint, index configuration, or SDK behavior?",
        "CONCEPTUAL": "Do you want a definition, a comparison, or an architecture explanation?",
        "TROUBLESHOOTING": "What symptom are you seeing: slow latency, memory errors, or low-quality results?",
        "ACCOUNT": "Is this about login access, billing, refunds, or workspace settings?",
    }
    return f"I need one more detail before retrieving. {prompts[route]}"


def step_back_query(query: str) -> str:
    route = route_query(query)
    if route == "TROUBLESHOOTING":
        return "What general factors cause retrieval and application performance issues?"
    if route == "TECHNICAL":
        return "What are the core technical components and configuration choices involved here?"
    if route == "ACCOUNT":
        return "What account, billing, or workspace policy applies to this request?"
    return "What higher-level concept or design principle explains this query?"


def multi_query_variations(query: str) -> list[str]:
    tokens = tokenize(query)
    variants = {query}
    expanded_terms: list[str] = []
    for token in tokens:
        expanded_terms.extend(SYNONYMS.get(token, []))

    if expanded_terms:
        variants.add(f"{query} {' '.join(expanded_terms[:3])}")
    if "slow" in tokens or "latency" in tokens:
        variants.add("application performance latency debugging retrieval tuning")
    if "password" in tokens:
        variants.add("account recovery forgot password reset email")
    if "api" in tokens:
        variants.add("endpoint sdk response parameter integration reference")
    if "clustered" in tokens or "graph" in tokens:
        variants.add("ivf hnsw vector index memory latency recall tradeoffs")
    if "ram" in tokens:
        variants.add("memory pressure graph index flat index resource limits")
    if "hypothetical" in tokens or "vague" in tokens:
        variants.add("hyde hypothetical document low recall weak query rewriting")
    if "ranker" in tokens:
        variants.add("reranking second stage ranking first pass retrieval relevance")
    if "evidence" in tokens:
        variants.add("wrong citations missing sources context retrieval debugging")
    if "compare" in tokens or "vs" in tokens:
        variants.add(query.replace("vs", "versus"))
        variants.add(f"{query} tradeoffs benefits drawbacks")
    return [item for item in variants if item]


def hyde_document(query: str, route: str) -> str:
    if route == "TROUBLESHOOTING":
        return (
            "Technical troubleshooting note: slow performance usually comes from high latency, "
            "inefficient indexing, missing filters, or memory-heavy search structures."
        )
    if route == "TECHNICAL":
        return (
            "Implementation note: the answer should mention configuration details, endpoints, "
            "SDK behavior, indexing settings, and exact system parameters."
        )
    if route == "ACCOUNT":
        return (
            "Support policy note: the answer should explain account access, refunds, billing, "
            "workspace ownership, or plan limitations."
        )
    return (
        "Concept note: the answer should explain architecture, design tradeoffs, retrieval strategy, "
        "and when to use each approach."
    )


def score_document(query: str, doc: Document) -> float:
    query_terms = content_tokens(query)
    doc_terms = content_tokens(f"{doc.title} {doc.text}")
    overlap = len(query_terms & doc_terms)
    route_bonus = 1.5 if route_query(query) == doc.route else 0.0
    phrase_bonus = 1.0 if any(term in doc.text.lower() for term in query_terms) else 0.0
    return overlap + route_bonus + phrase_bonus


def retrieve(query: str, docs: list[Document], k: int = 3) -> list[tuple[Document, float]]:
    ranked = sorted(
        ((doc, score_document(query, doc)) for doc in docs),
        key=lambda item: item[1],
        reverse=True,
    )
    filtered = [item for item in ranked if item[1] > 0]
    return filtered[:k]


def transformed_retrieve(query: str, docs: list[Document], k: int = 3) -> tuple[list[tuple[Document, float]], dict[str, object]]:
    route = route_query(query)
    variants = multi_query_variations(query)
    step_back = step_back_query(query)
    hyde = hyde_document(query, route)

    aggregate: dict[str, float] = {}
    doc_lookup = {doc.doc_id: doc for doc in docs}
    direct_results = retrieve(query, docs, k=k)
    for rank, (doc, score) in enumerate(direct_results, start=1):
        aggregate[doc.doc_id] = aggregate.get(doc.doc_id, 0.0) + (score * 3.0) + (2.0 / rank)

    search_queries = variants + [step_back, hyde]
    for rank_query in search_queries:
        results = retrieve(rank_query, docs, k=k)
        for rank, (doc, score) in enumerate(results, start=1):
            route_bonus = 0.75 if doc.route == route else 0.0
            aggregate[doc.doc_id] = aggregate.get(doc.doc_id, 0.0) + score + (1.0 / rank) + route_bonus

    ranked = sorted(
        ((doc_lookup[doc_id], score) for doc_id, score in aggregate.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:k], {
        "route": route,
        "multi_query": variants,
        "step_back": step_back,
        "hyde": hyde,
    }


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant) / len(relevant) if relevant else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def benchmark_pipeline(cases: list[QueryCase], docs: list[Document], k: int = 3) -> dict[str, object]:
    plain_scores: list[dict[str, float]] = []
    transformed_scores: list[dict[str, float]] = []
    route_correct = 0
    mode_correct = 0
    details: list[dict[str, object]] = []

    for case in cases:
        mode = classify_query(case.query)
        route = route_query(case.query)
        mode_correct += int(mode == case.expected_mode)
        route_correct += int(route == case.expected_route)

        plain = retrieve(case.query, docs, k=k)
        plain_ids = [doc.doc_id for doc, _ in plain]
        plain_metric = {
            "recall_at_3": recall_at_k(plain_ids, case.relevant_ids, k=3),
            "mrr": mrr(plain_ids, case.relevant_ids),
        }
        if mode == "complex":
            transformed, transforms = transformed_retrieve(case.query, docs, k=k)
        else:
            transformed = plain
            transforms = {
                "route": route,
                "multi_query": [case.query],
                "step_back": step_back_query(case.query),
                "hyde": hyde_document(case.query, route),
            }
        transformed_ids = [doc.doc_id for doc, _ in transformed]
        transformed_metric = {
            "recall_at_3": recall_at_k(transformed_ids, case.relevant_ids, k=3),
            "mrr": mrr(transformed_ids, case.relevant_ids),
        }
        if case.expected_mode == "complex":
            plain_scores.append(plain_metric)
            transformed_scores.append(transformed_metric)
        details.append(
            {
                "query": case.query,
                "mode": mode,
                "route": route,
                "plain_ids": plain_ids,
                "transformed_ids": transformed_ids,
                "plain": plain_metric,
                "transformed": transformed_metric,
                "transforms": transforms,
            }
        )

    return {
        "plain": {
            "recall_at_3": mean(item["recall_at_3"] for item in plain_scores),
            "mrr": mean(item["mrr"] for item in plain_scores),
        },
        "transformed": {
            "recall_at_3": mean(item["recall_at_3"] for item in transformed_scores),
            "mrr": mean(item["mrr"] for item in transformed_scores),
        },
        "route_accuracy": route_correct / len(cases),
        "mode_accuracy": mode_correct / len(cases),
        "details": details,
    }
