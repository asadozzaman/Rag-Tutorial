from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
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
    "our",
    "the",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "with",
    "within",
    "your",
}


@dataclass
class EvalExample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    retrieved_ids: list[str]
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


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def overlap_ratio(left: set[str], right: set[str]) -> float:
    return safe_divide(len(left & right), len(left))


def f1_overlap(left: set[str], right: set[str]) -> float:
    precision = overlap_ratio(left, right)
    recall = overlap_ratio(right, left)
    if not precision or not recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    retrieved_top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return safe_divide(len(retrieved_top_k & relevant), len(relevant))


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    retrieved_top_k = retrieved_ids[:k]
    if not retrieved_top_k:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_top_k if doc_id in relevant)
    return safe_divide(hits, len(retrieved_top_k))


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return safe_divide(dcg, idcg)


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    answer_terms = content_tokens(answer)
    context_terms = content_tokens(" ".join(contexts))
    return overlap_ratio(answer_terms, context_terms)


def answer_relevance_score(question: str, answer: str, ground_truth: str) -> float:
    question_terms = content_tokens(question)
    answer_terms = content_tokens(answer)
    truth_terms = content_tokens(ground_truth)

    question_alignment = overlap_ratio(question_terms, answer_terms)
    truth_alignment = f1_overlap(answer_terms, truth_terms)
    return (question_alignment + truth_alignment) / 2


def context_precision_score(answer: str, contexts: list[str]) -> float:
    answer_terms = content_tokens(answer)
    context_terms = content_tokens(" ".join(contexts))
    return overlap_ratio(context_terms, answer_terms)


def context_recall_score(ground_truth: str, contexts: list[str]) -> float:
    truth_terms = content_tokens(ground_truth)
    context_terms = content_tokens(" ".join(contexts))
    return overlap_ratio(truth_terms, context_terms)


def hallucination_risk(answer: str, contexts: list[str]) -> float:
    return 1.0 - faithfulness_score(answer, contexts)


def composite_quality(metrics: dict[str, float]) -> float:
    positive = [
        metrics["recall_at_3"],
        metrics["precision_at_3"],
        metrics["mrr"],
        metrics["ndcg_at_3"],
        metrics["faithfulness"],
        metrics["answer_relevance"],
        metrics["context_precision"],
        metrics["context_recall"],
    ]
    score = mean(positive)
    score *= 1.0 - (metrics["hallucination_risk"] * 0.35)
    return max(0.0, min(score, 1.0))


def evaluate_example(example: EvalExample) -> dict[str, float]:
    metrics = {
        "recall_at_3": recall_at_k(example.retrieved_ids, example.relevant_ids, k=3),
        "precision_at_3": precision_at_k(example.retrieved_ids, example.relevant_ids, k=3),
        "mrr": mrr(example.retrieved_ids, example.relevant_ids),
        "ndcg_at_3": ndcg_at_k(example.retrieved_ids, example.relevant_ids, k=3),
        "faithfulness": faithfulness_score(example.answer, example.contexts),
        "answer_relevance": answer_relevance_score(
            example.question,
            example.answer,
            example.ground_truth,
        ),
        "context_precision": context_precision_score(example.answer, example.contexts),
        "context_recall": context_recall_score(example.ground_truth, example.contexts),
    }
    metrics["hallucination_risk"] = hallucination_risk(example.answer, example.contexts)
    metrics["composite_quality"] = composite_quality(metrics)
    return metrics


def load_examples(csv_path: str | Path) -> list[EvalExample]:
    rows: list[EvalExample] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                EvalExample(
                    question=row["question"].strip(),
                    answer=row["answer"].strip(),
                    contexts=[part.strip() for part in row["contexts"].split("||") if part.strip()],
                    ground_truth=row["ground_truth"].strip(),
                    retrieved_ids=[part.strip() for part in row["retrieved_ids"].split("|") if part.strip()],
                    relevant_ids=[part.strip() for part in row["relevant_ids"].split("|") if part.strip()],
                )
            )
    return rows


def evaluate_dataset(examples: list[EvalExample]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, example in enumerate(examples, start=1):
        metrics = evaluate_example(example)
        results.append(
            {
                "query_id": index,
                "question": example.question,
                "answer": example.answer,
                "ground_truth": example.ground_truth,
                "contexts": example.contexts,
                "retrieved_ids": example.retrieved_ids,
                "relevant_ids": example.relevant_ids,
                "metrics": metrics,
            }
        )
    return results


def aggregate_metrics(results: list[dict[str, object]]) -> dict[str, float]:
    metric_names = list(results[0]["metrics"].keys()) if results else []
    return {
        metric_name: mean(float(result["metrics"][metric_name]) for result in results)
        for metric_name in metric_names
    }


def weakest_queries(results: list[dict[str, object]], top_n: int = 3) -> list[dict[str, object]]:
    ordered = sorted(results, key=lambda item: float(item["metrics"]["composite_quality"]))
    return ordered[:top_n]


def issue_tags(metrics: dict[str, float]) -> list[str]:
    tags: list[str] = []
    if metrics["recall_at_3"] < 0.67:
        tags.append("retrieval recall weak")
    if metrics["mrr"] < 0.5:
        tags.append("relevant docs ranked too low")
    if metrics["faithfulness"] < 0.75:
        tags.append("answer not grounded enough")
    if metrics["hallucination_risk"] > 0.25:
        tags.append("hallucination risk high")
    if metrics["answer_relevance"] < 0.65:
        tags.append("answer relevance weak")
    if metrics["context_recall"] < 0.70:
        tags.append("missing supporting context")
    return tags or ["healthy"]


def save_report(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
