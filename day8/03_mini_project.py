from __future__ import annotations

from pathlib import Path

from evaluation_utils import (
    aggregate_metrics,
    evaluate_dataset,
    issue_tags,
    load_examples,
    save_report,
    weakest_queries,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "sample_data" / "eval_examples.csv"
REPORT_PATH = ROOT / "reports" / "day8_quality_report.json"


def main() -> None:
    examples = load_examples(DATA_PATH)
    results = evaluate_dataset(examples)
    summary = aggregate_metrics(results)
    weakest = weakest_queries(results, top_n=4)

    print("=" * 94)
    print("DAY 8 MINI PROJECT: RAG QUALITY DASHBOARD")
    print("=" * 94)
    print(f"Loaded evaluation examples: {len(examples)}")
    print(f"Report output: {REPORT_PATH}")
    print()

    print("Average metrics")
    print("-" * 94)
    print(f"Recall@3:          {summary['recall_at_3']:.2f}")
    print(f"Precision@3:       {summary['precision_at_3']:.2f}")
    print(f"MRR:               {summary['mrr']:.2f}")
    print(f"NDCG@3:            {summary['ndcg_at_3']:.2f}")
    print(f"Faithfulness:      {summary['faithfulness']:.2f}")
    print(f"Answer Relevance:  {summary['answer_relevance']:.2f}")
    print(f"Context Precision: {summary['context_precision']:.2f}")
    print(f"Context Recall:    {summary['context_recall']:.2f}")
    print(f"Hallucination:     {summary['hallucination_risk']:.2f}")
    print(f"Composite Quality: {summary['composite_quality']:.2f}")
    print()

    print("Weakest queries")
    print("-" * 94)
    for item in weakest:
        metrics = item["metrics"]
        print(f"Q{item['query_id']}: {item['question']}")
        print(f"  Composite:      {metrics['composite_quality']:.2f}")
        print(f"  Faithfulness:   {metrics['faithfulness']:.2f}")
        print(f"  Hallucination:  {metrics['hallucination_risk']:.2f}")
        print(f"  Recall@3 / MRR: {metrics['recall_at_3']:.2f} / {metrics['mrr']:.2f}")
        print(f"  Issues:         {', '.join(issue_tags(metrics))}")
        print(f"  Answer:         {item['answer']}")
        print()

    report_payload = {
        "summary": summary,
        "weakest_queries": weakest,
    }
    save_report(REPORT_PATH, report_payload)
    print("Actionable insight:")
    print("Focus first on low-faithfulness and high-hallucination examples before tuning averages.")


if __name__ == "__main__":
    main()
