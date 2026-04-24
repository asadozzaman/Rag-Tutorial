from __future__ import annotations

from evaluation_utils import EvalExample, evaluate_example


def print_metrics(example: EvalExample) -> None:
    metrics = evaluate_example(example)
    print("=" * 86)
    print(f"Question: {example.question}")
    print(f"Answer:   {example.answer}")
    print("Retrieved IDs:", ", ".join(example.retrieved_ids))
    print("Relevant IDs: ", ", ".join(example.relevant_ids))
    print("-" * 86)
    print(f"Recall@3:          {metrics['recall_at_3']:.2f}")
    print(f"Precision@3:       {metrics['precision_at_3']:.2f}")
    print(f"MRR:               {metrics['mrr']:.2f}")
    print(f"NDCG@3:            {metrics['ndcg_at_3']:.2f}")
    print(f"Faithfulness:      {metrics['faithfulness']:.2f}")
    print(f"Answer Relevance:  {metrics['answer_relevance']:.2f}")
    print(f"Context Precision: {metrics['context_precision']:.2f}")
    print(f"Context Recall:    {metrics['context_recall']:.2f}")
    print(f"Hallucination:     {metrics['hallucination_risk']:.2f}")
    print(f"Composite Quality: {metrics['composite_quality']:.2f}")


def main() -> None:
    examples = [
        EvalExample(
            question="What is the refund policy?",
            answer="Refunds are available within 30 days, but digital products are non-refundable.",
            contexts=[
                "Refunds are processed within 30 days from purchase.",
                "Digital items are non-refundable unless required by law.",
            ],
            ground_truth="Customers can request refunds within 30 days, except digital products.",
            retrieved_ids=["refund_policy", "billing_terms", "faq_misc"],
            relevant_ids=["refund_policy", "billing_terms"],
        ),
        EvalExample(
            question="What programming languages does the API support?",
            answer="The API supports Python, JavaScript, Go, and Ruby SDKs.",
            contexts=[
                "Our API offers official SDKs in Python, JavaScript, Go, Ruby, and Java.",
                "Community SDKs exist for other languages.",
            ],
            ground_truth="Python, JavaScript, Go, Ruby, and Java are supported.",
            retrieved_ids=["api_sdks", "api_rate_limits", "faq_misc"],
            relevant_ids=["api_sdks"],
        ),
        EvalExample(
            question="Does the app support offline mode?",
            answer="Yes, the app supports full offline mode for all users.",
            contexts=[
                "The mobile app caches the last synced dashboard for 24 hours.",
                "Full offline editing is not currently supported.",
            ],
            ground_truth="The app does not support full offline mode; only limited cached viewing exists.",
            retrieved_ids=["mobile_cache", "roadmap_note", "product_overview"],
            relevant_ids=["mobile_cache", "roadmap_note"],
        ),
    ]

    print("=" * 86)
    print("DAY 8: RAG EVALUATION METRICS")
    print("=" * 86)
    print("This example measures retrieval quality, answer quality, and hallucination risk.")

    for example in examples:
        print_metrics(example)


if __name__ == "__main__":
    main()
