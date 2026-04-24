"""
Day 6: Production RAG prompt design demo.

This script demonstrates:
  - system prompt construction
  - context formatting
  - structured JSON output
  - conflict handling
  - low-context fallback

Run:
  python 02_code_example.py
"""

from __future__ import annotations

from prompt_utils import (
    ContextDoc,
    build_prompt_package,
    generate_structured_answer,
    retrieve_top_docs,
)


def main() -> None:
    docs = [
        ContextDoc("d1", "Refund Policy", "Refunds are available within 30 days of purchase. Digital products are non-refundable after download.", "policy.md", "billing"),
        ContextDoc("d2", "Billing FAQ", "Customers can request refunds by emailing billing@company.com. Processing takes 5 to 7 business days.", "faq.md", "billing"),
        ContextDoc("d3", "Defective Product Exception", "In cases of defective digital products, full refunds are guaranteed regardless of download status.", "terms.md", "billing"),
        ContextDoc("d4", "Old Subscription FAQ", "All subscription purchases are final sale and cannot be refunded.", "old_faq.md", "subscriptions"),
        ContextDoc("d5", "New Subscription Policy", "Subscriptions may be refunded within 14 days if the service was not heavily used.", "subscription_policy.md", "subscriptions"),
    ]

    questions = [
        "Can I get a refund on a downloaded digital product after 15 days?",
        "Are subscriptions always final sale?",
        "Do you support Apple Pay in Egypt?",
    ]

    print("=" * 78)
    print("DAY 6: PRODUCTION RAG PROMPT DESIGN")
    print("=" * 78)

    for question in questions:
        scored_docs = retrieve_top_docs(question, docs, k=3)
        prompt_package = build_prompt_package(question, scored_docs)
        response = generate_structured_answer(question, scored_docs, threshold=0.18)

        print(f"\nQuestion: {question}")
        print("\nSystem prompt:")
        print(prompt_package["system"])
        print("\nHuman prompt:")
        print(prompt_package["human"])
        print("\nStructured response:")
        print(response.to_json())
        print("-" * 78)


if __name__ == "__main__":
    main()
