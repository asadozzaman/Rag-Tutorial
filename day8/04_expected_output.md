# Day 8: Expected Output

## 02_code_example.py - Expected Output

Your exact numbers may vary slightly depending on token overlap, but the flow should look like this:

```text
======================================================================================
DAY 8: RAG EVALUATION METRICS
======================================================================================
This example measures retrieval quality, answer quality, and hallucination risk.

======================================================================================
Question: What is the refund policy?
...
Recall@3:          1.00
Precision@3:       0.67
MRR:               1.00
NDCG@3:            1.00
Faithfulness:      ...
Answer Relevance:  ...
Context Precision: ...
Context Recall:    ...
Hallucination:     ...
Composite Quality: ...

======================================================================================
Question: Does the app support offline mode?
...
Faithfulness:      lower
Hallucination:     higher
Composite Quality: lower
```

---

## 03_mini_project.py - Expected Output

```text
==============================================================================================
DAY 8 MINI PROJECT: RAG QUALITY DASHBOARD
==============================================================================================
Loaded evaluation examples: 10
Report output: ...\day8_quality_report.json

Average metrics
----------------------------------------------------------------------------------------------
Recall@3:          ...
Precision@3:       ...
MRR:               ...
NDCG@3:            ...
Faithfulness:      ...
Answer Relevance:  ...
Context Precision: ...
Context Recall:    ...
Hallucination:     ...
Composite Quality: ...

Weakest queries
----------------------------------------------------------------------------------------------
Q4: Does the app support offline mode?
  Composite:      ...
  Faithfulness:   ...
  Hallucination:  ...
  Recall@3 / MRR: ...
  Issues:         answer not grounded enough, hallucination risk high
  Answer:         ...

Actionable insight:
Focus first on low-faithfulness and high-hallucination examples before tuning averages.
```

---

## Verification Checklist

- [ ] Retrieval metrics are computed from retrieved and relevant document IDs
- [ ] Generation metrics are computed from answer, context, and ground truth
- [ ] Hallucination risk is estimated with a custom function
- [ ] Weakest queries are surfaced clearly
- [ ] A JSON quality report is saved in `reports/`

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Scores look too high for everything | Sample answers are too perfect | Add harder, imperfect examples |
| Hallucination score feels strict | Token overlap is a heuristic | Treat it as a teaching metric, not a final judge |
| Recall is high but answers are weak | Retriever works but prompting/generation is weak | Improve prompt instructions or answer synthesis |
| Faithfulness is low on correct answers | Wording differs too much from context | Add better normalization or a semantic judge later |
