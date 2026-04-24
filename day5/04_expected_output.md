# Day 5: Expected Output

## 02_code_example.py - Expected Output

Your exact scores depend on whether Gemini embeddings are available.

```text
==============================================================================
DAY 5: ADVANCED RETRIEVAL AND RERANKING
==============================================================================

Embedding model: hashing-semantic-baseline
Embedding build latency: 0.7 ms
Raw query: advanced Nora MMR retrieval article from 2024
Parsed semantic query: advanced Nora MMR retrieval article from 2024
Parsed filters: {'year': '2024', 'level': 'advanced', 'author': 'nora'}

Semantic retriever
------------------
  ...

BM25 retriever
--------------
  ...

Ensemble results
----------------
  ...

MMR diverse results
-------------------
  ...

Parent-child aggregated parents
-------------------------------
  ...

Reranked final results
----------------------
  ...

Takeaway:
  Retrieve broadly with more than one retriever, diversify if needed, then rerank down.
```

---

## 03_mini_project.py - Expected Output

```text
====================================================================================
DAY 5 MINI PROJECT: SMART DOCUMENT SEARCH WITH RERANKING
====================================================================================
Chunks loaded: 14
Evaluation queries: 6
Embedding model: hashing-semantic-baseline
Embedding build latency: 1.1 ms

Semantic retrieval
------------------
  ...

Keyword retrieval
-----------------
  ...

Ensemble retrieval
------------------
  ...

Reranked child results
----------------------
  ...

Parent results
--------------
  ...

A/B evaluation on labeled queries:
  q1: How do I blend keyword and semantic retrieval?
    expected=p1 before_rank=2 after_rank=1
  ...

A/B summary:
  Baseline semantic Hit@3: 1.00
  Ensemble+rerank Hit@3:   1.00
  Baseline semantic MRR:   0.92
  Ensemble+rerank MRR:     1.00
  Reranking improved ranking quality on this dataset.

Interactive mode. Type 'quit' to exit.
```

---

## Verification Checklist

- [ ] Semantic and BM25 results both appear
- [ ] Ensemble output merges both retrievers
- [ ] MMR returns diverse results instead of near-duplicates
- [ ] Self-query parsing extracts useful filters
- [ ] Parent-child aggregation returns parent sections
- [ ] A/B metrics print before vs after reranking

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `No GEMINI_API_KEY found` | Gemini is not configured | Use hashing fallback or add the key |
| No results after self-query parsing | Filters are too strict | Remove one filter and retry |
| Reranking does not help | Candidates are already poor | Increase first-stage retrieval depth |
| Duplicate parents dominate results | Child chunks are too similar | Use MMR before reranking |
| Low Hit@3 | Retrieval recall is weak | Fix retrieval before blaming generation |
