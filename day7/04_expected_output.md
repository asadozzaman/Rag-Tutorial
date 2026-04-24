# Day 7: Expected Output

## 02_code_example.py - Expected Output

Your exact text may vary slightly because this package uses a local LlamaIndex-style simulation, but the flow should look like this:

```text
==============================================================================
DAY 7: LLAMAINDEX-STYLE RAG PIPELINE
==============================================================================

Loaded documents: 2
Persisted index: ...\storage\day7_basic

Basic query: What are the key concepts in RAG?
...
Sources:
  Rag Fundamentals | chunk 1
  Retrieval Strategy Notes | chunk 1

Summary engine output:
Summary:
- ...
- ...

Sub-question engine output:
Sub-question: What is vector search?
...
Sub-question: What is keyword search?
...
Sub-question: Compare vector search and keyword search for RAG retrieval
...
```

---

## 03_mini_project.py - Expected Output

```text
======================================================================================
DAY 7 MINI PROJECT: MULTI-INDEX RAG ROUTER
======================================================================================
Tools available: technical_docs, faq_support, api_reference

Query: How do I reset my password?
Expected route: faq_support
Chosen route:   faq_support
Route correct:  True
Answer: ...
Sources:
  Account Recovery | ...

Query: What does the /v1/embeddings endpoint return?
Expected route: api_reference
Chosen route:   api_reference
Route correct:  True
...

Routing accuracy: 100.00% (15/15)
```

---

## Verification Checklist

- [ ] Documents are loaded and split into nodes
- [ ] Vector-style index persists and reloads
- [ ] Query engine returns source nodes
- [ ] Sub-question engine decomposes complex queries
- [ ] Router sends different queries to different indices
- [ ] Route accuracy is printed across diverse test queries

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Router keeps choosing one tool | Tool descriptions are too vague | Make tool metadata more distinct |
| Wrong route for FAQ/API queries | Category wording overlaps | Add clearer keywords to descriptions |
| Query answers look too short | `similarity_top_k` is too low | Increase it to 3 or 4 |
| Reload fails | Storage folder missing | Run the code example once to create `storage/` |
