# Day 14: Expected Output

## 02_code_example.py - Expected Output

Your exact wording may vary slightly, but the flow should look like this:

```text
========================================================================================================
DAY 14: ADVANCED RAG PATTERNS
========================================================================================================
GraphRAG snapshot
--------------------------------------------------------------------------------------------------------
Graph nodes: ...
LangChain neighbors: FAISS, RAG

RAPTOR snapshot
--------------------------------------------------------------------------------------------------------
Leaf nodes: ...
Topic summaries: ...
Root summary title: root summary

CRAG examples
--------------------------------------------------------------------------------------------------------
Query: What is LangChain used for?
  [  correct] score=...
  Action: use correct docs
  Answer: LangChain is used to build LLM applications ...

Query: What is the weather today?
  Action: fallback search
  Answer: Today's weather ...
```

---

## 03_mini_project.py - Expected Output

```text
================================================================================================================
DAY 14 MINI PROJECT: SELF-CORRECTING RAG SYSTEM
================================================================================================================
GraphRAG demo
----------------------------------------------------------------------------------------------------------------
Answer: LangChain is connected to FAISS through ...
Sources: LangChain + FAISS Integration

CRAG benchmark
----------------------------------------------------------------------------------------------------------------
Routing accuracy: ...
Correction rate: ...

Per-query actions
----------------------------------------------------------------------------------------------------------------
Query: How do transformers work in NLP?
  Expected: refine_and_reretrieve
  Actual:   refine_and_reretrieve
  Matched:  True
  Answer:   Transformers use self-attention ...
```

---

## Verification Checklist

- [ ] Graph index is built from document entities
- [ ] RAPTOR-style summaries exist at multiple levels
- [ ] CRAG evaluates retrieval and chooses `use_docs`, `refine_and_reretrieve`, or `fallback_search`
- [ ] Benchmark tracks correction rate across test queries
- [ ] Relationship query demo shows GraphRAG-style behavior

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Everything becomes ambiguous | Evaluation thresholds are too weak | Tighten overlap rules for correct vs ambiguous |
| Fallback never triggers | Retriever is too broad | Lower retrieval confidence or improve evaluation gating |
| Graph answers feel shallow | Entity links are sparse | Add richer entity metadata to the graph index |
| RAPTOR seems redundant | Docs are too short | Use a longer corpus where multi-level summaries matter |
