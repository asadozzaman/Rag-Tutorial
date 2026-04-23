# Day 4: Expected Output

## 02_code_example.py - Expected Output

Your exact similarity scores depend on whether Gemini embeddings are available.

```text
========================================================================
DAY 4: VECTOR INDEX COMPARISON
========================================================================

Embedding model: hashing-semantic-baseline
Index build embedding latency: 0.6 ms
Query: best typed language for browser web app development

Flat exact search
-----------------
  0.31 | d5 | {'lang': 'typescript', 'type': 'web'} | TypeScript adds types and tooling support...
  0.29 | d2 | {'lang': 'javascript', 'type': 'web'} | JavaScript powers interactive web applications...
  0.10 | d1 | {'lang': 'python', 'type': 'general'} | Python is a versatile programming language...

Clustered approximate search (1 probe)
--------------------------------------
  0.29 | d5 | {'lang': 'typescript', 'type': 'web'} | TypeScript adds types...
  0.10 | d1 | {'lang': 'python', 'type': 'general'} | Python is a versatile...

Metadata filtered search (systems only)
---------------------------------------
  0.14 | d4 | {'lang': 'go', 'type': 'systems'} | Go is often used for concurrent...
  0.12 | d3 | {'lang': 'rust', 'type': 'systems'} | Rust provides memory safety...

Reloaded persistent index
-------------------------
  0.35 | d6 | {'lang': 'sql', 'type': 'database'} | PostgreSQL with pgvector lets teams...

Takeaway:
  Flat search is exact. Clustered search is faster in spirit but can miss some results.
  Metadata filtering narrows the search space before ranking.
```

---

## 03_mini_project.py - Expected Output

```text
====================================================================================
DAY 4 MINI PROJECT: HYBRID SEARCH ENGINE
====================================================================================
Documents loaded: 10
Semantic model: hashing-semantic-baseline
Embedding build latency: 0.8 ms

Reloaded saved index
--------------------
  0.42 | doc07 | {'category': 'infra', 'author': 'nora', 'year': '2025', 'level': 'advanced'} ...

SEMANTIC | query='how do i scale vector search in production'
-------------------------------------------------------------
  ...

KEYWORD | query='bm25 keyword ranking'
--------------------------------------
  ...

HYBRID | query='metadata filters for finance reports' | filters={'category': 'finance'}
----------------------------------------------------------------------------------------
  ...

HYBRID | query='web retrieval for api docs' | filters={'category': 'docs', 'level': 'intermediate'}
-----------------------------------------------------------------------------------------------
  ...

Modes summary:
  semantic = vector similarity only
  keyword  = BM25 only
  hybrid   = clustered semantic + BM25 merged with RRF

Interactive mode. Modes: semantic, keyword, hybrid. Type 'quit' to exit.
```

---

## Verification Checklist

- [ ] Exact flat search returns relevant semantic matches
- [ ] Clustered approximate search returns similar but not always identical results
- [ ] Metadata filters change the result set correctly
- [ ] Saved index reloads from disk
- [ ] BM25 keyword search works
- [ ] Hybrid mode merges keyword and semantic rankings

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `No GEMINI_API_KEY found` | Gemini embeddings not configured | Add key to `.env`, or use hashing fallback |
| Approximate search misses a relevant doc | Low probe count | Increase `n_probe` |
| Hybrid results look odd | One retriever is noisy | Inspect semantic and BM25 results separately before fusion |
| Filters return no results | Filter key/value mismatch | Check exact metadata values in the CSV |
| Reload fails | Cache file missing | Run the script once to create `.search_cache` |
