# Day 9: Expected Output

## 02_code_example.py - Expected Output

Your exact ranking scores may vary slightly, but the flow should look like this:

```text
==========================================================================================
DAY 9: QUERY TRANSFORMATION TECHNIQUES
==========================================================================================
Query: Why is my vector search slow?
Route: TROUBLESHOOTING

Multi-query variations
------------------------------------------------------------------------------------------
- Why is my vector search slow?
- application performance latency debugging retrieval tuning
- ...

Step-back query
------------------------------------------------------------------------------------------
What general factors cause retrieval and application performance issues?

Plain retrieval
------------------------------------------------------------------------------------------
- troubleshoot_latency | Slow Retrieval Debug Guide | score=...
- tech_ivf | IVF Index Tuning | score=...

HyDE hypothesis
------------------------------------------------------------------------------------------
Technical troubleshooting note: slow performance usually comes from ...

Transformed retrieval
------------------------------------------------------------------------------------------
- troubleshoot_latency | Slow Retrieval Debug Guide | score=...
- troubleshoot_memory | Memory Pressure in Search | score=...
- tech_ivf | IVF Index Tuning | score=...
```

---

## 03_mini_project.py - Expected Output

```text
======================================================================================================
DAY 9 MINI PROJECT: INTELLIGENT QUERY PIPELINE
======================================================================================================
Loaded corpus documents: ...
Loaded benchmark queries: 20

Pipeline behavior examples
------------------------------------------------------------------------------------------------------
Query: How do I reset my password?
  Mode:  simple
  Route: ACCOUNT
  Action: retrieve account_password, ...

Query: Help, it is broken
  Mode:  ambiguous
  Route: TROUBLESHOOTING
  Action: I need one more detail before retrieving. ...

Benchmark summary
------------------------------------------------------------------------------------------------------
Mode accuracy:          ...
Route accuracy:         ...
Plain Recall@3:         ...
Plain MRR:              ...
Transformed Recall@3:   ...
Transformed MRR:        ...

Queries with biggest gain
------------------------------------------------------------------------------------------------------
Query: Compare IVF and HNSW for speed and memory tradeoffs
  Plain IDs:    ...
  Better IDs:   ...
  MRR gain:     ...
```

---

## Verification Checklist

- [ ] HyDE, multi-query, and step-back transformations are implemented
- [ ] Router supports 3 or more categories
- [ ] Ambiguous queries trigger a clarifying question
- [ ] Benchmark compares plain retrieval vs transformed retrieval
- [ ] At least some benchmark queries improve under transformed retrieval

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Router sends too many queries to one class | Keywords are too generic | Tighten route keywords per domain |
| No gain from transformations | Corpus wording already matches queries | Add harder benchmark queries with synonym mismatch |
| Ambiguous queries still retrieve documents | Classifier is too permissive | Add stronger ambiguity rules |
| HyDE text feels artificial | That is okay for learning | Treat it as a retrieval aid, not a final answer |
