# Day 15: Expected Output

## 02_code_example.py - Expected Output

Your exact latencies and cache values will differ, but the flow should look like this:

```text
==============================================================================================================
DAY 15: CAPSTONE RAG CORE MODULE
==============================================================================================================
Ingested chunks: ...

First query
--------------------------------------------------------------------------------------------------------------
Answer: To improve RAG precision, combine hybrid retrieval with reranking ...
Confidence: HIGH
Latency: ...ms
Cached: False (miss)
Citations: 3

Second query
--------------------------------------------------------------------------------------------------------------
Answer: To improve RAG precision, combine hybrid retrieval with reranking ...
Cached: True (exact)

Metrics
--------------------------------------------------------------------------------------------------------------
{'queries': 2, 'cache_hits': 1, ...}

Evaluation
--------------------------------------------------------------------------------------------------------------
{'accuracy': ..., 'avg_latency_ms': ..., 'citation_rate': ...}
```

---

## 03_mini_project.py - Expected Output

```text
====================================================================================================================
DAY 15 MINI PROJECT: ENTERPRISE DOCUMENT INTELLIGENCE PLATFORM
====================================================================================================================
Upload
--------------------------------------------------------------------------------------------------------------------
{'message': 'Uploaded ops_notes.md', 'chunks_added': 2, ...}

Conversation query 1
--------------------------------------------------------------------------------------------------------------------
{'answer': 'Semantic caching improves latency ...', 'confidence': 'HIGH', ...}

Conversation query 2 (history-aware)
--------------------------------------------------------------------------------------------------------------------
{'answer': 'Semantic caching improves latency ...', 'trace': {'standalone_query': 'How does semantic caching help latency?', ...}}

Repeated precision query (cache demo)
--------------------------------------------------------------------------------------------------------------------
{'answer': 'To improve RAG precision ...', 'cached': True, 'cache_type': 'exact', ...}

Streaming preview
--------------------------------------------------------------------------------------------------------------------
data: {'token': 'CRAG', 'confidence': 'HIGH'}
...

Evaluation
--------------------------------------------------------------------------------------------------------------------
{'accuracy': ..., 'avg_latency_ms': ..., 'citation_rate': ...}
```

---

## Verification Checklist

- [ ] Ingestion uses smart chunking with metadata
- [ ] Retrieval combines hybrid retrieval and reranking
- [ ] History-aware query rewriting works in a session
- [ ] CRAG-style correction runs when retrieval is weak
- [ ] API exposes query, stream, upload, health, metrics, and evaluation endpoints
- [ ] README, architecture doc, Dockerfile, and learning journal exist

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Follow-up questions ignore history | Session memory is not tracked | Rewrite queries using recent session context |
| Precision answers are vague | Reranking is too weak | Increase rerank weight or improve chunk quality |
| Fallback happens too often | Retrieval threshold is too strict | Tune correct vs ambiguous vs incorrect thresholds |
| Metrics look empty | Evaluation or endpoint calls were never made | Run the code example and mini project once |
