# Day 13: Expected Output

## 02_code_example.py - Expected Output

Your exact trace ids and latency values will differ, but the flow should look like this:

```text
======================================================================================================
DAY 13: PRODUCTION RAG API
======================================================================================================
First query
------------------------------------------------------------------------------------------------------
{'answer': 'Caching reduces latency and API costs significantly. ...',
 'sources': ['Caching Guide [kb]'],
 'cached': False,
 'cache_type': 'miss',
 'trace_id': '...'}

Second query (should hit cache)
------------------------------------------------------------------------------------------------------
{'answer': 'Caching reduces latency and API costs significantly. ...',
 'cached': True,
 'cache_type': 'exact',
 ...}

Health
------------------------------------------------------------------------------------------------------
{'status': 'healthy', 'documents': ..., 'cache_size': ..., 'cache_hit_rate': ...}
```

---

## 03_mini_project.py - Expected Output

```text
================================================================================================================
DAY 13 MINI PROJECT: PRODUCTION-READY RAG SERVICE
================================================================================================================
Upload
----------------------------------------------------------------------------------------------------------------
{'message': 'Uploaded ops_notes.txt', 'chunks_added': 2, 'documents': ...}

Exact or fresh query
----------------------------------------------------------------------------------------------------------------
{'answer': 'Circuit breakers ...', 'cached': False, ...}

Near-duplicate semantic cache query
----------------------------------------------------------------------------------------------------------------
{'answer': 'Circuit breakers ...', 'cached': True, 'cache_type': 'semantic', ...}

Streaming response preview
----------------------------------------------------------------------------------------------------------------
data: {'trace_id': '...', 'token': 'Caching'}
data: {'trace_id': '...', 'token': 'reduces'}
...

Health
----------------------------------------------------------------------------------------------------------------
{'status': 'healthy', 'cache_hit_rate': ..., 'avg_latency_ms': ...}
```

---

## Verification Checklist

- [ ] FastAPI app exposes `/query`, `/query/stream`, `/upload`, and `/health`
- [ ] Exact cache works for repeated queries
- [ ] Semantic cache works for near-duplicate queries
- [ ] Rate limiting is enforced per user
- [ ] Health endpoint reports cache and request metrics
- [ ] Dockerfile exists for deployment

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Every request misses cache | Query normalization is too strict | Add semantic similarity fallback |
| Streaming endpoint feels broken | Wrong media type or payload format | Return `text/event-stream` with SSE `data:` lines |
| Service slows down under repeated traffic | No cache or rate limit | Add both before scaling |
| Upload succeeds but retrieval ignores new docs | Index state not updated | Append uploaded chunks into the active service store |
