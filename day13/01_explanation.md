# Day 13: Production Deployment & Optimization

## Why Day 13 Matters

A notebook demo is not the same as a production system.

Production RAG has to handle:

- real traffic
- latency constraints
- failures
- cost control
- monitoring

Day 13 is about turning a RAG pipeline into a service.

---

## What Changes in Production

In a notebook, we mostly care about:

- correctness
- experimentation

In production, we also care about:

- API design
- retries and fallbacks
- caching
- streaming
- observability
- rate limits

That changes the architecture completely.

---

## Core Production Stack

Day 13 introduces a service shape like this:

```text
client
  ->
FastAPI gateway
  ->
cache check
  ->
retrieval
  ->
generation
  ->
response + logs + metrics
```

This is the practical foundation of a deployed RAG system.

---

## Caching

Caching is one of the easiest latency wins.

### Exact Cache

Use a normalized hash of the query.

Good for:

- repeated questions
- identical dashboard or support queries

### Semantic Cache

Use similarity between the new query and previous cached queries.

Good for:

- near-duplicate wording
- small phrasing changes

Example:

- "How does caching reduce latency?"
- "Why does caching lower response time?"

Those should often reuse the same answer.

---

## Streaming

Users perceive streaming as faster because they see tokens arriving before the whole answer is finished.

Even if total generation time is unchanged, perceived latency improves.

That is why streaming endpoints matter in production APIs.

---

## Rate Limiting

Without rate limiting, one noisy client can hurt everyone else.

A good production service should:

- cap requests per user
- fail clearly
- protect retrieval and generation resources

Day 13 uses a simple per-minute request window for that.

---

## Observability

You cannot debug production RAG without traces.

At minimum, log:

- query
- latency
- cache hit or miss
- token count
- trace id

That gives you enough visibility to answer questions like:

- why was this request slow?
- are users getting cache benefits?
- which queries are expensive?

---

## What You Should Learn Today

By the end of Day 13, you should be able to:

- expose a RAG pipeline through FastAPI
- add exact and semantic-style caching
- stream responses
- enforce rate limits
- collect health and request metrics

That is the baseline for production-ready RAG deployment.
