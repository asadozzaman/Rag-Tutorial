# Day 5: Retrieval Strategies and Reranking

## Why Day 5 Matters

By Day 4, you already have a working retrieval layer.

Day 5 is about improving retrieval quality, not just retrieval existence.

The big lesson is:

> a better LLM cannot rescue bad retrieval

If the right evidence never reaches the prompt, generation quality will stay weak.

---

## The Retrieval Quality Pyramid

Think of retrieval as layers:

1. base retrievers
2. retriever combination
3. reranking

The common production pattern is:

```text
retrieve broadly -> rerank precisely -> send only the best context to the LLM
```

---

## 1. Ensemble Retrieval

Ensemble retrieval combines more than one retriever.

Most common setup:

- semantic vector search
- BM25 keyword search

Why both matter:

- semantic search catches meaning
- BM25 catches exact words, acronyms, and identifiers

The blend usually beats either one alone.

---

## 2. MMR

MMR means Maximal Marginal Relevance.

Regular top-K retrieval can return near-duplicates:

- chunk A about refunds
- chunk B about refunds
- chunk C about refunds

That wastes context window.

MMR balances:

- relevance to the query
- diversity from already selected results

So instead of returning three copies of the same idea, it tries to cover more useful evidence.

---

## 3. Parent-Child Retrieval

Parent-child retrieval solves a common RAG problem.

Small chunks are good for embedding and retrieval, but they can be too small for generation.

So the pattern becomes:

1. embed small child chunks
2. retrieve relevant children
3. return the larger parent section to the LLM

This gives you:

- precise retrieval
- richer context at generation time

---

## 4. Self-Query Retrieval

Users often ask for both meaning and filters in the same sentence.

Example:

```text
Show advanced Nora articles from 2025 about retrieval.
```

A self-query retriever separates that into:

- semantic query: `about retrieval`
- filters:
  - `author = nora`
  - `year = 2025`
  - `level = advanced`

This is powerful because it lets you combine semantics with structured constraints.

---

## 5. Reranking

Reranking is usually the highest-ROI upgrade in a RAG system.

Typical flow:

1. retrieve top 10 or top 20 candidates quickly
2. rerank them with a slower but smarter model
3. keep only top 3 or top 5

Why it works:

- first-stage retrieval is broad and cheap
- reranking is precise and expensive
- together they give strong quality without searching everything slowly

---

## Bi-Encoder vs Cross-Encoder

### Bi-encoder

This is your embedding retriever.

- query is embedded alone
- document is embedded alone
- similarity is computed afterward

Pros:

- fast
- scalable

Cons:

- less precise on fine distinctions

### Cross-encoder

This scores query-document pairs jointly.

- query and document are processed together
- better understanding of pairwise relevance

Pros:

- high precision

Cons:

- much slower

That is why cross-encoders are normally used only for reranking a small candidate set.

---

## Day 5 Production Pattern

The standard practical setup is:

1. retrieve with BM25 + vector
2. fetch 3x to 5x more documents than needed
3. apply MMR if you want diversity
4. rerank the candidate set
5. send top 3 to the LLM

---

## What You Will Build Today

### Code example

- semantic retriever
- BM25 retriever
- weighted ensemble
- MMR selection
- self-query filter parsing
- reranked final results

### Mini project

A smart technical search system that:

1. loads article chunks
2. retrieves with BM25 + semantic search
3. uses parent-child aggregation
4. reranks top candidates
5. compares before vs after reranking on labeled queries

---

## Day 5 Success Checklist

- [ ] I can explain why ensemble retrieval helps
- [ ] I understand how MMR reduces redundancy
- [ ] I know when to use parent-child retrieval
- [ ] I can parse filters from a natural-language query
- [ ] I understand why reranking improves precision
- [ ] I can measure reranking lift with an A/B comparison
