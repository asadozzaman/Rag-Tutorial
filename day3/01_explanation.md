# Day 3: Embeddings Deep Dive

## Where Day 3 Fits

Day 1 taught the full RAG pipeline.

Day 2 focused on loading documents and chunking them well.

Day 3 focuses on the next critical step:

`text chunk -> embedding vector`

An embedding is a list of numbers that represents meaning. RAG uses those numbers to search by semantic similarity instead of only keyword overlap.

---

## What Is an Embedding?

An embedding model converts text into a vector.

```text
"How do I return a product?" -> [0.12, -0.04, 0.88, ...]
"What is your refund policy?" -> [0.10, -0.02, 0.84, ...]
"Best pizza in Dhaka" -> [-0.42, 0.31, 0.05, ...]
```

The first two should be close together because their meaning is similar. The pizza sentence should be far away.

That is the core idea:

> similar meaning should produce nearby vectors

---

## Why Embeddings Matter in RAG

During ingestion:

1. chunk the document
2. embed every chunk
3. store vectors in FAISS or another vector database

During query time:

1. embed the user query
2. compare the query vector to stored chunk vectors
3. retrieve the closest chunks
4. pass those chunks to the LLM

If the embedding model is weak for your domain, retrieval will return the wrong evidence.

---

## Embedding Quality Dimensions

### 1. Semantic accuracy

Does the model understand meaning, or does it mostly match exact words?

Example:

```text
Query: "How do I get my money back?"
Good match: "Refunds are available within 30 days."
```

A keyword-only system may miss this because "money back" and "refund" are different words.

---

### 2. Dimensionality

Dimensionality means how many numbers are in each vector.

Examples:

- 256 dimensions: smaller, cheaper, faster
- 768 dimensions: common for many open-source embedding models
- 1536+ dimensions: often richer, but bigger to store

Storage estimate:

```text
storage GB = documents * dimensions * 4 bytes / 1,000,000,000
```

One million 768-dimensional float32 vectors need about 3.07 GB before index overhead.

---

### 3. Latency

Embedding happens during ingestion and query time.

Ingestion latency matters for large document collections.

Query latency matters for user experience.

If embedding the query is slow, every chat response feels slow.

---

### 4. Domain fit

General models may struggle with specialized language.

Examples:

- legal clauses
- medical notes
- source code
- finance reports
- scientific papers

That is why you benchmark on your own data instead of trusting leaderboard scores blindly.

---

## Similarity Search

After embedding, you compare vectors.

The most common measure is cosine similarity.

```text
cosine similarity = how close two vectors point in the same direction
```

Scores usually range from:

- close to `1.0`: very similar
- around `0.0`: unrelated
- below `0.0`: opposite direction, usually not useful for retrieval

---

## Normalization

Normalization scales a vector so its length becomes 1.

Why it helps:

- cosine similarity becomes more stable
- vector size matters less than vector direction
- many vector stores expect normalized embeddings for cosine-style search

For RAG, normalize unless your chosen model or vector database says otherwise.

---

## Benchmarking Embedding Models

Do not pick an embedding model only from a blog post.

Build a small evaluation set:

```text
query -> relevant document ids
```

Example:

```text
Query: "How do I reduce hallucination?"
Relevant docs: ["rag-grounding", "retrieval-context"]
```

Then measure:

- `Recall@K`: did the correct document appear in the top K?
- `MRR`: how high was the first correct result?
- `Latency`: how long embedding took
- `Dimensions`: how large each vector is

---

## Metrics You Need Today

### Recall@K

Recall@K asks:

> Did at least one relevant document appear in the top K results?

If `Recall@3 = 1.0`, every test query found a relevant document within the top 3.

---

### MRR

MRR means Mean Reciprocal Rank.

If the first relevant result is:

- rank 1 -> score `1.0`
- rank 2 -> score `0.5`
- rank 3 -> score `0.333`
- not found -> score `0.0`

MRR rewards models that put relevant results near the top.

---

## Day 3 Recommended Workflow

1. Start with a simple similarity comparison.
2. Compare at least one real embedding model with simple local baselines.
3. Create a labeled query-document dataset.
4. Calculate Recall@K and MRR.
5. Choose the model based on your data, not generic claims.

---

## Production Notes

- Cache embeddings aggressively.
- Batch embeddings during ingestion.
- Keep the embedding model name in your metadata.
- If you switch embedding models, rebuild the whole index.
- Do not mix embeddings from different models in the same vector index.
- Measure retrieval quality before changing the LLM.

---

## Day 3 Success Checklist

- [ ] I can explain what embeddings are
- [ ] I can calculate cosine similarity
- [ ] I understand why normalization matters
- [ ] I can compare embedding models on the same dataset
- [ ] I can compute Recall@K and MRR
- [ ] I know why changing embedding models requires re-indexing
