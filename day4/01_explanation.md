# Day 4: Vector Databases and Indexing

## Why Day 4 Matters

Day 3 taught you how embeddings represent meaning.

Day 4 answers the next question:

> once I have vectors, where do I store them and how do I search them efficiently?

This is the job of the vector database or vector index.

---

## What a Vector Database Does

A vector database stores embeddings and helps you retrieve the nearest vectors to a query.

At minimum, it needs to support:

- storing vectors
- searching by similarity
- attaching metadata
- filtering results
- persisting the index

Many real systems also add:

- approximate nearest neighbor search
- compression
- hybrid keyword plus semantic retrieval
- distributed storage

---

## Common Vector Database Options

### FAISS

- library, not a full database
- excellent for local experiments and custom indexing
- very fast
- limited database-style features unless you build them around it

Best for:

- research
- local prototypes
- custom retrieval pipelines

---

### Chroma

- friendly developer experience
- strong local prototyping choice
- supports metadata and persistence

Best for:

- learning
- fast prototypes
- local semantic search apps

---

### Qdrant

- production-ready vector database
- strong metadata filtering
- hybrid search support
- good balance between ease and scale

Best for:

- production RAG with filtering
- hybrid retrieval systems

---

### Pinecone

- managed cloud vector database
- easy to operate at scale
- less infrastructure burden

Best for:

- teams that want managed infrastructure
- cloud-first production setups

---

### pgvector

- vector search inside PostgreSQL
- good when your team already lives in SQL
- simpler operationally for some products

Best for:

- apps already built around Postgres
- smaller or moderate-scale production workloads

---

## Index Types

### Flat Index

Flat search compares the query against every vector.

Pros:

- exact
- simple
- best recall

Cons:

- search cost grows linearly with the number of vectors

Good for:

- small datasets
- debugging
- gold-standard comparison

---

### IVF

IVF means Inverted File Index.

Idea:

1. cluster vectors into lists
2. at query time, find the closest clusters
3. search only inside those clusters

Pros:

- much faster than flat search on larger datasets

Cons:

- approximate, so recall can drop
- needs tuning

Key knob:

- `n_probe`: how many clusters to search

More probes means better recall, but slower search.

---

### HNSW

HNSW is a graph-based approximate nearest neighbor index.

Pros:

- great speed/recall trade-off
- widely used in production

Cons:

- more complex than flat search
- tuning matters

Good default production idea:

- if you need approximate search and your system supports HNSW, start there

---

### PQ

Product Quantization compresses vectors.

Pros:

- lower memory usage

Cons:

- lower accuracy

Best for:

- very large collections where memory matters

---

## Metadata Filtering

Vector similarity alone is often not enough.

Example:

```text
Query: "summarize the 2024 finance report"
```

You may want results filtered to:

- `category = finance`
- `year = 2024`

Metadata filtering is important because it:

- reduces search space
- improves precision
- enables product features like author, team, date, or document type filters

---

## Hybrid Search

The best retrieval systems usually combine:

- semantic vector search
- keyword search such as BM25
- metadata filters

Why:

- vector search captures meaning
- BM25 captures exact terms
- filters enforce hard constraints

Then you merge the rankings.

One simple method is Reciprocal Rank Fusion:

```text
score = sum(1 / (k + rank_i))
```

This rewards documents that rank well across more than one retriever.

---

## What You Will Build Today

### Code example

- compare flat vector search and clustered approximate search
- add metadata filtering
- inspect how approximate search can trade recall for speed

### Mini project

Build a hybrid search engine with:

1. semantic vector search
2. BM25 keyword search
3. metadata filters
4. reciprocal rank fusion
5. semantic-only, keyword-only, and hybrid modes

---

## Production Notes

- Start with exact or simple indexing while your dataset is small.
- Introduce approximate search only when latency or scale demands it.
- Persist indices to disk so you do not rebuild on every restart.
- Keep metadata clean and normalized.
- Measure recall before and after index changes.

---

## Day 4 Success Checklist

- [ ] I can explain flat vs approximate vector search
- [ ] I understand what IVF and HNSW are for
- [ ] I can attach and filter metadata
- [ ] I can combine semantic and keyword retrieval
- [ ] I know how RRF merges rankings
- [ ] I can persist and reload a vector index
