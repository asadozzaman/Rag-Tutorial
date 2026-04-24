# Day 7: LlamaIndex Deep Dive

## Why Day 7 Matters

Until now, you built the main parts of RAG by hand:

- chunking
- embeddings
- retrieval
- prompting

Day 7 introduces a framework that is built specifically for RAG:

`LlamaIndex`

LlamaIndex is strong because it treats data ingestion, nodes, indices, query engines, and routing as first-class concepts.

---

## Core LlamaIndex Mental Model

The basic flow is:

```text
documents -> nodes -> index -> query engine -> answer
```

More specifically:

1. load documents
2. parse them into nodes
3. build one or more indices
4. create query engines over those indices
5. optionally route or decompose complex questions

---

## Documents vs Nodes

### Document

A document is the original source unit:

- file
- page
- article
- section

### Node

A node is the chunked unit used inside the framework.

Nodes usually contain:

- chunk text
- metadata
- links back to the parent document

Why this matters:

- retrieval works better on smaller nodes
- answers are easier to trace back to original sources

---

## Index Types

### Vector Index

Use this when semantic retrieval matters.

Good for:

- conceptual questions
- broad meaning-based search

### Summary Index

Use this when you want high-level overviews or compact summaries.

Good for:

- long documents
- overview-style questions

### Router Pattern

If you have multiple data domains, do not force every query through one index.

Instead:

1. build separate indices
2. create a router
3. let the system choose the most relevant engine

---

## Query Engines

A query engine is the layer that:

- retrieves nodes
- synthesizes an answer
- returns source nodes

This is one of the most useful abstractions in LlamaIndex, because it cleanly separates retrieval from answer generation behavior.

---

## Response Modes

Different query tasks want different answer synthesis styles.

Examples:

- `compact`: short direct response
- `refine`: iterative answer improvement
- `tree_summarize`: summary over multiple pieces

The retrieval may be the same, but the synthesis style changes the user experience.

---

## Persistence

A production RAG system should not rebuild its index every time it starts.

Persisting the index means:

- save processed nodes and index data to disk
- reload later
- keep startup fast and reproducible

This is one of the most important production habits.

---

## SubQuestion Query Engine

Complex questions often contain multiple hidden questions.

Example:

```text
Compare vector search and keyword search for RAG retrieval.
```

That can be decomposed into:

- what is vector search?
- what is keyword search?
- how do they compare in RAG?

Sub-question decomposition helps the system gather better evidence before synthesizing a final answer.

---

## Router Query Engine

If you have different corpora such as:

- technical docs
- support FAQ
- API reference

then routing matters.

Example:

```text
Query: "How do I reset my password?"
Route -> FAQ index

Query: "What does the /v1/embeddings endpoint return?"
Route -> API reference index
```

Routing improves both relevance and speed because each query is sent to the most useful index.

---

## LangChain vs LlamaIndex

### LangChain

- broad orchestration framework
- agents, tools, chains, workflows
- RAG is one use case

### LlamaIndex

- purpose-built for retrieval and indexing
- stronger data abstractions
- good query engine and router concepts

Many production teams use both:

- LlamaIndex for data and retrieval
- LangChain or another orchestration layer for workflows

---

## What You Will Build Today

### Code example

- local document loader
- node parser
- vector-style index
- persistence and reload
- basic query engine
- sub-question query engine

### Mini project

A multi-index router with:

1. technical documentation index
2. FAQ/support index
3. API reference index
4. automatic routing
5. route accuracy check on diverse queries

---

## Day 7 Success Checklist

- [ ] I understand documents vs nodes
- [ ] I know what a query engine does
- [ ] I can explain vector index vs summary index
- [ ] I know why persistence matters
- [ ] I can decompose complex questions into sub-questions
- [ ] I can route a query to the right index
