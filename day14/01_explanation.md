# Day 14: Advanced Patterns - GraphRAG, RAPTOR & CRAG

## Why Day 14 Matters

Basic RAG works well for many tasks, but some question types still break it:

- relationship questions
- very long documents
- weak or failed retrieval

Day 14 introduces three advanced patterns for those gaps:

- `GraphRAG`
- `RAPTOR`
- `CRAG`

---

## GraphRAG

GraphRAG adds a knowledge graph on top of documents.

The flow is:

```text
documents
   ->
entity extraction
   ->
graph edges
   ->
graph traversal + retrieval
```

Why it helps:

- relationship questions become easier
- multi-entity reasoning becomes more explicit
- global summaries can use graph structure, not only chunk similarity

Example:

- "How are LangChain and FAISS related?"

Plain chunk retrieval may find nearby text.

GraphRAG can explicitly use the entity connection.

---

## RAPTOR

RAPTOR means hierarchical retrieval over summaries.

The idea is:

```text
leaf chunks
   ->
cluster
   ->
summary
   ->
cluster those summaries
   ->
higher-level summary
```

Why it helps:

- long documents can be retrieved at different zoom levels
- broad questions can use summary nodes
- detailed questions can still use leaf chunks

This is very useful for long reports, books, and research papers.

---

## CRAG

CRAG means `Corrective RAG`.

It adds a decision step after retrieval:

```text
query
   ->
retrieve
   ->
evaluate retrieval quality
   ->
correct / ambiguous / incorrect
```

Then the system reacts:

- `correct` -> use the docs as they are
- `ambiguous` -> refine the query and retrieve again
- `incorrect` -> fall back to another source

This is one of the most practical advanced patterns because it helps the system heal itself when retrieval is weak.

---

## When To Use Each Pattern

### Use GraphRAG for:

- relationship queries
- entity-heavy domains
- organization, legal, biomedical, or research graphs

### Use RAPTOR for:

- very long documents
- questions that need multiple abstraction levels
- synthesis across large sections

### Use CRAG for:

- high-stakes queries
- unreliable or incomplete retrieval
- systems that need safer failure handling

---

## What You Should Learn Today

By the end of Day 14, you should be able to:

- explain when GraphRAG is better than plain retrieval
- understand hierarchical summary retrieval in RAPTOR
- build a self-correcting CRAG pipeline with branching logic

That is the core of advanced RAG design.
