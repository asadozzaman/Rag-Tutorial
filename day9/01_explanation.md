# Day 9: Advanced RAG - Query Transformation & Routing

## Why Day 9 Matters

A lot of RAG failures start before retrieval even begins.

The user asks:

- vague questions
- overloaded questions
- multi-part questions
- questions using words that do not match your document wording

If you only send the raw user query into retrieval, quality can drop fast.

Day 9 is about fixing the query before retrieval.

---

## The Three Main Transformation Ideas

### 1. HyDE

`HyDE` means `Hypothetical Document Embeddings`.

Instead of embedding the user question directly, you first generate a hypothetical answer-like paragraph, then retrieve with that richer text.

Why it helps:

- answer-shaped text is often closer to real documents than the raw question
- short vague questions become more descriptive

Example:

```text
Query: Why is my vector search slow?

HyDE-style expansion:
"Slow vector search is often caused by flat indices, high memory pressure,
poor filtering, inefficient ANN settings, or expensive reranking."
```

That expanded text overlaps better with technical docs.

---

### 2. Multi-Query

A single query wording can miss good documents.

So instead of one search, generate several variations:

- original wording
- synonym version
- troubleshooting version
- conceptual version

Then merge the results.

Why it helps:

- improves recall
- handles different wording styles
- is usually the safest transformation to add first

---

### 3. Step-Back Prompting

Sometimes the user asks a very narrow question, but the answer requires broader context.

So you create a more general version of the question first.

Example:

```text
Specific: Why is my FAISS search slow on large datasets?
Step-back: What factors affect vector retrieval performance at scale?
```

Now retrieval can pick up broader architecture docs that the narrow query missed.

---

## Query Routing

Not every question should go through the same retrieval path.

A router decides which path or knowledge source to use.

Typical categories:

- `TECHNICAL`
- `CONCEPTUAL`
- `TROUBLESHOOTING`
- `ACCOUNT`

Routing matters because:

- technical queries want implementation docs
- conceptual queries want architecture or explanation docs
- troubleshooting queries want fix-oriented docs
- account queries want policy/support docs

If routing is wrong, even good retrieval can fail.

---

## Query Modes

For this day, it helps to classify the query itself:

### Simple

Use direct retrieval.

Example:

- "How do I reset my password?"

### Complex

Use transformation plus retrieval.

Example:

- "Compare IVF and HNSW for speed and memory tradeoffs"

### Ambiguous

Ask a clarifying question before retrieval.

Example:

- "Help, it is broken"

That is not enough information to retrieve safely.

---

## Good Day 9 Mental Model

Think of Day 9 like this:

```text
user query
   ->
classify intent
   ->
simple / complex / ambiguous
   ->
direct retrieval / transformed retrieval / clarification
   ->
better final context
```

This is a major step toward production-ready RAG.

---

## What You Should Learn Today

By the end of Day 9, you should be able to:

- rewrite weak queries into stronger retrieval queries
- route different query types to different search behavior
- detect when a query is too ambiguous to answer safely
- benchmark transformed retrieval against plain retrieval

That is the core of query intelligence in RAG.
