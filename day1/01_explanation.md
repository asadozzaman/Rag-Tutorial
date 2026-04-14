# Day 1: RAG Foundations & Mental Model

## What is RAG?

**RAG = Retrieval-Augmented Generation**

LLMs like GPT-4 or Claude have three fundamental problems:

| Problem | Example |
|---------|---------|
| **Hallucination** | "The company was founded in 2019" (actually 2015) — the model invents facts when it doesn't know |
| **Stale Knowledge** | Can't answer about anything after its training cutoff date |
| **No Domain Knowledge** | Knows nothing about YOUR company docs, internal wikis, proprietary data |

RAG solves all three by giving the LLM a **"cheat sheet"** at query time. Instead of relying
on memorized knowledge, it fetches relevant documents from your own data and injects them
into the prompt.

---

## The End-to-End Pipeline

RAG has two phases: **Ingestion** (done once, ahead of time) and **Retrieval** (done per query).

### Phase A: Ingestion (Offline — runs before any user query)

```
Raw Documents (.pdf, .txt, .html, .csv)
         |
         v
  +---------------+
  | 1. LOAD       |  Read files into raw text
  |   documents   |  (PyPDFLoader, TextLoader, WebBaseLoader)
  +-------+-------+
          |
          v
  +---------------+
  | 2. CHUNK      |  Split into small, meaningful pieces
  |   documents   |  (e.g., 200-500 tokens per chunk)
  +-------+-------+
          |
          v
  +---------------+
  | 3. EMBED      |  Convert each chunk into a vector
  |   chunks      |  (list of numbers representing meaning)
  +-------+-------+
          |
          v
  +---------------+
  | 4. STORE      |  Save vectors in a vector database
  |   in index    |  (FAISS, Chroma, Pinecone)
  +---------------+
```

### Phase B: Query (Online — runs every time a user asks a question)

```
User asks: "What is our refund policy?"
         |
         v
  +---------------+
  | 1. EMBED      |  Convert query into a vector using the
  |   the query   |  SAME embedding model as ingestion
  +-------+-------+
          |
          v
  +---------------+
  | 2. RETRIEVE   |  Find top-K most similar chunks
  |   from index  |  via cosine similarity in vector DB
  +-------+-------+
          |
          v
  +---------------+
  | 3. AUGMENT    |  Build a prompt:
  |   the prompt  |  "Given this context: {chunks}, answer: {query}"
  +-------+-------+
          |
          v
  +---------------+
  | 4. GENERATE   |  LLM reads context + query and generates
  |   answer      |  a grounded, factual response
  +---------------+
```

---

## Key Concept: Embeddings & Similarity Search

**Embedding** = converting text into a list of numbers (a vector) that captures its meaning.

```
"dog"    -> [0.12, -0.45, 0.78, 0.33, ...]   (1536 dimensions)
"puppy"  -> [0.11, -0.43, 0.80, 0.31, ...]   (very close to "dog"!)
"rocket" -> [-0.67, 0.22, -0.15, 0.89, ...]  (far from "dog")
```

When you search, the vector DB finds chunks whose vectors are **closest** to your query's vector.
This is **semantic search** — it matches by meaning, not keywords.

- "How to return a product?" matches "Our refund policy allows returns within 30 days"
- Keyword search would miss this because no words overlap!

---

## RAG vs Fine-Tuning vs Prompting

| | Prompting | Fine-Tuning | RAG |
|---|---|---|---|
| **How** | Paste context into prompt | Retrain model on your data | Retrieve docs dynamically |
| **Data size** | Small (fits in one prompt) | Medium to large | Unlimited |
| **Update data** | Edit the prompt | Retrain ($$, slow) | Update the vector DB |
| **Cites sources** | Only if you provide them | No | Yes (retrieved chunks) |
| **Cost** | Low | High (training) | Medium (embeddings + DB) |
| **Best for** | Small, static context | Teaching style/tone | Large/changing knowledge bases |

### When to use what:

- **Prompting only**: You have <10 pages of context that rarely changes
- **Fine-Tuning**: You want the model to adopt a specific writing style or behavior
- **RAG**: You have a large, evolving knowledge base and need source attribution
- **RAG + Fine-Tuning**: Fine-tune for style, RAG for knowledge (advanced)

---

## Key Insight

> RAG doesn't replace the LLM. It gives the LLM a **librarian** — someone who fetches the right
> books before the LLM writes its answer.

Without RAG: LLM guesses from memory (might hallucinate)
With RAG: LLM reads relevant documents first, then answers (grounded in facts)

---

## Vocabulary Cheat Sheet

| Term | Meaning |
|------|---------|
| **Embedding** | A vector (list of numbers) representing text meaning |
| **Vector Store / Vector DB** | Database optimized for storing and searching embeddings |
| **Chunk** | A small piece of a larger document |
| **Retriever** | Component that fetches relevant chunks for a query |
| **Top-K** | How many chunks to retrieve (K=2 to K=5 is typical) |
| **Cosine Similarity** | Math that measures how "close" two vectors are |
| **FAISS** | Facebook's fast vector similarity search library |
| **LangChain** | Framework for building LLM-powered applications |

---

## Reading List (in order)

1. Pinecone — What is RAG? (beginner overview)
   https://www.pinecone.io/learn/retrieval-augmented-generation/
2. LangChain RAG Tutorial (hands-on code)
   https://python.langchain.com/docs/tutorials/rag/
3. Lewis et al. — Original RAG Paper (2020, academic)
   https://arxiv.org/abs/2005.11401
4. Anthropic — RAG with Claude
   https://docs.anthropic.com/en/docs/build-with-claude/retrieval-augmented-generation
