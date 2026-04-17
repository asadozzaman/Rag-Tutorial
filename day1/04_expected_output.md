# Day 1: Expected Output

## 02_code_example.py — Expected Output

```
============================================================
  DAY 1: YOUR FIRST RAG PIPELINE
============================================================

Knowledge base: 6 documents

[CHUNK]  Created 6 chunks from 6 documents
         Average chunk size: 231 chars
         Smallest chunk: 198 chars
         Largest chunk: 272 chars

[EMBED]  Creating embeddings with text-embedding-3-small...
[STORE]  Vector store created with FAISS

[CHAIN]  RetrievalQA chain ready (k=2)

============================================================
  QUERYING THE RAG PIPELINE
============================================================

────────────────────────────────────────────────────────────
  Question 1: Why does RAG help with hallucination?
────────────────────────────────────────────────────────────

  Answer: RAG helps with hallucination by providing relevant source
  documents as context when generating answers. Instead of relying
  solely on its training data (which may be incomplete or outdated),
  the model can reference actual retrieved documents, producing
  grounded and factual responses. This reduces the chance of the
  LLM generating plausible but incorrect information.

  Retrieved 2 source chunks:
    [1] "LLMs can hallucinate — generating plausible but incorrect information — especia..."
    [2] "RAG stands for Retrieval-Augmented Generation. It combines information retrieva..."

────────────────────────────────────────────────────────────
  Question 2: What is a vector database and how does it work?
────────────────────────────────────────────────────────────

  Answer: A vector database is a specialized database that stores
  document embeddings as high-dimensional vectors. It enables fast
  similarity search by comparing the mathematical distance between
  vectors. When you search, the database finds vectors that are
  closest to your query vector, meaning the content is semantically
  similar. FAISS and Chroma are popular examples.

  Retrieved 2 source chunks:
    [1] "Vector databases like FAISS and Chroma store document embeddings as high-dimens..."
    [2] "The embedding model converts text into numerical vectors. Similar texts produce..."

────────────────────────────────────────────────────────────
  Question 3: How does chunking affect retrieval quality?
────────────────────────────────────────────────────────────

  Answer: Chunking directly affects retrieval quality because it
  determines the granularity of what the retriever can find.
  Good chunking preserves semantic meaning within each piece,
  so the retriever returns coherent, relevant passages. Common
  strategies include fixed-size, recursive, and semantic chunking,
  each with different trade-offs for accuracy and coherence.

  Retrieved 2 source chunks:
    [1] "Chunking is the process of splitting large documents into smaller pieces. Good ..."
    [2] "In a RAG pipeline, the retriever fetches the top-K most relevant chunks for a g..."

============================================================
  DONE! You've built your first RAG pipeline.
============================================================


--- BONUS: Manual Retrieval (no LLM) ---

Query: "What is an embedding?"
Retrieved 2 chunks:

  Chunk 1:
  The embedding model converts text into numerical vectors. Similar texts
  produce vectors that are close together in vector space, enabling
  semantic search rather than simple keyword matching.

  Chunk 2:
  Vector databases like FAISS and Chroma store document embeddings as
  high-dimensional vectors, enabling fast similarity search. FAISS was
  developed by Facebook AI Research and is widely used for its speed.
```

---

## 03_mini_project.py — Expected Output

```
=======================================================
  ELI5 RAG Bot — Quantum Computing Edition
=======================================================

Loading article from: https://en.wikipedia.org/wiki/Quantum_computing
Loaded 1 page(s), 85,432 characters
Split into 198 chunks (size=500, overlap=50)
Creating embeddings (this may take a moment)...
Index saved to ./faiss_eli5_index/

ELI5 chain ready!

=======================================================
  DEMO: Watch the ELI5 bot answer questions
=======================================================

  Q: What is quantum computing?
  A: Imagine you have a really special computer that doesn't use regular
     switches like your light switch at home. Instead, it uses tiny
     magical things called "qubits" that can be ON and OFF at the same
     time — like a spinning coin that's both heads AND tails! This lets
     it solve really tricky puzzles much faster than a regular computer.
     (Used 3 source chunks)
-------------------------------------------------------

  Q: What is a qubit?
  A: A qubit is like a magic marble! A regular marble can only be in
     one box — the left box OR the right box. But a qubit-marble is
     special: it can be in BOTH boxes at the same time until you look
     at it! When you peek, it picks just one. That's what makes quantum
     computers so cool.
     (Used 3 source chunks)
-------------------------------------------------------

  Q: Why is quantum computing faster than regular computers?
  A: Think about a maze! A regular computer tries one path, hits a wall,
     goes back, and tries another path — one at a time. A quantum
     computer is like having a bunch of tiny explorers that can try ALL
     the paths at the same time! So for really tricky mazes, it finds
     the way out much, much faster.
     (Used 3 source chunks)
-------------------------------------------------------

  Q: Who invented quantum computing?
  A: A really smart man named Richard Feynman had the idea a long time
     ago, in the 1980s! He noticed that regular computers have a really
     hard time pretending to be tiny particles. So he said, "Hey, what
     if we built a computer that USES those tiny particles instead?"
     And that's how the idea of quantum computing started!
     (Used 3 source chunks)
-------------------------------------------------------

  Q: What problems can quantum computers solve?
  A: Quantum computers are really good at special puzzles that are super
     hard for regular computers! Things like figuring out secret codes,
     finding the best route for a delivery truck, or understanding how
     tiny medicine molecules work inside your body. But they're not
     better at everything — your regular computer is still great for
     watching cartoons!
     (Used 3 source chunks)
-------------------------------------------------------

=======================================================
  INTERACTIVE MODE — Ask anything!
  Type 'quit' to exit
=======================================================

You: What is superposition?
Bot: Imagine you have a light switch. Normally it's either UP (on) or
     DOWN (off), right? Well, in quantum world, it's like the switch
     can be UP and DOWN at the SAME TIME! It only picks one when you
     look at it. Scientists call this "superposition" — it just means
     being in more than one state at once, like a magic trick!

You: quit
Bye!
```

---

## How to Verify Your Output

Your exact text will differ (LLMs are non-deterministic), but check for:

1. **Code Example (`02_code_example.py`)**
   - [ ] Chunks are created (count printed)
   - [ ] FAISS vector store is built without errors
   - [ ] All 3 questions get answers
   - [ ] Each answer cites 2 source chunks
   - [ ] Answers are relevant to the question (not random)
   - [ ] Bonus manual retrieval works

2. **Mini Project (`03_mini_project.py`)**
   - [ ] Wikipedia article loads (large character count)
   - [ ] Chunks created (expect 150-250 chunks)
   - [ ] FAISS index saved to disk
   - [ ] Demo questions get simple, kid-friendly answers
   - [ ] Interactive mode accepts your own questions
   - [ ] Second run loads from saved index (faster startup)

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: Missing API key` | `GEMINI_API_KEY` was not set | Check `.env` has `GEMINI_API_KEY=...` |
| `ModuleNotFoundError: langchain_google_genai` | Package not installed | `pip install langchain-google-genai` |
| `ModuleNotFoundError: faiss` | FAISS not installed | `pip install faiss-cpu` |
| `ConnectionError` on WebBaseLoader | Can't reach Wikipedia | Check internet; try a different URL |
| `bs4` / `lxml` not found | Missing HTML parser | `pip install beautifulsoup4 lxml` |
| Very slow embedding step | Large article, many chunks | Normal for first run; saved index is reused after |
