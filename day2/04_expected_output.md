# Day 2: Expected Output

## 02_code_example.py - Expected Output

Your exact numbers will vary, especially for the web page length, but the flow should look like this:

```text
================================================================
DAY 2: DOCUMENT LOADING AND CHUNKING COMPARISON
================================================================

Loaded sources:
  Markdown chars: 1200
  Text chars:     900
  CSV chars:      500
  Web chars:      6000

Fixed-size chunks
-----------------
Chunks: 18
Average size: 470 chars
Smallest: 190 chars
Largest: 500 chars

Recursive chunks
----------------
Chunks: 16
Average size: 455 chars
Smallest: 210 chars
Largest: 499 chars

Markdown-aware chunks
---------------------
Chunks: 4
Average size: 260 chars
Smallest: 160 chars
Largest: 340 chars

Boundary preview: Fixed-size
  Chunk 1: ...
  Chunk 2: ...

Boundary preview: Recursive
  Chunk 1: ...
  Chunk 2: ...

Boundary preview: Markdown-aware
  Chunk 1: ...
  Chunk 2: ...

Observations:
  1. Fixed-size chunking is simple, but often cuts in awkward places.
  2. Recursive chunking usually preserves sentence boundaries better.
  3. Markdown-aware chunking keeps section meaning, which helps retrieval.
```

---

## 03_mini_project.py - Expected Output

```text
==================================================================
DAY 2 MINI PROJECT: MULTI-FORMAT DOCUMENT INGESTER
==================================================================

Loading sources:
  Loaded 1 document(s) from: ...architecture.txt
  Loaded 4 document(s) from: ...retrieval_notes.md
  Loaded 4 document(s) from: ...faq.csv
  Loaded 1 document(s) from: https://en.wikipedia.org/wiki/Retrieval-augmented_generation

Created 22 chunks
Average chunk size: 380 chars
Smallest chunk: 84 chars
Largest chunk: 500 chars
Too short (<50 chars): 0
Too long (>800 chars): 0

Chunk breakdown by source type:
  csv        4
  markdown   6
  text       3
  web        9

Creating embeddings with models/gemini-embedding-001...
Saved new FAISS index to: ...faiss_multi_format_index

==================================================================
DEMO QUERIES
==================================================================

Question: Why is recursive chunking usually a strong default?
Answer: Recursive chunking is usually a strong default because it tries
to split on natural boundaries like paragraphs and sentences before
falling back to smaller separators. That helps preserve meaning inside
each chunk and usually improves retrieval quality [Source 1] [Source 2].
Sources:
[Source 1] markdown | ...
[Source 2] web | ...
[Source 3] text | ...
[Source 4] csv | ...
------------------------------------------------------------------

Question: What metadata should I preserve for RAG chunks?
Answer: You should preserve metadata such as source, source type, page,
section, row, or URL because it helps with citations, debugging, and
filtering retrieved documents later [Source 1] [Source 2].
Sources:
[Source 1] markdown | ...
[Source 2] text | ...
[Source 3] web | ...
[Source 4] csv | ...
------------------------------------------------------------------

Interactive mode. Type 'quit' to exit.

You: Why can bad chunking hurt retrieval?
Bot: Bad chunking hurts retrieval because a chunk can either become too
broad and mix unrelated topics or too small and lose important context.
That makes it harder for similarity search to return the most useful
evidence [Source 1] [Source 2].
Sources:
[Source 1] web | ...
[Source 2] markdown | ...
[Source 3] text | ...
[Source 4] csv | ...

You: quit
Bye!
```

---

## Verification Checklist

### 02_code_example.py

- [ ] All sample sources load without error
- [ ] You see chunk counts for three strategies
- [ ] Recursive boundaries look more natural than fixed-size boundaries
- [ ] Markdown-aware chunks preserve sections instead of slicing blindly

### 03_mini_project.py

- [ ] Mixed source types load into one pipeline
- [ ] Chunk statistics print successfully
- [ ] FAISS index is created or reused
- [ ] Demo queries return grounded answers
- [ ] Sources are displayed with metadata
- [ ] Interactive mode works

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: Missing API key` | `GEMINI_API_KEY` is not set | Set `GEMINI_API_KEY` in environment or `.env` |
| `ModuleNotFoundError: pypdf` | PDF dependency missing | `pip install pypdf` |
| `requests.exceptions.HTTPError` | Site blocked or unavailable | Retry later or change URL |
| `ValueError: Unsupported source type` | File extension not handled | Use `.txt`, `.md`, `.csv`, `.pdf`, or a URL |
| Retrieval feels weak | Chunking is poor | Inspect chunk sizes and boundaries before changing models |
