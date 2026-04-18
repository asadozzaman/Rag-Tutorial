# Day 2: Document Loading and Chunking Mastery

## Why Day 2 Matters

Day 1 gave you the mental model:

`Load -> Chunk -> Embed -> Index -> Retrieve -> Generate`

Day 2 focuses on the part that silently decides whether your RAG system feels smart or frustrating:

`document loading + chunking`

Most beginner RAG systems fail here, not in the LLM layer.

---

## Part 1: Document Loading

Real RAG systems rarely use one clean text file. They usually ingest:

- PDFs
- TXT files
- Markdown knowledge bases
- CSV exports
- Web pages
- API responses

Each source behaves differently.

| Source | Common problem |
|---|---|
| PDF | Broken layout, missing reading order, page artifacts |
| Web page | Extra navigation text, ads, unrelated HTML |
| Markdown | Losing headers destroys section meaning |
| CSV | Rows are short and structured, so naive chunking can be wasteful |
| TXT | Easy to load, but no built-in structure |

The lesson: before chunking, know what kind of document you are dealing with.

---

## Part 2: Why Chunking Matters

Chunking means splitting a document into smaller pieces that can be embedded and retrieved.

Bad chunking causes two opposite problems:

### Chunk too large

- one chunk contains too many topics
- retrieval becomes blurry
- context window gets wasted
- the answer may include irrelevant details

### Chunk too small

- key ideas get split apart
- answer context becomes fragmented
- the retriever may fetch half an explanation
- the model may answer with missing context

The goal is not "small chunks". The goal is:

> one chunk = one meaningful unit of context

---

## Part 3: Core Chunking Strategies

## 1. Fixed-size chunking

Split every `N` characters or tokens.

Example:

```text
Chunk 1 = chars 0-499
Chunk 2 = chars 450-949
Chunk 3 = chars 900-1399
```

Pros:

- simple
- fast
- predictable

Cons:

- often cuts through sentences or paragraphs
- ignores document structure

Good for:

- fast baseline experiments
- raw text when you need a simple starting point

---

## 2. Recursive chunking

This is usually the best default.

It tries to split on better boundaries first:

```text
\n\n -> \n -> sentence boundary -> space -> fallback character split
```

Pros:

- preserves meaning better than fixed-size
- works well on plain text, web text, PDFs
- good default for most RAG projects

Cons:

- still not fully aware of document structure

Good for:

- most Day 1 and Day 2 projects
- general-purpose RAG pipelines

---

## 3. Markdown-aware or document-aware chunking

This respects structure such as headers and sections.

Example:

```markdown
# API Overview
## Authentication
## Rate Limits
## Error Codes
```

Instead of cutting blindly, you keep content grouped by section.

Pros:

- preserves section meaning
- makes retrieval more interpretable
- great for technical docs and handbooks

Cons:

- needs structured source documents

Good for:

- docs sites
- internal wikis
- README collections
- policy manuals

---

## 4. Semantic chunking

This groups text by meaning instead of only by character boundaries.

Pros:

- can produce very coherent chunks
- useful for advanced retrieval systems

Cons:

- more expensive
- more complex to implement
- unnecessary for many beginner systems

Good for:

- high-quality production systems
- documents with long flowing prose

---

## Part 4: Why Overlap Matters

Overlap means the end of one chunk is repeated at the start of the next.

Example:

```text
chunk_size = 500
chunk_overlap = 50
```

Why overlap helps:

- prevents losing meaning at chunk boundaries
- keeps related sentences connected
- improves retrieval when the answer sits near the edge of a chunk

A good beginner baseline:

- `chunk_size = 500`
- `chunk_overlap = 50`

That is close to a 10 percent overlap.

---

## Part 5: Recommended Defaults by Source Type

| Source type | Good default strategy |
|---|---|
| TXT | RecursiveCharacterTextSplitter |
| Web page | RecursiveCharacterTextSplitter after cleaning text |
| PDF | RecursiveCharacterTextSplitter after extraction |
| Markdown | MarkdownHeaderTextSplitter, then recursive if sections are too large |
| CSV | Keep row metadata; split only if row text is long |

---

## Part 6: Metadata Is Part of Retrieval Quality

Do not store chunk text alone.

Store metadata on every chunk:

- `source`
- `source_type`
- `page`
- `row`
- `section`
- `url`

Why it matters:

- lets you show citations
- makes debugging easier
- enables filtering later
- helps you understand which source types are performing best

---

## Part 7: What You Should Build Today

### Code example

Compare chunking strategies on the same content:

- fixed-size
- recursive
- markdown-aware

Look at:

- chunk counts
- average chunk lengths
- whether chunk boundaries look natural

### Mini project

Build a multi-format ingester that:

1. loads TXT, Markdown, CSV, PDF, and web URLs
2. chooses the best splitter per source
3. preserves metadata
4. stores everything in one FAISS index
5. lets you query across all sources
6. reports chunk quality stats

---

## Part 8: Production Notes

- Start with recursive chunking before trying advanced methods.
- If retrieval is poor, inspect chunks before changing models.
- Metadata is not optional in real systems.
- Chunk size should be chosen based on document type and question style.
- A chunking bug can make a strong embedding model look weak.

---

## Day 2 Success Checklist

- [ ] I can explain why chunking matters
- [ ] I know the difference between fixed, recursive, and markdown-aware chunking
- [ ] I can load more than one document format
- [ ] I can preserve source metadata in my chunks
- [ ] I can build a single FAISS index from mixed sources
- [ ] I can inspect chunk quality instead of guessing
