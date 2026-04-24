# Day 6: Prompt Engineering for RAG

## Why Day 6 Matters

By Day 5, you already know how to retrieve stronger documents.

Day 6 is about the bridge between retrieval and final answer quality:

`prompt design`

Even with great retrieval, a poor prompt can still produce:

- vague answers
- missing citations
- hallucinations
- failure when context is empty
- confusion when sources conflict

---

## The Four Parts of a RAG Prompt

### 1. System prompt

This defines behavior.

Example goals:

- use only retrieved context
- cite sources
- say when evidence is missing
- handle contradictions explicitly

---

### 2. Context block

This is the retrieved evidence.

Good context formatting usually includes:

- source label
- source title or metadata
- clean chunk text

Example:

```text
[Source 1] Refund Policy
Refunds are available within 30 days...
```

---

### 3. User question

This should remain clear and separate from context.

Do not bury the actual question in a giant blob of text.

---

### 4. Output format

You often want the answer in a predictable format such as:

- JSON
- markdown with citations
- answer plus confidence
- answer plus reasoning

Structured output matters because it makes downstream app logic much easier.

---

## Core RAG Prompt Decisions

### Use only context, or allow world knowledge?

For factual enterprise RAG, the safer pattern is:

> answer from retrieved context only

That reduces hallucination and makes answers easier to audit.

---

### How should the system behave when context is weak?

Never assume retrieval is always good.

Your prompt needs an escape hatch such as:

```text
If the context does not contain enough evidence, say so clearly.
```

This is one of the most important prompt rules in RAG.

---

### How should contradictions be handled?

Retrieved sources may disagree.

A production-quality prompt should say something like:

```text
If the retrieved sources conflict, mention the disagreement instead of pretending they align.
```

---

### Should sources be cited inline?

Usually yes.

Inline citations like `[Source 1]` help users:

- verify claims
- trust the answer
- inspect edge cases

---

## Confidence and Graceful Failure

Good RAG systems do not always answer with full confidence.

Useful confidence labels:

- `HIGH`: directly supported by context
- `MEDIUM`: supported, but with some inference or nuance
- `LOW`: weak evidence or fallback response

Graceful failure means the system says:

- what it knows
- what it does not know
- why confidence is limited

---

## Conversation Memory for Follow-Ups

Users rarely ask isolated questions.

Example:

```text
User: What is the refund policy?
User: What about digital products?
```

The second question depends on the first.

A RAG chatbot often needs to:

1. keep a small chat history
2. rewrite the follow-up as a fuller query
3. retrieve again using that rewritten query

This is not just memory for memory’s sake. It improves retrieval precision.

---

## Practical Day 6 Prompt Template

The production-friendly pattern is:

1. instruction block
2. context block
3. question
4. formatting rules

Example behavior:

- answer only from context
- cite sources
- admit insufficient information
- mention contradictions
- output JSON

---

## What You Will Build Today

### Code example

- construct a production RAG prompt
- inject retrieved context
- generate structured JSON-style output
- detect contradiction or insufficient evidence

### Mini project

Build a small support chatbot that:

1. retrieves top-3 docs
2. checks if retrieval is relevant enough
3. falls back gracefully when context is weak
4. cites sources inline
5. handles follow-up questions using memory

---

## Day 6 Success Checklist

- [ ] I can explain the four parts of a RAG prompt
- [ ] I know how to format context clearly
- [ ] I can force structured output
- [ ] I can handle no-context and conflict cases
- [ ] I understand why source citations matter
- [ ] I can rewrite a follow-up question using history
