# Day 11: Conversational RAG & Memory

## Why Day 11 Matters

Normal RAG works fine for one-turn questions.

But real users talk like this:

- "Tell me about Python."
- "What about its web frameworks?"
- "Which one is best for APIs?"
- "Does it support async?"

A plain retriever sees only the last sentence.

That causes failure because:

- `its` has no meaning alone
- `one` has no meaning alone
- follow-up questions depend on earlier turns

Day 11 solves that with memory and history-aware retrieval.

---

## The Core Problem

Conversational RAG needs to understand references across turns.

This is called:

- `coreference resolution`
- `history-aware retrieval`
- `contextual query reformulation`

Example:

```text
User: Tell me about Python
User: What web frameworks does it have?
```

The retriever should not search for:

```text
it web frameworks
```

It should search for:

```text
What web frameworks does Python have?
```

That rewrite is the key idea.

---

## Day 11 Architecture

The mental model is:

```text
chat history + new query
    ->
contextualizer
    ->
standalone query
    ->
retriever
    ->
compressed context
    ->
answer
```

This adds one important step before retrieval:

- turn the follow-up into a standalone question

---

## Memory Types

### Short-Term Memory

This is the recent chat history in the current session.

Useful for:

- pronouns like `it`, `they`, `that`
- recent user preferences
- ongoing topic continuity

### Summarized Memory

When the chat grows long, sending every message becomes expensive.

So a common trick is:

- keep the most recent turns
- summarize older turns

This preserves context while reducing token cost.

---

## Contextual Compression

Even after retrieval, not every sentence is useful.

So you can compress the context by keeping only the most relevant sentences for the rewritten query.

This helps because it:

- reduces noise
- improves grounding
- makes multi-turn answers cleaner

---

## What Good Day 11 Behavior Looks Like

A good conversational RAG system should:

- rewrite vague follow-ups before retrieval
- remember the active topic across turns
- recommend sources for each answer
- let the user clear history when needed

That last part matters because old context can become harmful if the topic changes.

---

## What You Should Learn Today

By the end of Day 11, you should be able to:

- add session memory to a RAG system
- reformulate follow-up questions into standalone retrieval queries
- handle coreference like `it`, `its`, and `one`
- compress retrieved context before answering

That is the foundation of conversational RAG.
