# Day 8: Evaluation & Metrics for RAG

## Why Day 8 Matters

If you cannot measure your RAG system, you cannot improve it safely.

A RAG pipeline can fail in at least two major places:

- retrieval finds the wrong context
- generation writes an answer that is weak, incomplete, or hallucinated

Day 8 teaches you how to measure both sides separately.

---

## The Two Big Buckets of RAG Evaluation

### 1. Retrieval Metrics

These metrics answer:

- Did we retrieve the right documents?
- Were the useful documents ranked high enough?

Important retrieval metrics:

- `Recall@K`: how many relevant documents appeared in the top K
- `Precision@K`: how much of the top K was actually useful
- `MRR`: how early the first relevant document appeared
- `NDCG@K`: whether the ranking quality is good, not just whether a hit exists

If retrieval is weak, even a perfect LLM prompt will struggle.

---

### 2. Generation Metrics

These metrics answer:

- Is the answer grounded in the retrieved context?
- Does the answer actually address the user question?
- Is the answer making up unsupported claims?

Important generation metrics:

- `Faithfulness`: answer stays supported by context
- `Answer Relevance`: answer is actually about the question
- `Context Precision`: retrieved context is useful for the answer
- `Context Recall`: retrieved context covers the important truth
- `Hallucination Risk`: how much of the answer seems unsupported

---

## Mental Model

Think of RAG evaluation like this:

```text
user question
    ->
retriever quality
    ->
available evidence
    ->
generator quality
    ->
final answer quality
```

If the retriever fails:

- answer may be irrelevant
- answer may become vague
- hallucination risk rises

If the generator fails:

- answer may ignore good context
- answer may overstate facts
- citations may look correct while reasoning is wrong

---

## RAGAS, DeepEval, and Custom Metrics

Your roadmap mentions three approaches:

### RAGAS

Useful when you want standard RAG metrics such as:

- faithfulness
- answer relevancy
- context precision
- context recall

### DeepEval

Useful when you want testing-style evaluation workflows and LLM-judge style checks.

### Custom Metrics

Very important in real projects.

Why?

Because your system often has domain-specific failure modes:

- wrong refund rules
- outdated product plan details
- missing API parameter behavior
- policy contradictions

So even if you use RAGAS or DeepEval, you still usually add custom checks.

---

## What Each Metric Tells You

### Recall@K

High recall means:

- the retriever usually finds the needed evidence

Low recall means:

- chunking may be bad
- embeddings may be weak
- metadata filters may be too strict
- query rewriting may be needed

---

### MRR

High MRR means:

- the first useful document appears near the top

Low MRR means:

- relevant docs exist but ranking is poor

This often points to:

- weak scoring
- missing reranking
- poor fusion strategy

---

### Faithfulness

High faithfulness means:

- the answer mostly stays inside what the context supports

Low faithfulness means:

- answer generation is inventing details
- prompt constraints are too weak
- retriever did not provide enough support

---

### Context Precision vs Context Recall

`Context Precision` asks:

- how much of the retrieved context was actually useful

`Context Recall` asks:

- whether the retrieved context covered the important truth

This distinction matters a lot:

- low precision means noisy retrieval
- low recall means missing evidence

---

## A Good Day 8 Workflow

For each evaluation example, store:

- question
- generated answer
- retrieved context
- ground-truth answer
- retrieved document ids
- relevant document ids

Then compute:

1. retrieval metrics
2. generation metrics
3. a final review of the worst examples

Do not stop at averages.

Always inspect the weakest queries individually.

---

## What You Should Learn Today

By the end of Day 8, you should be able to:

- tell whether a failure came from retrieval or generation
- measure ranking quality with retrieval metrics
- estimate grounding and hallucination risk
- build an evaluation dashboard that shows weak spots clearly

That is the real skill of Day 8.
