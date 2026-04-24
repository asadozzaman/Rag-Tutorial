# Day 10: Agentic RAG & Tool Use

## Why Day 10 Matters

So far your RAG systems have mostly followed a fixed pipeline:

```text
query -> retrieve -> answer
```

That is useful, but rigid.

Real questions are often more dynamic:

- some questions need retrieval only
- some need retrieval plus math
- some need internal data plus outside information
- some need multiple steps before a final answer

Day 10 introduces `agentic RAG`.

---

## From Pipeline to Agent

### Traditional RAG

Always does the same thing:

1. retrieve context
2. generate answer

### Agentic RAG

Makes decisions during the process:

1. think about the request
2. choose a tool
3. inspect the result
4. decide the next step
5. answer when enough evidence exists

That is the big shift.

---

## ReAct Mental Model

The classic pattern is:

```text
Thought -> Action -> Observation -> Thought -> Action -> Observation -> Final Answer
```

This matters because one result can change the next action.

Example:

```text
Question: How does our churn compare to the industry average?

Thought: I need our churn first.
Action: search knowledge base
Observation: Our churn is 5.5%

Thought: Now I need the industry average.
Action: search web
Observation: Industry average is 6.8%

Thought: Now I should compare them numerically.
Action: calculate
Observation: 6.8 - 5.5 = 1.3

Final Answer: Our churn is 1.3 points lower than the industry average.
```

That is agentic behavior.

---

## Tools in Day 10

This day uses three core tools:

### 1. Knowledge Base Search

Use this for:

- company facts
- product details
- internal metrics
- policy details

### 2. Web Search

Use this for:

- industry averages
- market benchmarks
- external comparisons

### 3. Calculator

Use this for:

- projections
- comparisons
- growth calculations
- percentage differences

The key lesson is not the tools themselves.

The real lesson is deciding `when` to use them.

---

## When Agents Help

Agentic RAG is especially helpful when:

- a question is multi-hop
- evidence comes from multiple sources
- calculations are required
- the system should choose between acting and answering

Agentic RAG is less useful when:

- the question is simple and one retrieval step is enough
- tool calls add latency without improving quality

That tradeoff matters in production.

---

## Production Mindset

Agents are powerful, but they can become messy quickly.

So in real systems you usually need:

- `max_iterations` to stop runaway loops
- logging for each tool call
- structured tool outputs
- safety checks before math or external claims

Day 10 should teach you controlled agent behavior, not blind autonomy.

---

## What You Should Learn Today

By the end of Day 10, you should be able to:

- build an agent that chooses tools instead of always following one path
- combine retrieval, web lookup, and calculation
- handle multi-step questions with intermediate observations
- log the tool-use chain clearly

That is the foundation of agentic RAG.
