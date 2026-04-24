# Day 10: Expected Output

## 02_code_example.py - Expected Output

Your exact wording may vary slightly, but the flow should look like this:

```text
==============================================================================================
DAY 10: AGENTIC RAG WITH TOOLS
==============================================================================================
Query: What was our Q1 revenue?
Chosen tools: search_knowledge_base
----------------------------------------------------------------------------------------------
TOOL: search_knowledge_base
INPUT: What was our Q1 revenue?
OUTPUT: Q1 Revenue Summary: Company revenue in Q1 2024 was $5.2M, up 15% year over year. | ...

Final answer:
Company revenue in Q1 2024 was $5.2M, up 15% year over year.

==============================================================================================
Query: If Q1 revenue grows at the same 15% rate, what will Q2 revenue be?
Chosen tools: search_knowledge_base, calculate
...
OUTPUT: 5200000.0 * (1 + 0.15) = 5980000.0000

Final answer:
If the same 15% growth continues, projected Q2 revenue is about $5,980,000.
```

---

## 03_mini_project.py - Expected Output

```text
==========================================================================================================
DAY 10 MINI PROJECT: RESEARCH ASSISTANT AGENT
==========================================================================================================
Knowledge base docs: ...
External snippets:   ...

Query: How does our churn rate compare to the industry average?
Tool plan: search_knowledge_base, search_web, calculate
  Step 1: search_knowledge_base
    Observation: Churn KPI: Customer churn rate decreased from 8.0% to 5.5% in Q1. | ...
  Step 2: search_web
    Observation: SaaS churn benchmark: Industry average annualized small-business SaaS churn is around 6.8%. | ...
  Step 3: calculate
    Observation: 6.8 - 5.5 = 1.3000
Answer: Our Q1 churn was 5.5%, while the industry benchmark is 6.8%, so we are lower by 1.3 percentage points.
```

---

## Verification Checklist

- [ ] Agent chooses tools based on the question
- [ ] Knowledge base search is used for internal company facts
- [ ] Web search is used for outside benchmarks
- [ ] Calculator is used for projections or comparisons
- [ ] Multi-step questions show an inspectable tool trace

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Agent uses too many tools | Routing rules are too broad | Tighten when web search or calculation is triggered |
| Projection answer is wrong | Numeric extraction failed | Check money and percentage parsing |
| External comparison is missing | Web snippets do not match the query wording | Add stronger topic keywords to web data |
| Tool trace is hard to read | Outputs are too raw | Keep tool outputs short and structured |
