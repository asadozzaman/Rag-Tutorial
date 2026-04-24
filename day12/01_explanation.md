# Day 12: Multimodal RAG & Structured Data

## Why Day 12 Matters

RAG is not only about plain text.

Real documents often contain:

- prose
- tables
- charts
- figures
- code snippets

If your system only retrieves paragraphs, it misses a lot of value.

Day 12 teaches you how to route different questions to different data backends.

---

## The Main Idea

Different content types need different retrieval strategies.

### Text

Use:

- chunking
- embedding
- vector-style retrieval

Best for:

- narrative explanations
- policies
- qualitative insights

### Tables

Use:

- structured storage
- SQL
- DataFrame analysis

Best for:

- totals
- rankings
- growth calculations
- filtering by product, region, or quarter

### Images

Use:

- captions
- OCR
- vision models

Best for:

- semantic image lookup
- chart explanation
- figure references

### Code

Use:

- AST or symbol-aware parsing
- repository search
- metadata over functions, classes, and files

Best for:

- implementation questions
- usage examples
- debugging a codebase

---

## Text-to-SQL

Text-to-SQL means:

- take a natural-language question
- generate a SQL query
- run it safely
- return the result

Example:

```text
Question: Which region had the highest revenue in Q1?

SQL:
SELECT region, SUM(revenue)
FROM sales
WHERE quarter = 'Q1'
GROUP BY region
ORDER BY SUM(revenue) DESC
LIMIT 1;
```

This is much better than stuffing a whole table into an LLM prompt.

---

## DataFrame Q&A

Sometimes your data lives in:

- CSV
- Excel
- pandas-like tabular structures

In that case, a good pattern is:

1. inspect the schema
2. generate analysis code
3. execute it safely
4. return the result

This is especially useful for:

- ad hoc analysis
- inventory calculations
- grouped aggregations

---

## The Hybrid Router

Some questions need only one backend:

- "What is total revenue by product?" -> table route
- "Summarize the expansion strategy." -> text route

Some need both:

- "What was Widget A's Q1 revenue and what does the report say about North region demand?" -> hybrid route

That is the Day 12 pattern:

```text
question
   ->
router
   ->
text / structured / hybrid
   ->
answer
```

---

## Safety Matters

Structured-data RAG is powerful, but it adds risk.

Important production habits:

- allow only `SELECT` SQL
- reject destructive SQL
- prefer read-only database access
- log generated SQL
- validate outputs before presenting them

That is why safe execution matters as much as retrieval quality here.

---

## What You Should Learn Today

By the end of Day 12, you should be able to:

- route questions to text retrieval or SQL analysis
- generate safe SQL for table questions
- run DataFrame-style analysis code for CSV questions
- combine structured and unstructured evidence in one answer

That is the foundation of multimodal and structured-data RAG.
