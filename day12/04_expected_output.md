# Day 12: Expected Output

## 02_code_example.py - Expected Output

Your exact ordering may vary slightly, but the flow should look like this:

```text
================================================================================================
DAY 12: TEXT-TO-SQL AND TABLE Q&A
================================================================================================
Q: What is the total revenue by product?
SQL: SELECT product, ROUND(SUM(revenue), 2) AS total_revenue ...
Result: Widget B: $600,000; Widget A: $445,000

Q: Which region had the highest revenue in Q1?
SQL: SELECT region, ROUND(SUM(revenue), 2) AS total_revenue ...
Result: North had the highest Q1 revenue at $350,000.

Q: Show quarter-over-quarter growth for Widget A in North
SQL: SELECT quarter, revenue FROM sales ...
Result: Q1 to Q2 growth was $25,000 (16.7%).

DataFrame-style Q&A
------------------------------------------------------------------------------------------------
Question: What is the total value of inventory (price x stock) per product?
Generated code: result = {row['product']: ...}
Answer: {'Widget A': 14995.0, 'Widget B': 9998.0, 'Widget C': 19990.0}
```

---

## 03_mini_project.py - Expected Output

```text
==============================================================================================================
DAY 12 MINI PROJECT: HYBRID DOCUMENT Q&A SYSTEM
==============================================================================================================
Question: Summarize the North region demand outlook.
Route: text
Sources: Market Outlook, North Region Highlights
Answer: ...

Question: What is the total revenue by product?
Route: structured
SQL: SELECT product, ROUND(SUM(revenue), 2) AS total_revenue ...
Answer: Widget B: $600,000; Widget A: $445,000

Question: What was Widget A's Q1 revenue and what does the report say about North region demand?
Route: hybrid
SQL: SELECT product, region, revenue, quarter, year ...
Sources: North Region Highlights, Market Outlook
Answer: ... Report context: ...
```

---

## Verification Checklist

- [ ] Text-to-SQL pipeline runs with safe `SELECT` queries only
- [ ] DataFrame-style Q&A generates and executes analysis code
- [ ] Router sends questions to `text`, `structured`, or `hybrid`
- [ ] Hybrid answers combine table output with report prose
- [ ] SQL used for table questions is shown for inspection

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Generated SQL is unsafe | Validation is missing | Only allow `SELECT` statements and block destructive keywords |
| Table answers are vague | Formatter is too generic | Add question-specific result formatting |
| Router sends everything to SQL | Query intent rules are too broad | Add text-focused markers like outlook, strategy, summary |
| Hybrid answer misses narrative context | Text retrieval is too weak | Add better report chunks and more specific retrieval terms |
