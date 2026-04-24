# Day 6: Expected Output

## 02_code_example.py - Expected Output

Your exact text may vary slightly because the answer builder is heuristic, but the flow should look like this:

```text
==============================================================================
DAY 6: PRODUCTION RAG PROMPT DESIGN
==============================================================================

Question: Can I get a refund on a downloaded digital product after 15 days?

System prompt:
You are a precise research assistant. Answer using ONLY the provided context...

Human prompt:
CONTEXT:
[Source 1] Refund Policy | policy.md | score=...
...

Structured response:
{
  "answer": "... [Source 1] ... [Source 3]",
  "confidence": "MEDIUM",
  "sources": ["policy.md", "faq.md", "terms.md"],
  "reasoning": "...",
  "used_context": true
}
------------------------------------------------------------------------------

Question: Are subscriptions always final sale?
...
"answer": "The retrieved sources do not fully agree..."
...

Question: Do you support Apple Pay in Egypt?
...
"confidence": "LOW"
"used_context": false
```

---

## 03_mini_project.py - Expected Output

```text
====================================================================================
DAY 6 MINI PROJECT: RAG CHATBOT WITH GRACEFUL FAILURE HANDLING
====================================================================================
Support docs loaded: 8

User query: What is the refund policy for digital products?
...
Answer:
Digital products are non-refundable after download [Source 1] ...
Confidence: MEDIUM

User query: What about defective downloads?
Rewritten with memory: What is the refund policy for digital products? Follow-up: What about defective downloads?
...

User query: Are subscriptions always final sale?
...
Answer:
The retrieved sources do not fully agree...

User query: Do you offer offline retail pickup in Brazil?
...
Answer:
I don't have specific information about that in the retrieved context...
Confidence: LOW

Interactive mode. Type 'quit' to exit.
```

---

## Verification Checklist

- [ ] System prompt and context block are both shown
- [ ] Structured output contains answer, confidence, sources, and reasoning
- [ ] Weak retrieval triggers a fallback answer
- [ ] Source conflict is acknowledged instead of hidden
- [ ] Follow-up questions are rewritten using memory
- [ ] Answers cite sources inline

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Fallback triggers too often | Relevance threshold is too high | Lower the threshold slightly |
| The chatbot overanswers | Prompt rules are too loose | Strengthen the "use only context" instruction |
| Follow-up retrieval feels off | Memory rewrite is too weak | Include more of the last user turn |
| Conflicts are missed | Detection logic is too narrow | Add domain-specific conflict rules |
