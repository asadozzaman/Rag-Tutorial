# Day 11: Expected Output

## 02_code_example.py - Expected Output

Your exact wording may vary slightly, but the flow should look like this:

```text
==================================================================================================
DAY 11: CONVERSATIONAL RAG WITH MEMORY
==================================================================================================
User: Tell me about Python
Standalone query: Tell me about Python
Bot: Python is a general-purpose programming language ...
Sources:
  - Python Overview (python_intro)

User: What web frameworks does it have?
Standalone query: What web frameworks does Python have?
Bot: Python's main web frameworks include Django, Flask, and FastAPI.
Sources:
  - Python Web Frameworks (python_frameworks)

User: Which one is best for APIs?
Standalone query: Which Python web framework is best for APIs?
Bot: FastAPI is usually the best fit for APIs ...
```

---

## 03_mini_project.py - Expected Output

```text
==============================================================================================================
DAY 11 MINI PROJECT: DOCUMENT CHAT ASSISTANT
==============================================================================================================
Session 1
--------------------------------------------------------------------------------------------------------------
User: What about its web frameworks?
Standalone: What web frameworks does Python have?
Answer: Python's main web frameworks include Django, Flask, and FastAPI.
Source count: 1

User: Does it support async?
Standalone: Does FastAPI support async?
Answer: Yes. FastAPI supports async request handlers ...

Clear history
--------------------------------------------------------------------------------------------------------------
History reset complete.

Session 2
--------------------------------------------------------------------------------------------------------------
User: What about its frontend frameworks?
Standalone: What about JavaScript's frontend frameworks?
Answer: JavaScript frontend frameworks include React, Vue, and Angular.
```

---

## Verification Checklist

- [ ] Follow-up questions are reformulated into standalone queries
- [ ] Coreference like `it`, `its`, and `one` is handled
- [ ] Retrieved context is compressed before answering
- [ ] Memory can be cleared between conversations
- [ ] A Streamlit app exists for document chat with upload support

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Follow-up query stays vague | Contextualizer missed the topic | Improve entity detection from recent turns |
| Wrong topic after many turns | Memory is too long or too noisy | Keep recent turns and summarize older ones |
| Answers use irrelevant sentences | Compression is too loose | Keep only sentences that overlap with rewritten query |
| Old topic leaks into a new session | History was not reset | Add a clear-history action |
