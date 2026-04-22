# Day 3: Expected Output

## 02_code_example.py - Expected Output

Your exact similarity scores will differ, especially for Gemini embeddings.

```text
==================================================================
DAY 3: EMBEDDING MODEL SIMILARITY COMPARISON
==================================================================

Query: How can I build a RAG pipeline?

=== keyword-overlap-baseline ===
Dimensions: 31
Embedding latency: 0.2 ms
   0.3536 | Building RAG systems with chunking embeddings and vector search
   0.0000 | Step-by-step guide to retrieval augmented generation
   0.0000 | Best pizza restaurants in New York City
   0.0000 | Introduction to neural network training loops
   0.0000 | How to prepare strong coffee at home

=== hashing-baseline ===
Dimensions: 256
Embedding latency: 0.2 ms
   0.2500 | Building RAG systems with chunking embeddings and vector search
   0.1250 | Step-by-step guide to retrieval augmented generation
   0.0000 | Best pizza restaurants in New York City
  -0.1250 | Introduction to neural network training loops
  -0.1250 | How to prepare strong coffee at home

=== gemini-embedding-001 ===
Dimensions: 768
Embedding latency: 850.0 ms
   0.7400 | Building RAG systems with chunking embeddings and vector search
   0.7100 | Step-by-step guide to retrieval augmented generation
   0.5100 | Introduction to neural network training loops
   0.4200 | How to prepare strong coffee at home
   0.3900 | Best pizza restaurants in New York City

Takeaway:
  The best embedding model should rank the RAG-related documents highest.
  Local baselines are useful for learning, but real semantic models usually win.
```

If you do not have `GEMINI_API_KEY`, the script skips Gemini and still runs the local baselines.

---

## 03_mini_project.py - Expected Output

```text
======================================================================================
DAY 3 MINI PROJECT: EMBEDDING BENCHMARK SUITE
======================================================================================
Queries: 6
Documents: 12
Models: keyword-overlap-baseline, hashing-baseline, gemini-embedding-001

Top results for keyword-overlap-baseline:
  q1: How do I reduce hallucination in a RAG app?
    top1=d1 score=0.4082 text=RAG reduces hallucination by grounding answers in retrieved context...
  ...

Top results for hashing-baseline:
  q1: How do I reduce hallucination in a RAG app?
    top1=d1 score=0.3651 text=RAG reduces hallucination by grounding answers in retrieved context...
  ...

Top results for gemini-embedding-001:
  q1: How do I reduce hallucination in a RAG app?
    top1=d1 score=0.8123 text=RAG reduces hallucination by grounding answers in retrieved context...
  ...

======================================================================================
EMBEDDING BENCHMARK RESULTS
======================================================================================
Model                            Dims      R@1      R@3      MRR   Latency ms
--------------------------------------------------------------------------------------
keyword-overlap-baseline          120     0.67     1.00     0.78          0.3
hashing-baseline                  256     0.67     1.00     0.78          0.4
gemini-embedding-001              768     0.83     1.00     0.92       1200.0

Recommendation:
  Start with gemini-embedding-001. It has the strongest balance of Recall@3,
  MRR, and latency on this dataset.

Production reminder:
  If you change embedding models, rebuild the whole vector index.
```

---

## Verification Checklist

- [ ] `02_code_example.py` prints similarity rankings for each model
- [ ] RAG-related documents rank higher than unrelated documents
- [ ] `03_mini_project.py` loads the labeled dataset
- [ ] Recall@1, Recall@3, MRR, latency, and dimensions are printed
- [ ] The recommendation is based on measured results
- [ ] The scripts still run if Gemini is unavailable

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `No GEMINI_API_KEY found` | API key is not loaded | Add `GEMINI_API_KEY` to `.env`, or use local baselines only |
| `ModuleNotFoundError: langchain_google_genai` | Missing package | `pip install -r requirements.txt` |
| Low Recall@1 | Dataset is hard or model is weak | Check top 3 and inspect query-document wording |
| High latency | Remote embedding API call | Batch embeddings and cache results in production |
| You changed models but old FAISS index remains | Embeddings are not portable | Rebuild the vector index from original chunks |
