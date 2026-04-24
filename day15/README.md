# Day 15 Capstone: Enterprise Document Intelligence Platform

This folder is the final capstone for the 15-day RAG roadmap.

It combines:

- smart chunking and ingestion
- hybrid retrieval and reranking
- session memory and history-aware query rewriting
- CRAG-style corrective branching
- structured answers with citations and confidence
- evaluation, metrics, FastAPI, streaming, and Docker

## Main Files

- `capstone_rag.py`: core orchestration logic
- `day15_app.py`: FastAPI service
- `02_code_example.py`: direct core-module demo
- `03_mini_project.py`: end-to-end API demo via test client
- `ARCHITECTURE.md`: system design notes
- `learning_journal.md`: 15-day summary

## Quick Start

```powershell
cd "C:\Users\asado\OneDrive\Desktop\Rag Learning\day15"
python 02_code_example.py
python 03_mini_project.py
```

To run the API locally:

```powershell
uvicorn day15_app:app --reload
```

## Endpoints

- `POST /query`
- `GET /query/stream`
- `POST /upload`
- `GET /health`
- `GET /metrics`
- `GET /evaluate`

## Design Notes

- Retrieval is local and deterministic so the capstone remains runnable.
- Hybrid retrieval combines semantic-style overlap with keyword matching.
- Reranking narrows the final context to a smaller, higher-precision set.
- Session memory rewrites vague follow-ups into standalone queries.
- CRAG-style logic refines weak retrieval or falls back safely.

## Why This Matters

This capstone is meant to be a portfolio-ready summary of the roadmap:

- simple enough to run and inspect
- broad enough to show end-to-end system design
- structured enough to extend into a real project later
