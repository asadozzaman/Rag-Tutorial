# RAG Learning Progress

This repository tracks a full 15-day Retrieval-Augmented Generation learning journey based on the roadmap in `rag-mastery-roadmap.jsx`.

All day folders from `day1` through `day15` are present in the workspace and represent the current learning progress.

## Overall Status

- Status: `Day 1` to `Day 15` completed
- Roadmap source: `rag-mastery-roadmap.jsx`
- Structure: each day is organized as a self-contained learning module

## Standard Day Structure

Most day folders include:

- `01_explanation.md`: concept walkthrough
- `02_code_example.py`: focused lesson example
- `03_mini_project.py`: hands-on build
- `04_expected_output.md`: output reference
- helper utilities such as `*_utils.py` or app/service modules
- `sample_data/`: small local datasets used by the lesson

## Day-by-Day Progress

| Day | Topic | Status | Main Outcome |
|---|---|---|---|
| 1 | RAG Foundations & Mental Model | Complete | Core RAG architecture and first mental model |
| 2 | Document Loading & Chunking Mastery | Complete | Multi-format loading and chunking strategy comparison |
| 3 | Embeddings Deep Dive | Complete | Embedding benchmarking and semantic retrieval foundations |
| 4 | Vector Databases & Indexing | Complete | Indexing, persistence, and hybrid search basics |
| 5 | Retrieval Strategies & Reranking | Complete | Ensemble retrieval, MMR, reranking, and retrieval evaluation |
| 6 | Prompt Engineering for RAG | Complete | Grounded prompting, citations, and safer answer generation |
| 7 | LlamaIndex Deep Dive | Complete | LlamaIndex-style indices, query engines, and routing |
| 8 | Evaluation & Metrics for RAG | Complete | Retrieval metrics, generation metrics, and hallucination checks |
| 9 | Advanced RAG: Query Transformation & Routing | Complete | HyDE, multi-query, step-back prompting, and query routing |
| 10 | Agentic RAG & Tool Use | Complete | Tool-using retrieval agents with calculations and multi-hop flow |
| 11 | Conversational RAG & Memory | Complete | History-aware retrieval, memory, and multi-turn chat behavior |
| 12 | Multimodal RAG & Structured Data | Complete | Text-to-SQL, DataFrame Q&A, and hybrid text + table routing |
| 13 | Production Deployment & Optimization | Complete | FastAPI service, caching, streaming, health checks, and Docker |
| 14 | Advanced Patterns: GraphRAG, RAPTOR & CRAG | Complete | Graph retrieval, hierarchical summaries, and self-correcting RAG |
| 15 | Capstone: End-to-End Production RAG System | Complete | Portfolio-style full-stack RAG capstone with docs and deployment |

## Folder Highlights

### `day1` to `day5`

These folders build the RAG core:

- foundations
- chunking
- embeddings
- indexing
- retrieval quality

### `day6` to `day10`

These folders expand the query and answer pipeline:

- prompting
- framework abstractions
- evaluation
- query transformation
- agentic tool use

### `day11` to `day15`

These folders focus on advanced and production-style systems:

- conversation memory
- structured and multimodal retrieval
- deployment and observability
- advanced corrective patterns
- full capstone integration

## Final Capstone

The most complete build is in `day15`.

It combines:

- smart ingestion
- hybrid retrieval
- reranking
- session memory
- CRAG-style correction
- evaluation
- FastAPI endpoints
- Docker configuration
- architecture and learning documentation

Important capstone files:

- `day15/capstone_rag.py`
- `day15/day15_app.py`
- `day15/README.md`
- `day15/ARCHITECTURE.md`
- `day15/learning_journal.md`

## Recommended Reading Order

If you want to revisit the full path:

1. Start with `day1`
2. Move sequentially through `day15`
3. For each day, read `01_explanation.md` first
4. Run `02_code_example.py`
5. Finish with `03_mini_project.py`

## Current Review Notes

After reviewing the folder structure:

- all roadmap day folders from `day1` to `day15` exist
- the repository is organized consistently around the roadmap
- the final capstone documentation already lives inside `day15`
- this root README now provides a single progress overview across the whole journey
