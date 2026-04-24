from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from capstone_rag import CapstoneRAG


ROOT = Path(__file__).resolve().parent
DOCS_PATH = ROOT / "sample_data" / "docs.csv"
EVAL_PATH = ROOT / "sample_data" / "eval_queries.csv"
rag: CapstoneRAG | None = None


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    use_cache: bool = True


class UploadRequest(BaseModel):
    filename: str
    content: str
    doc_type: str = "markdown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = CapstoneRAG()
    rag.ingest_csv(DOCS_PATH)
    yield


app = FastAPI(title="Day 15 Capstone RAG", lifespan=lifespan)


def get_rag() -> CapstoneRAG:
    if rag is None:
        raise RuntimeError("RAG system is not initialized")
    return rag


@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = await get_rag().query(
            question=request.question,
            session_id=request.session_id,
            use_cache=request.use_cache,
        )
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "cached": result.cached,
            "cache_type": result.cache_type,
            "latency_ms": round(result.latency_ms, 2),
            "citations": [citation.__dict__ for citation in result.citations],
            "trace": result.trace.__dict__,
            "token_count": result.token_count,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/query/stream")
async def stream(question: str, session_id: str = "default", use_cache: bool = True):
    return StreamingResponse(
        get_rag().stream_query(question=question, session_id=session_id, use_cache=use_cache),
        media_type="text/event-stream",
    )


@app.post("/upload")
async def upload(request: UploadRequest):
    try:
        count = get_rag().upload_text(request.filename, request.content, request.doc_type)
        return {"message": f"Uploaded {request.filename}", "chunks_added": count, "metrics": get_rag().metrics_snapshot()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
async def health():
    return {"status": "healthy", **get_rag().metrics_snapshot()}


@app.get("/evaluate")
async def evaluate():
    return get_rag().evaluate(EVAL_PATH)


@app.get("/metrics")
async def metrics():
    return get_rag().metrics_snapshot()
