from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from production_rag import ProductionRAGService


ROOT = Path(__file__).resolve().parent
SAMPLE_DOCS = ROOT / "sample_data" / "docs.csv"
service: ProductionRAGService | None = None


class QueryRequest(BaseModel):
    query: str
    user_id: str = "demo"
    top_k: int = 3
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    cached: bool
    cache_type: str
    token_count: int
    trace_id: str


class UploadResponse(BaseModel):
    message: str
    chunks_added: int
    documents: int


class UploadRequest(BaseModel):
    filename: str
    content: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = ProductionRAGService.from_sample_file(SAMPLE_DOCS)
    yield


app = FastAPI(title="Production RAG API", lifespan=lifespan)


def get_service() -> ProductionRAGService:
    if service is None:
        raise RuntimeError("Service not initialized")
    return service


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        result = await get_service().query(
            query=request.query,
            user_id=request.user_id,
            top_k=request.top_k,
            use_cache=request.use_cache,
        )
        return QueryResponse(**result.__dict__)
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.get("/query/stream")
async def query_stream(query: str, user_id: str = "demo", top_k: int = 3, use_cache: bool = True):
    try:
        generator = get_service().stream_query(query=query, user_id=user_id, top_k=top_k, use_cache=use_cache)
        return StreamingResponse(generator, media_type="text/event-stream")
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.post("/upload", response_model=UploadResponse)
async def upload_document(request: UploadRequest):
    try:
        added = get_service().upload_text(request.filename, request.content)
        health = get_service().health()
        return UploadResponse(
            message=f"Uploaded {request.filename}",
            chunks_added=added,
            documents=int(health["documents"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload failed: {exc}") from exc


@app.get("/health")
async def health():
    return get_service().health()
