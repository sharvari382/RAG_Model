import time
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .config import settings
from .models import UploadResponse, QueryRequest, QueryResponse, DocumentStatus
from .document_processor import generate_doc_id
from .rag_pipeline import pipeline
# TEMP: synchronous ingestion for local testing (no Redis/Celery needed)
def ingest_document_task(
    doc_id: str,
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    from .document_processor import read_pdf_bytes, read_txt_bytes, simple_chunk
    from .rag_pipeline import pipeline

    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        text = read_pdf_bytes(file_bytes)
    elif ext == "txt":
        text = read_txt_bytes(file_bytes)
    else:
        raise ValueError("Unsupported file type")

    chunks = simple_chunk(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pipeline.index_document(doc_id, chunks)
    return {"doc_id": doc_id, "chunks_indexed": len(chunks)}


app = FastAPI(title="RAG-Based Question Answering System")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

DOC_STATUS: Dict[str, DocumentStatus] = {}


def get_status(doc_id: str) -> DocumentStatus:
    return DOC_STATUS.get(doc_id, DocumentStatus.PROCESSING)


@app.post("/upload", response_model=UploadResponse)
@limiter.limit(settings.rate_limit_upload)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    ext = file.filename.split(".")[-1].lower()
    if ext not in {"pdf", "txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT are supported.")
    if chunk_size <= 0 or chunk_overlap < 0:
        raise HTTPException(status_code=400, detail="Invalid chunk parameters.")

    doc_id = generate_doc_id()
    content = await file.read()

    DOC_STATUS[doc_id] = DocumentStatus.PROCESSING
   ingest_document_task(doc_id, content, file.filename, chunk_size, chunk_overlap)

    return UploadResponse(doc_id=doc_id, status=DocumentStatus.PROCESSING)


@app.get("/status/{doc_id}", response_model=UploadResponse)
async def document_status(doc_id: str):
    status = get_status(doc_id)
    return UploadResponse(doc_id=doc_id, status=status)


@app.post("/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit_query)
async def query_documents(request: Request, payload: QueryRequest):
    t0 = time.time()
    result = pipeline.answer(
        question=payload.question,
        top_k=payload.top_k,
        doc_ids=payload.doc_ids,
    )

    return QueryResponse(
        answer=result["answer"],
        relevant_chunks=result["chunks"],
        scores=result["scores"],
        latency_ms=result["latency_ms"],
    )
