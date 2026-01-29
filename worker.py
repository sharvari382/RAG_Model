import os
from celery import Celery

from app.document_processor import (
    read_pdf_bytes,
    read_txt_bytes,
    simple_chunk,
)
from app.rag_pipeline import pipeline

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("rag_worker", broker=REDIS_URL, backend=REDIS_URL)


@celery_app.task
def ingest_document_task(
    doc_id: str,
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
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
