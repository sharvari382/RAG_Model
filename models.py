from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class UploadResponse(BaseModel):
    doc_id: str
    status: DocumentStatus


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=512)
    top_k: int = Field(5, ge=1, le=10)
    doc_ids: Optional[List[str]] = None


class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: List[str]
    scores: List[float]
    latency_ms: float
