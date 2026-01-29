from typing import List, Dict
import uuid
from pypdf import PdfReader
from io import BytesIO


def read_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)


def read_txt_bytes(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def simple_chunk(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    Chunk by words with overlap.
    512-word chunks balance semantic coherence and retrieval granularity.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap
    return chunks


def build_chunk_metadata(doc_id: str, chunks: List[str]) -> List[Dict]:
    metas = []
    for i, ch in enumerate(chunks):
        metas.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{i}",
                "text": ch,
            }
        )
    return metas


def generate_doc_id() -> str:
    return str(uuid.uuid4())
