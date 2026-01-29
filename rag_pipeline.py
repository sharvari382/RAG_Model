import time
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from .config import settings
from .vector_store import vector_store


class RAGPipeline:
    def __init__(self):
        # all-MiniLM-L6-v2 -> 384-dim embeddings, fast and good for semantic search.
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embedder.encode(texts, normalize_embeddings=True))

    def index_document(self, doc_id: str, chunks: List[str]):
        embeddings = self.embed_texts(chunks)
        metas = [{"doc_id": doc_id, "text": c} for c in chunks]
        vector_store.add(embeddings, metas)

    def retrieve(self, question: str, top_k: int = 5, doc_ids: List[str] | None = None) -> List[Tuple[float, str]]:
        q_emb = self.embed_texts([question])
        raw_results = vector_store.search(q_emb, k=top_k * 2)
        filtered: List[Tuple[float, str]] = []
        for score, meta in raw_results:
            if doc_ids and meta.get("doc_id") not in doc_ids:
                continue
            filtered.append((score, meta["text"]))
            if len(filtered) >= top_k:
                break
        return filtered

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        joined_context = "\n\n".join(contexts)
        prompt = (
            "You are a precise assistant.\n"
            "Answer the question using ONLY the context.\n"
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{joined_context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def answer(self, question: str, top_k: int = 5, doc_ids: List[str] | None = None):
        t0 = time.time()
        retrieved = self.retrieve(question, top_k=top_k, doc_ids=doc_ids)
        chunks = [t for _, t in retrieved]
        scores = [float(s) for s, _ in retrieved]
        if not chunks:
            return {
                "answer": "I could not find any relevant information in the indexed documents.",
                "chunks": [],
                "scores": [],
                "latency_ms": (time.time() - t0) * 1000,
            }
        answer = self.generate_answer(question, chunks)
        latency_ms = (time.time() - t0) * 1000
        return {
            "answer": answer,
            "chunks": chunks,
            "scores": scores,
            "latency_ms": latency_ms,
        }


pipeline = RAGPipeline()
