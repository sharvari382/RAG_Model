import os
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np

from .config import settings


class LocalVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: List[Dict] = []

        if os.path.exists(settings.faiss_index_path) and os.path.exists(settings.metadata_path):
            self._load()

    def _load(self):
        self.index = faiss.read_index(settings.faiss_index_path)
        with open(settings.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def _persist(self):
        faiss.write_index(self.index, settings.faiss_index_path)
        with open(settings.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadatas)
        self._persist()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[float, Dict]]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        results: List[Tuple[float, Dict]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            score = float(1.0 / (1.0 + dist))  # similarity proxy
            results.append((score, self.metadata[idx]))
        return results


vector_store = LocalVectorStore()
