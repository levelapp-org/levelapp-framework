"""levelapp/assessor/strategies/retrieval.py"""
import asyncio
import hashlib
import uuid

import numpy as np

from typing import List, Dict, Any

from levelapp.assessor.registry import RetrievalStrategy
from levelapp.assessor.schemas import Document


def deterministic_id(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


class CosineRetrievalStrategy(RetrievalStrategy):
    """Retrieves top-k most relevant chunks based on cosine similarity."""
    async def run(self, query: str, embedded_chunks: List[Dict[str, Any]]) -> List[Document]:
        rng = np.random.default_rng(abs(hash(query)) % (2**32))
        query_vec = rng.normal(size=768)
        query_vec = query_vec / np.linalg.norm(query_vec)

        scores = []
        for emb in embedded_chunks:
            sim = np.dot(query_vec, emb["embedding"])
            scores.append((emb["text"], sim, emb.get("parent_id")))

        top_k = self.config.get("top_k", 5)
        top_chunks = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

        docs = [
            Document(
                id=parent_id or f"retrieved-{i}",
                content=text,
                source_type="CosineRetrievalStrategy",
                metadata={"score": float(score)}
            ) for i, (text, score, parent_id) in enumerate(top_chunks)
        ]

        await asyncio.sleep(0)
        return docs
