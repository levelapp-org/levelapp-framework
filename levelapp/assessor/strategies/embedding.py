"""levelapp/assessor/strategies/embedding.py"""
import asyncio
import numpy as np

from typing import List, Dict, Any

from levelapp.assessor.registry import BaseStrategy


class MockEmbeddingStrategy(BaseStrategy):
    """Simulates embedding generation for each chunk."""
    async def run(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(42)
        embeddings = []

        for chunk in chunks:
            vec = rng.standard_normal(768)
            vec /= np.linalg.norm(vec)
            embeddings.append(
                {
                    "text": chunk["text"],
                    "embedding": vec,
                    "parent_id": chunk.get("parent_id"),
                    "source_type": "MockEmbeddingStrategy",
                }
            )

        return embeddings
