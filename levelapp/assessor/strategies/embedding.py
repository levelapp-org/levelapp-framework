"""levelapp/assessor/strategies/embedding.py"""
import asyncio
import numpy as np

from typing import List, Dict, Any

from levelapp.assessor.registry import BaseStrategy


class MockEmbeddingStrategy(BaseStrategy):
    """Simulates embedding generation for each chunk."""
    async def run(self, chunks: List[str]) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(42)
        embeddings = []
        for idx, chunk in enumerate(chunks):
            vector = rng.normal(size=768)  # deterministic random vector
            embeddings.append(
                {
                    "chunk_id": idx,
                    "text": chunk,
                    "embedding": vector / np.linalg.norm(vector),
                }
            )

        await asyncio.sleep(0)
        return embeddings
