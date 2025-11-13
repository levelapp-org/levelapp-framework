"""levelapp/assessor/strategies/baseline/embedding.py"""
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from levelapp.assessor.registry import BaseStrategy


class BaselineEmbeddingStrategy(BaseStrategy):
    """Compute embeddings using SentenceTransformer."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name="SentenceTransformer", config=config)
        model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)

    async def run(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        return [
            {**chunk, "embedding": emb}
            for chunk, emb in zip(chunks, embeddings)
        ]