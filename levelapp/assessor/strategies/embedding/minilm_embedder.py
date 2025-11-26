"""levelapp/assessor/strategies/embedding/minilm_embedder.py"""
import numpy as np

from typing import List

from levelapp.assessor.strategies.embedding.base import EmbeddingStrategy
from levelapp.assessor.schemas import Document
from levelapp.aspects.logger import logger


class MiniLMEmbedder(EmbeddingStrategy):
    """MiniLM embedding strategy for basic profile."""
    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None
        self._tokenizer = None

    async def _initialize(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

        except ImportError as e:
            raise ImportError(f"[MiniLMEmbedder] requires sentence-transformers") from e

    async def close(self) -> None:
        """Clean up model resources."""
        self._model = None
        self._tokenizer = None

    async def run(self, chunks: List[Document]) -> List[List[float]]:
        """Generate embeddings for document chunks."""
        if not self._model:
            raise RuntimeError(f"[MiniLMEmbedder] Embedding model not initialized. Call initialize() first.")

        if not chunks:
            logger.warning("[MiniLMEmbedder] No chunks to embed.")
            return []

        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]

        try:
            # Generate embeddings
            embeddings = self._model.encode(texts, convert_to_numpy=True)

            # Convert to list of lists for serialization
            embeddings_list = embeddings.tolist()

            return embeddings_list

        except Exception as e:
            raise RuntimeError(f"[MiniLMEmbedder] Failed to generate embeddings: {e}")
