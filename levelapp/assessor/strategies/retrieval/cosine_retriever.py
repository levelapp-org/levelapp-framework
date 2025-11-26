"""levelapp/assessor/strategies/retrieval/cosine_retriever.py"""
import numpy as np

from typing import List

from levelapp.assessor.strategies.retrieval.base import RetrievalStrategy
from levelapp.assessor.schemas import Document


class CosineRetriever(RetrievalStrategy):
    """Cosine similarity-based retriever for basic profile."""
    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.top_k = self.config.get("top_k", 5)
        self._documents = []
        self._embeddings = None

        if self.top_k < 1 or self.top_k > 100:
            raise ValueError("[CosineRetriever] top_k must be between 1 and 100]")

    async def _initialize(self) -> None:
        """Initialize the retriever (no heavy setup needed)."""
        self._documents = []
        self._embeddings = None

    async def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the retriever."""
        if len(documents) != len(embeddings):
            raise ValueError("[CosineRetriever] Number of documents and embeddings do not match.")

        self._documents = documents
        self._embeddings = embeddings

        # Normalize embeddings for cosine similarity
        if len(self._embeddings) > 0:
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            self._embeddings = self._embeddings / norms

    async def run(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve most relevant documents using cosine similarity."""
        if top_k is None:
            top_k = self.top_k

        if not self._documents or self._embeddings is None:
            return None

        # To be replaced with generating query embeddings and compute cosine similarity.
        query_lower = query.lower()
        scored_documents = []

        for i, doc in enumerate(self._documents):
            # Simple keyword matching score
            doc_text_lower = doc.content.lower()
            score = sum(1 for word in query_lower.split() if word in doc_text_lower)

            # Normalize score by query length
            if score > 0:
                score = score / len(query_lower.split())

            scored_documents.append((score, doc))

        # Sort by score descending and take top_k
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        top_documents = [doc for score, doc in scored_documents[:top_k] if score > 0]

        return top_documents

    async def _cleanup(self) -> None:
        """Clean up retriever resources."""
        self._documents = []
        self._embeddings = None
