"""levelapp/assessor/strategies/baseline/retriever.py"""
import numpy as np

from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from levelapp.assessor.schemas import Document
from levelapp.assessor.registry import RetrievalStrategy


class BaselineRetrievalStrategy(RetrievalStrategy):
    """Retrieve top-k most relevant chunks by cosine similarity."""

    async def run(self, query: str, embedded_chunks: List[Dict[str, Any]]) -> List[Document]:
        model = SentenceTransformer(self.config.get("embedder", "sentence-transformers/all-MiniLM-L6-v2"))
        query_vec = np.array(model.encode(query))

        # TODO-0: For the sake of not going insane, create some DTO to ensure that the data is transferred properly
        # TODO-0: between the different levels.
        embeddings = np.array([c["embedding"] for c in embedded_chunks])
        similarities = embeddings @ query_vec

        top_k = self.config.get("top_k", 5)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            Document(
                id=embedded_chunks[i]["parent_id"],
                content=embedded_chunks[i]["text"],
                source="BaselineRetrievalStrategy",
                metadata={"score": float(similarities[i])}
            )
            for i in top_indices
        ]
