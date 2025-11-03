"""levelapp/metrics/embeddings/sentence_transformer.py"""
import numpy as np

from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from levelapp.core.base import BaseMetric


class SentenceEmbeddingMetric(BaseMetric):
    """Lightweight embeddings similarity using TF-IDF cosine similarity."""
    def __init__(self, **kwargs):
        super().__init__(processor=kwargs.get("processor"), score_cutoff=kwargs.get("score_cutoff"))
        self.vectorizer = TfidfVectorizer()

    def compute(self, generated: str, reference: str) -> Dict[str, Any]:
        self._validate_inputs(generated=generated, reference=reference)

        corpus = [reference, generated]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        # clamping for numerical stability
        similarity = float(np.clip(similarity, 0.0, 1.0))

        return {
            "similarity": similarity,
            "metadata": self._build_metadata(backend="scikit", vectorizer="TF-IDF"),
        }
