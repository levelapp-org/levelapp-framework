"""tests/metrics/test_sentence_transformer.py"""
import pytest
from levelapp.metrics.embeddings.sentence_transformer import SentenceEmbeddingMetric


@pytest.fixture
def scikit_metric():
    return SentenceEmbeddingMetric()


def test_tfidf_similarity_self(scikit_metric):
    """Self-similarity must be near 1."""
    result = scikit_metric.compute("Gabagool?! Over here!", "Gabagool?! Over here!")
    assert result["similarity"] > 0.95


def test_tfidf_similarity_diff(scikit_metric):
    """Different texts should yield lower similarity."""
    result = scikit_metric.compute("Gabagool?! Over here!", "All this from a slice of Gabagool?!")
    assert result["similarity"] < 0.8


def test_metadata_integrity(scikit_metric):
    """Ensure consistent metadata dictionary."""
    result = scikit_metric.compute("Ohhh!", "Ohhh!")
    metadata = result["metadata"]
    assert metadata["type"] == "SentenceEmbeddingMetric"
    assert "timestamp" in metadata
