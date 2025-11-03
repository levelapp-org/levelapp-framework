import pytest
import importlib

from levelapp.metrics.embedding import EmbeddingMetric


@pytest.mark.parametrize("backend", ["torch", "scikit"])
def test_explicit_backend_loading(monkeypatch, backend):
    """Ensure explicit backend loads correctly and returns a similarity score."""
    metric = EmbeddingMetric(backend=backend)
    result = metric.compute(
        "woke up this morning got yourself a gun",
        "woke up this morning got yourself a gun"
    )
    assert "similarity" in result
    assert 0.0 <= result["similarity"] <= 1.0
    assert "metadata" in result
    assert result["metadata"]["inputs"]


def test_auto_backend_detection(monkeypatch):
    """Ensure backend auto-detection prefers Torch if available."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: True if name in ("torch", "transformers") else None)
    metric = EmbeddingMetric()
    assert metric.backend_name == "torch"


def test_scikit_fallback(monkeypatch):
    """Ensure fallback to scikit when torch not available."""
    def mock_find_spec(name):
        if name in ("torch", "transformers"):
            return None

        if name == "sklearn":
            return True

        return None

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)
    metric = EmbeddingMetric()
    assert metric.backend_name == "scikit"


def test_no_backend_available(monkeypatch):
    """Ensure clear ImportError when neither backend is available."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: False)
    with pytest.raises(ImportError):
        EmbeddingMetric()
