"""levelapp/metrics/embedding.py"""
import importlib

from __future__ import annotations
from typing import Any, Dict

from levelapp.core.base import BaseMetric


class EmbeddingMetric(BaseMetric):
    """
    Abstract embedding metric that dynamically delegates to a backend implementation (Torch or Scikit).
    """
    def __init__(self, backend: str | None = None, **kwargs: Any):
        """
        Initialize the embedding metric.

        Args:
            backend (str, optional): Embedding metric backend 'torch' or 'scikit'. Defaults to None.
        """
        super().__init__(processor=kwargs.get("processor"), score_cutoff=kwargs.get("score_cutoff"))
        self.backend_name = backend or self._detect_backend()
        self.backend = self._load_backend(self.backend_name)(**kwargs)

    def _detect_backend(self) -> str:
        """Auto-detect which embedding backend to use."""
        if importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"):
            return "torch"

        elif importlib.util.find_spec("sklearn"):
            return "scikit"

        raise ImportError(
            "No embedding backend available. Install with 'pip install levelapp[embedding]' "
            "for Torch support, or ensure scikit-learn is installed."
        )

    def _load_backend(self, backend: str) :
        if backend == "torch":
            module = importlib.import_module("levelapp.metrics.embedding.torch_base")
            return getattr(module, "TorchEmbeddingMetric")

        elif backend == "scikit":
            module = importlib.import_module("levelapp.metrics.embedding.sentence_transformer")
            return getattr(module, "SentenceEmbeddingMetric")

        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

    def compute(self, generated: str, reference: str) -> Dict[str, Any]:
        """Delegate to selected backend implementation."""
        return self.backend.compute(generated, reference)