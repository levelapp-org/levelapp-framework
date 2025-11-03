"""levelapp/metrics/embeddings.py"""
from __future__ import annotations

import importlib

from importlib import util
from typing import Any, Dict

from levelapp.core.base import BaseMetric


class EmbeddingMetric(BaseMetric):
    """
    Abstract embeddings metric that dynamically delegates to a backend implementation (Torch or Scikit).
    """
    def __init__(self, backend: str | None = None, **kwargs: Any):
        """
        Initialize the embeddings metric.

        Args:
            backend (str, optional): Embedding metric backend 'torch' or 'scikit'. Defaults to None.
        """
        super().__init__(processor=kwargs.get("processor"), score_cutoff=kwargs.get("score_cutoff"))
        self.backend_name = backend or self._detect_backend()
        self.backend = self._load_backend(self.backend_name)(**kwargs)

    @staticmethod
    def _detect_backend() -> str:
        """Auto-detect which embeddings backend to use."""
        if util.find_spec("torch") and util.find_spec("transformers"):
            return "torch"

        elif util.find_spec("sklearn"):
            return "scikit"

        raise ImportError(
            "No embeddings backend available. Install with 'pip install levelapp[embeddings]' "
            "for Torch support, or ensure scikit-learn is installed."
        )

    @staticmethod
    def _load_backend(backend: str):
        if backend == "torch":
            module = importlib.import_module("levelapp.metrics.embeddings.torch_based")
            return getattr(module, "TorchEmbeddingMetric")

        elif backend == "scikit":
            module = importlib.import_module("levelapp.metrics.embeddings.sentence_transformer")
            return getattr(module, "SentenceEmbeddingMetric")

        else:
            raise ValueError(f"Unknown embeddings backend: {backend}")

    def compute(self, generated: str, reference: str) -> Dict[str, Any]:
        """Delegate to selected backend implementation."""
        return self.backend.compute(generated, reference)
