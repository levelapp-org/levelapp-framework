"""levelapp/metrics/rag/retrieval.py"""
import math

from typing import List, Dict, Any

from levelapp.core.base import BaseMetric


def _docs_ids(docs: List[Dict[str, Any]]) -> List[str]:
    """Extract document IDs from list of dicts or Pydantic models."""
    return [d["id"] if isinstance(d, dict) else getattr(d, "id", None) for d in docs]


class PrecisionMetric(BaseMetric):
    """Precision@k: relevant / retrieved."""

    def compute(self, expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> Dict[str, Any]:
        expected_ids = set(_docs_ids(docs=expected))
        actual_ids = set(_docs_ids(docs=actual))

        if not actual_ids:
            score = 0.0
        else:
            score = float(len(expected_ids.intersection(actual_ids)) / len(actual_ids)) if actual_ids else 0.0

        return {
            "score": score,
            "metadata": {
                "num_expected": len(expected_ids),
                "num_actual": len(actual_ids),
            }
        }


class RecallMetric(BaseMetric):
    """Recall@k: relevant retrieved/ total relevant."""

    def compute(self, expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> Dict[str, Any]:
        expected_ids = set(_docs_ids(docs=expected))
        actual_ids = set(_docs_ids(docs=actual))

        if not actual_ids:
            score = 0.0
        else:
            score = float(len(expected_ids.intersection(actual_ids)) / len(expected_ids)) if expected_ids else 0.0

        return {
            "score": score,
            "metadata": {
                "num_expected": len(expected_ids),
                "num_actual": len(actual_ids),
            }
        }


class NDCGMetric(BaseMetric):
    """Normalize Discounted Cumulative Gain."""

    def compute(self, expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> Dict[str, Any]:
        expected_ids = set(_docs_ids(docs=expected))
        actual_ids = set(_docs_ids(docs=actual))

        relevance = {doc_id: 1.0 - (i / len(expected_ids)) for i, doc_id in enumerate(expected_ids)}
        dcg = sum(relevance.get(doc_id, 0) / math.log2(i + 2) for i, doc_id in enumerate(actual_ids))
        ideal_dcg = sum(relevance[doc_id] / math.log2(i + 2) for i, doc_id in enumerate(expected_ids))

        score = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        return {
            "score": score,
            "metadata": {
                "num_expected": len(expected_ids),
                "num_actual": len(actual_ids),
            }
        }


class RedundancyMetric(BaseMetric):
    """Fraction of duplicate documents in retrieval results."""

    def compute(self, expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> Dict[str, Any]:
        seen = set()
        redundant = 0

        for doc in actual:
            content = doc["content"] if isinstance(doc, dict) else getattr(doc, "content", "")

            if content in seen:
                redundant += 1
            else:
                seen.add(content)

        score = redundant / max(1, len(actual))

        return {
            "score": score,
            "metadata": {
                "redundant": redundant,
                "total": len(actual),
            }
        }


RAG_RETRIEVAL_METRICS = {
    "precision": PrecisionMetric,
    "recall": RecallMetric,
    "ndcg": NDCGMetric,
    "redundancy": RedundancyMetric,
}