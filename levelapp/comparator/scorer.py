"""'comparator/scorer.py':"""
import numpy as np

from collections import namedtuple
from typing import List, Tuple, Callable, cast, Protocol, Optional, Dict

from rapidfuzz import distance, process, utils, fuzz

from levelapp.comparator.schemas import MetricConfig, EntityMetric, SetMetric
from levelapp.aspects import logger

ComputedScores = namedtuple(
    typename="ComputedScores",
    field_names=["ref", "ext", "e_metric", "e_score"],
)
ComparisonResults = namedtuple(
    typename="ComparisonResults",
    field_names=["ref", "ext", "e_metric", "e_score", "s_metric", "s_score"]
)


class Scorer(Protocol):
    def __call__(self, ref: str, ext: str) -> float:
        ...


class MetricsManager:
    """Manages scorer registration, score computation, metric configuration."""

    def __init__(self, metrics_mapping: Dict[str, MetricConfig] | None = None):
        self._scorers: Dict[str, Callable] = {}
        self._metrics_mapping = metrics_mapping or {}
        self._initialize_scorers()

    @property
    def metrics_mapping(self) -> Dict[str, MetricConfig]:
        return self._metrics_mapping

    @metrics_mapping.setter
    def metrics_mapping(self, value: Dict[str, MetricConfig]):
        self._metrics_mapping = value

    def _initialize_scorers(self) -> None:
        """Register existing scorers to prevent residual state."""
        self._scorers.clear()

        self.register_scorer(
            EntityMetric.LEV_NORM.value,
            distance.Levenshtein.normalized_similarity
        )
        self.register_scorer(
            EntityMetric.JARO_WINKLER.value,
            distance.JaroWinkler.normalized_similarity,
        )
        self.register_scorer(
            EntityMetric.TOKEN_SET_RATIO.value,
            fuzz.token_set_ratio,
        )
        self.register_scorer(
            EntityMetric.TOKEN_SET_RATIO.value,
            fuzz.token_set_ratio,
        )

        self.register_scorer(
            EntityMetric.WRATIO.value,
            fuzz.WRatio
        )

    def register_scorer(self, name: str, scorer: Callable) -> None:
        """
        Register a scorer

        Args:
            name (str): name of the scorer.
            scorer (Callable): scorer to register.

        Raises:
            ValueError: if the scorer is not a callable.
        """
        self._scorers[name] = scorer
        logger.info(f"[MetricsManager] Registered scorer: {name}")

    def get_scorer(self, name: str) -> Callable:
        """
        Retrieve a scorer by name.

        Args:
            name (str): name of the scorer.

        Returns:
            Callable: scorer.

        Raises:
            ValueError: if the passed name is not registered.
        """
        try:
            scorer = self._scorers.get(name)
            logger.info(f"[get_scorer] Retrieved scorer: {name}")
            return scorer

        except KeyError:
            raise ValueError(f"[MetricsManager] '{name}' is not registered")

    def get_metrics_config(self, field: str) -> MetricConfig:
        """
        Retrieve the metrics configuration for a given field.

        Args:
            field (str): field name.

        Returns:
            MetricConfig: metrics configuration for the given field.
        """
        default_config = MetricConfig(
            field_name=field,
            entity_metric=EntityMetric.LEV_NORM,
            set_metric=SetMetric.ACCURACY,
            threshold=1
        )
        return self._metrics_mapping.get(field, default_config)

    def compute_entity_scores(
            self,
            reference_seq: List[str],
            extracted_seq: List[str],
            scorer: EntityMetric = EntityMetric.LEV_NORM,
            pairwise: bool = True
    ) -> List[ComputedScores]:
        """
        Compute the distance/similarity between ref/seq sequence entities.

        Args:
            reference_seq (List[str]): The reference sequence.
            extracted_seq (List[str]): The extracted sequence.
            scorer (str): Name of the scorer to use (e.g., 'levenshtein', 'jaro_winkler').
            pairwise (bool): Whether to use pairwise distances or not.

        Returns:
            List[Tuple[str, str, np.float32]]: List of (reference, extracted, score) tuples.
        """
        if not reference_seq or not extracted_seq:
            return [
                ComputedScores(
                    ref=reference_seq,
                    ext=extracted_seq,
                    e_metric=scorer.value,
                    e_score=np.nan,
                )
            ]

        if scorer not in EntityMetric.list():
            logger.warning(f"[MetricsManager] Scorer name <{scorer}> is not supported.]")
            raise ValueError(f"[MetricsManager] Scorer <{scorer}> is not registered.")

        max_len = max(len(reference_seq), len(extracted_seq))
        reference_padded = reference_seq + [""] * (max_len - len(reference_seq))
        extracted_padded = extracted_seq + [""] * (max_len - len(extracted_seq))

        scorer_func = cast(Callable, self.get_scorer(name=scorer.value))

        if pairwise:
            scores_ = process.cpdist(
                queries=reference_padded,
                choices=extracted_padded,
                scorer=scorer_func,
                processor=utils.default_process,
                workers=-1,
            )
            scores = scores_.flatten()
            res = [
                ComputedScores(
                    ref=reference_padded[i],
                    ext=extracted_padded[i],
                    e_metric=scorer.value,
                    e_score=scores[i]
                ) for i in range(len(scores))
            ]

        else:
            scores_ = process.cdist(
                queries=reference_padded,
                choices=extracted_padded,
                scorer=scorer_func,
                processor=utils.default_process,
                workers=-1,
            )
            scores = np.max(scores_, axis=1)
            max_idx = np.argmax(scores_, axis=1)
            res = [
                ComputedScores(
                    ref=reference_padded[i],
                    ext=extracted_padded[max_idx[i]],
                    e_metric=scorer.value,
                    e_score=scores[i]
                ) for i in range(len(scores))
            ]

        return res

    @staticmethod
    def compute_set_scores(
            data: List[ComputedScores],
            scorer: SetMetric = SetMetric.F1_SCORE,
            threshold: float = 1.0,
    ) -> ComparisonResults:
        """
        Compute evaluation metrics from similarity scores and return results as named tuples.

        Args:
            data: List of tuples containing reference string, extracted string, and similarity score.
            scorer: Metric to compute.
            threshold: Similarity threshold for considering a match.

        Returns:
            List[ComparisonResults]: List of named tuples containing reference, extracted, score, and metric value.
        """
        if not data:
            return ComparisonResults("", "", None, None, None, None)

        ref = [_.ref for _ in data]
        ext = [_.ext for _ in data]
        entity_scores = np.array([_.e_score for _ in data], dtype=np.float32)
        entity_metric = data[0].e_metric

        matches = np.count_nonzero(entity_scores >= threshold)

        if len(data) == 1:
            entity_scores = entity_scores.tolist()
            set_scores = np.array(
                [1 if score >= threshold else 0 for score in entity_scores], dtype=np.float32
            ).tolist()
            return ComparisonResults(
                ref=ref,
                ext=ext,
                e_metric=entity_metric,
                e_score=entity_scores,
                s_metric=None,
                s_score=set_scores
            )

        tp = matches
        fp = len(ref) - int(matches)
        fn = len(ext) - int(matches)

        if scorer == SetMetric.ACCURACY:
            accuracy = (tp / len(entity_scores)) if len(entity_scores) > 0 else 0.0
            return ComparisonResults(
                ref=ref,
                ext=ext,
                e_metric=entity_metric,
                e_score=entity_scores,
                s_metric=scorer.value,
                s_score=accuracy
            )

        if scorer == SetMetric.F1_SCORE:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            return ComparisonResults(
                ref=ref,
                ext=ext,
                e_metric=entity_metric,
                e_score=entity_scores,
                s_metric=scorer.value,
                s_score=f1
            )
