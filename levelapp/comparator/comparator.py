"""'comparator/service.py':"""
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple, Literal

from pydantic import BaseModel

from levelapp.comparator.extractor import DataExtractor
from levelapp.core.base import BaseProcess
from levelapp.comparator.scorer import MetricsManager, ComparisonResults
from levelapp.comparator.schemas import EntityMetric, SetMetric, MetricConfig
from levelapp.comparator.utils import format_evaluation_results


class MetadataComparator(BaseProcess):
    """Metadata comparator component."""

    def __init__(
            self,
            reference: BaseModel | None = None,
            generated: BaseModel | None = None,
            metrics_manager: MetricsManager | None = None,
    ):
        """
        Initialize the MetadataComparator.

        Args:
            reference (BaseModel): Reference BaseModel
            generated (BaseModel): Extracted BaseModel
            metrics_manager (MetricsManager): MetricsManager
        """
        self.extractor = DataExtractor()

        self._reference = reference
        self._generated = generated
        self._metrics_manager = metrics_manager

        self._evaluation_data: List[
            Tuple[str, list[str], list[str], Any, Any, Any, Any, float]
        ] = []

    @property
    def reference_data(self) -> BaseModel:
        return self._reference

    @property
    def generated_data(self) -> BaseModel:
        return self._generated

    @property
    def metrics_manager(self) -> MetricsManager:
        return self._metrics_manager

    @reference_data.setter
    def reference_data(self, value: BaseModel):
        self._reference = value

    @generated_data.setter
    def generated_data(self, value: BaseModel):
        self._generated = value

    @metrics_manager.setter
    def metrics_manager(self, value: MetricsManager):
        self._metrics_manager = value

    def _get_score(self, field: str) -> Tuple[EntityMetric, SetMetric, float]:
        """
        Retrieve the scoring metric and threshold for a given field.

        Args:
            field: The field for which to retrieve the metric and threshold.

        Returns:
            A tuple containing the scoring metric and its threshold.
        """
        if self._metrics_manager:
            config = self._metrics_manager.get_metrics_config(field=field)
        else:
            config = MetricConfig()

        return config.entity_metric, config.set_metric, config.threshold

    def _format_results(
            self,
            output_type: Literal["json", "csv"] = "json"
    ) -> Dict[int, Any]:
        """
        Format the internal evaluation data for reporting or storage.

        Args:
            output_type: 'json' returns a list of dictionaries; 'csv' returns a DataFrame.

        Returns:
            Formatted evaluation results or None if no data.
        """
        formatted_results = format_evaluation_results(self._evaluation_data, output_type=output_type)

        return dict(enumerate(formatted_results))

    def evaluate(
            self,
            reference_list: List[str],
            extracted_list: List[str],
            entity_metric: EntityMetric,
            set_metric: SetMetric,
            threshold: float,
    ) -> ComparisonResults:
        """
        Evaluates pairwise similarity between elements in two lists using fuzzy matching.

        Args:
            reference_list: Ground-truth list of strings.
            extracted_list: Extracted list of strings to compare.
            entity_metric (EntityMetric): entity-level comparison metric.
            set_metric (SetMetric): set-level comparison metric.
            threshold: Similarity threshold (0–100) for considering a match.

        Returns:
            A dict with accuracy, precision, recall, and F1-score.
        """
        if not (reference_list or extracted_list):
            return ComparisonResults("", "", entity_metric.value, None, set_metric.value, None)

        scores = self._metrics_manager.compute_entity_scores(
            reference_seq=reference_list,
            extracted_seq=extracted_list,
            scorer=entity_metric,
            pairwise=False
        )

        return self._metrics_manager.compute_set_scores(
            data=scores,
            scorer=set_metric,
            threshold=threshold,
        )

    def _recursive_compare(
        self,
        ref_node: Any,
        ext_node: Any,
        results: Dict[str, Dict[str, float]],
        prefix: str = "",
        threshold: float = 99.0,
    ) -> None:
        """
        Recursively compare extracted vs. reference metadata nodes.

        Args:
            ref_node: dict or list (from deep_extract reference metadata)
            ext_node: dict or list (from deep_extract extracted metadata)
            results: Dict to accumulate comp_results keyed by hierarchical attribute paths.
            prefix: str, current path prefix to form hierarchical keys.
        """
        # Case 1: Both nodes are dicts -> recurse on keys
        if isinstance(ref_node, Mapping) and isinstance(ext_node, Mapping):
            all_keys = set(ref_node.keys()) | set(ext_node.keys())
            for key in all_keys:
                new_prefix = f"{prefix}.{key}" if prefix else key
                ref_subnode = ref_node.get(key, [])
                ext_subnode = ext_node.get(key, [])
                self._recursive_compare(
                    ref_node=ref_subnode,
                    ext_node=ext_subnode,
                    results=results,
                    prefix=new_prefix,
                    threshold=threshold,
                )

        # Case 2: Leaf nodes (lists) -> evaluate directly
        else:
            # Defensive: convert to list if not list
            ref_list = ref_node if isinstance(ref_node, list) else [ref_node]
            ext_list = ext_node if isinstance(ext_node, list) else [ext_node]

            # Convert all to strings for consistent fuzzy matching
            ref_list_str = list(map(str, ref_list))
            ext_list_str = list(map(str, ext_list))

            entity_metric_, set_metric_, threshold = self._get_score(field=prefix)

            # Evaluate similarity metrics
            comp_results = self.evaluate(
                reference_list=ref_list_str,
                extracted_list=ext_list_str,
                entity_metric=entity_metric_,
                set_metric=set_metric_,
                threshold=threshold,
            )

            if comp_results:
                self._evaluation_data.append(
                    (
                        prefix,
                        ref_list_str,
                        ext_list_str,
                        comp_results.e_metric,
                        comp_results.e_score,
                        comp_results.s_metric,
                        comp_results.s_score,
                        threshold,
                    )
                )

            results[prefix] = comp_results or {"accuracy": 0}

    def run(self, indexed_mode: bool = False) -> Dict[int, Any]:
        """
        Launch a metadata comparison process between reference and extracted data.

        Args:
            indexed_mode: Flag to use indexed mode for metadata extraction.

        Returns:
            Dictionary with comparison results, keyed by attribute paths.
        """
        self._evaluation_data.clear()

        ref_data = self.extractor.deep_extract(model=self.reference_data, indexed=indexed_mode)
        ext_data = self.extractor.deep_extract(model=self.generated_data, indexed=indexed_mode)

        results: Dict[str, Dict[str, float]] = {}

        self._recursive_compare(
            ref_node=ref_data,
            ext_node=ext_data,
            results=results,
            prefix="",
            threshold=1,
        )

        formatted_results = self._format_results()

        return formatted_results
