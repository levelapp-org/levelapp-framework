"""'comparator/service.py':"""
from collections import defaultdict

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple, Literal

from pydantic import BaseModel

from levelapp.comparator.metrics import MetricsManager, ComparisonResults
from levelapp.comparator.schemas import EntityMetric, SetMetric
from levelapp.comparator.utils import format_evaluation_results


class MetadataComparator:
    def __init__(self, reference: BaseModel, extracted: BaseModel, metrics_manager: MetricsManager):
        self.reference = reference
        self.extracted = extracted
        self.metrics_manager = metrics_manager
        self._evaluation_data: List[
            Tuple[str, list[str], list[str], Any, Any, Any, Any, float]
        ] = []

    def _get_score(self, field: str) -> Tuple[EntityMetric, SetMetric, float]:
        """
        Retrieve the scoring metric and threshold for a given field.

        Args:
            field: The field for which to retrieve the metric and threshold.

        Returns:
            A tuple containing the scoring metric and its threshold.
        """
        config = self.metrics_manager.get_metrics_config(field=field)
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

        return {i: row for i, row in enumerate(formatted_results)}

    def _handle_model(
        self, model: BaseModel, prefix: str, result: Dict[str, List[str]]
    ) -> None:
        """
        Extract values from a Pydantic model recursively.

        Args:
            model: Pydantic BaseModel instance.
            prefix: Current field path.
            result: Dictionary to store field paths and value lists.
        """
        for field_name, field_info in type(model).model_fields.items():
            field_value = getattr(model, field_name)
            new_prefix = f"{prefix}.{field_name}" if prefix else field_name
            self._extract_field_values(
                value=field_value, prefix=new_prefix, result=result
            )

    def _handle_sequence(
        self,
        sequence: Sequence,
        prefix: str,
        result: Dict[str, List[str]],
        indexed: bool = False,
    ) -> None:
        """
        Extract values from a sequence (list or tuple) recursively.

        Args:
            sequence: List or tuple of values.
            prefix: Current field path.
            result: Dictionary to store field paths and value lists.
            indexed: Switch parameter to select the extraction approach.
        """
        if not sequence:
            result[prefix] = []

        if indexed:
            for i, item in enumerate(sequence):
                new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                self._extract_field_values(value=item, prefix=new_prefix, result=result)
        else:
            for i, item in enumerate(sequence):
                self._extract_field_values(
                    value=item, prefix=prefix, result=result, indexed=indexed
                )

    def _extract_field_values(
        self,
        value: Any,
        prefix: str,
        result: Dict[str, List[str]],
        indexed: bool = False,
    ) -> None:
        """
        Recursively extract values from a field, storing them in result with field path as key.

        Args:
            value: The value to extract (BaseModel, dict, list, or primitive).
            prefix: The current field path (e.g., 'documents.tribunal_members').
            result: Dictionary to store field paths and their value lists.
            indexed: Switch parameter to select the extraction approach.
        """
        if isinstance(value, BaseModel):
            self._handle_model(model=value, prefix=prefix, result=result)

        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            self._handle_sequence(
                sequence=value, prefix=prefix, result=result, indexed=indexed
            )

        else:
            result[prefix].append(value)

    def deep_extract(
        self, model: BaseModel, indexed: bool = False
    ) -> Dict[str, List[str]]:
        """
        Extracts data in a recursive way from pydantic model.

        Args:
            model: An instance of a BaseModel.
            indexed: Switch parameter to select the extraction approach.

        Returns:
            A dictionary where keys are attribute names and values are lists of string values.
        """
        result: Dict[str, List[str]] = defaultdict(list)
        for field_name, field_info in type(model).model_fields.items():
            field_value = getattr(model, field_name)
            self._extract_field_values(
                value=field_value, prefix=field_name, result=result, indexed=indexed
            )

        return result

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

        scores = self.metrics_manager.compute_entity_scores(
            reference_seq=reference_list,
            extracted_seq=extracted_list,
            scorer=entity_metric,
            pairwise=False
        )

        return self.metrics_manager.compute_set_scores(
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

    def compare_metadata(self, indexed_mode: bool = False) -> Dict[int, Any]:
        """
        Launch a metadata comparison process between reference and extracted data.

        Args:
            indexed_mode: Flag to use indexed mode for metadata extraction.

        Returns:
            Dictionary with comparison results, keyed by attribute paths.
        """
        self._evaluation_data.clear()

        ref_data = self.deep_extract(model=self.reference, indexed=indexed_mode)
        ext_data = self.deep_extract(model=self.extracted, indexed=indexed_mode)

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
