"""levelapp/comparator/extractor.py"""

from collections import defaultdict
from collections.abc import Sequence
from typing import List, Dict, Any
from pydantic import BaseModel


class DataExtractor:
    """
    Extracts primitive values from nested Pydantic models, dicts, and sequences.
    """
    def deep_extract(
            self, model: BaseModel,
            indexed: bool = False
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
