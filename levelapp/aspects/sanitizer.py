"""'levelapp/aspects/sanitizers.py'"""
import re
import json
from typing import Dict, Any, Callable


class JSONSanitizer:
    def __init__(
            self,
            type_conversions: Dict[str, Callable[[str], Any]] = None,
            default_values: Dict[str, Any] = None,
            remove_nulls: bool = True,
            raise_on_missing: bool = False,
    ):
        """
        Initialize the sanitizer with optional transformation rules.

        Args:
            type_conversions: Map of field names to functions that convert their values.
            default_values: Map of default values for missing or null fields.
            remove_nulls: If True, null-valued fields will be dropped.
            raise_on_missing: If True, an error is raised when a required field is missing.
        """
        self.type_conversions = type_conversions or {}
        self.default_values = default_values or {}
        self.remove_nulls = remove_nulls
        self.raise_on_missing = raise_on_missing

    def sanitize(self, data: Any) -> Any:
        """
        Entry point for sanitization logic.

        Args:
            data: Input data (expected to be a dict or list of dicts).

        Returns:
            Sanitized data with transformations and corrections applied.
        """
        if isinstance(data, dict):
            return self._sanitize_dict(data)

        elif isinstance(data, list):
            return [self._sanitize_dict(item) for item in data]

        else:
            return self._sanitize_value(data)

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a dictionary recursively.

        Args:
            data: Dictionary to be cleaned.

        Returns:
            A sanitized version of the dictionary.
        """
        sanitized_data = {}
        for key, value in data.items():
            if value is None:
                if self.remove_nulls:
                    continue

                elif key in self.default_values:
                    value = self.default_values[key]

                elif self.raise_on_missing:
                    raise ValueError(f'[_sanitize_dict] Missing value for key "{key}"')

            if isinstance(value, (dict, list)):
                sanitized_data[key] = self.sanitize(value)

            else:
                sanitized_data[key] = self._sanitize_field(key, value)

        for key, default in self.default_values.items():
            if key not in sanitized_data:
                sanitized_data[key] = default

        return sanitized_data

    def _sanitize_field(self, key: str, value: Any) -> Any:
        """
        Apply type conversion to a single field if configured.

        Args:
            key: Field name.
            value: Original value.

        Returns:
            Sanitized and type-converted value.
        """
        if key in self.type_conversions:
            try:
                value = self.type_conversions[key](value)

            except Exception as e:
                raise ValueError(f"[_sanitized_field] Failed to convert field {key} to type {type(value)}: {e}")

        return self._sanitize_value(value)

    def _sanitize_value(self, value: Any) -> Any:
        """
        Ensure a value is JSON-serializable and safely encoded.

        Args:
            value: Raw value from input.

        Returns:
            Cleaned value, suitable for JSON serialization.
        """
        if isinstance(value, str):
            return self._escape_special_characters(value)

        else:
            try:
                json.dumps(value)
                return value

            except (TypeError, ValueError):
                return str(value)

    @staticmethod
    def _escape_special_characters(value: Any) -> str:
        """
        Escape non-UTF-8 or invalid characters in string data.

        Args:
            value: A string that may contain unsafe characters.

        Returns:
            UTF-8-safe, sanitized string.
        """
        try:
            value.encode("utf-8").decode("utf-8")
            return value

        except UnicodeDecodeError:
            return value.encode("utf-8", errors="replace").decode("utf-8")

    @staticmethod
    def strip_code_fences(text: str) -> str:
        """
        Remove triple backticks and language hints from code blocks.

        Args:
            text: Input string potentially containing code fences.

        Returns:
            String with code fences removed.
        """
        return re.sub(r"^(```[a-zA-Z]*\n?)$", "", text.strip(), flags=re.MULTILINE)

    def safe_load_json(self, text: str) -> Dict[str, Any]:
        """
        Safely parse JSON from a string, even if surrounded by extra spaces/newlines.

        Args:
            text: Input string containing JSON.

        Returns:
            Parsed JSON as a dictionary, or empty dict on failure.
        """
        try:
            return json.loads(text.strip())

        except json.JSONDecodeError:
            return self.sanitize(data=text)
