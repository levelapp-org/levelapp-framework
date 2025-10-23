"""levelapp/endpoint/parsers.py"""
from typing import List, Dict, Any

from levelapp.config.endpoint_ import ResponseMappingConfig
from levelapp.endpoint.schemas import RequestSchemaConfig


class RequestPayloadBuilder:
    def build(self, schema: List[RequestSchemaConfig], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds nested JSON payloads using dot-notation paths.

        Args:
            schema (List[RequestSchemaConfig]): List of request schema configurations.
            context (Dict[str, Any]): Context for building the payload.

        Returns:
            payload (Dict[str, Any]): Request payload.
        """
        payload = {}

        for field_config in schema:
            value = self._resolve_value(config=field_config, context=context)
            if value is None and field_config.required:
                raise ValueError(f"Required field '{field_config.field_path}' has no value")

            self._set_nested_value(obj=payload, path=field_config.field_path, value=value)

        return payload

    @staticmethod
    def _resolve_value(config: RequestSchemaConfig, context: Dict[str, Any]) -> Any:
        """
        Resolve value based on type: static, env, or dynamic.

        Args:
            config (RequestSchemaConfig): Request schema configuration.
            context (Dict[str, Any]): Context for building the payload.

        Returns:
            Any: Value resolved.
        """
        if config.value_type == "static":
            return config.value
        elif config.value_type == "env":
            import os
            return os.getenv(config.value)
        elif config.value_type == "dynamic":
            return context.get(config.value, None)

        return config.value

    @staticmethod
    def _set_nested_value(obj: Dict, path: str, value: Any) -> None:
        parts: List[str] = path.split(".")
        for part in parts[:-1]:
            obj = obj.setdefault(part, {})

        obj[parts[-1]] = value


class ResponseDataExtractor:
    """Extracts data from API response using mapping-based config."""
    def extract(
            self,
            response_data: Dict[str, Any],
            mappings: List[ResponseMappingConfig]
    ) -> Dict[str, Any]:
        """
        Extracts data from API response using mapping-based config.

        Args:
            response_data (Dict[str, Any]): API response data.
            mappings (List[ResponseMappingConfig]): List of response mappings.

        Returns:
            Dict[str, Any]: Extracted data.
        """
        result: Dict[str, Any] = {}

        for mapping in mappings:
            try:
                value = self._extract_by_path(obj=response_data, path=mapping.field_path, default=mapping.default)
                result[mapping.extract_as] = value

            except Exception as e:
                print(f"Failed to extract '{mapping.field_path}':\n{e}")
                result[mapping.extract_as] = mapping.default

        return result

    @staticmethod
    def _extract_by_path(obj: Dict, path: str, default: Any = "N/A") -> Any:
        """
        Extracts value using JSON path-like notation.
        """
        parts = path.split(".")
        current = obj

        for part in parts:
            if not isinstance(current, dict):
                print("[extract_by_path][WARNING] the response data is not a dict.")
                return default

            try:
                if '[' in part and ']' in part:
                    key, idx = part.split('[')
                    idx = int(idx.rstrip(']'))
                    current = current[key][idx] if key else current[idx]
                else:
                    if part not in current:
                        print(f"[extract_by_path][WARNING] Key '{part}' is missing from response.")
                        return default
                    current = current.get(part)

            except (KeyError, IndexError, TypeError, AttributeError) as e:
                print(f"[extract_by_path][ERROR] Error type <{e.__class__.__name__}> : {e.args[0]}")
                return default

        return current
