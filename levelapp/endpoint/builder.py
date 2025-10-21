"""levelapp/endpoint/builder.py"""
from typing import List, Dict, Any

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
            return context.get(config.field_path, None)

        return config.value


    @staticmethod
    def _set_nested_value(obj: Dict, path: str, value: Any) -> None:
        parts: List[str] = path.split(".")
        for part in parts[:-1]:
            obj = obj.setdefault(part, {})

        obj[parts[-1]] = value
