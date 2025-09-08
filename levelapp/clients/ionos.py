"""levelapp/clients/ionos.py"""
import os
import uuid

from typing import Dict, Any
from levelapp.core.base import BaseChatClient


class IonosClient(BaseChatClient):
    """
    Client for interacting with the IONOS LLM API.

    This implementation adapts requests and responses to the IONOS
    API format, including payload structure, headers, and response parsing.

    Attributes:
        model_id (str): Model identifier to target (from IONOS dashboard or env).
        base_url (str): Base endpoint for IONOS API, e.g. https://api.ionos.ai.
        api_key (str): Authentication token for the IONOS API.
        top_k (int): Sampling parameter; number of top tokens to consider.
        top_p (float): Sampling parameter; nucleus probability cutoff.
        temperature (float): Sampling randomness.
        max_tokens (int): Maximum tokens allowed in completion.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_id = kwargs.get('model_id') or os.getenv('IONOS_MODEL_ID')
        self.base_url = kwargs.get('base_url') or os.getenv("IONOS_BASE_URL")
        self.api_key = kwargs.get('api_key') or os.environ.get("IONOS_API_KEY")
        self.top_k = kwargs.get('top_k') or 5
        self.top_p = kwargs.get('top_p') or 0.5
        self.temperature = kwargs.get('temperature') or 0.0
        self.max_tokens = kwargs.get('max_tokens') or 150

        if not self.api_key:
            raise ValueError("IONOS API key not set.")

    @property
    def endpoint_path(self) -> str:
        """
        API-specific endpoint path for inference calls.

        Example:
            "models/{model_id}/predictions"
        """
        return f"models/{self.model_id}/predictions"

    def _build_endpoint(self) -> str:
        """
        Construct the full API endpoint URL.

        Returns:
            str: Concatenation of base_url and endpoint_path.
        """
        return f"{self.base_url}/{self.endpoint_path.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for the IONOS API request.

        Returns:
            Dict[str, str]: Headers with authentication and content type.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        """
        Construct the JSON payload for the IONOS API.

        Args:
            message (str): User input or prompt to evaluate.

        Returns:
            Dict[str, Any]: Payload containing properties and sampling options.
        """
        return {
            "properties": {"input": message},
            "option": {
                "top-k": self.top_k,
                "top-p": self.top_p,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": uuid.uuid4().int & ((1 << 16) - 1),
            },
        }

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and normalize the IONOS API response.

        - Extracts model output from `properties.output`.
        - Strips any code fences or formatting noise.
        - Attempts to JSON-parse the result (safe fallback if invalid).
        - Collects token usage metadata.

        Args:
            response (Dict[str, Any]): Raw JSON response from IONOS.

        Returns:
            Dict[str, Any]: {
                "output": Parsed model output (dict or str),
                "metadata": {
                    "input_tokens": int,
                    "output_tokens": int
                }
            }
        """
        input_tokens = response.get("metadata", {}).get("inputTokens", -1)
        output_tokens = response.get("metadata", {}).get("outputTokens", -1)
        output = response.get("properties", {}).get("output", "")
        cleaned = self.sanitizer.strip_code_fences(output)
        parsed = self.sanitizer.safe_load_json(text=cleaned)
        return {"output": parsed, "metadata": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
