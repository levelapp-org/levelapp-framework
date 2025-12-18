""" levelapp/clients/huggingface.py"""
import os

from typing import List, Dict, Any

from levelapp.core.base import BaseChatClient


class HuggingFaceClient(BaseChatClient):
    """
    Client for interacting with HuggingFace's Chat Completions API.

    This implementation adapts requests and responses to the HuggingFace Router API format,
    which is OpenAI-compatible but with HuggingFace-specific endpoint.

    Attributes:
        model (str): Target model ID (default: "openai/gpt-oss-120b:fastest").
        base_url (str): Base endpoint for HuggingFace API (default: "https://huggingface.co/v1").
        api_key (str): Authentication token for the HuggingFace API
        max_tokens (int): Maximum tokens allowed in the completion.
    """
    SUPPORTED_PROVIDERS: List[str] = [
        "cerebras",
        "cohere",
        "featherless-ai",
        "fireworks-ai",
        "groq",
        "hf-inference",
        "hyperbolic",
        "nebius",
        "novita",
        "nscale",
        "ovhcloud",
        "publicai",
        "sambanova",
        "scaleway",
        "together",
        "zai-org"
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model") or "openai/gpt-oss-120b"
        self.base_url = kwargs.get("base_url") or "https://router.huggingface.co/v1"
        self.api_key = kwargs.get("api_key") or os.getenv("HF_TOKEN")
        self.max_tokens = kwargs.get("max_tokens") or 1024

        _provider = os.getenv("HUGGINGFACE_PROVIDER")
        self.provider = kwargs.get("provider") or _provider or "auto"

        self._validate_provider(self.provider)

        if not self.api_key:
            raise ValueError("HuggingFace API token not set (HF_TOKEN env var).")

    def _validate_provider(self, provider):
        """
        Validate that the provided provider string is supported.

        Args:
            provider (str): The provider string to validate.

        Raises:
            ValueError: If the provider is not in the SUPPORTED_PROVIDERS list.
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            supported_str = ", ".join(self.SUPPORTED_PROVIDERS)
            raise ValueError(
                f"[HuggingFaceClient] Unsupported HuggingFace provider '{provider}'. "
                f"Supported providers: {supported_str}."
            )

    @property
    def endpoint_path(self) -> str:
        """
        API-specific endpoint path for chat completions.

        Returns:
            str: "/chat/completions
        """
        return "/chat/completions"

    def _build_endpoint(self) -> str:
        """
        Construct the full API endpoint URL.

        Returns:
            str: Concatenated API endpoint URL.
        """
        return f"{self.base_url}/{self.endpoint_path.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for the HuggingFace API request.

        Returns:
            Dict[str, str]: HTTP headers with authentication and content type.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        """
        Construct the JSON payload for the HuggingFace Chat Completions API.

        Args:
            message (str): Message to send.

        Returns:
            Dict[str, Any]: JSON payload containing model ID and messages.
        """
        model_with_provider = self.model
        if self.provider and self.provider != "auto":
            model_with_provider = f"{self.model}:{self.provider}"

        payload = {
            "model": model_with_provider,
            "messages": [{"role": "user", "content": message}]
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        return payload

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and normalize the HuggingFace API response.

        - Extracts text output from 'choices[0].message.content'.
        - Attempts to JSON-parse the result if it contains structured content.
        - Collects token usage metadata from 'usage'.

        Args:
            response (Dict[str, Any]): Raw JSON response from HuggingFace.

        Returns:
            Dict[str, Any]: {
                "output": Parsed model output (dict or str),
                "metadata": {
                    "input_tokens": int,
                    "output_tokens": int,
                }
            }
        """
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            output = message.get("content", "")

        else:
            output = ""

        parsed = self.sanitizer.safe_load_json(text=output)

        return {"output": parsed, "metadata": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
