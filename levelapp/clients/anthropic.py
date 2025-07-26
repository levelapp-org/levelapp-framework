"""levelapp/clients/anthropic.py"""
import os

from typing import Dict, Any
from ..core.base import BaseChatClient


class AnthropicClient(BaseChatClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get('model') or "claude-sonnet-4-20250514"
        self.version = kwargs.get('version') or "2023-06-01"
        self.max_tokens = kwargs.get('max_tokens') or 1024
        self.base_url = kwargs.get("base_url") or "https://api.anthropic.com/v1"
        self.api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not set.")

    def _endpoint(self) -> str:
        return "/messages"

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json"
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": self.max_tokens
        }
