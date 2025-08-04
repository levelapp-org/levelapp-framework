"""levelapp/clients/mistral.py"""
import os

from typing import Dict, Any
from ..core.base import BaseChatClient


class MistralClient(BaseChatClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model") or "mistral-large-latest"
        self.base_url = kwargs.get('base_url') or "https://api.mistral.ai/v1"
        self.api_key = kwargs.get('api_key') or os.environ.get('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("Missing API key not set.")

    def _endpoint(self) -> str:
        return "/chat/completions"

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
        }
