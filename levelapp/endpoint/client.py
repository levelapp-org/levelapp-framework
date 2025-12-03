"""levelapp/endpoint/client.py"""
import os
import httpx
import asyncio
import backoff
import logging

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from levelapp.endpoint.schemas import HttpMethod, HeaderConfig, RequestSchemaConfig, ResponseMappingConfig


class EndpointConfig(BaseModel):
    """Complete endpoint configuration."""
    name: str
    base_url: str
    path: str
    method: HttpMethod
    headers: List[HeaderConfig] = Field(default_factory=list)
    request_schema: List[RequestSchemaConfig] = Field(default_factory=list)
    response_mapping: List[ResponseMappingConfig] = Field(default_factory=list)
    timeout: int = Field(default=120)
    retry_count: int = Field(default=3)
    retry_backoff: float = Field(default=1.0)

    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.startswith('/'):
            return f"/{v}"
        return v


@dataclass
class APIClient:
    """HTTP client for REST API interactions"""
    config: EndpointConfig
    client: httpx.AsyncClient = field(init=False)
    logger: logging.Logger = field(init=False)
    request_queue: asyncio.Queue = field(init=False)
    semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self):
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self.logger = logging.getLogger(f"AsyncAPIClient.{self.config.name}")
        self.semaphore = asyncio.Semaphore(5)
        self.last_request_time = 0
        self.mix_request_interval = 1

    async def __aenter__(self) -> "APIClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.client.aclose()

    def _build_headers(self) -> Dict[str, str]:
        """Build headers with secure value resolution."""
        headers = {}

        for header in self.config.headers:
            if header.secure:
                value = os.getenv(header.value)
                if value is None:
                    self.logger.warning(f"Secure header '{header.name}' env var '{header.value}' not found")
                    continue
                headers[header.name] = value
            else:
                headers[header.name] = header.value

        return headers

    @backoff.on_exception(
        backoff.expo,
        (httpx.ConnectTimeout, httpx.WriteTimeout, httpx.NetworkError),
        max_tries=3,
        max_time=100
    )
    async def execute(
            self,
            payload: Dict[str, Any] | None = None,
            query_params: Dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Execute asynchronous REST API request with retry logic."""
        headers = self._build_headers()

        async with self.semaphore:
            now = asyncio.get_running_loop().time()
            time_since_last = now - self.last_request_time

            if time_since_last < self.mix_request_interval:
                await asyncio.sleep(delay=self.mix_request_interval - time_since_last)

            response = await self.client.request(
                method=self.config.method.value,
                url=self.config.path,
                json=payload,
                params=query_params,
                headers=headers,
            )
            # Disabled to prevent simulation interruption
            # response.raise_for_status()

            self.last_request_time = asyncio.get_running_loop().time()
            return response
