"""levelapp/endpoint/config.py"""
from typing import List

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
    timout: int = Field(default=30)
    retry_count: int = Field(default=3)
    retry_backoff: float = Field(default=1.0)

    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.startswith('/'):
            return f"/{v}"
        return v
