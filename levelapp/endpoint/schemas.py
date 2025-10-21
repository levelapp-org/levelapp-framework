"""levelapp/endpoint/schemas.py"""
from enum import Enum
from typing import Any

from pydantic import BaseModel


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class HeaderConfig(BaseModel):
    """Secure header configuration with environment variables support."""
    name: str
    value: str
    secure: bool = False

    class Config:
        frozen = True


class RequestSchemaConfig(BaseModel):
    """Schema Definition for request payload population."""
    field_path: str  # JSON path-like: "data.user.id"
    value: Any
    value_type: str = "static"
    required: bool = True


class ResponseMappingConfig(BaseModel):
    """Response data extraction mapping."""
    field_path: str
    extract_as: str
    default: Any = None
