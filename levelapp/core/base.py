"""levelapp/core/base.py"""
import asyncio
import datetime
import json

import httpx
import requests
import logging

from abc import ABC, abstractmethod

from pydantic import BaseModel
from typing import List, Dict, Any, Callable, TypeVar, Type

logger = logging.getLogger(__name__)

Model = TypeVar("Model", bound=BaseModel)
Context = TypeVar("Context")


class BaseProcess(ABC):
    """Interface for the evaluation classes."""
    @abstractmethod
    def run(self, **kwargs) -> Any:
        raise NotImplementedError


class BaseEvaluator(ABC):
    """Abstract base class for evaluator components."""
    @abstractmethod
    def evaluate(self, provider: str, generated_text: str, reference_text: str):
        """Evaluate system output to reference output."""
        raise NotImplementedError


class BaseChatClient(ABC):
    """Abstract base chat client for different LLM providers integration."""
    def __init__(self, **kwargs):
        self.base_url = kwargs.get("base_url")

    @staticmethod
    def _endpoint() -> str:
        return "/predictions"

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    @abstractmethod
    def _build_headers(self) -> Dict[str, str]:
        ...

    @abstractmethod
    def _build_payload(self, message: str) -> Dict[str, Any]:
        ...

    def call(self, message: str) -> Dict[str, Any]:
        url = self._build_url(self._endpoint())
        headers = self._build_headers()
        payload = self._build_payload(message)

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            raise
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            raise
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            raise
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected error occurred: {req_err}")
            raise

    async def acall(self, message: str) -> Dict[str, Any]:
        url = self._build_url(self._endpoint())
        headers = self._build_headers()
        payload = self._build_payload(message)

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as http_err:
            print(f"[IonosClient.acall] HTTP error: {http_err}")
            raise
        except httpx.RequestError as req_err:
            print(f"[IonosClient.acall] Request error: {req_err}")
            raise
        except httpx.TimeoutException as timeout_err:
            print(f"[IonosClient.acall] Timeout: {timeout_err}")
            raise
        except Exception as e:
            print(f"[IonosClient.acall] Unexpected error: {e}")
            raise


class BaseMetric(ABC):
    """Abstract base class for metrics collection."""

    def __init__(self, processor: Callable | None = None, score_cutoff: float | None = None):
        """
        Initialize the metric.

        Args:
            processor (Optional[Callable]): Optional function to preprocess strings before comparison.
            score_cutoff (Optional[float]): Minimum similarity score for an early match cutoff.
        """
        self.processor = processor
        self.score_cutoff = score_cutoff

    @abstractmethod
    def compute(self, generated: str, reference: str) -> Dict[str, Any]:
        """
        Evaluate the generated text against the reference text.

        Args:
            generated (str): The generated text to evaluate.
            reference (str): The reference text to compare against.

        Returns:
            Dict[str, Any]: Evaluation results including match level and justification.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Get the name of the metric.

        Returns:
            str: Name of the metric.
        """
        return self.__class__.__name__.lower()

    # TODO-0: You know what..We can remove this at some point.
    @staticmethod
    def _validate_inputs(generated: str, reference: str) -> None:
        """Validate that both inputs are strings."""
        if not (isinstance(generated, str) and isinstance(reference, str)):
            raise TypeError("Both 'generated' and 'reference' must be strings.")

    def _get_params(self) -> Dict[str, Any]:
        """Return a serializable dictionary of metric parameters."""
        return {
            "processor": repr(self.processor) if self.processor else None,
            "score_cutoff": self.score_cutoff
        }

    def _build_metadata(self, **extra_inputs) -> Dict[str, Any]:
        """Construct a consistent metadata dictionary."""
        return {
            "type": self.__class__.__name__,
            "params": self._get_params(),
            "inputs": extra_inputs,
            "timestamp": datetime.datetime.now()
        }


class BaseRepository(ABC):
    """
    Abstract base class for pluggable NoSQL data stores.
    Supports document-based operations with Pydantic model parsing.
    """

    @abstractmethod
    def connect(self) -> None:
        """Initialize connection or client."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close connection or client."""
        raise NotImplementedError

    @abstractmethod
    def retrieve_document(
            self,
            collection_id: str,
            section_id: str,
            sub_collection_id: str,
            document_id: str,
            model_type: Type[Model]
    ) -> Model | None:
        """
        Retrieve and parse a document from the datastore based on its type.

        Args:
            collection_id (str): Collection reference.
            section_id (str): Section reference.
            sub_collection_id (str): Sub-collection reference.
            document_id (str): Reference of the document to retrieve.
            model_type (Type[BaseModel]): Pydantic class to instantiate.

        Returns:
            Parsed model instance or None if document was not found.
        """
        raise NotImplementedError

    @abstractmethod
    def store_document(
            self,
            collection_id: str,
            section_id: str,
            sub_collection_id: str,
            document_id: str,
            data: Model
    ) -> None:
        """
        Store a pydantic model instance as a document.

        Args:
            collection_id (str): Collection reference.
            section_id (str): Section reference.
            sub_collection_id (str): Sub-collection reference.
            document_id (str): Reference of the document to store.
            data (Model): Pydantic model instance.
        """
        raise NotImplementedError

    @abstractmethod
    def query_collection(
            self,
            collection_id: str,
            section_id: str,
            sub_collection_id: str,
            filters: Dict[str, Any],
            model_type: Type[Model]
    ) -> List[Model]:
        """
        Query documents in a collection with optional filters.

        Args:
            collection_id (str): Collection reference.
            section_id (str): Section reference.
            sub_collection_id (str): Sub-collection reference.
            filters (Dict[str, Any]): Filters to apply to the query (implementation dependent).
            model_type (Type[BaseModel]): Pydantic class to instantiate.

        Returns:
            List[Model]: Query results.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_document(
            self,
            collection_id: str,
            section_id: str,
            sub_collection_id: str,
            document_id: str
    ) -> bool:
        """
        Delete a document.

        Args:
            collection_id (str): Collection reference.
            section_id (str): Section reference.
            sub_collection_id (str): Sub-collection reference.
            document_id (str): Reference of the document to delete.

        Returns:
            True if deleted, False if not.
        """
        raise NotImplementedError
