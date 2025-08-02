"""levelapp/core/base.py"""
import json
import httpx
import requests
import logging

from abc import ABC, abstractmethod

from pydantic import BaseModel
from typing import Dict, Any


logger = logging.getLogger(__name__)


class BaseSimulator(ABC):
    """Abstract base class for simulator components."""
    @abstractmethod
    def simulate(self):
        """Run a stress test simulation based on the provided configuration."""
        raise NotImplementedError


class BaseComparator(ABC):
    """Abstract base class for comparator components."""
    @abstractmethod
    def compare(self):
        """Compare system output against reference output."""
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
        url = self._build_url("/predictions")
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


class BaseDatastore(ABC):
    """Abstract base class for data stores."""
    @abstractmethod
    def fetch_document(
            self,
            user_id: str,
            collection_id: str,
            document_id: str,
            doc_type: str
    ) -> BaseModel:
        """
        Retrieve and parse a document from the datastore based on its type.

        Args:
            user_id (str): ID of the user.
            collection_id (str): Name of the collection.
            document_id (str): ID of the document to retrieve.
            doc_type (str): Type of document (e.g., scenario, bundle).

        Returns:
            BaseModel: Parsed Pydantic model representing the document.
        """
        pass

    @abstractmethod
    def fetch_stored_results(
            self,
            user_id: str,
            collection_id: str,
            project_id: str,
            category_id: str,
            batch_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve stored batch results for a specific user and batch ID.

        Args:
            user_id (str): ID of the user.
            collection_id (str): Main collection name.
            project_id (str): Project identifier.
            category_id (str): Category/sub-collection name.
            batch_id (str): Batch identifier.

        Returns:
            Dict[str, Any]: Dictionary containing the stored result data.
        """
        pass

    @abstractmethod
    def save_batch_test_results(
            self,
            user_id: str,
            project_id: str,
            batch_id: str,
            data: Dict[str, Any]
    ) -> None:
        """
        Store batch test results in the datastore for a specific user and batch.

        Args:
            user_id (str): ID of the user.
            project_id (str): Project identifier.
            batch_id (str): Batch identifier (used as document ID).
            data (Dict[str, Any]): Batch result data to store.
        """
        pass
