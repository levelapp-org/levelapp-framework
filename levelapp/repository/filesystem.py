import json
from pathlib import Path

from typing import List, Dict, Any, Type, TYPE_CHECKING

from pydantic.v1 import ValidationError

from levelapp.core.base import BaseRepository, Model
from levelapp.aspects import logger

if TYPE_CHECKING:
    from levelapp.workflow.config import WorkflowConfig


class FileSystemRepository(BaseRepository):
    """
    File-system implementation of BaseRepository.
    Persists Pydantic model data as JSON files under the configured base path.
    """
    def __init__(self, config: "WorkflowConfig | None" = None):
        self._CLASS_NAME = self.__class__.__name__

        self.config = config
        base_path = getattr(config.repository, "base_path", "./data") if config else "./data"
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{self.__class__.__name__}] Base path: {base_path}")

    def connect(self) -> None:
        """No-op for local storage."""
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{self._CLASS_NAME}] connected to {self.base_path}")

    def close(self) -> None:
        """No-op for local storage."""
        logger.info(f"[{self._CLASS_NAME}] Closed (no active connections)")

    def _compose_path(
            self,
            collection_id: str,
            section_id: str,
            sub_collection_id: str,
            document_id: str,
    ) -> Path:
        """
        Compose the hierarchical path for a document.

        Args:
            collection_id (str): the ID for the whole collection.
            section_id (str): the ID for the section.
            sub_collection_id (str): the ID for the sub collection.
            document_id (str): the ID for the document.

        Returns:
            Path: the composed path.
        """
        path = self.base_path / collection_id / section_id / sub_collection_id
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{document_id}.json"

    def retrieve_document(
            self,
            collection_id: str,
            section_id: str,
            sub_collection_id: str,
            document_id: str,
            model_type: Type[Model]
    ) -> Model | None:
        """
        Retrieve a document from the local JSON file system.

        Args:
            collection_id (str): the ID for the whole collection.
            section_id (str): the ID for the section.
            sub_collection_id (str): the ID for the sub collection.
            document_id (str): the ID for the document.
            model_type (Type[Model]): Pydantic model for parsing.

        Returns:
            Model | None: An instance of the provided model.
        """
        path = self._compose_path(collection_id, section_id, sub_collection_id, document_id)
        if not path.exists():
            logger.warning(f"[{self._CLASS_NAME}] Document '{path}' no found")
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return model_type.model_validate(data)

        except json.JSONDecodeError as e:
            logger.error(f"[{self._CLASS_NAME}] Failed to load the JSON file '{document_id}':\n{e}")
            return None

        except ValidationError as e:
            logger.error(f"[{self._CLASS_NAME}] Failed to instantiate a Pydantic model for file '{document_id}':\n{e}")
            return None

        except Exception as e:
            logger.exception(f"[{self._CLASS_NAME}] Unexpected error retrieving file '{document_id}':\n{e}")
            return None

    def store_document(
            self,
            collection_id: str,
            section_id: str,
            sub_collection_id: str,
            document_id: str,
            data: Model
    ) -> None:
        """
        Store a document as JSON file locally.

        Args:
            collection_id (str): the ID for the whole collection.
            section_id (str): the ID for the section.
            sub_collection_id (str): the ID for the sub collection.
            document_id (str): the ID for the document.
            data (Model): Pydantic model for parsing.
        """
        path = self._compose_path(collection_id, section_id, sub_collection_id, document_id)

        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"[{self._CLASS_NAME}] Stored document '{document_id}' in '{path}'")

        except Exception as e:
            logger.exception(f"[{self._CLASS_NAME}] Failed to store document '{document_id}' in '{path}':\n{e}'")

    def query_collection(
            self,
            collection_id: str,
            section_id: str,
            sub_collection_id: str,
            filters: Dict[str, Any],
            model_type: Type[Model]
    ) -> List[Model]:
        """
        Query all document in a sub collection, applying simple equality filters.

        Args:
            collection_id (str): the ID for the whole collection.
            section_id (str): the ID for the section.
            sub_collection_id (str): the ID for the sub collection.
            filters (Dict[str, Any]): Pydantic model for parsing.
            model_type (Type[Model]): Pydantic model for parsing.

        Returns:
            List[Model]: List of deserialized models that match the query.
        """
        path = self.base_path / collection_id / section_id / sub_collection_id

        if not path.exists():
            logger.warning(f"[{self._CLASS_NAME}] Sub-collection '{path}' not found")
            return []

        results = []
        try:
            for file in path.glob("*.json"):
                with file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                if all(data.get(k) == v for k, v in filters.items()):
                    results.append(model_type.model_validate(data))

        except json.JSONDecodeError as e:
            logger.error(f"[{self._CLASS_NAME}] Failed to read JSON files content:\n{e}")

        except ValidationError as e:
            logger.error(f"[{self._CLASS_NAME}] Failed to parse JSON files content:\n{e}")

        return results

    def delete_document(
            self,
            collection_id: str,
            section_id: str,
            sub_collection_id: str,
            document_id: str
    ) -> bool:
        """Delete a JSON document from the local file system."""
        path = self._compose_path(collection_id, section_id, sub_collection_id, document_id)

        if not path.exists():
            logger.warning(f"[{self._CLASS_NAME}] Document '{path}' not found")
            return False

        try:
            path.unlink()
            logger.info(f"[{self._CLASS_NAME}] Deleted document '{document_id}'")
            return True

        except FileNotFoundError:
            logger.warning(f"[{self._CLASS_NAME}] Document '{document_id}' not found")
            return False

        except Exception as e:
            logger.exception(f"[{self._CLASS_NAME}] Failed to delete document '{document_id}':\n{e}")
            return False
