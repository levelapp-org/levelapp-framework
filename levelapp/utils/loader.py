"""levelapp/utils/loader.py"""
import os
import yaml
import json
import logging

from pathlib import Path

from collections.abc import Mapping, Sequence
from typing import Any, Type, TypeVar, List, Optional, Dict, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, create_model, ValidationError

from rapidfuzz import utils


logger = logging.getLogger(__name__)
Model = TypeVar("Model", bound=BaseModel)


class DynamicModelBuilder:
    """
    Implements dynamic model builder.
    -docs here-
    """
    def __init__(self):
        self.model_cache: Dict[Tuple[str, str], Type[BaseModel]] = {}

    def clear_cache(self):
        self.model_cache.clear()

    @staticmethod
    def _sanitize_field_name(name: str) -> str:
        """
        Sanitize field names to be valid Python identifiers using rapidfuzz.
        Ensures non-empty names and handles numeric-starting names.
        """
        name = utils.default_process(name).replace(' ', '_')
        if not name:
            return "field_default"
        if name[0].isdigit():
            return f"field_{name}"
        return name

    def _get_field_type(self, value: Any, model_name: str, key: str) -> Tuple[Any, Any]:
        """
        Determine the field type and default value for a given value.
        Handles dictionaries, lists, and primitive types.
        """
        if isinstance(value, Mapping):
            nested_model = self.create_dynamic_model(model_name=f"{model_name}_{key}", data=value)
            return nested_model, ...

        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                return List[BaseModel], ...

            elif isinstance(value[0], Mapping):
                nested_model = self.create_dynamic_model(model_name=f"{model_name}_{key}", data=value[0])
                return List[nested_model], ...

            else:
                field_type = type(value[0]) if value[0] is not None else Any
                return List[field_type], ...

        else:
            field_type = Optional[type(value)] if value is not None else Optional[Any]
            return field_type, ...

    def create_dynamic_model(self, model_name: str, data: Any) -> Type[BaseModel]:
        """
        Create a Pydantic model dynamically from data.
        Supports nested dictionaries, lists, and primitives with caching.
        """
        model_name = self._sanitize_field_name(name=model_name)
        cache_key = (model_name, str(data) if not isinstance(data, dict) else str(sorted(data.keys())))

        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        if isinstance(data, Mapping):
            fields = {
                self._sanitize_field_name(name=key): self._get_field_type(value=value, model_name=model_name, key=key)
                for key, value in data.items()
            }
            model = create_model(model_name, **fields)

        else:
            field_type = Optional[type(data)] if data else Optional[Any]
            model = create_model(model_name, value=(field_type, None))

        self.model_cache[cache_key] = model

        return model


class DataLoader:
    def __init__(self):
        self.builder = DynamicModelBuilder()
        self._name = self.__class__.__name__
        load_dotenv()

    @staticmethod
    def load_configuration(path: str | None = None):
        try:
            if not path:
                path = os.getenv('WORKFLOW_CONFIG_PATH', 'no-file')

                if not os.path.exists(path):
                    raise FileNotFoundError(f"The provided configuration file path '{path}' does not exist.")

            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith((".yaml", ".yml")):
                    content = yaml.safe_load(f)

                elif path.endswith(".json"):
                    content = json.load(f)

                else:
                    raise ValueError("[WorkflowConfiguration] Unsupported file format.")

                return content

        except FileNotFoundError as e:
            raise FileNotFoundError(f"[EndpointConfig] Payload template file '{e.filename}' not found in path.")

        except yaml.YAMLError as e:
            raise ValueError(f"[EndpointConfig] Error parsing YAML file:\n{e}")

        except json.JSONDecodeError as e:
            raise ValueError(f"[EndpointConfig] Error parsing JSON file:\n{e}")

        except IOError as e:
            raise IOError(f"[EndpointConfig] Error reading file:\n{e}")

        except Exception as e:
            raise ValueError(f"[EndpointConfig] Unexpected error loading configuration:\n{e}")

    def load_data(
            self,
            data: Dict[str, Any],
            model_name: str = "ExtractedData"
    ) -> BaseModel | None:
        """
        Load data into a dynamically created Pydantic model instance.

        Args:
            data (Dict[str, Any]): The data to load.
            model_name (str, optional): The name of the model. Defaults to "ExtractedData".

        Returns:
            An Pydantic model instance.

        Raises:
            ValidationError: If a validation error occurs.
            Exception: If an unexpected error occurs.
        """
        try:
            self.builder.clear_cache()
            dynamic_model = self.builder.create_dynamic_model(model_name=model_name, data=data)
            model_instance = dynamic_model.model_validate(data)
            return model_instance

        except ValidationError as e:
            logger.exception(f"[{self._name}] Validation Error: {e.errors()}")

        except Exception as e:
            logger.error(f"[{self._name}] An error occurred: {e}")
