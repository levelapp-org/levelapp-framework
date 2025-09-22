"""levelapp/aspects/loader.py"""
import os
import yaml
import json
import logging

from collections.abc import Mapping, Sequence
from typing import Any, Type, TypeVar, List, Optional, Dict, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, create_model, ValidationError

from rapidfuzz import utils


logger = logging.getLogger(__name__)
Model = TypeVar("Model", bound=BaseModel)


class DynamicModelBuilder:
    """
    A utility for creating dynamic Pydantic models at runtime from arbitrary Python
    data structures (dicts, lists, primitives).

    Features:
    ---------
    - **Dynamic Model Generation**: Builds `pydantic.BaseModel` subclasses on the fly
      based on the structure of the input data.
    - **Recursive Nesting**: Handles arbitrarily nested dictionaries and lists by
      generating nested models automatically. (Don't freak out, it's for shallow traversal).
    - **Field Name Sanitization**: Ensures generated field names are valid Python identifiers.
    - **Caching**: Maintains a cache of previously generated models to avoid redundant
      re-construction and improve performance.

    Use Cases:
    ----------
    - Converting arbitrary JSON/dict data into structured, type-safe Pydantic models.
    - Prototyping with dynamic/unknown payloads where upfront schema definition is not feasible.

    Notes:
    ------
    - Field names are sanitized using `utils.default_process` (from rapidfuzz),
      replacing spaces with underscores and handling invalid identifiers.
    - Lists are typed based on their first element only; heterogeneous lists
      may not be fully captured (Sorry?).
    - Model caching is based on `(model_name, str(data) or str(sorted(keys)))`.
      This improves performance but may cause collisions if `str(data)` is ambiguous.
    """
    def __init__(self):
        """
        Initialize a DynamicModelBuilder instance.

        Attributes:
        -----------
        model_cache : Dict[Tuple[str, str], Type[BaseModel]]
            Cache of generated models keyed by (model_name, data_signature).
            Ensures models are reused instead of rebuilt.
        """
        self.model_cache: Dict[Tuple[str, str], Type[BaseModel]] = {}

    def clear_cache(self):
        """
        Clear the internal model cache.

        Use when schema changes are expected or to free memory in long-running processes.
        """
        self.model_cache.clear()

    @staticmethod
    def _sanitize_field_name(name: str) -> str:
        """
        Normalize and sanitize field names into valid Python identifiers.

        - Converts to lowercase and strips unwanted characters using `utils.default_process`.
        - Replaces spaces with underscores.
        - Ensures field names are not empty; substitutes `"field_default"` if so.
        - Prepends `"field_"` if the name starts with a digit.

        Args:
            name (str): The original field name from input data.

        Returns:
            str: A valid Python identifier for use in a Pydantic model.
        """
        name = utils.default_process(name).replace(' ', '_')
        if not name:
            return "field_default"
        if name[0].isdigit():
            return f"field_{name}"
        return name

    def _get_field_type(self, value: Any, model_name: str, key: str) -> Tuple[Any, Any]:
        """
        Infer the field type and default value for a given data value.

        Supported cases:
        - **Mapping (dict-like)**: Creates a nested dynamic model recursively.
        - **Sequence (list/tuple, non-string)**:
          - Empty list → `List[BaseModel]`
          - List of dicts → `List[<nested model>]`
          - List of primitives → `List[primitive type]`
        - **Primitive**: Wraps in `Optional` to allow nulls.

        Args:
            value (Any): The raw field value to inspect.
            model_name (str): Name of the parent model.
            key (str): Field name (before sanitization).

        Returns:
            Tuple[Any, Any]: A `(type, default_value)` tuple suitable for `create_model`.
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
        Create a dynamic Pydantic model from arbitrary input data.

        - Handles nested dictionaries and lists by recursively generating
          sub-models.
        - Uses caching to avoid rebuilding models for the same schema.

        Args:
            model_name (str): Suggested name of the generated model.
            data (Any): Input data (dict, list, or primitive).

        Returns:
            Type[BaseModel]: A dynamically created Pydantic model class.
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
    """Main utility tool for loading configuration and reference data"""
    def __init__(self):
        self.builder = DynamicModelBuilder()
        self._name = self.__class__.__name__
        load_dotenv()

    @staticmethod
    def load_raw_data(path: str | None = None):
        """
        Load raw data from JSON or YAML files.

        Args:
            path (str): path to the JSON or YAML file.

        Returns:
            A dictionary containing the raw data.

        Raises:
            FileNotFoundError: if the file was not found in the path.
            YAMLError: if there was an error parsing the YAML file.
            JSONDecoderError: if there was an error parsing the JSON file.
            IOError: if the file is corrupt and cannot be read.
        """
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

    def create_dynamic_model(
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
