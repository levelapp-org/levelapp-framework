"""levelapp/utils/data_loader.py"""
import json

from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError


Model = TypeVar("Model", bound=BaseModel)


def load_json_file(
    model: Type[Model],
    file_path: Path = Path("../data/conversation_example_1.json"),
) -> Model:
    """
    Load a JSON file and parse it into a Pydantic model instance.

    Args:
        model (Type[Model]): The Pydantic model class to instantiate.
        file_path (Path): Path to the JSON file. Defaults to 'config.json'.

    Returns:
        Model: An instance of the provided model with data from the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an IO error reading the file.
        ValueError: If the file contains invalid JSON.
        ValidationError: If the data doesn't validate against the model.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return model.model_validate(data)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {file_path}:\n{e}")

    except IOError as e:
        raise IOError(f"Error reading file {file_path}:\n{e}")

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}:\n{e}")

    except ValidationError:
        raise ValueError(f"Validation error while creating model {model.__name__}")
