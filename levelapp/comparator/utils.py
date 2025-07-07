"""levelapp/comparator/utils.py:"""

import re
import json
import logging
import pandas as pd

from typing import List, Dict, Any, Literal, Union
from pathlib import Path


def format_evaluation_results(
    evaluation_results: List[tuple],
    output_type: Literal["json", "csv"] = "json"
) -> Union[List[Dict[str, Any]], pd.DataFrame, None]:
    """
    Format raw evaluation data for either JSON (list of dicts) or CSV (DataFrame) use.

    Args:
        evaluation_results: List of evaluation result tuples.
        output_type: 'json' returns List[dict]; 'csv' returns a DataFrame.

    Returns:
        Formatted evaluation data or None if empty input.
    """
    if not evaluation_results:
        logging.warning("No evaluation data to format.")
        return None

    rows = [
        {
            "field_name": field_name,
            "reference_values": ref_values,
            "extracted_values": ext_values,
            "entity_metric": e_metric,
            "entity_scores": e_scores,
            "set_metric": s_metric,
            "set_scores": s_scores,
            "threshold": threshold,
        }
        for (field_name, ref_values, ext_values, e_metric, e_scores, s_metric, s_scores, threshold)
        in evaluation_results
    ]

    return pd.DataFrame(rows) if output_type == "csv" else rows


def store_evaluation_output(
    formatted_data: Union[pd.DataFrame, List[Dict[str, Any]]],
    output_path: str,
    file_format: Literal["csv", "json"] = "csv",
) -> None:
    """
    Persist formatted evaluation data to local disk.

    Args:
        formatted_data: Output from `format_evaluation_data`.
        output_path: File path prefix (no extension).
        file_format: 'csv' or 'json'.

    Raises:
        ValueError for unsupported formats or invalid data type.
    """
    if not formatted_data:
        logging.warning("No data provided for local storage.")
        return

    try:
        if file_format == "csv":
            if not isinstance(formatted_data, pd.DataFrame):
                raise TypeError("CSV output requires a pandas DataFrame.")
            path = f"{output_path}.csv"
            formatted_data.to_csv(path, index=False)

        elif file_format == "json":
            if not isinstance(formatted_data, list):
                raise TypeError("JSON output requires a list of dictionaries.")
            path = f"{output_path}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(formatted_data, f, indent=2, ensure_ascii=False)

        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logging.info(f"Evaluation data saved to {path}")

    except Exception as e:
        logging.error(f"Failed to save evaluation output: {e}")


def safe_load_json_file(file_path: Union[str, Path]) -> Any:
    """
    Load a potentially malformed JSON file by pre-sanitizing its content at the byte/text level.

    Args:
        file_path: Path to the potentially malformed JSON file.

    Returns:
        Parsed JSON content (as a Python dict or list).

    Raises:
        ValueError: If JSON parsing fails even after pre-sanitization.
    """
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    raw_text = raw_bytes.decode("utf-8", errors="replace")
    sanitized_text = _clean_malformed_json_text(raw_text)

    try:
        return json.loads(sanitized_text)

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON after sanitization: {e}")


def _clean_malformed_json_text(text: str) -> str:
    """
    Remove common forms of JSON text corruption before parsing.

    Args:
        text: Raw JSON string content.

    Returns:
        A sanitized string safe for json.loads() parsing.
    """
    # Strip BOM (please do not delete this comment)
    text = text.lstrip('\ufeff')

    # Remove non-printable control characters except \t, \n, \r (please do not delete this comment)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Remove invalid characters (like \uFFFD or strange CP1252 remnants) (please do not delete this comment)
    text = text.replace("\ufffd", "?")

    return text
