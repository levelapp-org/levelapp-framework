"""
'simulators/aspects.py': Utility functions for handling VLA interactions and requests.
"""
import re
import ast
import json
import httpx

from uuid import UUID
from string import Template
from typing import Any, Dict, List, Union, Iterable

from pydantic import ValidationError

from levelapp.clients import ClientRegistry
from levelapp.config.prompts import SUMMARIZATION_PROMPT_TEMPLATE
from levelapp.simulator.schemas import InteractionResults
from levelapp.aspects import MonitoringAspect, MetricType, logger


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")  # captures inner name(s) of ${...}


def _traverse_path(d: Dict[str, Any], path: str):
    """Traverse a dot-separated path (payload.metadata.budget) and return value or None."""
    parts = path.split(".")
    cur = d
    try:
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur
    except Exception:
        return None


def _recursive_find(container: Any, target_key: str):
    """
    Recursively search container (dicts/lists) for the first occurrence of target_key.
    Returns the value if found, else None.
    """
    if isinstance(container, dict):
        # direct hit
        if target_key in container:
            return container[target_key]
        # recurse into values
        for v in container.values():
            found = _recursive_find(v, target_key)
            if found is not None:
                return found
        return None

    if isinstance(container, list):
        for item in container:
            found = _recursive_find(item, target_key)
            if found is not None:
                return found
        return None

    # not a container
    return None


def _extract_placeholders(template_str: str) -> Iterable[str]:
    """Return list of placeholder names in a template string (inner contents of ${...})."""
    return [m.group(1) for m in _PLACEHOLDER_RE.finditer(template_str)]


def extract_interaction_details(
    response: str | Dict[str, Any],
    template: Dict[str, Any],
) -> InteractionResults:
    """
    Parse response (str or dict), look up placeholders recursively in the response and
    use Template.safe_substitute with a mapping built from those lookups.
    """
    try:
        response_dict = response if isinstance(response, dict) else json.loads(response)
        print(f"response:\n{response_dict}\n--")
        if not isinstance(response_dict, dict):
            raise ValueError("Response is not a valid dictionary")

        output: Dict[str, Any] = {}

        for out_key, tpl_str in template.items():
            # Build mapping for placeholders found in tpl_str
            placeholders = _extract_placeholders(tpl_str)
            mapping: Dict[str, str] = {}

            for ph in placeholders:
                value = None

                # 1) If ph looks like a dotted path, try explicit path traversal first
                if "." in ph:
                    value = _traverse_path(response_dict, ph)

                # 2) If not found yet, try recursive search for the bare key (last path segment)
                if value is None:
                    bare = ph.split(".")[-1]
                    value = _recursive_find(response_dict, bare)

                # Prepare mapping value for Template substitution:
                # - dict/list -> JSON string (so substitution yields valid JSON text)
                # - None -> empty string
                # - otherwise -> str(value)
                if isinstance(value, (dict, list)):
                    try:
                        mapping[ph] = json.dumps(value, ensure_ascii=False)
                    except Exception:
                        mapping[ph] = str(value)
                elif value is None:
                    mapping[ph] = ""
                else:
                    mapping[ph] = str(value)

            # Perform substitution using Template (safe_substitute: missing keys left intact)
            substituted = Template(tpl_str).safe_substitute(mapping)
            output[out_key] = substituted

        # Post-process generated_metadata if present: convert JSON text back to dict/list when possible
        raw_meta = output.get("generated_metadata", {})
        if isinstance(raw_meta, str) and raw_meta:
            # Try json first (since we used json.dumps above for mapping)
            try:
                output["generated_metadata"] = json.loads(raw_meta)
            except Exception:
                # fallback to ast.literal_eval (handles Python dict strings)
                try:
                    output["generated_metadata"] = ast.literal_eval(raw_meta)
                except Exception:
                    # if parsing fails, keep the original raw string or use an empty dict
                    output["generated_metadata"] = raw_meta

        # If generated_metadata is empty string, normalize to {}
        if output.get("generated_metadata") == "":
            output["generated_metadata"] = {}

        print(f"output:\n{output}\n---")
        # Return validated model
        return InteractionResults.model_validate(output)

    except json.JSONDecodeError as e:
        logger.error(f"[extract_interaction_details] Failed to parse JSON response: {e}")
        return InteractionResults()

    except ValidationError as e:
        logger.exception(f"[extract_interaction_details] InteractionResults validation failed: {e}")
        return InteractionResults()

    except Exception as e:
        logger.exception(f"[extract_interaction_details] Unexpected error: {e}")
        return InteractionResults()


@MonitoringAspect.monitor(name="interaction_request", category=MetricType.API_CALL)
async def async_interaction_request(
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
) -> httpx.Response | None:
    """
    Perform an asynchronous interaction request.

    Args:
        url (str): The URL to send the request to.
        headers (Dict[str, str]): The headers to include in the request.
        payload (Dict[str, Any]): The payload to send in the request.

    Returns:
        httpx.Response: The response from the interaction request, or None if an error occurred.
    """
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(url=url, headers=headers, json=payload)
            response.raise_for_status()

            return response

    except httpx.HTTPStatusError as http_err:
        logger.error(f"[async_interaction_request] HTTP error: {http_err.response.text}", exc_info=True)

    except httpx.RequestError as req_err:
        logger.error(f"[async_interaction_request] Request error: {str(req_err)}", exc_info=True)

    return None


@MonitoringAspect.monitor(
    name="average_calc",
    category=MetricType.SCORING,
    cached=True,
    maxsize=1000
)
def calculate_average_scores(scores: Dict[str, Union[List[float], float]]) -> Dict[str, float]:
    """
    Helper function that calculates the average scores for a dictionary of score lists.

    Args:
        scores (Dict[str, List[float]]): A dictionary where keys are identifiers and values are lists of scores.

    Returns:
        Dict[str, float]: A dictionary with average scores rounded to three decimal places.
    """
    result: Dict[str, float] = {}
    for field, value in scores.items():
        if isinstance(value, (int, float)):
            result[field] = value
        elif isinstance(value, list):
            result[field] = round((sum(value) / len(value)), 3) if value else 0.0
        else:
            raise TypeError(f"[calculate_average_scores] Unexpected type '{type(value)}' for field '{field}")

    return result


@MonitoringAspect.monitor(name="summarization", category=MetricType.API_CALL)
def summarize_verdicts(
        verdicts: List[str],
        judge: str,
        max_bullets: int = 5
) -> List[str]:
    client_registry = ClientRegistry()
    client = client_registry.get(provider=judge)

    try:
        verdicts = chr(10).join(verdicts)
        prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(max_bullets=max_bullets, judge=judge, verdicts=verdicts)
        response = client.call(message=prompt)
        parsed = client.parse_response(response=response)
        striped = parsed.get("output", "").strip("")
        bullet_points = [point.strip() for point in striped.split("- ") if point.strip()]

        return bullet_points[:max_bullets]

    except Exception as e:
        logger.error(f"[summarize_justifications] Error during summarization: {str(e)}", exc_info=True)
        return []


# if __name__ == '__main__':
#     template = {'generated_reply': '${agent_reply}', 'generated_metadata': '${generated_metadata}'}
#     response_dict = {
#         'agent_reply': "I'd be happy to help you book something for 10 AM.",
#         'generated_metadata': {'appointment_type': 'Cardiology', 'date': 'next Monday', 'time': '10 AM'}
#     }
#
#     result = extract_interaction_details(response_dict, template)
#     print(f"result: {result.model_dump()}")
