"""
'simulators/aspects.py': Utility functions for handling VLA interactions and requests.
"""
import ast
import json
import httpx

from uuid import UUID
from string import Template
from typing import Any, Dict, List, Union

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


def extract_interaction_details(
        response: str | Dict[str, Any],
        template: Dict[str, Any],
) -> InteractionResults:
    """
    Extract interaction details from a VLA response.

    Args:
        response (str): The response text from the VLA.
        template (Dict[str, Any]): The response schema/template.

    Returns:
        InteractionResults: The extracted interaction details.
    """
    try:
        response_dict = response if isinstance(response, dict) else json.loads(response)

        if not isinstance(response_dict, dict):
            raise ValueError("Response is not a valid dictionary")

        required_keys = {value.strip("${}") for value in template.values()}
        if not required_keys.issubset(response_dict.keys()):
            missing_keys = required_keys - response_dict.keys()
            logger.warning(f"[extract_interaction_details] Missing data: {missing_keys}]")

        output = {}
        for k, v in template.items():
            output[k] = Template(v).safe_substitute(response_dict)

        raw_value = output.get("generated_metadata", {})
        output["generated_metadata"] = ast.literal_eval(raw_value) if isinstance(raw_value, str) else raw_value

        return InteractionResults.model_validate(output)

    except json.JSONDecodeError as e:
        logger.error(f"[extract_interaction_details] Failed to extract details:\n{e}")
        return InteractionResults()

    except ValidationError as e:
        logger.exception(f"[extract_interaction_details] Failed to create an InteractionResults instance:\n{e}")
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
