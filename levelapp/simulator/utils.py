"""
'simulators/utils.py': Utility functions for handling VLA interactions and requests.
"""
import json
import logging

import httpx
import arrow

from uuid import UUID
from typing import Dict, Any, Optional, List

from openai import OpenAI
from pydantic import ValidationError

from levelapp.simulator.schemas import InteractionResults


logger = logging.getLogger("batch-test-cloud-function")


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def extract_interaction_details(
        response_text: str,
) -> InteractionResults:
    """
    Extract interaction details from a VLA response.

    Args:
        response_text (str): The response text from the VLA.

    Returns:
        InteractionResults: The extracted interaction details.
    """
    try:
        payload = {
            "generated_reply": response_text,
            "generated_metadata": {},
            "guardrail_details": {},
            "interaction_type": ""
        }

        return InteractionResults(
            generated_reply=payload.get("generated_reply", ""),
            generated_metadata=payload.get("generated_metadata", {}),
            guardrail_details=payload.get("guardrail_details", {}),
            interaction_type=payload.get("interaction_type", ""),
        )

    except json.JSONDecodeError as err:
        logger.error(f"[extract_interaction_details] JSON decoding error: {err}")
        return InteractionResults(
            generated_reply="VLA request failed.",
            generated_metadata={},
            guardrail_details={},
            interaction_type="",
        )

    except ValidationError as err:
        logger.error(f"[extract_interaction_details] Pydantic validation error: {err}")
        return InteractionResults(
            generated_reply="VLA request failed.",
            generated_metadata={},
            guardrail_details={},
            interaction_type="",
        )


async def async_interaction_request(
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
) -> Optional[httpx.Response]:
    """
    Perform an asynchronous VLA request.

    Args:
        url (str): The URL to send the request to.
        headers (Dict[str, str]): The headers to include in the request.
        payload (Dict[str, Any]): The payload to send in the request.

    Returns:
        Optional[httpx.Response]: The response from the VLA request, or None if an error occurred.
    """
    try:
        logger.info(f"[async_vla_request] VLA request payload:\n{payload}\n---")
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(url=url, headers=headers, json=payload)
            logger.info(f"[async_vla_request] VLA response:\n{response.text}\n---")
            response.raise_for_status()

            return response

    except httpx.HTTPStatusError as http_err:
        logger.error(f"[async_vla_request] HTTP error: {http_err.response.text}", exc_info=True)

    except httpx.RequestError as req_err:
        logger.error(f"[async_vla_request] Request error: {str(req_err)}", exc_info=True)

    return None


def parse_date_value(raw_date_value: Optional[str], default_date_value: Optional[str] = "") -> str:
    """
    Cleans and parses a dehumanized relative date string to ISO format.

    Args:
        raw_date_value (Optional[str]): The raw date value to parse.
        default_date_value (Optional[str]): The default value to return if parsing fails. Defaults to an empty string.

    Returns:
        str: The parsed date in ISO format, or the default value if parsing fails.
    """
    if not raw_date_value:
        logger.info(f"[parse_date_value] No raw value provided. returning default: '{default_date_value}'")
        return default_date_value

    clean = raw_date_value.replace("{{", "").replace("}}", "").replace("_", " ").strip()
    clean += 's' if not clean.endswith('s') else clean

    try:
        arw = arrow.utcnow()
        parsed_date = arw.dehumanize(clean).utcnow().format('YYYY-MM-DD')
        return parsed_date

    except arrow.parser.ParserError as e:
        logger.error(f"[parse_date_value] Failed to parse date: '{clean}'\nParserError: {str(e)}", exc_info=True)
        return default_date_value

    except ValueError as e:
        logger.error(f"[parse_date_value] Invalid date value: '{clean}'\nValueError: {str(e)}", exc_info=True)
        return default_date_value

    except Exception as e:
        logger.error(f"[parse_date_value] Unexpected error.\nException: {str(e)}", exc_info=True)
        return default_date_value


def calculate_average_scores(scores: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Helper function that calculates the average scores for a dictionary of score lists.

    Args:
        scores (Dict[str, List[float]]): A dictionary where keys are identifiers and values are lists of scores.

    Returns:
        Dict[str, float]: A dictionary with average scores rounded to three decimal places.
    """
    def average(values: List[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    return {key: average(values) for key, values in scores.items()}


def summarize_verdicts(verdicts: List[str], judge: str, max_bullets: int = 5) -> List[str]:
    """
    Summarize the justifications for each judge.

    Args:
        verdicts (List[str]): A list of justifications.
        judge (str): The judge or evaluator (provider) name for context.
        max_bullets (int): The maximum number of bullets allowed per judge.

    Returns:
        List[str]: The summarized justifications.
    """
    if not verdicts:
        return []

    prompt = f"""
    You are reviewing evaluation justifications from LL judges about replies generated by a virtual leasing agent.\n
    Each justification contains the judge's assessment of how well the agent's response matched the expected reply.\n
    Your task is to identify and summarize only the **negative points**, such as errors, misunderstandings,
    missing information, or failure to meet expectations.\n
    Return up to {max_bullets} bullet points. Be concise and start each point with '- '\n\n
    ---
    - Judge: {judge}
    - Justifications:\n{chr(10).join(verdicts)}\n
    """

    client = OpenAI()

    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        bullet_points = [point.strip() for point in result.split('- ') if point.strip()]

        return bullet_points

    except Exception as e:
        logger.error(f"[summarize_justifications] Error during summarization: {str(e)}", exc_info=True)
        return []
