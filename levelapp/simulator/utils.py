"""
'simulators/aspects.py': Utility functions for handling VLA interactions and requests.
"""
import httpx

from typing import Any, Dict, List, Union


from levelapp.clients import ClientRegistry
from levelapp.config.prompts import MULTI_TURN_EVALUATION_PROMPT_TEMPLATE
from levelapp.aspects import MonitoringAspect, MetricType, logger


def set_by_path(obj: Dict, path: str, value: Any) -> None:
    """
    Sets a value in a nested dictionary using JSON path-like notation.

    Args:
        obj (dict): Dictionary to modify.
        path (str): Path (e.g., "a.b[0].c") indicating where to set the value.
        value (Any): Value to assign at the specified path.

    Returns:
        None
    """
    parts = path.split(".")
    current = obj

    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1

        try:
            # Handle list index access, e.g., key[0] or [1]
            if '[' in part and ']' in part:
                key, idx = part.split('[')
                idx = int(idx.rstrip(']'))

                # If we have a key before the list
                if key:
                    if key not in current or not isinstance(current[key], list):
                        current[key] = []
                    while len(current[key]) <= idx:
                        current[key].append({})
                    target = current[key]
                else:
                    if not isinstance(current, list):
                        print("[set_by_path][WARNING] Expected a list at this level.")
                        return
                    while len(current) <= idx:
                        current.append({})
                    target = current

                if is_last:
                    target[idx] = value
                else:
                    if not isinstance(target[idx], dict):
                        target[idx] = {}
                    current = target[idx]

            else:
                # Regular dictionary key
                if is_last:
                    current[part] = value
                else:
                    if part not in current or not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]

        except (KeyError, IndexError, TypeError, AttributeError) as e:
            print(f"[set_by_path][ERROR] Error type <{e.__class__.__name__}> : {e.args[0]}")
            return


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
        interaction_summaries: List[str],
        verdicts: List[str],
        judge: str,
        max_bullets: int = 5
) -> List[str]:
    client_registry = ClientRegistry()
    client = client_registry.get(provider=judge)

    try:
        verdicts = chr(10).join(verdicts)
        dialogue_trace = chr(10).join(interaction_summaries)

        prompt = MULTI_TURN_EVALUATION_PROMPT_TEMPLATE.format(
            dialogue_trace=dialogue_trace,
            judge=judge,
            verdicts=verdicts,
            max_bullets=max_bullets,
        )

        response = client.call(message=prompt)
        parsed = client.parse_response(response=response)
        # striped = parsed.get("output", "").strip("")
        # bullet_points = [point.strip() for point in striped.split("- ") if point.strip()]

        return parsed

    except Exception as e:
        logger.error(f"[summarize_justifications] Error during summarization: {str(e)}", exc_info=True)
        return []


if __name__ == '__main__':
    interaction_summaries = [
        "[T0][A][task=Information Query][s=3.0][e=0.80][g=0]] Facts: [The agent provided comprehensive information and invited further questions., All key points covered]",
        "[T1][A][task=Service Transaction][s=3.0][e=0.80][g=0]] Facts: [The AGENT response precisely matches the expected reply., Exact match, The agent's reply is identical to the expected reply, perfectly confirming the surgical appointment requested by the user.]",
        "[T2][A][task=Information Query][s=3.0][e=0.80][g=0]] Facts: [The agent's reply closely matches the expected reply with precise information., Accurate and sufficient]"
    ]

    judge = "gemini"

    verdicts = [
        "The AGENT's reply fully matches and addresses the user's inquiry.",
        "The agent's reply matches the expected confirmation of the appointment.",
        "The AGENT's reply matches the EXPECTED response perfectly."
    ]

    max_bullets = 5

    summary = summarize_verdicts(
        interaction_summaries=interaction_summaries,
        judge=judge,
        verdicts=verdicts,
        max_bullets=max_bullets
    )

    print(summary)