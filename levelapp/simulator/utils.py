"""
'simulators/aspects.py': Utility functions for handling VLA interactions and requests.
"""
import httpx

from typing import Any, Dict, List, Union


from levelapp.clients import ClientRegistry
from levelapp.config.prompts import SUMMARIZATION_PROMPT_TEMPLATE
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
