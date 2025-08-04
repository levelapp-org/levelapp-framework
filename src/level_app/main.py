"""Example integration with MetricRegistry"""
from rapidfuzz import utils

from levelapp.clients import ClientRegistry
from levelapp.metrics import MetricRegistry
from levelapp.utils.monitoring import FunctionMonitor


if __name__ == '__main__':

    metric_registry = MetricRegistry()
    metrics = ["fuzzy_ratio", "partial_ratio", "exact_match", "levenshtein", "jaro_winkler", "hamming", "prefix_match"]

    for metric in metrics:
        metric_instance = metric_registry.get(metric, processor=utils.default_process)
        score = metric_instance.compute("hello!", "hello, world!")
        print(f"{metric} score: {score['score']:.2f}")

    # Get statistics
    stats = FunctionMonitor.get_stats("fuzzy_ratio")
    print("\nFunction Statistics:")
    print(f"Name: {stats['name']}")
    print(f"Arguments: {stats.get('execution_count', 'N/A')}")
    print(f"Cache Hits: {stats['cache_info'].hits if stats['cache_info'] else 0}")
    print(f"Cache Size: {stats['cache_info'].currsize if stats['cache_info'] else 0}")

    clients_registry = ClientRegistry()
    clients_get = clients_registry.get("openai")
    result = clients_get.call(message="What is the capital of France?")
    print(f"Client call result: {result}")

    print(f"Monitored functions:\n{FunctionMonitor.list_monitored_functions()}")

    stats = FunctionMonitor.get_stats("OpenAIClient.call")
    print("\nFunction Statistics:")
    print(f"Name: {stats['name']}")
    print(f"Arguments: {stats.get('execution_count', 'N/A')}")
    print(f"Cache Hits: {stats['cache_info'].hits if stats['cache_info'] else 0}")
    print(f"Cache Size: {stats['cache_info'].currsize if stats['cache_info'] else 0}")