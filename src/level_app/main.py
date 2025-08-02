"""Example integration with MetricRegistry"""
from levelapp.utils.monitoring import FunctionMonitor
from typing import Dict, Any

class BaseMetric:
    pass

class FuzzyRatio(BaseMetric):
    @FunctionMonitor.register(
        name="fuzzy_ratio_metric",
        cached=True,
        enable_timing=True
    )
    def compute(self, generated: str, reference: str) -> Dict[str, Any]:
        return {"score": 0.95}



if __name__ == '__main__':
    metric = FuzzyRatio()
    result = metric.compute("hello", "hello!")  # Auto-logged and cached
    print(result)  # Output: {'score': 0.95}
