"""levelapp/assessor/gauger.py"""
from typing import List

from levelapp.assessor.schemas import PipelineResult, EvaluationSummary, Document
from levelapp.metrics import MetricRegistry


class RetrievalGauger:
    """A quantitative evaluator for the retrieval stage of the RAG pipeline."""

    DEFAULT_METRICS = ["precision", "recall", "ndcg", "redundancy"]

    def __init__(self, metrics: List[str] = None):
        self.metric_names = metrics or self.DEFAULT_METRICS

    def evaluate(
            self,
            expected_docs: List[Document],
            actual_docs: List[Document],
            query: str
    ) -> EvaluationSummary:
        """
        Compute retrieval metrics by comparing actual retrieved documents
        against expected (reference) retrieved documents.
        """
        results = []

        for name in self.metric_names:
            metric_cls = MetricRegistry.get(name=name)
            metric_instance = metric_cls() if callable(metric_cls) else metric_cls  # ISSUE: 'BaseMetric' object is not callable

            output = metric_instance.compute(expected_docs, actual_docs)
            results.append(
                {
                    "name": name,
                    "score": output.get("score", 0.0),
                    "metadata": output.get("metadata", {}),
                }
            )

        avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

        # ISSUE: Parameter 'reference_run_id' unfilled
        # ISSUE: Parameter 'base_pipeline_id' unfilled
        # ISSUE: Parameter 'reference_run_id' unfilled
        # ISSUE: Parameter 'base_pipeline_id' unfilled
        return EvaluationSummary(
            query=query,
            comparative_metrics=results,
            report={"avg_score": avg_score}
        )

    # ISSUE: Use asynchronous features in this function or remove the `async` keyword.
    async def evaluate_pipeline(
            self,
            pipeline_result: PipelineResult,
            expected_docs: List[Document],
    ) -> PipelineResult:
        """
        Evaluate a Pipeline and attach computed metrics.
        """
        summary = self.evaluate(
            expected_docs=expected_docs,
            actual_docs=pipeline_result.retrieved_docs,
            query=pipeline_result.strategies.get("query", "N/A"),
        )
        pipeline_result.metrics = summary
        return pipeline_result
