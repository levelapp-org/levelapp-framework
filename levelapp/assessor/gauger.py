"""levelapp/assessor/gauger.py"""
import asyncio
from typing import List, Dict

from levelapp.assessor.schemas import PipelineResult, EvaluationSummary, Document
from levelapp.metrics import MetricRegistry


class ProfileGauger:
    """
    Gauges the performance of a full profile run, evaluating retrieval, generation,
    and overall pipeline quality based on the configured metric suite.
    """

    DEFAULT_SCOPES = ["retrieval", "generation"]
    DEFAULT_METRICS = {
        "retrieval": ["precision", "recall", "ndcg", "redundancy"],
        "generation": ["fluency", "faithfulness", "relevance"]
    }

    def __init__(
            self, 
            profile_name: str,
            metric_scopes: List[str] | None = None,
            metric_overrides: Dict[str, List[str]] | None = None,
    ):
        self.profile_name = profile_name
        self.metric_scopes = metric_scopes or self.DEFAULT_SCOPES
        self.metric_overrides = metric_overrides or self.DEFAULT_METRICS

    async def evaluate_profile(
            self,
            pipeline_result: PipelineResult,
            expected_docs: List[Document],
            query: str
    ) -> PipelineResult:
        """
        Perform evaluation across retrieval and generation stages for a given profile.
        """
        tasks = []
        
        for scope in self.metric_scopes:
            tasks.append(
                self._evaluate_scope(scope, expected_docs, pipeline_result, query)
            )
            
        scope_results = await asyncio.gather(*tasks)
        
        all_metrics = [m for scope in scope_results for m in scope["comparative_metrics"]]
        avg_score = sum(r["score"] for r in all_metrics) / len(all_metrics) if all_metrics else 0.0

        summary = EvaluationSummary(
            query=query,
            comparative_metrics=all_metrics,
            report={
                "profile": self.profile_name,
                "avg_score": avg_score,
                "scope_evaluated": self.metric_scopes,
            }
        )
        
        pipeline_result.metrics = summary
        
        return pipeline_result

    def _evaluate_scope(self, scope, expected_docs, pipeline_result, query):
        pass