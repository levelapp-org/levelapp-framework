"""levelapp/assessor/gauger.py"""
import asyncio
from typing import List, Dict, Any

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
        self.metric_sets = metric_overrides or self.DEFAULT_METRICS

    async def evaluate_profile(
            self,
            pipeline_result: PipelineResult,
            expected_docs: List[Document],
            query: str
    ) -> PipelineResult:
        """
        Perform evaluation across retrieval and generation stages for a given profile.
        """
        effective_scopes = []

        for scope in self.metric_scopes:
            if scope == "generation" and not pipeline_result.augmented_answer:
                continue

            effective_scopes.append(scope)

        if not effective_scopes:
            summary = EvaluationSummary(
                query=query,
                comparative_metrics=[],
                report={"profile": self.profile_name, "avg_score": 0.0, "scopes_evaluated": []},
            )
            pipeline_result.metrics = summary

            return pipeline_result

        scope_tasks = [
            asyncio.create_task(
                self._evaluate_scope(
                    scope=scope,
                    expected_docs=expected_docs,
                    pipeline_result=pipeline_result,
                    query=query
                )
            ) for scope in effective_scopes
        ]
            
        scope_results = await asyncio.gather(*scope_tasks, return_exceptions=False)
        
        all_metrics = [m for scope in scope_results for m in scope.get("comparative_metrics", [])]
        avg_score = (sum(r.get("score", 0.0) for r in all_metrics) / len(all_metrics)) if all_metrics else 0.0

        summary = EvaluationSummary(
            query=query,
            comparative_metrics=all_metrics,
            report={
                "profile": self.profile_name,
                "avg_score": avg_score,
                "metric_scopes": self.metric_scopes,
            }
        )
        
        pipeline_result.metrics = summary
        
        return pipeline_result

    async def _evaluate_scope(
            self,
            scope: str,
            expected_docs: List[Document],
            pipeline_result: PipelineResult,
            query: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single scope (retrieval, generation, etc.)
        """
        results: List[Dict[str, Any]] = []
        metric_names = self.metric_sets.get(scope, [])

        expected_list = [d.model_dump() for d in expected_docs] if expected_docs else []
        retrieved_list = [d.model_dump() for d in (pipeline_result.retrieved_docs or [])]

        for name in metric_names:
            try:
                metric_instance = MetricRegistry.get(name=name)

                if scope == "retrieval":
                    output = metric_instance.compute(
                        expected=expected_list,
                        actual=retrieved_list
                    )

                elif scope == "generation":
                    if not pipeline_result.augmented_answer:
                        results.append(
                            {
                                "scope": scope,
                                "name": name,
                                "score": 0.0,
                                "metadata": {"error": "no_generated_answer"},
                            }
                        )
                        continue

                    references = "\n".join(doc.content for doc in expected_docs)
                    output = metric_instance.compute(
                        generated=pipeline_result.augmented_answer,
                        reference=references,
                    )

                else:
                    results.append(
                        {
                            "scope": scope,
                            "name": name,
                            "score": 0.0,
                            "metadata": {"error": f"unsupported scope: '{scope}'"},
                        }
                    )
                    continue

                score = float(output.get("score", 0.0))
                metadata = output.get("metadata", {})
                results.append(
                    {
                        "scope": scope,
                        "name": name,
                        "score": score,
                        "metadata": metadata,
                    }
                )

            except KeyError:
                results.append(
                    {
                        "scope": scope,
                        "name": name,
                        "score": 0.0,
                        "metadata": {"error": f"Metric '{name}' not registered."},
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "scope": scope,
                        "name": name,
                        "score": 0.0,
                        "metadata": {"error": str(e)},
                    }
                )

        return {"scope": scope, "comparative_metrics": results}
