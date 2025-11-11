"""levelapp/assessor/orchestrator.py"""
from __future__ import annotations

import asyncio
import logging

from dataclasses import dataclass, field
from typing import Dict, Any, List

from levelapp.assessor.builder import ProfileBuilder
from levelapp.assessor.gauger import EmbeddingGauger, RetrievalGauger, GenerationGauger
from levelapp.assessor.registry import StrategyRegistry
from levelapp.assessor.schemas import Document
from levelapp.endpoint.client import EndpointConfig
from levelapp.endpoint.manager import EndpointConfigManager


logger = logging.getLogger(__name__)


@dataclass
class ProfileCard:
    name: str
    config: Dict[str, Any]


@dataclass
class PipelineResult:
    profile_card: ProfileCard
    retrieved_docs: List[Document] = field(default_factory=list)
    augmented_answer: str | None = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssessmentResults:
    profile_results: PipelineResult
    user_result: PipelineResult
    errors: List[str] = field(default_factory=list)


@dataclass
class ComparisonSummary:
    profile_aggregated: Dict[str, float]
    user_aggregated: Dict[str, float]
    delta: Dict[str, float]


class AssessmentOrchestrator:
    """
    Orchestrator that:
      - sets up the endpoint configuration for the user's system
      - builds a profile via ProfileBuilder
      - runs the predefined profile pipeline and the user's system in parallel
      - evaluates both outputs using gaugers and aggregates results
    """

    DEFAULT_TIMEOUT = 30.0

    def __init__(
            self,
            endpoint_config: EndpointConfig,
            registry: StrategyRegistry | None = None,
            embedding_gauger_cls=EmbeddingGauger,
            retrieval_gauger_cls=RetrievalGauger,
            generation_gauger_cls=GenerationGauger,
            timeout: float | None = None,
    ) -> None:
        self.endpoint_config = endpoint_config
        self.endpoint_cm = EndpointConfigManager()
        self.registry = registry or StrategyRegistry()
        self.embedding_gauger = embedding_gauger_cls
        self.retrieval_gauger = retrieval_gauger_cls
        self.generation_gauger = generation_gauger_cls
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        # runtime attributes
        self._profile_card: ProfileCard | None = None
        self._initialized = False

    def setup(self, endpoint_config: EndpointConfig) -> None:
        self.endpoint_config = endpoint_config

        if not self.endpoint_cm:
            self.endpoint_cm = EndpointConfigManager()

        self.endpoint_cm.set_endpoints(endpoints_config=[endpoint_config])
        self._initialized = True
        print(f"[AssessmentOrchestrator] Endpoint configured: '{endpoint_config.name}'")

    def build_profile(self, profile_name: str, profile_config: Dict[str, Any] | None = None):
        """
        Build a 'ProfileCard' using 'ProfileBuilder' and 'StrategyRegistry'.
        """
        print(f"[AssessmentOrchestrator] Building profile '{profile_name}'")

        builder = ProfileBuilder(registry=self.registry)

        # TODO-0: We need to find out how to pass the profile configuration to the builder.
        profile = builder.build(profile_name=profile_name)

        if isinstance(profile_config, ProfileCard):
            card = profile

        else:
            card = ProfileCard(name=profile_name, config=profile_config)

        self._profile_card = card
        logger.debug(f"[AssessmentOrchestrator] Profile build: '{card}'")

        return card

    async def run_evaluation(
            self,
            profile_card: ProfileCard,
            documents: List[Dict[str, Any]],
            query: str | None = None,
    ) -> AssessmentResults:
        """
        Run the predefined profile pipeline and the user's system in parallel,
        evaluate both, and return results
        """
        if not self._initialized:
            try:
                self.setup(endpoint_config=self.endpoint_config)
            except Exception as e:
                raise RuntimeError(f"[AssessmentOrchestrator] not initialized and auto-setup failed:\n{e}]")

        errors: List[str] = []
        profile_task = asyncio.create_task(
            self._run_profile_pipeline(profile_card, documents, query)
        )
        user_task = asyncio.create_task(
            self._run_user_system(documents, query)
        )

        try:
            profile_result, user_result = await asyncio.gather(profile_task, user_task)

        except Exception as e:
            logger.exception(f"[AssessmentOrchestrator] Parallel execution of profile/user pipeline failed:\n{e}]")

            for t in (profile_task, user_task):
                if not t.done():
                    t.cancel()

            raise

        try:
            await self._evaluate_both(profile_result, user_result, profile_card)

        except Exception as e:
            logger.exception(f"[AssessmentOrchestrator] Evaluation failed:\n{e}]")
            errors.append(str(e))

        return AssessmentResults(
            profile_results=profile_result,
            user_result=user_result,
            errors=errors,
        )

    async def _run_profile_pipeline(
            self,
            profile_card: ProfileCard,
            documents: List[Dict[str, Any]],
            query: str | None = None,
    ) -> PipelineResult:
        """
        Run the profile's pipeline using local strategies resolved from registry.
        """
        logger.debug(f"[AssessmentOrchestrator] Running profile pipeline: '{profile_card}'")

        cfg = profile_card.config
        chunker_cfg = cfg.get("chunker", {})
        embedder_cfg = cfg.get("embedder", {})
        retriever_cfg = cfg.get("retriever", {})
        generator_cfg = cfg.get("generator", {})

        chunker_cls = self.registry.get_strategy(level="chunking", name=chunker_cfg.get("name"))
        chunker = chunker_cls(name=chunker_cfg.get("name"), config=chunker_cfg.get("config", {}))
        embedder_cls = self.registry.get_strategy(level="embedder", name=embedder_cfg.get("name"))
        embedder = embedder_cls(name=embedder_cfg.get("name"), config=embedder_cls.get("config", {}))
        retriever_cls = self.registry.get_strategy(level="retriever", name=retriever_cfg.get("name"))
        retriever = retriever_cls(name=retriever_cfg.get("name"), config=embedder_cfg.get("config", {}))
        generator_cls = self.registry.get_strategy(level="generator", name=generator_cfg.get("name"))
        generator = generator_cls(name=generator_cfg.get("name"), config=generator_cfg.get("config", {}))

        # TODO-1: Maybe we can create a 'Chunk' model later?
        all_chunks: List[Dict[str, Any]] = []

        # TODO-2: Optionally, we can construct a pipeline to run the whole process in a clean way.
        for doc in documents:
            chunks = await chunker.run(doc) if asyncio.iscoroutine(doc) else chunker.run(doc)
            all_chunks.extend(chunks)

        embedding = await embedder.run(all_chunks) if asyncio.iscoroutine(all_chunks) else embedder.run(all_chunks)
        retrieved_docs = await retriever.run(embedding) if asyncio.iscoroutine(embedding) else embedder.run(embedding)

        generated = None
        if generator:
            generated = await generator.run(embedding) if asyncio.iscoroutine(embedding) else generator.run(embedding)

        profile_result = PipelineResult(
            profile_card=profile_card,
            retrieved_docs=retrieved_docs,
            augmented_answer=generated,
        )

        return profile_result

    async def run_user_system(
            self,
            documents: List[Dict[str, Any]],
            query: str | None = None
    ) -> PipelineResult:
        """
        Call the user system endpoint.
        """
        logger.debug("[AssessmentOrchestrator] Running user system]")
        request_payload = {
            "query": query,
            "documents": documents,
        }
        profile_card = ProfileCard(
            name="user_system",
            config={"request_payload": request_payload}
        )

        response_details: Dict[str, Any] = {}
        response = None

        try:
            # TODO-3: We need to account for the case where the user system accepts a URL or raw text as documents.

            response = await self.endpoint_cm.send_request(
                endpoint_config=self.endpoint_config,
                context=request_payload,
            )

            if response is None:
                logger.error("[AssessmentOrchestrator] No response received from user system")
                return PipelineResult(profile_card=profile_card)

            if response.status_code != 200:
                logger.error(f"[AssessmentOrchestrator] Request failed with status: {response.status_code}")
                result = PipelineResult(profile_card=profile_card,)

            mappings = self.endpoint_config.response_mapping
            response_details = self.endpoint_cm.extract_response_data(
                response=response,
                mappings=mappings,
            )

        except asyncio.TimeoutError:
            logger.exception("[AssessmentOrchestrator] User system call timed out.")

        except Exception as e:
            logger.exception(f"[AssessmentOrchestrator] Exception calling user system:\n{e}")

        finally:
        # TODO-4: We need to extract a 'list[Documents]' from the response content.
        # TODO-4: Or change the 'retrieved_docs' type to something more flexible (e.g., Dict[Any, Any]
            result = PipelineResult(
                profile_card=profile_card,
                retrieved_docs=response_details.get("retrieved_docs", {}),
                augmented_answer=response_details.get("augmented_answer", ""),
                # TODO-5: maybe we add the raw response here.
            )

        return result

    async def _evaluate_both(self, profile_result: PipelineResult, user_result: PipelineResult, profile_card: ProfileCard) -> None:
        """
        Use dedicated gaugers to evaluate retrieval/embedding/generation stages for both profile_result and user_result.
        Mutates PipelineResult.metrics with per-stage metric lists and aggregated score.
        """
        embedding_gauger = self.embedding_gauger()
        retrieval_gauger = self.retrieval_gauger()
        generation_gauger = self.generation_gauger()

        async def eval_retrieval(pr: PipelineResult, label: str):
            try:
                summary = retrieval_gauger.evaluate(  # TODO-5: Implement the 'evaluate' meth (no sh**t!)
                    # I don't like this ..
                    expected_docs=[d for d in profile_card.config.get("reference_docs", [])],
                    actual_docs=pr.retrieved_docs, query=profile_card.name
                )
                pr.metrics["retrieval"] = summary.model_dump() if hasattr(summary, "model_dump") else summary
            except Exception as e:
                logger.exception("Retrieval gauger failed for %s: %s", label, e)
                pr.metrics["retrieval"] = {"error": str(e)}

        async def eval_generation(pr: PipelineResult, label: str):
            try:
                if pr.augmented_answer:
                    summary = generation_gauger.evaluate(query=profile_card.name, generated=pr.augmented_answer, context_docs=pr.retrieved_docs)
                    pr.metrics["generation"] = summary.model_dump() if hasattr(summary, "model_dump") else summary
                else:
                    pr.metrics["generation"] = {"skipped": True}
            except Exception as e:
                logger.exception("Generation gauger failed for %s: %s", label, e)
                pr.metrics["generation"] = {"error": str(e)}

        await asyncio.gather(
            eval_retrieval(profile_result, "profile"),
            eval_generation(profile_result, "profile"),
            eval_retrieval(user_result, "user"),
            eval_generation(user_result, "user"),
        )

        try:
            if profile_card.config.get("check_embeddings", False):
                e_summary_profile = embedding_gauger.evaluate(expected_docs=[], actual_docs=profile_result.retrieved_docs, query=profile_card.name)
                profile_result.metrics["embedding"] = e_summary_profile.model_dump() if hasattr(e_summary_profile, "model_dump") else e_summary_profile

                e_summary_user = embedding_gauger.evaluate(expected_docs=[], actual_docs=user_result.retrieved_docs, query=profile_card.name)
                user_result.metrics["embedding"] = e_summary_user.model_dump() if hasattr(e_summary_user, "model_dump") else e_summary_user
        except Exception as e:
            logger.exception("Embedding gauger failed: %s", e)

        profile_result.metrics["aggregated"] = self.aggregate_scores(profile_result.metrics, profile_card)
        user_result.metrics["aggregated"] = self.aggregate_scores(user_result.metrics, profile_card)

    # TODO-5: Refactor this piece of sh**t method to reduce the complexity.
    def aggregate_scores(self, metrics_blob: Dict[str, Any], profile_card: ProfileCard) -> Dict[str, float]:
        """
        Aggregate per-scope metrics (retrieval/generation/embedding) using profile weights.
        Returns a dict of aggregated values (per-scope and global).
        """
        weights = profile_card.config.get("weights", {})
        strategy = profile_card.config.get("aggregation_strategy", "weighted")

        # collect numeric scores (we expect each scope metrics to contain 'avg_score' or similar)
        scope_scores: Dict[str, float] = {}
        for scope in ("retrieval", "generation", "embedding"):
            data = metrics_blob.get(scope)
            if not data:
                continue
            # metric normalization heuristics
            if isinstance(data, dict) and "report" in data:
                score = data["report"].get("avg_score") if isinstance(data["report"], dict) else None
            else:
                score = None

            # fallback tries
            score = float(score) if score is not None else 0.0
            scope_scores[scope] = score

        # aggregation
        if strategy == "weighted":
            total = 0.0
            for s, sc in scope_scores.items():
                w = float(weights.get(s, 1.0))  # default weight 1.0
                total += sc * w
            # normalize by sum of weights to produce 0..1-like metric
            weights_sum = sum(float(weights.get(s, 1.0)) for s in scope_scores.keys()) or 1.0
            global_score = total / weights_sum
        else:
            # simple average
            if scope_scores:
                global_score = sum(scope_scores.values()) / len(scope_scores)
            else:
                global_score = 0.0

        return {"global": float(global_score), "by_scope": scope_scores}

    # I am tired boss..
