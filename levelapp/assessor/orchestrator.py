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
