"""levelapp/assessor/orchestrator.py"""
import uuid

from typing import List, Dict, Any

from levelapp.assessor.gauger import ProfileGauger
from levelapp.assessor.registry import StrategyRegistry
from levelapp.assessor.schemas import Document, PipelineResult


class ProfileOrchestrator:
    def __init__(self, registry: StrategyRegistry, endpoint=None):
        self._registry = registry
        self._endpoint = endpoint

    async def run_profile(
            self,
            profile_name: str,
            query: str,
            documents: List[Document],
            profile_config: Dict[str, Any],
    ) -> PipelineResult:
        """
        Executes a single profile configuration.
        Example structure of profile_config:
        {
            "chunking": {"name": "semantic", "config": {"chunk_size": 300}},
            "embedding": {"name": "transformer", "config": {"model": "all-MiniLM-L6-v2"}},
            "retrieval": {"name": "faiss", "config": {"top_k": 5}},
            "generation": {"name": "llm_openai", "config": {"model": "gpt-4o-mini"}},
        }
        """
        pipeline_id = str(uuid.uuid4())

        chunker = self._build_strategy("chunking", profile_config)
        embedding = self._build_strategy("embedding", profile_config)
        retrieval = self._build_strategy("retrieval", profile_config)
        generator = self._build_strategy("generation", profile_config)

        all_chunks = []
        for doc in documents:
            chunks = await chunker.run(doc)
            all_chunks.extend(chunks)

        embeddings = await embedding.run(all_chunks)
        retrieved_docs = await retrieval.run(query, embeddings)

        gauger = ProfileGauger()
        evaluated_results = await gauger.evaluate_pipeline(
            pipeline_result=PipelineResult(
                pipeline_id=pipeline_id,
                strategies={
                    "profile": profile_name,
                    "chunking": chunker.name,
                    "embedding": embedding.name,
                    "retrieval": retrieval.name,
                    "generation": generator.name if generator else None,
                },
                retrieved_docs=retrieved_docs,
                augmented_answer=None,
            ),
            expected_docs=documents,
        )

        if generator:
            generated_answer = await generator.run(query, retrieved_docs)
            evaluated_results.augmented_answer = generated_answer

        return evaluated_results

    def _build_strategy(self, strategy_type: str, profile_config: Dict[str, Any], optional: bool = False):
        """Builds a strategy instance from registry + profile config."""
        strat_data = profile_config.get(strategy_type)

        if not strat_data:
            if optional:
                return None
            raise ValueError(f"[ProfileOrchestrator] Missing required config for '{strategy_type}' in profile")

        name = strat_data.get("name")
        config = strat_data.get("config", {})
        strategy_cls = self._registry.get(strategy_type=strategy_type, name=name)

        return strategy_cls(name=name, config=config)


if __name__ == '__main__':
    import asyncio

    from levelapp.assessor.strategies.chunking import SimpleChunkingStrategy
    from levelapp.assessor.strategies.embedding import MockEmbeddingStrategy
    from levelapp.assessor.strategies.retrieval import CosineRetrievalStrategy
    from levelapp.assessor.strategies.generation import MockGenerationStrategy

    registry = StrategyRegistry()
    registry.register("chunking", "simple", SimpleChunkingStrategy)
    registry.register("embedding", "mock", MockEmbeddingStrategy)
    registry.register("retrieval", "cosine", CosineRetrievalStrategy)
    registry.register("generation", "mock", MockGenerationStrategy)

    orchestrator = ProfileOrchestrator(registry=registry, endpoint=None)

    # === Example profile config ===
    basic_profile = {
        "chunking": {"name": "simple", "config": {"chunk_size": 300}},
        "embedding": {"name": "mock", "config": {}},
        "retrieval": {"name": "cosine", "config": {"top_k": 5}},
        "generation": {"name": "mock", "config": {}},
    }

    query_ = "What is the role of mitochondria?"
    documents_ = [
        Document(
            id="0001",
            content="Mitochondria are the powerhouses of the cell. They produce ATP.",
            source_type="document",
        ),
        Document(
            id="0002",
            content="Cells contain various organelles including mitochondria and ribosomes.",
            source_type="document",
        ),
    ]

    results = asyncio.run(
        orchestrator.run_profile(
            profile_name="basic",
            query=query_,
            documents=documents_,
            profile_config=basic_profile,
        )
    )

    print(results.model_dump_json(indent=2))
