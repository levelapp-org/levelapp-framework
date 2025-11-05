"""levelapp/assessor/orchestrator.py"""
import itertools
import asyncio
import uuid

from typing import List, Dict

from levelapp.assessor.gauger import RetrievalGauger
from levelapp.assessor.registry import StrategyRegistry
from levelapp.assessor.schemas import Document, PipelineResult, PipelineConfig


class GridSearchOrchestrator:
    def __init__(self, registry: StrategyRegistry, endpoint):
        self._registry = registry
        self._endpoint = endpoint

    async def run_grid_search(
            self,
            query: str,
            documents: List[Document],
            strategy_grid: Dict[str, List[str]],
    ) -> List[PipelineResult]:
        """
        Run all combinations of chunking/embedding/retrieval/generation strategies.
        """
        combinations = list(itertools.product(
            strategy_grid["chunking"],
            strategy_grid["embedding"],
            strategy_grid["retrieval"],
            strategy_grid.get("generation", [None]),
        ))

        results = []
        tasks = []

        for chunker, embedder, retriever, generator in combinations:
            pipeline_id = str(uuid.uuid4())
            tasks.append(self._execute_pipeline(
                pipeline_id=pipeline_id,
                query=query,
                documents=documents,
                chunker_name=chunker,
                embedder_name=embedder,
                retriever_name=retriever,
                generator_name=generator,
            ))

        results = await asyncio.gather(*tasks)
        return results

    async def _execute_pipeline(
            self,
            pipeline_id: str,
            query: str,
            documents: List[Document],
            chunker_name: str,
            embedder_name: str,
            retriever_name: str,
            generator_name: str | None = None,
    ) -> PipelineResult:
        chunker_cls = self._registry.get("chunking", chunker_name)
        embedder_cls = self._registry.get("embedding", embedder_name)
        retriever_cls = self._registry.get("retrieval", retriever_name)
        generator_cls = self._registry.get("generation", generator_name) if generator_name else None

        chunker = chunker_cls(name=chunker_name, config={})
        embedder = embedder_cls(name=embedder_name, config={})
        retriever = retriever_cls(name=retriever_name, config={})
        generator = generator_cls(name=generator_name, config={}) if generator_name else None

        all_chunks = []
        for doc in documents:
            chunks = await chunker.run(doc)
            all_chunks.extend(chunks)

        embeddings = await embedder.run(all_chunks)
        retrieved_docs = await retriever.run(query, embeddings)

        gauger = RetrievalGauger()

        evaluated_results = await gauger.evaluate_pipeline(
            pipeline_result=PipelineResult(
                pipeline_id=pipeline_id,
                strategies={
                    "chunking": chunker_name,
                    "embedding": embedder_name,
                    "retrieval": retriever_name,
                    "generation": generator_name,
                },
                retrieved_docs=retrieved_docs,
                augmented_answer=None
            ),
            expected_docs=documents,
        )

        generated_answer = None
        if generator_name:
            generated_answer = await generator.run(query, retrieved_docs)
            evaluated_results.augmented_answer = generated_answer

        return evaluated_results


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

    orchestrator = GridSearchOrchestrator(registry=registry, endpoint=None)

    query = "What is the role of mitochondria?"
    documents = [
        Document(
            id="0bd28cfec6794acf399466cf39f534bc",
            content="Mitochondria are the powerhouses of the cell. They produce ATP.",
            source_type="document",
        ),
        Document(
            id="88afd984e0241163fe472caffb6057c4",
            content="Cells contain various organelles including mitochondria and ribosomes.",
            source_type="document",
        )
    ]

    strategy_grid = {
        "chunking": ["simple"],
        "embedding": ["mock"],
        "retrieval": ["cosine"],
        "generation": ["mock"],
    }

    results_ = asyncio.run(orchestrator.run_grid_search(query, documents, strategy_grid))
    for res in results_:
        print(res.model_dump_json(indent=2))
        print("---")
