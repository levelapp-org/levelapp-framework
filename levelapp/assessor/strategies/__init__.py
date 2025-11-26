"""levelapp/assessor/strategies/__init__.py"""
from levelapp.assessor.registry import StrategyRegistry
from levelapp.assessor.schemas import MetricSpec


def register_basic_profile_strategies(registry: StrategyRegistry):
    """Register all strategies for the basic profile."""

    # Import strategies
    from .chunking.fixed_size_chunker import FixedSizeChunker
    from .embedding.minilm_embedder import MiniLMEmbedder
    from .retrieval.cosine_retriever import CosineRetriever
    from .generation.basic_model_generator import BasicModelGenerator

    # Register chunking strategy
    registry.register_strategy(
        strategy_class=FixedSizeChunker,
        name="fixed_size_chunker",
        description="Splits text into fixed-size chunks with configurable overlap",
        metrics=[
            MetricSpec(name="chunk_size_consistency", weight=0.3),
            MetricSpec(name="content_preservation", weight=0.7)
        ]
    )

    # Register embedding strategy
    registry.register_strategy(
        strategy_class=MiniLMEmbedder,
        name="minilm_embedder",
        description="Uses SentenceTransformer MiniLM model for efficient embeddings",
        metrics=[
            MetricSpec(name="embedding_quality", weight=0.8),
            MetricSpec(name="embedding_latency", weight=0.2)
        ]
    )

    # Register retrieval strategy
    registry.register_strategy(
        strategy_class=CosineRetriever,
        name="cosine_retriever",
        description="Retrieves documents using cosine similarity on embeddings",
        metrics=[
            MetricSpec(name="retrieval_precision", weight=0.6),
            MetricSpec(name="retrieval_recall", weight=0.4)
        ]
    )

    # Register generation strategy
    registry.register_strategy(
        strategy_class=BasicModelGenerator,
        name="basic_model_generator",
        description="Generates answers using HuggingFace transformer models",  # FIXED
        metrics=[
            MetricSpec(name="answer_relevance", weight=0.5),
            MetricSpec(name="answer_accuracy", weight=0.5)
        ]
    )
