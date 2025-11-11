"""levelapp/assessor/builder.py"""
from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import List, Dict, Any

from levelapp.assessor.registry import StrategyRegistry, BaseStrategy
from levelapp.assessor.schemas import MetricSpec

logger = logging.getLogger(__name__)


@dataclass
class ProfileTemplate:
    name: str
    description: str
    strategies: Dict[str, str]  # {level_name: strategy_name}
    aggregation_strategy: str = "weighted"
    weights: Dict[str, float] = field(default_factory=lambda: {"retrieval": 1.0, "generation": 1.0})
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileCard:
    name: str
    config: Dict[str, Any]
    description: str
    aggregation_strategy: str
    weights: Dict[str, float]


class ProfileBuilder:
    """
    Constructs a ProfileCard by assembling predefined strategies and metrics
    for each evaluation level: chunking, embedding, retrieval, generation.
    """
    def __init__(self, registry: StrategyRegistry) -> None:
        self.registry = registry
        self.available_profiles: Dict[str, ProfileTemplate] = self._load_predefined_profiles()
        self.strategy_map: Dict[str, BaseStrategy] = {}
        self.metric_map: Dict[str, List[MetricSpec]] = {}

    def build(
            self,
            profile_name: str,
            overrides: Dict[str, Any] | None = None
    ) -> ProfileCard:  # Issue0 - Expected type 'ProfileCard', got 'ProfileCard | None' instead
        """
        Assemble the strategies and metrics into a complete ProfileCard.
        """
        if profile_name not in self.available_profiles:
            raise KeyError(f"[ProfileBuilder] Profile '{profile_name}' is not defined.")

        template = self.available_profiles[profile_name]
        profile_config: Dict[str, Any] = {}

        for level, name in template.strategies.items():  # Issue1 - Missing return statement on some paths
            strategy_cls = self.registry.get_strategy(level=level, name=name)

            if not strategy_cls:
                raise ValueError(f"[ProfileBuilder] Strategy '{level}' is not defined.")

            strategy = self.registry.get_strategy(name=name, level=level)
            self.strategy_map[level] = strategy

            metrics = self.attach_metrics(strategy)
            self.metric_map[level] = metrics

            profile_config[level] = {
                "name": name,
                "config": getattr(strategy, "config", {}),
                "metrics": [m.model_dump() if hasattr(m, "model_dump") else m for m in metrics]
            }

        # Apply overrides (user-provided)
        if overrides:
            for k, v in overrides.items():
                profile_config[k] = {**profile_config.get(k, {}), **v}

        # Construct ProfileCard
        card = ProfileCard(
            name=template.name,
            description=template.description,
            config={
                **profile_config,
                "aggregation_strategy": template.aggregation_strategy,
                "weights": template.weights,
            },
            aggregation_strategy=template.aggregation_strategy,
            weights=template.weights,
        )

        self.validate_profile(card)
        logger.info(f"[ProfileBuilder] Profile '{profile_name}' built successfully.")

        return card

    def attach_metrics(self, strategy: BaseStrategy) -> List[MetricSpec]:
        """
        Attach metrics based on the strategy level.
        The registry holds metric suites per level.
        """
        try:
            metric_suite = self.registry.resolve_metrics(level=strategy.level, name=strategy.name)

            if not metric_suite:
                logger.debug(f"[ProfileBuilder] No metrics found for strategy '{strategy.level}'.")
                return []

            return metric_suite

        except Exception as e:
            logger.warning(f"[ProfileBuilder] Failed to attach metrics for '{strategy.name}':\n{e}")
            return []

    @staticmethod
    def validate_profile(profile_card: ProfileCard) -> None:
        """
        Ensure profile structure integrity and metric completeness.
        """
        required_levels = {"embedding", "retrieval", "generation"}
        missing = [lvl for lvl in required_levels if lvl not in profile_card.config]

        if missing:
            raise ValueError(f"[ProfileBuilder] Profile '{profile_card.name}' missing levels:\n{missing}")

        weights = profile_card.weights

        if not weights or not all(isinstance(v, (float, int)) for v in weights.values()):
            raise ValueError(f"[ProfileBuilder] Invalid weight schema for profile '{profile_card.name}'.")

        logger.debug(f"[ProfileBuilder] Validation passed for profile '{profile_card.name}'.")

    @staticmethod
    def _load_predefined_profiles() -> Dict[str, ProfileTemplate]:
        """
        Predefine available profiles for evaluation purposes.
        Extendable in future through configuration or YAML/JSON definitions (is it though..?)
        """
        return {
            "quality": ProfileTemplate(
                name="quality",
                description="Emphasizes accuracy and answer fidelity over latency or cost.",
                strategies={
                    "chunking": "semantic_splitter",
                    "embedding": "openai_text_embedding",
                    "retrieval": "bm25_retriever",
                    "generation": "gpt4_generator"
                },
                weights={"retrieval": 1.0, "generation": 1.5},
            ),
            "efficiency": ProfileTemplate(
                name="efficiency",
                description="Optimized for performance and latency rather than completeness.",
                strategies={
                    "chunking": "fixed_size_chunker",
                    "embedding": "fast_embedder",
                    "retrieval": "hybrid_retriever",
                    "generation": "fast_llm_generator"
                },
                aggregation_strategy="weighted",
                weights={"retrieval": 1.0, "generation": 0.8},
            ),
        }

        # for now ..


if __name__ == '__main__':
    registry_ = StrategyRegistry()
    builder_ = ProfileBuilder(registry_)

    profile_card_ = builder_.build(profile_name="quality")
    print(profile_card_)
