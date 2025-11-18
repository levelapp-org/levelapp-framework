"""levelapp/assessor/builder.py"""
from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, field

from levelapp.assessor.registry import StrategyRegistry, BaseStrategy
from levelapp.assessor.schemas import MetricSpec, StrategyConfig
from levelapp.aspects.logger import logger


class ProfileTemplateBuilder(ABC):
    """Abstract base for providing profile templates - enables multiple sources."""
    @abstractmethod
    def get_template(self, profile_name: str) -> ProfileTemplate:
        pass

    @abstractmethod
    def list_available_profiles(self) -> List[str]:
        pass


@dataclass(frozen=True)
class ProfileTemplate:
    name: str
    description: str
    strategies: Dict[str, str]  # {level_name: strategy_name}
    aggregation_strategy: str = "weighted"
    weights: Dict[str, float] = field(default_factory=lambda: {"retrieval": 1.0, "generation": 1.0})
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LevelConfig:
    """Represents a fully configured level with strategy and metrics."""
    strategy_name: str
    strategy_config: Dict[str, Any]
    metrics: List[MetricSpec]


@dataclass(frozen=True)
class ProfileCard:
    """Immutable representation of a fully configured profile."""
    name: str
    description: str
    levels: Dict[str, LevelConfig]
    aggregation_strategy: str
    weights: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DefaultProfileTemplateProvider(ProfileTemplateBuilder):
    """Default in-code implementation of profile templates."""
    def __init__(self) -> None:
        self._templates = self._load_predefined_profiles()

    def get_template(self, profile_name: str) -> ProfileTemplate:
        if profile_name not in self._templates:
            raise KeyError(
                f"[DefaultProfileTemplateProvider] Profile '{profile_name}' not found. "
                f"Available profiles: {self._templates.keys()}'"
            )

        return self._templates[profile_name]

    def list_available_profiles(self) -> List[str]:
        return list(self._templates.keys())

    @staticmethod
    def _load_predefined_profiles() -> Dict[str, ProfileTemplate]:
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
                metadata={"priority": "accuracy", "cost_tier": "high"},
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
                metadata={"priority": "speed", "cost_tier": "low"},
            ),
        }


class ProfileBuilder:
    """
    Constructs a ProfileCard by assembling predefined strategies and metrics
    for each evaluation level: chunking, embedding, retrieval, generation.
    """
    def __init__(
            self,
            registry: StrategyRegistry,
            template_builder: ProfileTemplateBuilder | None = None,
    ) -> None:
        self.registry = registry
        self.template_builder = template_builder or DefaultProfileTemplateProvider()

    def build(
            self,
            profile_name: str,
            config: Dict[str, Any] | None = None
    ) -> ProfileCard:  # Issue0 - Expected type 'ProfileCard', got 'ProfileCard | None' instead
        """
        Assemble the strategies and metrics into a complete ProfileCard.

        Args:
            profile_name (str): Name of the predefined profile to build.
            config (Dict[str, Any]): Optional configuration overrides for strategies.

        Returns:
            ProfileCard: fully configured profile card.

        Raises:
            KeyError: if profile_name does not exist.
            ValueError: if profile configuration is invalid.
        """
        logger.info(f"[ProfileBuilder] Building profile '{profile_name}'")

        # Get template
        template = self.template_builder.get_template(profile_name=profile_name)

        # Build level configuration
        level_configs = self._build_level_configs(strategies=template.strategies, config=config)

        # Validate before construction
        self._validate_profile_structure(level_configs, template)

        # Construct the immutable ProfileCard
        profile_card = ProfileCard(
            name=template.name,
            description=template.description,
            levels=level_configs,
            aggregation_strategy=template.aggregation_strategy,
            weights=template.weights,
            metadata=template.metadata,
        )

        logger.info(f"[ProfileBuilder] Profile '{profile_name}' built successfully.")

        return profile_card

    def _build_level_configs(
            self,
            strategies: Dict[str, str],
            config: Dict[str, Any],
    ) -> Dict[str, LevelConfig]:
        """
        Build configuration for each level.

        Args:
            strategies (Dict[str, str]): Dictionary of strategy name and strategy configuration.
            config (Dict[str, Any]): Optional configuration overrides for strategies.

        Returns:
            level config: Configuration for each level.

        Raises:
            ValueError: if strategy configuration is invalid.
        """
        level_configs = {}

        for level_name, strategy_name in strategies.items():
            # Get strategy from registry
            strategy_cls = self.registry.get_strategy(level=level_name, name=strategy_name)
            if not strategy_cls:
                raise ValueError(f"[ProfileBuilder] Strategy '{strategy_name}' not found for level '{level_name}'.")

            strategy_config = self._get_strategy_config(
                strategy_cls=strategy_cls,
                config=config.get(level_name, {}))

            # Attach metrics
            metrics = self._resolve_metrics(level=level_name, strategy_name=strategy_name)

            # Create level configuration
            level_configs[level_name] = LevelConfig(
                strategy_name=strategy_name,
                strategy_config=strategy_config,
                metrics=metrics,
            )

            logger.debug(f"[ProfileBuilder] Configured level '{level_name}' with strategy '{strategy_name}'.'")

        return level_configs

    @staticmethod
    def _get_strategy_config(
            strategy_cls: Any,
            config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and merge strategy configuration.

        Args:
            strategy_cls (Any): Strategy class.
            config (Dict[str, Any]): Optional configuration overrides for strategies.

        Returns:
            Strategy configuration: Strategy configuration.
        """
        base_config = getattr(strategy_cls, "config", {})
        return {**base_config, **config}

    def _resolve_metrics(
            self,
            level: str,
            strategy_name: str
    ) -> List[MetricSpec]:
        """
        Resolve metrics for a given strategy.

        Args:
            level (str): Strategy name.
            strategy_name (str): Strategy name.

        Returns:
            List of metric specifications.
        """
        try:
            metrics = self.registry.resolve_metrics(level=level, name=strategy_name)
            return metrics or []

        except Exception as e:
            logger.warning(f"[ProfileBuilder] Failed to resolve metrics for {level}.{strategy_name}:\n{e}")
            return []

    @staticmethod
    def _validate_profile_structure(
            level_configs: Dict[str, LevelConfig],
            template: ProfileTemplate
    ) -> None:
        """
        Validate the profile structure.

        Args:
            level_configs (Dict[str, LevelConfig]): Level configurations.
            template (ProfileTemplate): Template configuration.
        """
        required_levels = {"embedding", "retrieval", "generation"}
        missing_levels = required_levels - level_configs.keys()
        if missing_levels:
            raise ValueError(f"[ProfileBuilder] Profile missing required levels: {missing_levels}")

        # Validate weights
        if not template.weights or not all(isinstance(w, (int, float)) for w in template.weights.values()):
            raise ValueError(f"[ProfileBuilder] Profile weights must be a float or int.")

        # Validate aggregation strategy
        valid_aggregations = {"weighted", "average", "max"}
        if template.aggregation_strategy not in valid_aggregations:
            raise ValueError(f"[ProfileBuilder] Invalid aggregation strategy: {template.aggregation_strategy}")

    def list_available_profiles(self) -> List[str]:
        return self.template_builder.list_available_profiles()


if __name__ == '__main__':
    # Example usage
    registry = StrategyRegistry()
    builder = ProfileBuilder(registry)

    # List available profiles
    print("Available profiles:", builder.list_available_profiles())

    # Build a profile
    profile_card = builder.build(profile_name="quality")
    print(f"Built profile: {profile_card.name}")
