"""levelapp/assessor/builder.py"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from levelapp.assessor.registry import StrategyRegistry, StrategyInfo
from levelapp.assessor.schemas import MetricSpec, ProfileTemplate, ProfileCard, LevelConfig
from levelapp.aspects.logger import logger


class ProfileTemplateBuilder(ABC):
    """Abstract base for providing profile templates - enables multiple sources."""
    @abstractmethod
    def get_template(self, profile_name: str) -> ProfileTemplate:
        pass

    @abstractmethod
    def list_available_profiles(self) -> List[str]:
        pass


class DefaultProfileTemplateProvider(ProfileTemplateBuilder):
    """Default in-code implementation of profile templates."""
    def __init__(self) -> None:
        self._templates = self._load_predefined_profiles()

    def get_template(self, profile_name: str) -> ProfileTemplate:
        if profile_name not in self._templates:
            raise KeyError(
                f"[DefaultProfileTemplateProvider] Profile '{profile_name}' not found. "
                f"Available profiles: {list(self._templates.keys())}"  # Fixed: list() for proper display
            )

        return self._templates[profile_name]

    def list_available_profiles(self) -> List[str]:
        return list(self._templates.keys())

    @staticmethod
    def _load_predefined_profiles() -> Dict[str, ProfileTemplate]:
        return {
            "basic": ProfileTemplate(
                name="basic",
                description="Minimal compute profile for fast, deterministic testing",
                strategies={
                    "chunking": "fixed_size_chunker",
                    "embedding": "minilm_embedder",
                    "retrieval": "cosine_retriever",
                    "generation": "basic_model_generator",
                },
                aggregation_strategy="weighted",
                weights={"retrieval": 1.0, "generation": 1.0},
                config_params={
                    "chunking": {"chunk_size": 300, "overlap": 50},
                    "retrieval": {"top_k": 5},
                    "generation": {"model": "distilgpt2", "temperature": 0.1},
                },
                metadata={"priority": "accuracy", "cost_tier": "high"},
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
    ) -> ProfileCard:
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
        level_configs = self._build_level_configs(
            strategies=template.strategies,
            config=config or {}
        )
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
            config (Dict[str, Any]): Configuration overrides for strategies.

        Returns:
            level config: Configuration for each level.

        Raises:
            ValueError: if strategy configuration is invalid.
        """
        level_configs = {}

        for level_name, strategy_name in strategies.items():
            strategy_info = self.registry.get_strategy_info(level=level_name, name=strategy_name)
            if not strategy_info:
                raise ValueError(f"[ProfileBuilder] Strategy '{strategy_name}' not found for level '{level_name}'.")

            strategy_config = self._get_strategy_config(
                strategy_info=strategy_info,
                level_config=config.get(level_name, {})
            )

            # Attach metrics
            metrics = self._resolve_metrics(level=level_name, strategy_name=strategy_name)

            # Create level configuration
            level_configs[level_name] = LevelConfig(
                strategy_name=strategy_name,
                strategy_config=strategy_config,
                metrics=metrics,
            )

            logger.debug(f"[ProfileBuilder] Configured level '{level_name}' with strategy '{strategy_name}'.")

        return level_configs

    @staticmethod
    def _get_strategy_config(
            strategy_info: StrategyInfo,
            level_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract and merge strategy configuration.

        Args:
            strategy_info (Any): Strategy info object containing config schema.
            level_config (Dict[str, Any]): Level-specific configuration.

        Returns:
            Strategy configuration: Merged strategy configuration.
        """
        base_config = {}

        # If config schema exists, use its defaults
        if strategy_info.config_schema:
            try:
                # Create instance with empty config to get defaults
                schema_instance = strategy_info.config_schema()
                base_config = schema_instance.model_dump()

            except ValueError:
                base_config = {}

        # Merge with level-specific config
        return {**base_config, **level_config}

    def _resolve_metrics(
            self,
            level: str,
            strategy_name: str
    ) -> List[MetricSpec]:
        """
        Resolve metrics for a given strategy.

        Args:
            level (str): Strategy level.
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
            raise ValueError("[ProfileBuilder] Profile weights must be a float or int.")

        # Validate aggregation strategy
        valid_aggregations = {"weighted", "average", "max"}
        if template.aggregation_strategy not in valid_aggregations:
            raise ValueError(f"[ProfileBuilder] Invalid aggregation strategy: {template.aggregation_strategy}")

    def list_available_profiles(self) -> List[str]:
        return self.template_builder.list_available_profiles()


if __name__ == '__main__':
    # Example usage
    from levelapp.assessor.strategies import register_basic_profile_strategies

    registry_ = StrategyRegistry()
    register_basic_profile_strategies(registry=registry_)
    builder_ = ProfileBuilder(registry_)

    # List available profiles
    print("Available profiles:", builder_.list_available_profiles())

    # Build a profile
    profile_card_ = builder_.build(profile_name="basic")
    print(f"Built profile: {profile_card_}\n---")
    print(f"Profile levels: {list(profile_card_.levels.keys())}\n---")
