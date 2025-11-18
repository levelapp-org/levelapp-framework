"""levelapp/assessor/__init__.py"""
from levelapp.assessor.registry import StrategyRegistry
from levelapp.assessor.strategies import register_mock_strategies, register_baseline_strategies


registry = StrategyRegistry()
register_mock_strategies(registry)
register_baseline_strategies(registry)
