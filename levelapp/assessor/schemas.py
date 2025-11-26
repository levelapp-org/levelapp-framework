"""levelapp/assessor/schemas.py"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from levelapp.endpoint.client import EndpointConfig


class Document(BaseModel):
    # TODO: We can add either a random ID or provide the ID.
    content: str
    source: str = Field(default="-")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class MetricSpec(BaseModel):
    name: str
    weight: float = 1.0


@dataclass(frozen=True)
class ProfileTemplate:
    name: str
    description: str
    strategies: Dict[str, str]  # {level_name: strategy_name}
    aggregation_strategy: str = "weighted"
    weights: Dict[str, float] = field(default_factory=lambda: {"retrieval": 1.0, "generation": 1.0})
    config_params: Dict[str, Any] = field(default=dict)
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


class EvaluationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# @dataclass(frozen=True)
# class EvaluationRequest:
#     """Immutable request for evaluation."""
#     profile_name: str
#     user_system_endpoint: EndpointConfig
#     test_queries: List[str]
#     evaluation_metrics: List[str] = field(default_factory=list)
#     max_retries: int = 3
#     timeout_seconds: int = 300
#     metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRequest:
    """Immutable request for evaluation."""
    profile_name: str
    endpoint_config_name: str
    test_queries: List[str]
    evaluation_metrics: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyOutput:
    """Standardized output from strategy execution."""
    level: str
    strategy_name: str
    output: Any
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """Complete result from running a RAG pipeline."""
    query: str
    strategy_outputs: Dict[str, StrategyOutput]  # level -> output
    final_answer: str
    source_documents: List[Document]
    total_execution_time: float
    success: bool = True
    error: str | None = None


@dataclass
class EvaluationResult:
    """Result for a single query evaluation."""
    query: str
    profile_result: PipelineResult
    user_system_result: PipelineResult
    metric_scores: Dict[str, float]  # metric_name -> score
    comparison_results: Dict[str, Any]


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report."""
    model_config = ConfigDict(frozen=True)

    evaluation_id: str
    status: EvaluationStatus
    profile_card: ProfileCard
    evaluation_request: EvaluationRequest
    results: List[EvaluationResult]
    summary_metrics: Dict[str, float]
    execution_time: float
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)


class StrategyConfig(BaseModel):
    name: str
    stype: str  # e,g. "chunking", "embedding", "retrieval", "generation"
    parameters: Dict[str, Any] = Field(default_factory=dict)


# # TODO-0: Hmmm?
# class PipelineConfig(BaseModel):
#     chunking: StrategyConfig
#     embedding: StrategyConfig
#     retrieval: StrategyConfig
#     generation: StrategyConfig | None = None
#
#
# class EvaluationSummary(BaseModel):
#     query: str
#     reference_run_id: str | None = None
#     base_pipeline_id: str | None = None
#     comparative_metrics: List[Dict[str, Any]] = Field(default_factory=list)
#     report: Dict[str, Any] = Field(default_factory=dict)
#
#
# class PipelineResult(BaseModel):
#     pipeline_id: str
#     strategies: Dict[str, str]
#     retrieved_docs: List[Document]
#     augmented_answer: str | None = None
#     metrics: EvaluationSummary | None = None


class MetricType(Enum):
    SIMILARITY = "similarity"  # Higher is better (0-1)
    ACCURACY = "accuracy"      # Higher is better (0-1)
    LATENCY = "latency"        # Lower is better (0-1)
    QUALITY = "quality"        # Higher is better (0-1)
    EFFICIENCY = "efficiency"  # Higher is better (0-1)


@dataclass(frozen=True)
class GaugerConfig:
    """Configuration for gauger behavior and thresholds."""
    similarity_threshold: float = 0.7
    latency_threshold_ms: float = 5000.0
    min_answer_length: int = 10
    max_answer_length: int = 1000
    enable_fallback_metrics: bool = True
