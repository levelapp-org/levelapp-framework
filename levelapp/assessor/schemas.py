"""levelapp/assessor/schemas.py"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    content: str
    source_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategyConfig(BaseModel):
    name: str
    stype: str  # e,g. "chunking", "embedding", "retrieval", "generation"
    parameters: Dict[str, Any] = Field(default_factory=dict)


# TODO-0: Hmmm?
class PipelineConfig(BaseModel):
    chunking: StrategyConfig
    embedding: StrategyConfig
    retrieval: StrategyConfig
    generation: StrategyConfig | None = None


class EvaluationSummary(BaseModel):
    query: str
    reference_run_id: str | None = None
    base_pipeline_id: str | None = None
    comparative_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    report: Dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    pipeline_id: str
    strategies: Dict[str, str]
    retrieved_docs: List[Document]
    augmented_answer: str | None = None
    metrics: EvaluationSummary | None = None
