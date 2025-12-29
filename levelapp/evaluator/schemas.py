"""levelapp/evaluator/schemas.py"""
from pydantic import BaseModel, Field, model_validator, computed_field
from typing import List, Dict, Any
from enum import Enum

from levelapp.aspects.logger import logger


class SentimentType(str, Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"


class TaskMetadata(BaseModel):
    """Structured metadata extracted per interaction."""
    task_type: str = Field(..., description="Inferred task category (e.g., 'book_appointment')")
    task_success: bool = Field(..., description="Whether the agent fulfilled the user's request")
    user_sentiment: SentimentType = Field(..., description="Sentiment of the USER_MESSAGE")


class Evidence(BaseModel):
    covered_points: List[str] = Field(
        default_factory=list,
        max_length=3,
        description="Key points covered (≤3 items)"
    )
    missing_or_wrong: List[str] = Field(
        default_factory=list,
        max_length=3,
        description="Key points missed or contradicted (≤3 items)"
    )


class GriceanMaximResult(BaseModel):
    violated: bool = Field(..., description="Whether this maxim was violated")
    justification: str = Field(
        default="",
        max_length=120,  # ~15 words
        description="Concise evidence (≤15 words)"
    )


class GriceanAnalysis(BaseModel):
    quantity: GriceanMaximResult = Field(
        default_factory=lambda: GriceanMaximResult(violated=False, justification="")
    )
    quality: GriceanMaximResult = Field(
        default_factory=lambda: GriceanMaximResult(violated=False, justification="")
    )
    relation: GriceanMaximResult = Field(
        default_factory=lambda: GriceanMaximResult(violated=False, justification="")
    )
    manner: GriceanMaximResult = Field(
        default_factory=lambda: GriceanMaximResult(violated=False, justification="")
    )

    @computed_field
    @property
    def violation_count(self) -> int:
        return sum([
            self.quantity.violated,
            self.quality.violated,
            self.relation.violated,
            self.manner.violated
        ])

    @computed_field
    @property
    def score(self) -> float:
        """0.0 (all violated) → 1.0 (none violated)"""
        return round(1.0 - (self.violation_count / 4.0), 3)


class JudgeEvaluationResults(BaseModel):
    provider: str = Field(..., description="The provider name, e.g., 'openai', 'ionos'")
    score: int = Field(..., ge=0, le=3, description="Evaluation score between 0 and 3")
    label: str = Field(..., pattern=r"^(Poor|Moderate|Good|Excellent)$", description="evaluation results label")
    verdict: str = Field(..., min_length=10, max_length=200, description="1-2 sentence verdict")
    evidence: Evidence = Field(default_factory=Evidence, description="Detailed evidence for the evaluation")
    raw_response: Dict[str, Any] = Field(default_factory=dict, description="Full unprocessed response")
    task_metadata: TaskMetadata | None = Field(..., description="Structured interaction metadata")
    gricean: GriceanAnalysis = Field(
        default_factory=GriceanAnalysis,
        description="Gricean maxims analysis (turn-level)"
    )

    @computed_field
    @property
    def engagement_score(self) -> float:
        """
        Composite Engagement Score (0.0–1.0):
        engagement = 0.4 * task_success + 0.3 * (score/3) + 0.2 * sentiment_bonus + 0.1 * gricean_score
        """
        task_contrib = 0.4 * (1.0 if self.task_metadata.task_success else 0.0)
        quality_contrib = 0.3 * (self.score / 3.0)

        sent_bonus = 0.0
        if self.task_metadata.user_sentiment == SentimentType.positive:
            sent_bonus = 0.1
        elif self.task_metadata.user_sentiment == SentimentType.negative:
            sent_bonus = -0.1

        sentiment_contrib = 0.2 * sent_bonus
        gricean_contrib = 0.1 * self.gricean.score  # Pragmatic fluency bonus

        return max(0.0, min(1.0, round(
            task_contrib + quality_contrib + sentiment_contrib + gricean_contrib,
            3
        )))

    @model_validator(mode='after')
    def validate_completeness(self) -> 'JudgeEvaluationResults':
        if not isinstance(self.task_metadata, TaskMetadata):
            raise ValueError("task_metadata must be a TaskMetadata instance")
        if not isinstance(self.gricean, GriceanAnalysis):
            raise ValueError("gricean must be a GriceanAnalysis instance")
        return self

    @classmethod
    def from_parsed(cls, provider: str, parsed: Dict[str, Any], raw: Dict[str, Any]) -> "JudgeEvaluationResults":
        content = parsed.get("output", {})

        # Parse task metadata
        task_meta_dict = content.get("task_metadata", {})
        try:
            task_meta = TaskMetadata(**task_meta_dict)
        except Exception as e:
            raise ValueError(f"Failed to parse task_metadata: {task_meta_dict} | Error: {e}")

        # Parse Gricean (fallback to default if missing/invalid)
        gricean_data = content.get("gricean", {})
        gricean_obj = GriceanAnalysis()
        try:
            # Validate each maxim individually to allow partial success
            quantity = GriceanMaximResult(**gricean_data.get("quantity", {"violated": False, "justification": ""}))
            quality = GriceanMaximResult(**gricean_data.get("quality", {"violated": False, "justification": ""}))
            relation = GriceanMaximResult(**gricean_data.get("relation", {"violated": False, "justification": ""}))
            manner = GriceanMaximResult(**gricean_data.get("manner", {"violated": False, "justification": ""}))
            gricean_obj = GriceanAnalysis(quantity=quantity, quality=quality, relation=relation, manner=manner)
        except Exception as e:
            logger.warning(f"Gricean parsing partial failure (using defaults): {e}")

        return cls(
            provider=provider,
            score=int(content.get("score", 0)),
            label=str(content.get("label", "Poor")),
            verdict=str(content.get("justification", "N/A")),
            evidence=Evidence(**content.get("evidence", {})),
            raw_response=raw,
            task_metadata=task_meta,
            gricean=gricean_obj,
        )
