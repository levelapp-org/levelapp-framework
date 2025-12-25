"""
levelapp/simulator/schemas.py

Defines Pydantic models for simulator-related data structures,
including test configurations, batch metadata, and evaluation results.
"""
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime

from typing import Dict, Any, List
from pydantic import BaseModel, Field, computed_field

from levelapp.evaluator.evaluator import JudgeEvaluationResults


# Input DTOs:
class InteractionLevel(str, Enum):
    """Enum representing the type of interaction."""
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    FINAL = "final"


class Interaction(BaseModel):
    """Represents a single interaction within a conversation."""
    id: UUID = Field(default_factory=uuid4, description="Interaction identifier")
    user_message_path: str = Field(..., description="Path of the user message in the request payload")
    user_message: str = Field(..., description="The user's query message")
    reference_reply: str = Field(..., description="The preset reference message")
    interaction_type: InteractionLevel = Field(default=InteractionLevel.INITIAL, description="Type of interaction")
    reference_metadata: Dict[str, Any] = Field(default_factory=dict, description="Expected metadata")
    guardrail_flag: Any = Field(default=False, description="Flag for guardrail signaling")
    request_payload: Dict[str, Any] = Field(default_factory=dict, description="Additional request payload")


class ConversationScript(BaseModel):
    """Represents a basic conversation with multiple interactions."""
    id: UUID = Field(default_factory=uuid4, description="Conversation identifier")
    interactions: List[Interaction] = Field(default_factory=list, description="List of interactions")
    domain_context: str = Field(default="real_estate", description="Domain context of the conversation")
    description: str = Field(default="no-description", description="A short description of the conversation")
    details: Dict[str, str] = Field(default_factory=dict, description="Conversation details")
    variable_request_schema: bool = Field(default=False, description="The payload schema changes for each request")
    uuid_field: str | None = Field(default=None, description="field that requires a UUID value")


class ScriptsBatch(BaseModel):
    id: UUID = Field(default_factory=uuid4, description="Batch identifier")
    scripts: List[ConversationScript] = Field(default_factory=list, description="List of conversation scripts")


# Output DTOs:
class InteractionEvaluationResults(BaseModel):
    """Model representing the evaluation result of an interaction."""
    judge_evaluations: Dict[str, JudgeEvaluationResults] = Field(default_factory=dict)
    metadata_evaluation: Dict[str, float] = Field(default_factory=dict)
    guardrail_flag: int | None = Field(default=None)


class TurnSummary(BaseModel):
    """Compact, evaluation-rich turn representation for multi-turn analysis."""
    turn_index: int = Field(..., description="0-based turn index")
    role: str = Field(..., pattern=r"^(U|A)$", description="'U'=user, 'A'=agent")
    task_type: str = Field(default="other", description="Inferred task (e.g., 'Information Inquiry')")
    task_success: bool = Field(default=False, description="Task completed this turn?")
    score: float = Field(ge=0, le=3, description="Judge score (0-3)")
    engagement: float = Field(ge=0, le=1, description="Turn engagement (0.0-1.0)")
    gricean_violations: int = Field(ge=0, le=4, description="Gricean maxims violated (0-4)")
    sentiment: str = Field(default="neutral", description="User sentiment (if role='U')")
    # TODO-0: Change 'key_facts' to 'justification' (which will be changed to 'verdicts')
    key_facts: List[str] = Field(default_factory=list, description="Extracted facts for retention")
    guardrail_triggered: bool = Field(default=False, description="Guardrail flag raised?")

    @computed_field
    @property
    def compact_repr(self) -> str:
        """
        Token-optimized string for LLM consumption.
        Format:
        [T{idx}][U/A][s={score:.1f}][e={eng:.2f}][g={gricean_violations}][sent={sent}]
        Facts: [..]
        Text: \"..."\
        """
        role_tag = "U" if self.role == "U" else "A"
        sent_tag = f"[sent={self.sentiment}]" if self.role == "U" else ""

        header = (
            f"[T{self.turn_index}]"
            f"[{role_tag}]"
            f"[task={self.task_type}]"
            f"[s={self.score:.1f}]"
            f"[e={self.engagement:.2f}]"
            f"[g={self.gricean_violations}]"
            f"{sent_tag}]")

        facts_line = f"Facts: [{', '.join(self.key_facts)}]" if self.key_facts else ""

        parts = [header]
        if facts_line:
            parts.append(facts_line)

        return " ".join(parts)


# ---- Interaction Details Models ----
class SingleInteractionResults(BaseModel):
    """Represents metadata extracted from a VLA interaction."""
    conversation_id: str = Field(description="Conversation identifier")
    user_message: str = Field(default="", description="The user's query message")
    generated_reply: str = Field(default="Interaction request failed", description="The generated reply message")
    reference_reply: str = Field(default="", description="The preset reference message")
    generated_metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted metadata")
    reference_metadata: Dict[str, Any] = Field(default_factory=dict, description="Expected metadata")
    guardrail_details: bool | None = Field(default=None, description="Flag for guardrail signaling")
    evaluation_results: InteractionEvaluationResults = Field(default_factory=InteractionEvaluationResults)
    turn_summary: TurnSummary | None = Field(default=None)



class InteractionEvaluationResults(BaseModel):
    """Model representing the evaluation result of an interaction."""
    judge_evaluations: Dict[str, JudgeEvaluationResults] | None = Field(default_factory=dict)
    metadata_evaluation: Dict[str, float] | None = Field(default_factory=dict)
    guardrail_flag: int | None = Field(default=None)


class SingleInteractionResults(BaseModel):
    """Represents metadata extracted from VLA interaction."""
    conversation_id: str = Field(description="Conversation identifier")
    user_message: str = Field(default="", description="The user's query message")
    generated_reply: str = Field(default="Interaction request failed", description="The generated reply message")
    reference_reply: str = Field(default="", description="The preset reference message")
    generated_metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted metadata")
    reference_metadata: Dict[str, Any] = Field(default_factory=dict, description="Expected metadata")
    guardrail_details: bool | None = Field(default=None, description="Flag for guardrail signaling")
    evaluation_results: InteractionEvaluationResults = Field(default_factory=InteractionEvaluationResults)


class SingleAttemptResults(BaseModel):
    attempt_nbr: int = Field(default=1, description="The attempt number")
    attempt_id: str = Field(default=None, description="The attempt ID")
    script_id: str = Field(default=None, description="The script ID")
    total_duration: float = Field(default=None, description="Total duration")
    interaction_results: List[SingleInteractionResults] = Field(default_factory=list)
    evaluation_verdicts: Dict[str, List[str]] = Field(default_factory=dict)
    average_scores: Dict[str, float] = Field(default_factory=dict)
    interaction_summaries: List[str] = Field(default_factory=list)


class AllAttemptsResults(BaseModel):
    script_id: str = Field(default=None, description="The script ID")
    attempts: List[SingleAttemptResults] = Field(default_factory=list)
    average_scores: Dict[str, float] = Field(default_factory=dict)


class SimulationResults(BaseModel):
    # Collected data
    started_at: datetime = datetime.now()
    finished_at: datetime
    # Collected Results
    evaluation_summary: Dict[str, Any] | None = Field(default_factory=dict, description="Evaluation result")
    average_scores: Dict[str, Any] | None = Field(default_factory=dict, description="Average scores")
    script_results: List[AllAttemptsResults] | None = Field(default_factory=list, description="detailed results")

    @computed_field
    @property
    def batch_id(self) -> str:
        return str(uuid4())

    @computed_field
    @property
    def elapsed_time(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()
