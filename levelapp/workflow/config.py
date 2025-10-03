"""levelapp/workflow/config.py: Contains modular workflow configuration components."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from levelapp.config.endpoint import EndpointConfig
from levelapp.core.schemas import WorkflowType, RepositoryType, EvaluatorType


class ProcessConfig(BaseModel):
    project_name: str
    workflow_type: WorkflowType
    evaluation_params: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    evaluators: List[EvaluatorType]
    providers: List[str] = Field(default_factory=list)
    metrics_map: Dict[str, str] | None = Field(default_factory=dict)


class ReferenceDataConfig(BaseModel):
    path: str | None
    data: Dict[str, Any] | None = Field(default_factory=dict)


class RepositoryConfig(BaseModel):
    type: RepositoryType | None = None
    project_id: str | None = None
    database_name: str = Field(default="(default)")

    class Config:
        extra = "allow"


class WorkflowConfig(BaseModel):
    """
    Static workflow configuration. Maps directly to YAML sections.
    Supports both file-based loading and in-memory dictionary creation.
    """
    process: ProcessConfig
    evaluation: EvaluationConfig
    reference_data: ReferenceDataConfig
    endpoint: EndpointConfig
    repository: RepositoryConfig

    class Config:
        extra = "allow"

    @classmethod
    def load(cls, path: Optional[str] = None) -> "WorkflowConfig":
        """Load workflow configuration from a YAML/JSON file."""
        from levelapp.aspects.loader import DataLoader

        loader = DataLoader()
        config_dict = loader.load_raw_data(path=path)
        return cls.model_validate(config_dict)

    @classmethod
    def from_dict(cls, content: Dict[str, Any]) -> "WorkflowConfig":
        """Load workflow configuration from an in-memory dict."""
        return cls.model_validate(content)

    def set_reference_data(self, content: Dict[str, Any]) -> None:
        """Load referer data from an in-memory dict."""
        self.reference_data.data = content
