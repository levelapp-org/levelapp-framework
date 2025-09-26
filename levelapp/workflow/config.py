"""levelapp/workflow/config.py: Contains modular workflow configuration components."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from levelapp.config.endpoint import EndpointConfig
from levelapp.workflow.schemas import WorkflowType, RepositoryType, EvaluatorType


class ProcessConfig(BaseModel):
    project_name: str
    workflow_type: WorkflowType
    evaluation_params: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    evaluators: List[EvaluatorType]
    providers: List[str] = Field(default_factory=list)
    metrics_map: Dict[str, str] | None = Field(default_factory=dict)


class RepositoryConfig(BaseModel):
    type: RepositoryType
    project_id: str
    database_name: str
    reference_data: Dict[str, Any] = Field(default=dict)
    source: str = "LOCAL"  # LOCAL, REMOTE, IN_MEMORY
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


if __name__ == '__main__':
    # Load from YAML
    config = WorkflowConfig.load(path="../../src/data/workflow_config.yaml")
    print(config.model_dump())

    # Load from in-memory dict
    config_dict = {
        "process": {"project_name": "test-project", "workflow_type": "SIMULATOR", "evaluation_params": {"attempts": 2}},
        "evaluation": {"evaluators": ["JUDGE"], "providers": ["openai"]},
        "endpoint": {"base_url": "http://127.0.0.1:8000", "api_key": "key", "model_id": "model"},
        "repository": {"type": "FIRESTORE", "source": "IN_MEMORY", "metrics_map": {"field_1": "EXACT"}},
    }
    config_in_memory = WorkflowConfig.from_dict(config_dict)
    print(config_in_memory.model_dump())

