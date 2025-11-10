"""levelapp/workflow/config.py: Contains modular workflow configuration components."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from levelapp.aspects import logger
from levelapp.endpoint.client import EndpointConfig
from levelapp.core.schemas import WorkflowType, RepositoryType, EvaluatorType


# TODO-0: This whole configuration schema needs to be adjusted!!
# 1. Split the sections into the different modules: ASSESSOR, COMPARATOR, SIMULATOR
# 2. Keep the 'PROCESS', 'REPOSITORY', and 'ENDPOINT' as is.
# 3. The workflow type (workflow_type: SIMULATOR) will decide which module/modules will be configured.
# 4. Each module section will contain its own configuration separately.
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


class AssessorConfig(BaseModel):
    """
    Configuration for the Assessor module.
    Defines how performance profiles and metric evaluations are aggregated.
    """
    profile_targets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    aggregation_strategy: str = Field(default="WEIGHTED_AVERAGE")
    weights: Dict[str, float] = Field(default_factory=dict)
    output_format: str = Field(default="JSON")
    save_path: str | None = Field(default=None)

    class Config:
        extra = "allow"


class WorkflowConfig(BaseModel):
    """
    Static workflow configuration. Maps directly to YAML sections.
    Supports both file-based loading and in-memory dictionary creation.
    """
    process: ProcessConfig
    endpoint: EndpointConfig
    evaluation: EvaluationConfig
    reference_data: ReferenceDataConfig
    repository: RepositoryConfig
    assessor: AssessorConfig | None = None  # for backward compatibility (for now)

    class Config:
        extra = "allow"

    @classmethod
    def load(cls, path: str | None = None) -> "WorkflowConfig":
        """
        Load workflow configuration from a YAML/JSON file.

        Args:
            path (str): YAML/JSON configuration file path.

        Returns:
            WorkflowConfig: An instance of WorkflowConfig.
        """
        from levelapp.aspects.loader import DataLoader

        loader = DataLoader()
        config_dict = loader.load_raw_data(path=path)
        logger.info(f"[{cls.__name__}] Workflow configuration loaded from '{path}' file content")
        return cls.model_validate(config_dict)

    @classmethod
    def from_dict(cls, content: Dict[str, Any]) -> "WorkflowConfig":
        """
        Load workflow configuration from an in-memory dict.

        Args:
            content (dict): Workflow configuration content.

        Returns:
            WorkflowConfig: An instance of WorkflowConfig.
        """
        logger.info(f"[{cls.__name__}] Workflow configuration loaded from provided content")
        return cls.model_validate(content)

    def set_reference_data(self, content: Dict[str, Any]) -> None:
        """
        Load referer data from an in-memory dict.

        Args:
            content (dict): Workflow configuration content.

        """
        self.reference_data.data = content
        logger.info(f"[{self.__class__.__name__}] Reference data loaded from provided content")


if __name__ == '__main__':
    workflow_config = WorkflowConfig.load(path="../../src/data/workflow_config.yaml")
    print(f"Workflow Configuration:\n{workflow_config.model_dump_json(indent=2)}")
