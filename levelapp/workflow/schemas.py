from enum import Enum
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Dict, Any

from levelapp.config.endpoint import EndpointConfig
from levelapp.core.base import BaseRepository, BaseEvaluator
from levelapp.aspects import DataLoader


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return [e.value for e in cls]


class WorkflowType(ExtendedEnum):
    SIMULATOR = "SIMULATOR"
    COMPARATOR = "COMPARATOR"
    ASSESSOR = "ASSESSOR"


class RepositoryType(ExtendedEnum):
    FIRESTORE = "FIRESTORE"
    FILESYSTEM = "FILESYSTEM"


class EvaluatorType(ExtendedEnum):
    JUDGE = "JUDGE"
    REFERENCE = "REFERENCE"
    RAG = "RAG"


class WorkflowConfig:
    """Configuration for a workflow, loaded from JSON/YAML via DataLoader."""

    # Class-level constant
    _fields_list: List[str] = [
        "project_name",
        "evaluation_params",
        "workflow",
        "repository",
        "evaluators",
        "reference_data",
    ]

    def __init__(
        self,
        workflow: WorkflowType,
        repository: RepositoryType,
        evaluators: List[EvaluatorType],
        endpoint_config: EndpointConfig,
        inputs: Dict[str, Any],
    ):
        self.workflow = workflow
        self.repository = repository
        self.evaluators = evaluators
        self.endpoint_config = endpoint_config
        self.inputs = inputs

    @classmethod
    def load(cls, path: str | None = None) -> "WorkflowConfig":
        """Load and validate workflow configuration from a file."""
        loader = DataLoader()
        config_dict = loader.load_raw_data(path=path)
        model_config: BaseModel = loader.create_dynamic_model(data=config_dict, model_name="WorkflowConfiguration")

        cls._check_fields(model_config)
        cls._check_values(model_config)

        workflow = WorkflowType(model_config.workflow)
        repository = RepositoryType(model_config.repository)

        if isinstance(model_config.evaluators, str):
            print(f"evaluators: {model_config.evaluators}")
            evaluators = [EvaluatorType(model_config.evaluators)]
        else:
            evaluators = [EvaluatorType(e) for e in model_config.evaluators]

        evaluation_params = model_config.evaluation_params.model_dump()
        reference_data_path = getattr(model_config.reference_data, "path", None)
        endpoint_config = EndpointConfig.model_validate(model_config.endpoint_configuration.model_dump())

        return cls(
            workflow=workflow,
            repository=repository,
            evaluators=evaluators,
            endpoint_config=endpoint_config,
            inputs={'reference_data_path': reference_data_path, 'evaluation_params': evaluation_params},
        )

    @classmethod
    def _check_fields(cls, config: BaseModel) -> None:
        for field_name in cls._fields_list:
            if field_name not in config.model_fields:
                raise ValueError(f"[WorkflowConfig] Field '{field_name}' missing in configuration")

    @staticmethod
    def _check_values(config: BaseModel) -> None:
        if config.workflow not in WorkflowType.list():
            raise ValueError(f"[WorkflowConfig] Unsupported workflow type '{config.workflow}'")
        if config.repository not in RepositoryType.list():
            raise ValueError(f"[WorkflowConfig] Unsupported repository type '{config.repository}'")

        evals = config.evaluators
        if isinstance(evals, str):
            evals = [evals]

        for e in evals:
            if e not in EvaluatorType.list():
                raise ValueError(f"[WorkflowConfig] Unsupported evaluator type '{config.evaluators}'")


@dataclass(frozen=True)
class WorkflowContext:
    """Immutable data holder for workflow execution context."""
    config: WorkflowConfig
    repository: BaseRepository
    evaluators: Dict[str, BaseEvaluator]
    endpoint_config: EndpointConfig
    inputs: Dict[str, Any]
