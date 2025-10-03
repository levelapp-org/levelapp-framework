"""levelapp/workflow/context.py: Builds runtime WorkflowContext from WorkflowConfig."""
from typing import Dict, Callable

from levelapp.workflow.config import WorkflowConfig
from levelapp.core.base import BaseRepository, BaseEvaluator
from levelapp.workflow.runtime import WorkflowContext
from levelapp.core.schemas import EvaluatorType, RepositoryType

from levelapp.repository.firestore import FirestoreRepository
from levelapp.evaluator.evaluator import JudgeEvaluator, MetadataEvaluator


class WorkflowContextBuilder:
    """Builds a runtime WorkflowContext from a WorkflowConfig."""

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config

        # Map repository type to constructor that accepts the WorkflowConfig
        self.repository_map: Dict[RepositoryType, Callable[[WorkflowConfig], BaseRepository]] = {
            RepositoryType.FIRESTORE: lambda cfg: FirestoreRepository(cfg),
        }

        # Map evaluator type to constructor that accepts the WorkflowConfig
        self.evaluator_map: Dict[EvaluatorType, Callable[[WorkflowConfig], BaseEvaluator]] = {
            EvaluatorType.JUDGE: lambda cfg: JudgeEvaluator(config=cfg),
            EvaluatorType.REFERENCE: lambda cfg: MetadataEvaluator(config=cfg),
        }

    def build(self) -> WorkflowContext:
        """
        Build a runtime WorkflowContext from the static WorkflowConfig.
        Supports in-memory reference data if provided.
        """
        # Repository instance
        repository_type = self.config.repository.type
        repository = self.repository_map.get(repository_type)(self.config)

        # Evaluator instances
        evaluators: Dict[EvaluatorType, BaseEvaluator] = {
            ev: self.evaluator_map.get(ev)(self.config) for ev in self.config.evaluation.evaluators
        }

        # Providers and endpoint
        providers = self.config.evaluation.providers
        endpoint_config = self.config.endpoint

        # Inputs include reference data path or in-memory dict
        inputs = {}
        if self.config.reference_data.data:
            inputs["reference_data"] = self.config.reference_data.data
        else:
            inputs["reference_data_path"] = self.config.reference_data.path
            print(f"[WorkflowContextBuilder] reference data path: {inputs['reference_data_path']}")

        return WorkflowContext(
            config=self.config,
            repository=repository,
            evaluators=evaluators,
            providers=providers,
            endpoint_config=endpoint_config,
            inputs=inputs,
        )
