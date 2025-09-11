from typing import Callable
from levelapp.workflow.schemas import WorkflowType, RepositoryType, EvaluatorType, WorkflowConfig, WorkflowContext
from levelapp.core.base import BaseRepository, BaseEvaluator
from levelapp.workflow.base import BaseWorkflow

from levelapp.repository.firestore import FirestoreRepository
from levelapp.evaluator.evaluator import JudgeEvaluator


class MainFactory:
    """Central factory for repositories, evaluators, and workflows."""

    _repository_map: dict[RepositoryType, Callable[[WorkflowConfig], BaseRepository]] = {
        RepositoryType.FIRESTORE: lambda cfg: FirestoreRepository(),
    }

    _evaluator_map: dict[EvaluatorType, Callable[[WorkflowConfig], BaseEvaluator]] = {
        EvaluatorType.JUDGE: lambda cfg: JudgeEvaluator(),
    }

    _workflow_map: dict[WorkflowType, Callable[["WorkflowContext"], BaseWorkflow]] = {}

    @classmethod
    def create_repository(cls, config: WorkflowConfig) -> BaseRepository:
        fn = cls._repository_map.get(config.repository)
        if not fn:
            raise NotImplementedError(f"Repository {config.repository} not implemented")
        return fn(config)

    @classmethod
    def create_evaluator(cls, config: WorkflowConfig) -> BaseEvaluator:
        fn = cls._evaluator_map.get(config.evaluator)
        if not fn:
            raise NotImplementedError(f"Evaluator {config.evaluator} not implemented")
        return fn(config)

    @classmethod
    def create_workflow(cls, wf_type: WorkflowType, context: "WorkflowContext") -> BaseWorkflow:
        fn = cls._workflow_map.get(wf_type)
        if not fn:
            raise NotImplementedError(f"Workflow {wf_type} not implemented")
        return fn(context)

    @classmethod
    def register_workflow(cls, wf_type: WorkflowType, builder: Callable[["WorkflowContext"], BaseWorkflow]) -> None:
        cls._workflow_map[wf_type] = builder
