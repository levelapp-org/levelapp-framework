from typing import Callable, Dict
from levelapp.workflow.schemas import WorkflowType, RepositoryType, EvaluatorType, WorkflowConfig, WorkflowContext
from levelapp.core.base import BaseRepository, BaseEvaluator
from levelapp.workflow.base import BaseWorkflow

from levelapp.repository.firestore import FirestoreRepository
from levelapp.evaluator.evaluator import JudgeEvaluator, MetadataEvaluator


class MainFactory:
    """Central factory for repositories, evaluators, and workflows."""

    _repository_map: dict[RepositoryType, Callable[[WorkflowConfig], BaseRepository]] = {
        RepositoryType.FIRESTORE: lambda cfg: FirestoreRepository(),
    }

    _evaluator_map: dict[EvaluatorType, Callable[[WorkflowConfig], BaseEvaluator]] = {
        EvaluatorType.JUDGE: lambda cfg: JudgeEvaluator(),
        EvaluatorType.REFERENCE: lambda cfg: MetadataEvaluator(),
        # Next is the RAG evaluator..
    }

    _workflow_map: dict[WorkflowType, Callable[["WorkflowContext"], BaseWorkflow]] = {}

    @classmethod
    def create_repository(cls, config: WorkflowConfig) -> BaseRepository:
        fn = cls._repository_map.get(config.repository)
        if not fn:
            raise NotImplementedError(f"Repository {config.repository} not implemented")
        return fn(config)

    @classmethod
    def create_evaluator(cls, config: WorkflowConfig) -> Dict[EvaluatorType, BaseEvaluator]:
        evaluators: dict[EvaluatorType, BaseEvaluator] = {}
        for ev in config.evaluators:
            fn = cls._evaluator_map.get(ev)
            if not fn:
                raise NotImplementedError(f"Evaluator {config.evaluators} not implemented")
            evaluators[ev] = fn(config)
        return evaluators

    @classmethod
    def create_workflow(cls, wf_type: WorkflowType, context: "WorkflowContext") -> BaseWorkflow:
        fn = cls._workflow_map.get(wf_type)
        if not fn:
            raise NotImplementedError(f"Workflow {wf_type} not implemented")
        return fn(context)

    @classmethod
    def register_workflow(cls, wf_type: WorkflowType, builder: Callable[["WorkflowContext"], BaseWorkflow]) -> None:
        cls._workflow_map[wf_type] = builder
