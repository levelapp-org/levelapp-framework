"""levelapp/workflow/runtime.py: contains the workflow runtime context component."""
from dataclasses import dataclass
from typing import Dict, List, Any

from levelapp.config import EndpointConfig
from levelapp.core.base import BaseRepository, BaseEvaluator
from levelapp.workflow.config import WorkflowConfig
from levelapp.core.schemas import EvaluatorType


@dataclass(frozen=True)
class WorkflowContext:
    """Immutable data holder for workflow execution context."""
    config: WorkflowConfig
    repository: BaseRepository
    evaluators: Dict[EvaluatorType, BaseEvaluator]
    providers: List[str]
    endpoint_config: EndpointConfig
    inputs: Dict[str, Any]