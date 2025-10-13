"""levelapp/workflow/factory.py: Creates workflows using WorkflowContext."""
from typing import Dict, Callable

from levelapp.core.schemas import WorkflowType
from levelapp.workflow.base import SimulatorWorkflow, ComparatorWorkflow, BaseWorkflow
from levelapp.workflow.runtime import WorkflowContext


class MainFactory:
    """Central factory for workflows."""

    _workflow_map: Dict[WorkflowType, Callable[[WorkflowContext], BaseWorkflow]] = {
        WorkflowType.SIMULATOR: lambda ctx: SimulatorWorkflow(ctx),
        WorkflowType.COMPARATOR: lambda ctx: ComparatorWorkflow(ctx),
    }

    @classmethod
    def create_workflow(cls, context: WorkflowContext) -> BaseWorkflow:
        """Create workflow using the given runtime context."""
        wf_type = context.config.process.workflow_type
        builder = cls._workflow_map.get(wf_type)
        if not builder:
            raise NotImplementedError(f"Workflow '{wf_type}' not implemented")
        return builder(context)

    @classmethod
    def register_workflow(cls, wf_type: WorkflowType, builder: Callable[[WorkflowContext], BaseWorkflow]) -> None:
        """Register a new workflow implementation."""
        cls._workflow_map[wf_type] = builder
