from levelapp.workflow.factory import MainFactory
from levelapp.workflow.entities import WorkflowType
from levelapp.workflow.base import SimulatorWorkflow, ComparatorWorkflow

MainFactory.register_workflow(WorkflowType.SIMULATOR, lambda ctx: SimulatorWorkflow(ctx))
MainFactory.register_workflow(WorkflowType.COMPARATOR, lambda ctx: ComparatorWorkflow(ctx))
