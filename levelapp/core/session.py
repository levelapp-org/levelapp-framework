"""levelapp/core/session.py"""
import threading

from dataclasses import dataclass, field
from typing import Dict, List, Any

from datetime import datetime
from humanize import precisedelta

from levelapp.workflow import MainFactory
from levelapp.workflow.base import BaseWorkflow
from levelapp.workflow.schemas import WorkflowConfig, WorkflowContext
from levelapp.aspects import FunctionMonitor, MetricType, ExecutionMetrics, MonitoringAspect, logger


@dataclass
class SessionMetadata:
    """Metadata for an evaluation session."""
    session_name: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    total_executions: int = 0
    total_duration: float = 0.0
    steps: Dict[str, 'StepMetadata'] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if the session is currently active."""
        return self.ended_at is None

    @property
    def duration(self) -> float | None:
        """Calculate the duration of the session in seconds."""
        if not self.is_active:
            return (self.ended_at - self.started_at).total_seconds()
        return None


@dataclass
class StepMetadata:
    """Metadata for a specific step within an evaluation session."""
    step_name: str
    session_name: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    memory_peak_mb: float | None = None
    error_count: int = 0
    procedures_stats: List[ExecutionMetrics] | None = None

    @property
    def is_active(self) -> bool:
        """Check if the step is currently active."""
        return self.ended_at is None

    @property
    def duration(self) -> float | None:
        """Calculate the duration of the step in seconds."""
        if not self.is_active:
            return (self.ended_at - self.started_at).total_seconds()
        return None


class StepContext:
    """Context manager for an evaluation step within an EvaluationSession."""
    def __init__(self, session: "EvaluationSession", step_name: str, category: MetricType):
        self.session = session
        self.step_name = step_name
        self.category = category
        self.step_meta: StepMetadata | None = None
        self.full_step_name = f"{session.session_name}.{step_name}"
        self._monitored_func = None
        self._func_gen = None

    def __enter__(self):
        with self.session.lock:
            self.step_meta = StepMetadata(
                step_name=self.step_name,
                session_name=self.session.session_name,
                started_at=datetime.now()
            )
            self.session.session_metadata.steps[self.step_name] = self.step_meta

        # Wrap with FunctionMonitor
        self._monitored_func = self.session.monitor.monitor(
            name=self.full_step_name,
            category=self.category,
            enable_timing=True,
            track_memory=True,
        )(self._step_wrapper)

        # Start monitoring
        self._func_gen = self._monitored_func()
        next(self._func_gen)  # Enter monitoring
        return self  # returning self allows nested instrumentation

    def _step_wrapper(self):
        yield  # Actual user step execution happens here

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            next(self._func_gen)  # Exit monitoring
        except StopIteration:
            pass

        with self.session.lock:
            self.step_meta.ended_at = datetime.now()
            if exc_type:
                self.step_meta.error_count += 1
            self.session.session_metadata.total_executions += 1
            if self.step_meta.duration:
                self.session.monitor.update_procedure_duration(name=self.full_step_name, value=self.step_meta.duration)
                self.session.session_metadata.total_duration += self.step_meta.duration

        return False


class EvaluationSession:
    """Context manager for LLM evaluation sessions with integrated monitoring."""
    def __init__(
            self,
            session_name: str = "test-session",
            monitor: FunctionMonitor | None = None,
            workflow_config: WorkflowConfig | None = None
    ):
        """
        Initialize Evaluation Session.

        Args:
            session_name (str): Name of the session
            monitor (FunctionMonitor): Function monitoring aspect
            workflow_config (WorkflowConfig): Workflow configuration.
        """
        self._NAME = self.__class__.__name__

        self.session_name = session_name
        self.monitor = monitor or MonitoringAspect
        self.workflow_config = workflow_config
        self.workflow_type = workflow_config.workflow

        self.workflow: BaseWorkflow | None = None

        self.session_metadata = SessionMetadata(session_name=session_name)
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    def __enter__(self):
        self.session_metadata.started_at = datetime.now()

        # Instantiate workflow if not already
        if not self.workflow:
            if not self.workflow_config:
                raise ValueError(f"{self._NAME}: Workflow configuration must be provided")

            context = WorkflowContext(
                config=self.workflow_config,
                repository=MainFactory.create_repository(self.workflow_config),
                evaluators=MainFactory.create_evaluator(self.workflow_config),
                endpoint_config=self.workflow_config.endpoint_config,
                inputs=self.workflow_config.inputs
            )
            self.workflow = MainFactory.create_workflow(self.workflow_type, context)

        logger.info(
            f"[{self._NAME}] Starting evaluation session: {self.session_name}, "
            f"Workflow: '{self.workflow.name}'"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session_metadata.ended_at = datetime.now()
        logger.info(
            f"[{self._NAME}] Completed session '{self.session_name}' "
            f"in {self.session_metadata.duration:.2f}s"
        )

        if exc_type:
            logger.error(f"[{self._NAME}] Session ended with error: {exc_val}", exc_info=True)
        return False

    def step(self, step_name: str, category: MetricType = MetricType.CUSTOM) -> StepContext:
        """Create a monitored evaluation step."""
        return StepContext(self, step_name, category)

    def run(self):
        if not self.workflow:
            raise RuntimeError(f"{self._NAME} Workflow not initialized")

        with self.step(step_name="setup", category=MetricType.SETUP):
            self.workflow.setup()

        with self.step(step_name="load_data", category=MetricType.DATA_LOADING):
            self.workflow.load_data()

        with self.step(step_name="execute", category=MetricType.EXECUTION):
            self.workflow.execute()

        with self.step(step_name=f"{self.session_name}.collect_results", category=MetricType.RESULTS_COLLECTION):
            self.workflow.collect_results()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "session": {
                "name": self.session_name,
                "duration": precisedelta(self.session_metadata.duration, suppress=['minutes']),
                "start_time": self.session_metadata.started_at.isoformat(),
                "end_time": self.session_metadata.ended_at.isoformat(),
                "steps": len(self.session_metadata.steps),
                "errors": sum(s.error_count for s in self.session_metadata.steps.values())
            },
            "stats": self.monitor.get_all_stats()
        }
