import asyncio

from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any

from levelapp.core.base import BaseProcess
from levelapp.simulator.simulator import ConversationSimulator
from levelapp.aspects.loader import DataLoader
from levelapp.workflow.schemas import WorkflowContext


class BaseWorkflow(ABC):
    """Abstract base class for evaluation workflows."""

    def __init__(self, name: str, context: WorkflowContext) -> None:
        self.name = name
        self.context = context
        self.process: BaseProcess | None = None
        self._input_data: Any | None = None
        self._results: Any | None = None
        self._initialized: bool = False

    def setup(self) -> None:
        """Validate and initialize workflow-specific settings."""
        if self._initialized:
            return
        self.process = self._setup_process(context=self.context)
        self._initialized = True

    def load_data(self) -> None:
        """Load and preprocess input data."""
        if not self._initialized:
            raise RuntimeError(f"[{self.name}] Workflow not initialized. Call setup() first.")
        self._input_data = self._load_input_data(context=self.context)

    def execute(self) -> None:
        """Run the workflow evaluation steps."""
        if not self._input_data:
            raise RuntimeError(f"[{self.name}] No reference data available.")

        if asyncio.iscoroutinefunction(self.process.run):
            self._results = asyncio.run(self.process.run(**self._input_data))
        else:
            self._results = self.process.run(**self._input_data)

    async def aexecute(self) -> None:
        if not self._input_data:
            raise RuntimeError(f"[{self.name}] No reference data available.")

        if asyncio.iscoroutinefunction(self.process.run):
            self._results = await self.process.run(**self._input_data)
        else:
            loop = asyncio.get_running_loop()
            func = partial(self.process.run, **self._input_data)
            self._results = await loop.run_in_executor(None, func)

    def collect_results(self) -> Any:
        """Return unified results structure."""
        return self._results

    @abstractmethod
    def _setup_process(self, context: WorkflowContext) -> BaseProcess:
        raise NotImplementedError

    @abstractmethod
    def _load_input_data(self, context: WorkflowContext) -> Any:
        raise NotImplementedError


class SimulatorWorkflow(BaseWorkflow):
    def __init__(self, context: WorkflowContext) -> None:
        super().__init__(name="ConversationSimulator", context=context)
        print(f"[SimulatorWorkflow] Context inputs:\n{context.inputs}")

    def _setup_process(self, context: WorkflowContext) -> BaseProcess:
        simulator = ConversationSimulator()
        simulator.setup(
            repository=context.repository,
            evaluator=context.evaluator,
            endpoint_config=context.endpoint_config,
        )
        return simulator

    def _load_input_data(self, context: WorkflowContext) -> Any:
        loader = DataLoader()
        reference_data_path = context.inputs.get("reference_data_path", "no-path-provided")
        file_path = Path(reference_data_path)

        if not file_path.exists():
            raise FileNotFoundError(f"[{self.name}] Reference data file not found.")

        evaluation_params = context.inputs.get("evaluation_params", {})
        data_config = loader.load_configuration(path=reference_data_path)
        scripts_batch = loader.load_data(data=data_config, model_name="ScriptsBatch")

        return {"test_batch": scripts_batch, "attempts": evaluation_params.get("attempts", 1)}


class ComparatorWorkflow(BaseWorkflow):
    def _setup_process(self, context: WorkflowContext) -> BaseProcess:
        pass

    def _load_input_data(self, context: WorkflowContext) -> Any:
        pass

    def __init__(self, context: WorkflowContext) -> None:
        super().__init__(name="MetadataComparator", context=context)
