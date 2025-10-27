import asyncio

from abc import ABC, abstractmethod
from pydantic import ValidationError
from functools import partial
from pathlib import Path
from typing import Any, Dict

from levelapp.core.base import BaseProcess
from levelapp.endpoint.client import EndpointConfig
from levelapp.endpoint.manager import EndpointConfigManager
from levelapp.simulator.schemas import ScriptsBatch
from levelapp.simulator.simulator import ConversationSimulator
from levelapp.workflow.runtime import WorkflowContext
from levelapp.aspects.loader import DataLoader


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
            self._results = await loop.run_in_executor(None, func, None)

    def collect_results(self) -> Any:
        """
        Return unified results structure.

        Returns:
            The simulation results.
        """
        return self._results

    @abstractmethod
    async def test_connection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method for testing endpoint connection.

        Args:
            context (Dict[str, Any]): The context (request payload) to test connectivity with.

        Returns:
            The test connectivity result.
        """

    @abstractmethod
    def _setup_process(self, context: WorkflowContext) -> BaseProcess:
        """
        Abstract method for setting up the configured process.

        Args:
            context (WorkflowContext): The workflow context.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_input_data(self, context: WorkflowContext) -> Any:
        """
        Abstract method for loading reference data.

        Args:
            context (WorkflowContext): The workflow context.
        """
        raise NotImplementedError


class SimulatorWorkflow(BaseWorkflow):
    def __init__(self, context: WorkflowContext) -> None:
        super().__init__(name="ConversationSimulator", context=context)

    def _setup_process(self, context: WorkflowContext) -> BaseProcess:
        """
        Concrete implementation for setting up the simulation workflow.

        Args:
            context (WorkflowContext): The workflow context for the simulation workflow.

        Returns:
            ConversationSimulator instance.
        """
        simulator = ConversationSimulator()
        simulator.setup(
            endpoint_config=context.endpoint,
            evaluators=context.evaluators,
            providers=context.providers,
        )

        return simulator

    async def test_connection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs a connectivity test of the configured endpoint.

        Args:
            context (Dict[str, Any]): The request payload to send for testing.

        Returns:
            The test connectivity result.
        """
        endpoint_cm = EndpointConfigManager()
        endpoint_cm.set_endpoints(endpoints_config=[self.context.endpoint])
        tester = endpoint_cm.get_tester(endpoint_name=self.context.endpoint.name)
        results = await tester.test(context=context)

        return results

    def _load_input_data(self, context: WorkflowContext) -> Dict[str, Any]:
        """
        Concrete implementation for loading the reference data.

        Args:
            context (WorkflowContext): The workflow context for the simulation workflow.

        Returns:
            Dict[str, Any]: The reference data.
        """
        loader = DataLoader()
        if "reference_data" in context.inputs:
            data_config = context.inputs["reference_data"]
        else:
            reference_data_path = context.inputs.get("reference_data_path", "no-path-provided")

            if not reference_data_path:
                raise RuntimeError(f"[{self.name}] No reference data available.")

            file_path = Path(reference_data_path)

            if not file_path.exists():
                raise FileNotFoundError(f"[{self.name}] Reference data file not found.")

            data_config = loader.load_raw_data(path=reference_data_path)

        try:
            scripts_batch = ScriptsBatch.model_validate(data_config)

        except ValidationError as e:
            raise RuntimeError(f"[{self.name}] Validation error: {e}")

        attempts = context.config.process.evaluation_params.get("attempts", 1)

        return {"test_batch": scripts_batch, "attempts": attempts}


class ComparatorWorkflow(BaseWorkflow):
    def __init__(self, context: WorkflowContext) -> None:
        super().__init__(name="MetadataComparator", context=context)

    def _setup_process(self, context: WorkflowContext) -> BaseProcess:
        raise NotImplementedError

    def _load_input_data(self, context: WorkflowContext) -> Any:
        raise NotImplementedError
