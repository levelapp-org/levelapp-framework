"""levelapp/core/workflow.py"""
from pathlib import Path
from typing import Any, Dict

from levelapp.config.interaction_request import EndpointConfig
from levelapp.core.evaluator import InteractionEvaluator
from levelapp.simulator.schemas import ScriptsBatch
from levelapp.utils.data_loader import load_json_file

from levelapp.core.base import BaseWorkflow, BaseSimulator, BaseComparator
from levelapp.core.comparator import MetadataComparator
from levelapp.core.simulator import ConversationSimulator


class SimulatorWorkflow(BaseWorkflow):

    def __init__(self) -> None:
        super().__init__(name="ConversationSimulator")
        self.simulator: BaseSimulator | None = None

    def setup(self, config: Dict[str, Any] | None = None) -> None:
        """
        Set up the simulator component.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        if not config.get("simulator_config"):
            self.config = {
                # TODO-0: Complete the StorageService impl. and add a default value to 'storage_service'.
                "storage_service": None,
                "evaluation_service": InteractionEvaluator(),
                "endpoint_configuration": EndpointConfig(),
            }
            print(f"[SimulatorWorkflow] configuration:\n{self.config}")
        self.simulator = ConversationSimulator(**self.config)

    def load_data(self, config: Dict[str, Any]) -> None:
        file_path = Path(config.get("file_path", "no-file-path"))
        if not file_path.exists():
            raise FileNotFoundError(f"No file path was provide (default value: {file_path})")

        self.data = load_json_file(model=ScriptsBatch, file_path=file_path)

    def execute(self, config: Dict[str, Any]) -> None:
        if not (self.simulator and self.data):
            raise RuntimeError("[SimulatorWorkflow] Workflow not properly initialized.")
        config["test_batch"] = self.data
        self.results = self.simulator.simulate(**config)

    def collect_results(self) -> Any:
        return self.results


class ComparatorWorkflow(BaseWorkflow):
    """Workflow for metadata extraction evaluation."""

    def __init__(self) -> None:
        super().__init__(name="MetadataComparator")
        self.comparator: BaseComparator = MetadataComparator()

    def setup(self, config: Dict[str, Any]) -> None:
        # TODO-1: Add a default config for the comparator workflow.
        self.config = config
        self.comparator = MetadataComparator(**self.config)

    def load_data(self, config: Any) -> None:
        self.data = config.load()

    def execute(self, config: dict) -> None:
        if not (self.comparator and self.data):
            raise RuntimeError("[ComparatorWorkflow] Workflow not properly initialized.")
        self.results = self.comparator.compare()

    def collect_results(self) -> Any:
        return self.results
