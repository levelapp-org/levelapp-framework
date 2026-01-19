"""levelapp/core/evaluator.py"""
import threading

from functools import lru_cache
from typing import List, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    AsyncRetrying,
    RetryError,
)

from levelapp.clients import ClientRegistry
from levelapp.comparator import MetricsManager, MetadataComparator
from levelapp.config.prompts import EVAL_PROMPT_TEMPLATE
from levelapp.core.base import BaseEvaluator, BaseChatClient
from levelapp.aspects import MonitoringAspect, MetricType, logger, DataLoader

if TYPE_CHECKING:
    from levelapp.workflow.config import WorkflowConfig


class Evidence(BaseModel):
    """Evidence details for evaluation."""
    covered_points: List[str] = Field(
        default_factory=list,
        description="Key points covered the agent reply covered (<= 3 items)"
    )
    missing_or_wrong: List[str] = Field(
        default_factory=list,
        description="Key points the agent reply missed or contradicted (<= 3 items)"
    )


class JudgeEvaluationResults(BaseModel):
    """Structured result of an interaction evaluation."""
    provider: str = Field(..., description="The provider name, e.g., 'openai', 'ionos'")
    score: int = Field(..., ge=0, le=3, description="Evaluation score between 0 and 3")
    label: str = Field(..., description="The label of the evaluation result")
    justification: str = Field(..., description="Short explanation of the evaluation result")
    evidence: Evidence = Field(default_factory=Evidence, description="Detailed evidence for the evaluation")
    raw_response: Dict[str, Any] = Field(..., description="Full unprocessed API response", exclude=True)
    metadata: Dict[str, Any] = Field(..., description="Metadata about the evaluation result")

    @classmethod
    def from_parsed(cls, provider: str, parsed: Dict[str, Any], raw: Dict[str, Any]) -> "JudgeEvaluationResults":
        """
        Build a model instance from the provided data.

        Args:
            provider (str): The provider name.
            parsed (Dict[str, Any]): The parsed response data.
            raw (Dict[str, Any]): The raw response data.

        Returns:
            JudgeEvaluationResults: The constructed evaluation result instance.
        """
        content = parsed.get("output", {})
        metadata = parsed.get("metadata", {})
        return cls(
            provider=provider,
            score=content.get("score", 0),
            label=content.get("label", "N/A"),
            justification=content.get("justification", "N/A"),
            evidence=Evidence(**content.get("evidence", {})),
            raw_response=raw,
            metadata=metadata,
        )


class JudgeEvaluator(BaseEvaluator):
    """LLM-as-a-judge evaluator class"""
    def __init__(self, config: "WorkflowConfig | None" = None):
        """
        Initialize the JudgeEvaluator.

        Args:
            config (WorkflowConfig | None): The configuration of the workflow.
        """
        if config:
            self.config = config
            self.providers = config.evaluation.providers

        self.prompt_template = EVAL_PROMPT_TEMPLATE
        self.client_registry = ClientRegistry

    def select_client(self, provider: str) -> BaseChatClient:
        """
        Select an LLM client to use for the evaluation.

        Args:
            provider (str): The provider name.

        Returns:
            client (BaseChatClient): The LLM client to use for the evaluation.
        """
        if provider not in self.client_registry.list_providers():
            logger.warning(f"[JudgeEvaluator] {provider} is not registered. Defaulting to 'OpenAI'.")
            return self.client_registry.get(provider="openai")

        return self.client_registry.get(provider=provider)

    @lru_cache(maxsize=1024)
    def _build_prompt(self, user_input: str, generated_text: str, reference_text: str) -> str:
        """
        Build the prompt used for the evaluation.

        Args:
            user_input (str): The user input.
            generated_text (str): The generated text.
            reference_text (str): The reference text.

        Returns:
            A string containing the prompt.
        """
        return self.prompt_template.format(
            user_input=user_input,
            generated_text=generated_text,
            reference_text=reference_text
        )

    @retry(
        retry=retry_if_exception_type((TimeoutError, ValueError, RuntimeError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def evaluate(
            self,
            generated_data: str,
            reference_data: str,
            user_input: str,
            provider: str,
    ) -> JudgeEvaluationResults | None:
        """
        Synchronous evaluation for the generated data.

        Args:
            generated_data (str): The generated data.
            reference_data (str): The reference data.
            user_input (str): The user input.
            provider (str): The LLM provider user for evaluation.

        Returns:
            JudgeEvaluationResults instance containing the evaluation results.

        Raises:
            Exception: If the evaluation failed.
        """
        prompt = self._build_prompt(
            user_input=user_input,
            generated_text=generated_data,
            reference_text=reference_data
        )
        client = self.select_client(provider=provider)

        try:
            response = client.call(message=prompt)
            logger.info(f"[{provider}] Evaluation: {response}\n{'---' * 10}")
            parsed = client.parse_response(response=response)
            return JudgeEvaluationResults.from_parsed(provider=provider, parsed=parsed, raw=response)

        except Exception as e:
            logger.error(f"[{provider}] Evaluation failed: {e}", exc_info=True)
            return JudgeEvaluationResults(
                provider=provider,
                score=0,
                label="N/A",
                justification="N/A",
                evidence=Evidence(covered_points=[], missing_or_wrong=[]),
                raw_response={},
                metadata={}
            )

    @MonitoringAspect.monitor(name="judge_evaluation", category=MetricType.API_CALL)
    async def async_evaluate(
            self,
            generated_data: str,
            reference_data: str,
            user_input: str,
            provider: str,
    ) -> JudgeEvaluationResults | None:
        """
        Synchronous evaluation for the generated data.

        Args:
            generated_data (str): The generated data.
            reference_data (str): The reference data.
            user_input (str): The user input.
            provider (str): The LLM provider user for evaluation.

        Returns:
            JudgeEvaluationResults instance containing the evaluation results.

        Raises:
            RetryError: If the evaluation failed.
        """
        prompt = self._build_prompt(
            user_input=user_input,
            generated_text=generated_data,
            reference_text=reference_data
        )
        client = self.select_client(provider=provider)

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((TimeoutError, ValueError, RuntimeError)),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=True,
            ):
                with attempt:
                    response = await client.acall(message=prompt)
                    parsed = client.parse_response(response=response)
                    return JudgeEvaluationResults.from_parsed(provider=provider, parsed=parsed, raw=response)

        except RetryError as e:
            logger.error(f"[{provider}] Async evaluation failed after retries: {e}", exc_info=True)
            return JudgeEvaluationResults(
                provider=provider,
                score=0,
                label="N/A",
                justification="N/A",
                evidence=Evidence(covered_points=[], missing_or_wrong=[]),
                raw_response={},
                metadata={}
            )


class MetadataEvaluator(BaseEvaluator):
    """Metadata evaluator class."""
    def __init__(self, config: "WorkflowConfig | None" = None):
        """
        Initialize the MetadataEvaluator.

        Args:
            config (WorkflowConfig | None): The workflow configuration.
        """
        if config:
            self.config = config
            self.metrics_map = config.evaluation.metrics_map

        self.data_loader = DataLoader()
        self.comparator = MetadataComparator()
        self.metrics_manager = MetricsManager()

    def evaluate(
            self,
            generated_data: str | Dict[str, Any],
            reference_data: str | Dict[str, Any],
            metrics_mapping: Any | None = None,
    ) -> Dict[str, float]:
        """
        Synchronous evaluation for the generated data.

        Args:
              generated_data (str): The generated data.
              reference_data (str): The reference data.
              metrics_mapping (dict): A dictionary mapping metric names to metrics.

        Returns:
              A dict containing the evaluation results.
        """
        gen_data = self.data_loader.create_dynamic_model(data=generated_data, model_name="GeneratedMetadata")
        ref_data = self.data_loader.create_dynamic_model(data=reference_data, model_name="ReferenceMetadata")

        if metrics_mapping:
            self.comparator.metrics_manager = metrics_mapping
        else:
            logger.info(f"[MetadataEvaluator] Metric map: {self.metrics_map}")
            self.comparator.metrics_manager = self.metrics_map

        self.comparator.metrics_manager = self.metrics_manager
        self.comparator.generated_data = gen_data
        self.comparator.reference_data = ref_data

        output = self.comparator.run(indexed_mode=False)
        results: Dict[str, float] = {}
        logger.info(f"[MetadataEvaluator] Metadata Evaluation Output:\n{output}]")

        for k, v in output.items():
            field = v.get("field_name", "N/A")
            score = v.get("set_scores", -1)

            if score is None:
                results[field] = -1
                continue

            try:
                val = score[0] if isinstance(score, list) else score
                results[field] = float(val)

            except (TypeError, ValueError):
                results[field] = -1

        return results

    async def async_evaluate(
            self,
            generated_data: str | Dict[str, Any],
            reference_data: str | Dict[str, Any],
            **kwargs
    ):
        """Not implemented yet."""
        raise NotImplementedError()


class SimilarityEvaluator(BaseEvaluator):
    """
    Evaluator that computes semantic similarity between generated and reference text
    using Sentence-Transformers and Cosine Similarity.
    """
    _model_lock = threading.Lock()
    _similarity_model = None

    def __init__(self, config: "WorkflowConfig | None" = None):
        """
        Initialize the SimilarityEvaluator.

        Args:
            config (WorkflowConfig | None): The workflow configuration.
        """
        if config:
            self.config = config

        self.model_name = "all-MiniLM-L6-v2"
        self.warmup()

    @classmethod
    def _load_model(cls, model_name: str) -> SentenceTransformer:
        """
        Thread-safe lazy loader for the Sentence-Transformer model.
        Implements the warm-up pattern: model loads on first use.

        Args:
            model_name (str): The name of the Sentence-Transformer model.

        Returns:
            SentenceTransformer: Loaded model instance.
        """
        if cls._similarity_model is None:
            with cls._model_lock:
                if cls._similarity_model is None:
                    logger.info(
                        f"[SimilarityEvaluator] Downloading and preparing Sentence-Transformer model '{model_name}'..."
                        f"This might take a moment..."
                    )
                    cls._similarity_model = SentenceTransformer(model_name)
                    logger.info(f"[SimilarityEvaluator] Model '{model_name}' is ready.")

        return cls._similarity_model

    def warmup(self) -> None:
        """
        Explicit warm-up method to load the model before evaluation.
        """
        logger.info("[SimilarityEvaluator] Warming up...")
        self._load_model(self.model_name)
        logger.info("[SimilarityEvaluator] Model loaded.")

    def evaluate(
            self,
            generated_data: str,
            reference_data: str,
            **kwargs
    ) -> Dict[str, float]:
        """
        Compute semantic similarity between generated and reference text.

        Args:
            generated_data (str): The generated text reply.
            reference_data (str): The reference text reply.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, float]: A dictionary with single key 'semantic_similarity' score (0.0-1.0).
        """
        model = self._load_model(self.model_name)

        try:
            embeddings = model.encode([generated_data, reference_data])
            gen_embedding = embeddings[0].reshape(1, -1)
            ref_embedding = embeddings[1].reshape(1, -1)

            similarity_score = cosine_similarity(gen_embedding, ref_embedding)

            return {"semantic_similarity": float(similarity_score.item())}

        except RuntimeError as e:
            logger.error(f"[SimilarityEvaluator] Failed to compute semantic similarity:\n{e}\n---", exc_info=True)
            return {"semantic_similarity": -1.0}

        except Exception as e:
            logger.error(f"[SimilarityEvaluator] Evaluation failed:\n{e}\n---", exc_info=True)
            return {"semantic_similarity": -1.0}

    async def async_evaluate(
            self,
            generated_data: str,
            reference_data: str,
            **kwargs
    ) -> Dict[str, float]:
        """
        Async version of evaluate. Since encoding is CPU-bounded and the model isn't async-native,
        we run the synchronous 'evaluate' method in a thread pool.

        Args:
            generated_data (str): The generated text reply.
            reference_data (str): The reference text reply.
            **kwargs: Additional keyword arguments.
        """
        import asyncio
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            None,
            lambda: self.evaluate(generated_data, reference_data, **kwargs)
        )


if __name__ == '__main__':
    import asyncio

    generated_text = "Yes, we recommend completing any necessary paperwork before your appointment to ensure a smooth visit. You can usually find the forms on our website or we can provide them when you arrive."
    reference_text = "Yes, we recommend completing any necessary paperwork before your appointment to ensure a smooth visit. You can usually find the forms on our website or arrive a bit early to fill them out at the clinic."

    similarity_evaluator = SimilarityEvaluator()
    similarity_evaluator.warmup()
    res = asyncio.run(
        similarity_evaluator.async_evaluate(generated_data=generated_text, reference_data=reference_text)
    )
    print(f"Evaluation results: {res}")