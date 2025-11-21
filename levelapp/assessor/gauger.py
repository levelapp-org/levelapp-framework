"""levelapp/assessor/gauger.py"""
import asyncio
import statistics

from abc import ABC, abstractmethod
from typing import Dict, List, Any

from levelapp.assessor.schemas import Document, GaugerConfig, PipelineResult

from levelapp.aspects.logger import logger


class EvaluationGauger(ABC):
    """Abstract base for evaluation metrics calculators."""

    @abstractmethod
    async def calculate(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        """Calculate metrics for comparison"""
        pass

    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """List of metrics this gauger supports."""
        pass


class BaseGauger(EvaluationGauger, ABC):
    """Base class with common functionality for all gaugers."""
    def __init__(self, config: GaugerConfig | None = None):
        self.config = config or GaugerConfig()
        self._cache: Dict[str, Any] = {}

    async def calculate(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        """Calculate metrics with error handling and caching."""
        cache_key = f"{query}_{id(profile_result)}_{id(user_system_result)}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            metrics = await self._calculate_metrics(profile_result, user_system_result, query)
            self._cache[cache_key] = metrics
            return metrics

        except Exception as e:
            logger.error(f"Gauger {self.__class__.__name__} failed for query '{query}':\n{e}")
            return await self._calculate_fallback_metrics(profile_result, user_system_result, query)

    @abstractmethod
    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        """Strategy-specific metric calculation."""
        raise NotImplementedError

    async def _calculate_fallback_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        """Provide basic metrics when primary calculation fails."""
        if not self.config.enable_fallback_metrics:
            return {}

        return {
            f"fallback_{metric}": 0.5 for metric in self.supported_metrics
        }


class AnswerQualityGauger(BaseGauger):
    """Evaluates the quality of generated answers."""
    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        metrics = {}

        # Answer similarity between profile and user system
        similarity = await self._calculate_answer_similarity(
            profile_result.final_answer,
            user_system_result.final_answer,
        )
        metrics["answer_similarity"] = similarity

        # Answer relevance to query
        relevance = await self._calculate_answer_relevance(
            user_system_result.final_answer,
            query
        )
        metrics["answer_relevance"] = relevance

        # Answer coherence and fluency
        coherence = await self._calculate_answer_coherence(
            user_system_result.final_answer
        )
        metrics["answer_coherence"] = coherence

        # Factual consistency with retrieved documents
        factual_consistency = await self._calculate_factual_consistency(
            user_system_result.final_answer,
            user_system_result.source_documents
        )
        metrics["factual_consistency"] = factual_consistency

        return metrics

    @staticmethod
    async def _calculate_answer_similarity(answer1: str, answer2: str) -> float:
        """Calculate semantic similarity between two answers."""
        if not answer1 or not answer2:
            return 0.0

        # Simple implementation using token overlap (to be replaced with RapidFuzz Token-based similarity)
        tokens1 = set(answer1.lower().split())
        tokens2 = set(answer2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    @staticmethod
    async def _calculate_answer_relevance(answer: str, query: str) -> float:
        """Calculate how relevant the answer is to the query."""
        if not answer or not query:
            return 0.0

        # Simple keyword-based relevance (to be replaced with embedding similarity)
        query_tokens = set(answer.lower().split())
        answer_tokens = set(query.lower().split())

        if not query_tokens:
            return 0.0

        matching_tokens = len(query_tokens.intersection(answer_tokens))
        return matching_tokens / len(query_tokens) if matching_tokens > 0 else 0.0

    @staticmethod
    async def _calculate_answer_coherence(answer: str) -> float:
        """Evaluate answer coherence and fluency."""
        if not answer:
            return 0.0

        # Simple heuristic-based coherence scoring (to be replace with LM perplexity or grammar checking)
        sentences = [s.strip() for s in answer.split('.') if s.strip()]

        if len(sentences) <= 1:
            return 0.8  # If single sentence then it is usually coherent?

        # Check for transition words and logical flow (not sure about this)
        transition_words = {"however", "therefore", "furthermore", "additionally", "consequently"}
        found_transitions = sum(1 for word in transition_words if word in answer.lower())

        coherence_score = min(0.3 + (found_transitions * 0.1) + (len(sentences) * 0.05), 1.0)

        return coherence_score

    async def _calculate_factual_consistency(self, answer: str, sources: List[Document]) -> float:
        """Check if answer is consistent with source documents."""
        if not answer or not sources:
            return 0.0

        # Simple implementation (to be replaced with NER or LLM-as-a-judge)
        source_content = " ".join([doc.content for doc in sources])
        source_entities = self._extract_key_entities(source_content)
        answer_entities = self._extract_key_entities(answer)

        if not source_entities:
            return 0.5  # No entities to verify against

        matching_entities = len(source_entities.intersection(answer_entities))
        return matching_entities / len(source_entities) if matching_entities > 0 else 0.0

    @staticmethod
    def _extract_key_entities(text: str) -> set:
        """Simple entity extraction (placeholder for proper NER)"""
        # We can use either SpaCy, NLTKm or NER models like NuNER
        important_words = {word.lower() for word in text.split() if len(word) > 4 and word[0].isupper()}  # for now xD
        return important_words

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "answer_similarity",
            "answer_relevance",
            "answer_coherence",
            "factual_consistency",
        ]


class RetrievalQualityGauger(BaseGauger):
    """Evaluates the quality of document retrieval."""
    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        metrics = {}

        # Document relevance scoring
        relevance_score = await self._calculate_document_relevance(
            user_system_result.source_documents,
            query
        )
        metrics["retrieval_relevance"] = relevance_score

        # Document diversity
        diversity_score = await self._calculate_document_diversity(
            user_system_result.source_documents,
        )
        metrics["retrieval_diversity"] = diversity_score

        # Coverage of different aspects
        coverage_score = await self._calculate_query_coverage(
            user_system_result.source_documents,
            query
        )
        metrics["retrieval_coverage"] = coverage_score

        return metrics

    async def _calculate_document_relevance(self, documents: List[Document], query: str) -> float:
        """Calculate average relevance of retrieved documents to query."""
        if not documents:
            return 0.0

        relevance_scores = []

        for doc in documents:
            score = await self._calculate_single_document_relevance(doc.content, query)
            relevance_scores.append(score)

        return statistics.mean(relevance_scores) if relevance_scores else 0.0

    @staticmethod
    async def _calculate_single_document_relevance(document: str, query: str) -> float:
        """Calculate relevance between a single document and query."""
        if not document or not query:
            return 0.0

        # Simple TF-IDF like scoring (to be replaced with embedding similarity or cross-encoder models)
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())

        if not query_terms:
            return 0.0

        matching_terms = len(query_terms.intersection(doc_terms))
        return matching_terms / len(query_terms) if matching_terms > 0 else 0.0

    async def _calculate_document_diversity(self, documents: List[Document]) -> float:
        """Calculate diversity among retrieved documents."""
        if len(documents) <= 1:
            return 1.0  # Single document is maximally diverse by default

        # Simple content-based diversity (to be replaced with topic modeling or embedding diversity)
        all_content = " ".join([doc.content for doc in documents])
        total_tokens = len(set(all_content.split()))

        individual_tokens = [len(set(doc.content.split())) for doc in documents]
        avg_individual_tokens = statistics.mean(individual_tokens)

        if avg_individual_tokens == 0:
            return 0.0

        diversity = total_tokens / len(documents) if total_tokens > 0 else 0.0
        return min(diversity, 1.0)

    async def _calculate_query_coverage(self, documents: List[Document], query: str) -> float:
        """Calculate how well documents cover different aspects of the query."""
        if not documents or not query:
            return 0.0

        # Simple aspect coverage based on query terms (to be replaced with aspect extraction and matching)
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0

        covered_terms = set()
        for doc in documents:
            doc_terms = set(doc.content.lower().split())
            covered_terms.update(query_terms.intersection(doc_terms))

        return len(covered_terms) / len(documents) if len(covered_terms) > 0 else 0.0

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "retrieval_relevance",
            "retrieval_diversity",
            "retrieval_coverage",
        ]


class PerformanceGauger(BaseGauger):
    """Evaluates system performance metrics."""
    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str,
    ) -> Dict[str, float]:
        metrics = {}

        # Latency comparison
        latency_ratio = await self._calculate_latency_ratio(
            profile_result.total_execution_time,
            user_system_result.total_execution_time
        )
        metrics["latency_ratio"] = latency_ratio

        # Throughput estimation
        throughput_score = await self._calculate_throughput_score(
            user_system_result.total_execution_time,
            len(user_system_result.final_answer) if user_system_result.final_answer else 0
        )
        metrics["throughput_score"] = throughput_score

        # Resource efficiency
        resource_score = await self._calculate_resource_efficiency(
            profile_result,
            user_system_result,
        )
        metrics["resource_efficiency"] = resource_score

        return metrics

    @staticmethod
    async def _calculate_latency_ratio(profile_time: float, user_system_time: float) -> float:
        """Calculate latency ratio (lower is better for user system)."""
        if profile_time <= 0:
            return 1.0  # to avoid division by zero

        ratio = user_system_time / profile_time

        # Convert to score where 1.0 is perfect, lower scores for slower performance
        if ratio <= 1.0:
            return 1.0  # User system is faster or equal

        else:
            return 1.0 / ratio  # Penalize slower performance

    @staticmethod
    async def _calculate_throughput_score(execution_time: float, answer_length: int) -> float:
        """Calculate throughput efficiency."""
        if execution_time <= 0:
            return 0.0

        # Characters per second (normalized)
        chars_per_second = answer_length / execution_time

        # Normalize to 0-1 scale (assuming >100 chars/second is excellent)
        return min(chars_per_second / 100.0, 1.0)

    @staticmethod
    async def _calculate_resource_efficiency(
            profile_result: PipelineResult,
            user_system_result: PipelineResult
    ) -> float:
        """Calculate resource efficiency based on answer quality per time unit."""
        # Simple heuristic combining answer quality and latency (to be replaced with resource monitoring)
        answer_length = len(user_system_result.final_answer) if user_system_result.final_answer else 0
        execution_time = user_system_result.total_execution_time

        if execution_time <= 0 or answer_length <= 0:
            return 0.0

        efficiency = answer_length / execution_time

        # Normalize to 0-1 scale
        return min(efficiency / 50.0, 1.0)  # Assuming 50 chars/second is good efficiency xD

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "latency_ratio",
            "throughput_efficiency",
            "resource_efficiency",
        ]


class SemanticGauger(BaseGauger):
    """Advanced semantic evaluation using LLM-based judgements."""
    def __init__(self, config: GaugerConfig | None = None, llm_client: Any = None):
        super().__init__(config)
        self.llm_client = llm_client  # To be replace with JudgeEvaluator

    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        metrics = {}

        # LLM-as-a-judge answer quality assessment
        if self.llm_client:
            try:
                quality_score = await self._llm_answer_quality_assessment(
                    query,
                    user_system_result.final_answer,
                    user_system_result.source_documents,
                )
                metrics["llm_quality_score"] = quality_score

                # Comparative assessment
                comparative_score = await self._llm_comparative_assessment(
                    query,
                    profile_result.final_answer,
                    user_system_result.final_answer,
                )
                metrics["comparative_quality"] = comparative_score

            except Exception as e:
                logger.warning(f"[SemanticGauger] LLM-based assessment failed with exception:\n{e}")
                # Fall back to basic metrics
                metrics.update(
                    await self._calculate_fallback_metrics(
                        profile_result=profile_result,
                        user_system_result=user_system_result,
                        query=query,
                    )
                )

        else:
            metrics.update(
                await self._calculate_fallback_metrics(
                    profile_result=profile_result,
                    user_system_result=user_system_result,
                    query=query,
                )
            )

        return metrics

    async def _llm_answer_quality_assessment(
            self,
            query: str,
            answer: str,
            sources: List[Document]
    ) -> float:
        """Use LLM to assess answer quality (placeholder implementation)."""
        await asyncio.sleep(0.01)  # simulate LLM call (until we implement it later)

        # Mock implementation
        if not answer:
            return 0.0

        # Simple heuristic based on answer characteristics
        score = 0.5
        if len(answer) > 50:
            score += 0.2

        if any(doc.content in answer for doc in sources):
            score += 0.3

        return min(score, 1.0)

    @staticmethod
    async def _llm_comparative_assessment(
            query: str,
            profile_answer: str,
            user_answer: str
    ) -> float:
        """Use LLM to compare two answers (placeholder implementation)."""
        await asyncio.sleep(0.01)

        if not user_answer:
            return 0.0

        if not profile_answer:
            return 0.5

        # Simple comparison heuristic
        user_len = len(user_answer)
        profile_len = len(profile_answer)

        if user_len > profile_len * 1.5:
            return 0.8  # More detailed answer

        elif user_len < profile_len * 0.5:
            return 0.3  # Too brief

        else:
            return 0.5  # Comparable

    @staticmethod
    async def _fallback_semantic_metrics(
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        """Fallback semantic metrics when LLM is unavailable."""
        return {
            "semantic_similarity": 0.7,
            "context_appropriateness": 0.6
        }

    @property
    def supported_metrics(self) -> List[str]:
        base_metrics = ["semantic_similarity", "context_appropriateness"]

        if self.llm_client:
            base_metrics.extend(["llm_quality_score", "comparative_quality"])

        return base_metrics


class GaugerFactory:
    """Factory for creating and configuring gaugers."""

    @staticmethod
    def create_default_gaugers(config: GaugerConfig | None = None) -> List[BaseGauger]:
        """Create a default set of gaugers for comprehensive evaluation."""
        config = config or GaugerConfig()

        return [
            AnswerQualityGauger(config=config),
            RetrievalQualityGauger(config=config),
            PerformanceGauger(config=config),
            SemanticGauger(config=config),  # LLM client will be injected once it is implemented
        ]

    @staticmethod
    def create_gauger_by_type(gauger_type: str, config: GaugerConfig | None = None) -> BaseGauger:
        """Create specific gauger by type name."""
        config = config or GaugerConfig()
        gauger_map = {
            "answer_quality": AnswerQualityGauger,
            "retrieval_quality": RetrievalQualityGauger,
            "performance": PerformanceGauger,
            "semantic": SemanticGauger,
        }

        if gauger_type not in gauger_map:
            raise ValueError(f"[GaugerFactory] Unknown gauger type: {gauger_type}")

        return gauger_map[gauger_type](config=config)


class DefaultEvaluationGauger(EvaluationGauger):
    """Default gauger implementation (placeholder)."""

    async def calculate(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        # Mock
        return {
            "answer_similarity": 0.75,
            "retrieval_precision": 0.8,
            "generation_quality": 0.7,
            "latency_score": 0.8
        }

    @property
    def supported_metrics(self) -> List[str]:
        return ["answer_similarity", "retrieval_precision", "generation_quality", "latency_score"]


# Update the orchestrator to use real gaugers
def update_orchestrator_with_gaugers():
    """Helper function to update orchestrator with real gaugers"""

    class EnhancedAssessmentOrchestrator:
        """Orchestrator enhanced with real gaugers"""

        def __init__(self, registry, profile_builder=None, gauger_config=None):
            from levelapp.assessor.orchestrator import AssessmentOrchestrator
            self._orchestrator = AssessmentOrchestrator(registry, profile_builder)
            self._orchestrator._gaugers.clear()  # Remove placeholder

            # Add real gaugers
            real_gaugers = GaugerFactory.create_default_gaugers(gauger_config)
            for gauger in real_gaugers:
                self._orchestrator.register_gauger(gauger)

        def __getattr__(self, name):
            # Delegate to underlying orchestrator
            return getattr(self._orchestrator, name)

    return EnhancedAssessmentOrchestrator


# Example usage
async def demonstrate_gaugers():
    """Demonstrate gauger functionality"""
    from levelapp.assessor.orchestrator import PipelineResult, Document

    # Create mock results
    profile_result = PipelineResult(
        query="What is machine learning?",
        strategy_outputs={},
        final_answer="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        source_documents=[Document(content="Machine learning algorithms build models based on sample data.")],
        total_execution_time=1.5
    )

    user_result = PipelineResult(
        query="What is machine learning?",
        strategy_outputs={},
        final_answer="Machine learning is a field of AI where computers learn from data to make predictions or decisions without explicit programming for every task.",
        source_documents=[
            Document(content="Machine learning uses statistical techniques."),
            Document(content="Deep learning is a type of machine learning.")
        ],
        total_execution_time=2.0
    )

    # Test gaugers
    gauger = AnswerQualityGauger()
    metrics = await gauger.calculate(profile_result, user_result, "What is machine learning?")
    print("Answer Quality Metrics:", metrics)


if __name__ == "__main__":
    asyncio.run(demonstrate_gaugers())
