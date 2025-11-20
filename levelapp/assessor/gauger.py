"""levelapp/assessor/gauger.py"""
import statistics
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Any

from levelapp.assessor.orchestrator import PipelineResult
from levelapp.aspects.logger import logger
from levelapp.assessor.schemas import Document


class MetricType(Enum):
    SIMILARITY = "similarity"  # Higher is better (0-1)
    ACCURACY = "accuracy"      # Higher is better (0-1)
    LATENCY = "latency"        # Lower is better (0-1)
    QUALITY = "quality"        # Higher is better (0-1)
    EFFICIENCY = "efficiency"  # Higher is better (0-1)


@dataclass(frozen=True)
class GaugerConfig:
    """Configuration for gauger behavior and thresholds."""
    similarity_threshold: float = 0.7
    latency_threshold_ms: float = 5000.0
    min_answer_length: int = 10
    max_answer_length: int = 1000
    enable_fallback_metrics: bool = True


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

    async def _calculate_single_document_relevance(self, document: str, query: str) -> float:
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
        return metrics


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