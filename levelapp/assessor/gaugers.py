"""levelapp/assessor/gauger.py"""
import asyncio
import statistics

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any

from rapidfuzz import fuzz

from levelapp.assessor.schemas import Document, GaugerConfig, PipelineResult
from levelapp.aspects.logger import logger

SENTENCE_TRANSFORMERS_AVAILABLE = False
LANGUAGE_TOOLS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True

except ImportError:
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")


try:
    import language_tool_python
    from language_tool_python import LanguageTool
    LANGUAGE_TOOLS_AVAILABLE = True

except ImportError:
    logger.warning("LanguageTool not available. Install with: pip install language-tool-python")


@dataclass
class SimilarityConfig:
    """Configuration for similarity calculation."""
    similarity_threshold: float = 0.7
    min_answer_length: int = 10
    embedding_model: str = "all-MiniLM-L6-v2"
    enable_semantic_similarity: bool = True
    enable_fuzzy_matching: bool = True


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

    async def initialize(self) -> None:
        """Initialize gauger resources. Override if needed."""
        pass

    async def cleanup(self) -> None:
        """Clean up gauger resources. Override if needed."""
        pass


class BaseGauger(EvaluationGauger, ABC):
    """Base class with common functionality for all gaugers."""

    def __init__(self, config: GaugerConfig | None = None):
        self.config = config or GaugerConfig()
        self._cache: Dict[str, Any] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize gauger resources."""
        if not self._initialized:
            await self._initialize()
            self._initialized = True

    async def _initialize(self) -> None:
        """Gauger-specific initialization. Override in subclasses."""
        pass

    async def cleanup(self) -> None:
        """Clean up gauger resources."""
        if self._initialized:
            await self._cleanup()
            self._initialized = False

    async def _cleanup(self) -> None:
        """Gauger-specific cleanup. Override in subclasses."""
        pass

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
            # Ensure gauger is initialized
            if not self._initialized:
                await self.initialize()

            metrics = await self._calculate_metrics(profile_result, user_system_result, query)
            self._cache[cache_key] = metrics
            return metrics

        except Exception as e:
            logger.error(f"Gauger {self.__class__.__name__} failed for query '{query}': {e}")
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

    def __init__(self, config: GaugerConfig | None = None, similarity_config: SimilarityConfig | None = None):
        super().__init__(config)
        self.similarity_config = similarity_config or SimilarityConfig()
        self._embedding_model: SentenceTransformer | None = None
        self._language_tool: LanguageTool | None = None

    async def _initialize(self) -> None:
        """Initialize models for quality assessment."""
        if self.similarity_config.enable_semantic_similarity and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer(self.similarity_config.embedding_model)
                logger.info(f"[AnswerQualityGauger] Loaded embedding model: {self.similarity_config.embedding_model}")
            except Exception as e:
                logger.warning(f"[AnswerQualityGauger] Failed to load embedding model: {e}")

        if LANGUAGE_TOOLS_AVAILABLE:
            try:
                self._language_tool = language_tool_python.LanguageTool('en-US')
            except Exception as e:
                logger.warning(f"[AnswerQualityGauger] Failed to initialize LanguageTool: {e}")

    async def _cleanup(self) -> None:
        """Clean up model resources."""
        self._embedding_model = None
        if self._language_tool:
            self._language_tool.close()
            self._language_tool = None

    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        metrics = {}

        # Enhanced answer similarity with multiple methods
        semantic_similarity = await self._calculate_semantic_similarity(
            profile_result.final_answer,
            user_system_result.final_answer
        )
        metrics["semantic_similarity"] = semantic_similarity

        # Fuzzy string matching for surface similarity
        fuzzy_similarity = await self._calculate_fuzzy_similarity(
            profile_result.final_answer,
            user_system_result.final_answer  # FIXED: Compare answers, not answer with query
        )
        metrics["fuzzy_similarity"] = fuzzy_similarity

        # Answer relevance using semantic similarity to query
        relevance_score = await self._calculate_semantic_relevance(
            user_system_result.final_answer,
            query
        )
        metrics["answer_relevance"] = relevance_score

        # Grammar and fluency assessment
        fluency_score = await self._calculate_fluency_score(
            user_system_result.final_answer
        )
        metrics["answer_fluency"] = fluency_score

        # Factual consistency with advanced matching
        consistency_score = await self._calculate_factual_consistency(
            user_system_result.final_answer,
            user_system_result.source_documents
        )
        metrics["factual_consistency"] = consistency_score  # FIXED: Correct metric name

        # Answer completeness (length and structure)
        completeness_score = await self._calculate_completeness_score(
            user_system_result.final_answer,
            query
        )
        metrics["answer_completeness"] = completeness_score

        return metrics

    async def _calculate_semantic_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if not answer1 or not answer2:
            return 0.0

        # FIXED: Corrected condition
        if not self._embedding_model or not SENTENCE_TRANSFORMERS_AVAILABLE:
            # Fallback to token-based similarity
            return await self._calculate_token_similarity(answer1, answer2)

        try:
            # Encode both answers
            embeddings = self._embedding_model.encode([answer1, answer2], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning(f"[AnswerQualityGauger] Semantic similarity failed: {e}")
            return await self._calculate_token_similarity(answer1, answer2)

    @staticmethod
    async def _calculate_fuzzy_similarity(answer1: str, answer2: str) -> float:
        """Calculate fuzzy string similarity using multiple methods."""
        if not answer1 or not answer2:
            return 0.0

        try:
            # Use multiple fuzzy matching techniques
            ratio = fuzz.ratio(answer1, answer2) / 100.0
            partial_ratio = fuzz.partial_ratio(answer1, answer2) / 100.0
            token_sort_ratio = fuzz.token_sort_ratio(answer1, answer2) / 100.0

            # Weighted average favoring token-based matching
            combined_score = (ratio * 0.3 + partial_ratio * 0.3 + token_sort_ratio * 0.4)
            return max(0.0, min(1.0, combined_score))
        except Exception as e:
            logger.warning(f"[AnswerQualityGauger] Fuzzy similarity failed: {e}")
            return 0.0

    async def _calculate_semantic_relevance(self, answer: str, query: str) -> float:
        """Calculate semantic relevance between answer and query."""
        if not answer or not query:
            return 0.0

        if not self._embedding_model:
            return await self._calculate_keyword_relevance(answer, query)

        try:
            embeddings = self._embedding_model.encode([answer, query], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning(f"[AnswerQualityGauger] Semantic relevance failed: {e}")
            return await self._calculate_keyword_relevance(answer, query)

    async def _calculate_fluency_score(self, answer: str) -> float:
        """Calculate grammar and fluency score using LanguageTool."""
        if not answer or len(answer.split()) < 3:
            return 0.5  # Neutral score for very short answers

        if not self._language_tool or not LANGUAGE_TOOLS_AVAILABLE:
            return await self._calculate_heuristic_fluency(answer)

        try:
            matches = self._language_tool.check(answer)
            error_count = len(matches)
            word_count = len(answer.split())

            # Score based on errors per word (lower is better)
            error_ratio = error_count / max(word_count, 1)
            fluency_score = max(0.0, 1.0 - min(error_ratio * 10, 1.0))  # Normalize
            return fluency_score
        except Exception as e:
            logger.warning(f"[AnswerQualityGauger] Fluency check failed: {e}")
            return await self._calculate_heuristic_fluency(answer)

    async def _calculate_factual_consistency(self, answer: str, sources: List[Document]) -> float:
        """Advanced factual consistency using semantic matching."""
        if not answer or not sources:
            return 0.0

        if not self._embedding_model:
            return await self._calculate_basic_factual_consistency(answer, sources)

        try:
            # Encode answer and all source documents
            answer_embedding = self._embedding_model.encode([answer], convert_to_tensor=True)[0]  # FIXED: variable name
            source_texts = [doc.content for doc in sources]
            source_embeddings = self._embedding_model.encode(source_texts, convert_to_tensor=True)  # FIXED: correct parameter

            # Calculate similarities
            similarities = util.pytorch_cos_sim(answer_embedding, source_embeddings)[0]
            max_similarity = similarities.max().item()

            # Normalize and threshold
            consistency_score = max(0.0, min(1.0, max_similarity * 1.5))  # Scale for better distribution
            return consistency_score
        except Exception as e:
            logger.warning(f"[AnswerQualityGauger] Advanced consistency check failed: {e}")
            return await self._calculate_basic_factual_consistency(answer, sources)

    @staticmethod
    async def _calculate_completeness_score(answer: str, query: str) -> float:
        """Calculate answer completeness based on length and structure."""
        if not answer:
            return 0.0

        word_count = len(answer.split())
        sentence_count = len([s for s in answer.split('.') if s.strip()])

        # Base score from length (normalized)
        length_score = min(word_count / 50.0, 1.0)  # 50 words = perfect length

        # Structure bonus for multiple sentences
        structure_bonus = min(sentence_count * 0.1, 0.3)

        # Question-specific completeness
        question_indicators = any(word in query.lower() for word in ['how', 'why', 'explain', 'describe'])
        if question_indicators and sentence_count < 2:
            length_score *= 0.7  # Penalize short answers for explanatory questions

        return max(0.0, min(1.0, length_score + structure_bonus))

    # Fallback methods
    @staticmethod
    async def _calculate_token_similarity(answer1: str, answer2: str) -> float:
        """Token-based similarity fallback."""
        tokens1 = set(answer1.lower().split())
        tokens2 = set(answer2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    async def _calculate_keyword_relevance(answer: str, query: str) -> float:
        """Keyword-based relevance fallback."""
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())

        if not query_terms:
            return 0.0

        matching_terms = len(query_terms.intersection(answer_terms))
        return matching_terms / len(query_terms) if len(query_terms) > 0 else 0.0

    @staticmethod
    async def _calculate_heuristic_fluency(answer: str) -> float:
        """Heuristic fluency fallback."""
        sentences = [s.strip() for s in answer.split('.') if s.strip()]

        if len(sentences) <= 1:
            return 0.8

        # Simple checks for basic fluency indicators
        transition_words = {"however", "therefore", "furthermore", "additionally", "consequently"}  # FIXED: spelling
        found_transitions = sum(1 for word in transition_words if word in answer.lower())

        return min(0.7 + (found_transitions * 0.1) + (len(sentences) * 0.05), 1.0)

    async def _calculate_basic_factual_consistency(self, answer: str, sources: List[Document]) -> float:
        """Basic factual consistency fallback."""
        source_content = " ".join([doc.content for doc in sources])
        source_entities = self._extract_key_entities(source_content)
        answer_entities = self._extract_key_entities(answer)

        if not source_entities:
            return 0.5

        matching_entities = len(source_entities.intersection(answer_entities))
        return matching_entities / len(source_entities) if len(source_entities) > 0 else 0.0

    @staticmethod
    def _extract_key_entities(text: str) -> set:
        """Enhanced entity extraction."""
        # Extract capitalized phrases and longer words as potential entities
        words = text.split()
        entities = set()

        current_entity = []
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                current_entity.append(word)
            elif current_entity:
                entities.add(' '.join(current_entity).lower())
                current_entity = []

        if current_entity:
            entities.add(' '.join(current_entity).lower())

        return entities

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "semantic_similarity",
            "fuzzy_similarity",
            "answer_relevance",
            "answer_fluency",
            "factual_consistency",
            "answer_completeness"
        ]


class RetrievalQualityGauger(BaseGauger):
    """Enhanced retrieval quality evaluation."""

    def __init__(self, config: GaugerConfig | None = None):
        super().__init__(config)
        self._embedding_model: SentenceTransformer | None = None

    async def _initialize(self) -> None:
        """Initialize models for retrieval assessment."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"[RetrievalQualityGauger] Failed to load embedding model: {e}")

    async def _cleanup(self) -> None:
        """Clean up model resources."""
        self._embedding_model = None

    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        metrics = {}

        # Document relevance with semantic matching
        relevance_score = await self._calculate_semantic_relevance(
            user_system_result.source_documents,
            query
        )
        metrics["retrieval_relevance"] = relevance_score

        # Document diversity using embedding variance
        diversity_score = await self._calculate_embedding_diversity(
            user_system_result.source_documents
        )
        metrics["retrieval_diversity"] = diversity_score

        # Coverage of different aspects
        coverage_score = await self._calculate_semantic_coverage(
            user_system_result.source_documents,
            query
        )
        metrics["retrieval_coverage"] = coverage_score

        # Document novelty (non-redundancy)
        novelty_score = await self._calculate_novelty_score(
            user_system_result.source_documents
        )
        metrics["retrieval_novelty"] = novelty_score

        return metrics

    async def _calculate_semantic_relevance(self, documents: List[Document], query: str) -> float:
        """Calculate average relevance of documents to query."""
        if not documents or not query:
            return 0.0

        if not self._embedding_model:
            return await self._calculate_keyword_relevance(documents, query)

        try:
            query_embedding = self._embedding_model.encode([query], convert_to_tensor=True)[0]
            doc_texts = [doc.content for doc in documents]
            doc_embeddings = self._embedding_model.encode(doc_texts, convert_to_tensor=True)

            similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
            avg_similarity = similarities.mean().item()
            return max(0.0, min(1.0, avg_similarity))
        except Exception as e:
            logger.warning(f"[RetrievalQualityGauger] Semantic relevance failed: {e}")
            return await self._calculate_keyword_relevance(documents, query)

    async def _calculate_embedding_diversity(self, documents: List[Document]) -> float:
        """Calculate diversity using embedding variance."""
        if len(documents) <= 1:
            return 1.0

        if not self._embedding_model or not SENTENCE_TRANSFORMERS_AVAILABLE:
            return await self._calculate_content_diversity(documents)

        try:
            doc_texts = [doc.content for doc in documents]
            embeddings = self._embedding_model.encode(doc_texts, convert_to_tensor=True)

            # Calculate pairwise similarities - FIXED: use pytorch_cos_sim
            similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

            # Diversity = 1 - average similarity (excluding diagonal)
            mask = ~torch.eye(len(documents), dtype=torch.bool, device=similarity_matrix.device)
            avg_similarity = similarity_matrix[mask].mean().item()
            diversity = 1.0 - avg_similarity

            return max(0.0, min(1.0, diversity))
        except Exception as e:
            logger.warning(f"[RetrievalQualityGauger] Embedding diversity failed: {e}")
            return await self._calculate_content_diversity(documents)

    @staticmethod
    async def _calculate_semantic_coverage(documents: List[Document], query: str) -> float:
        """Calculate how well documents cover different query aspects."""
        if not documents or not query:
            return 0.0

        # Simple implementation (to be replaced with aspect extraction)
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0

        covered_terms = set()
        for doc in documents:
            doc_terms = set(doc.content.lower().split())
            covered_terms.update(query_terms.intersection(doc_terms))

        coverage = len(covered_terms) / len(query_terms)
        return max(0.0, min(1.0, coverage))

    @staticmethod
    async def _calculate_novelty_score(documents: List[Document]) -> float:
        """Calculate how novel/non-redundant the documents are."""
        if len(documents) <= 1:
            return 1.0

        try:
            # Use fuzzy matching to detect near-duplicates
            doc_contents = [doc.content for doc in documents]
            duplicate_pairs = 0
            total_pairs = 0

            for i in range(len(doc_contents)):
                for j in range(i+1, len(doc_contents)):
                    similarity = fuzz.token_sort_ratio(doc_contents[i], doc_contents[j]) / 100.0
                    if similarity > 0.8:  # Threshold for near-duplicates
                        duplicate_pairs += 1
                    total_pairs += 1

            novelty = 1.0 - (duplicate_pairs / max(total_pairs, 1))
            return max(0.0, min(1.0, novelty))
        except Exception as e:
            logger.warning(f"[RetrievalQualityGauger] Novelty calculation failed: {e}")
            return 0.5

    # Fallback methods
    @staticmethod
    async def _calculate_keyword_relevance(documents: List[Document], query: str) -> float:
        """Keyword-based relevance fallback."""
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0

        relevance_scores = []
        for doc in documents:
            doc_terms = set(doc.content.lower().split())
            matching_terms = len(query_terms.intersection(doc_terms))
            score = matching_terms / len(query_terms) if matching_terms > 0 else 0.0
            relevance_scores.append(score)

        return statistics.mean(relevance_scores) if relevance_scores else 0.0

    @staticmethod
    async def _calculate_content_diversity(documents: List[Document]) -> float:
        """Content-based diversity fallback."""
        if len(documents) <= 1:
            return 1.0

        all_content = " ".join([doc.content for doc in documents])
        total_tokens = len(set(all_content.split()))
        # FIXED: Correct list comprehension
        avg_tokens_per_doc = statistics.mean([len(set(doc.content.split())) for doc in documents])

        if avg_tokens_per_doc == 0:
            return 0.0

        diversity = total_tokens / (avg_tokens_per_doc * len(documents))
        return min(diversity, 1.0)

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "retrieval_relevance",
            "retrieval_diversity",
            "retrieval_coverage",
            "retrieval_novelty"
        ]


class PerformanceGauger(BaseGauger):
    """Enhanced performance evaluation with real metrics."""

    async def _calculate_metrics(
            self,
            profile_result: PipelineResult,
            user_system_result: PipelineResult,
            query: str
    ) -> Dict[str, float]:
        metrics = {}

        # Latency comparison with normalization
        latency_score = await self._calculate_latency_score(
            profile_result.total_execution_time,
            user_system_result.total_execution_time
        )
        metrics["latency_score"] = latency_score

        # Throughput efficiency
        throughput_score = await self._calculate_throughput_efficiency(
            user_system_result.total_execution_time,
            len(user_system_result.final_answer) if user_system_result.final_answer else 0
        )
        metrics["throughput_efficiency"] = throughput_score

        # Cost efficiency (placeholder for actual cost calculation)
        cost_score = await self._calculate_cost_efficiency(
            profile_result,
            user_system_result
        )
        metrics["cost_efficiency"] = cost_score

        # Stability score (variance in performance)
        stability_score = await self._calculate_stability_score(
            profile_result,
            user_system_result
        )
        metrics["stability_score"] = stability_score

        return metrics

    @staticmethod
    async def _calculate_latency_score(profile_time: float, user_time: float) -> float:
        """Calculate normalized latency score."""
        if profile_time <= 0 or user_time <= 0:
            return 0.5

        # Score based on relative performance
        if user_time <= profile_time:
            # User system is faster or equal - perfect score
            return 1.0
        else:
            # Penalize slower performance with exponential decay
            ratio = user_time / profile_time
            # Score decays from 1.0 to 0.0 as ratio increases from 1.0 to 5.0
            score = max(0.0, 1.0 - (ratio - 1.0) / 4.0)
            return score

    @staticmethod
    async def _calculate_throughput_efficiency(execution_time: float, answer_length: int) -> float:
        """Calculate throughput efficiency in characters per second."""
        if execution_time <= 0:
            return 0.0

        chars_per_second = answer_length / execution_time

        # Normalize: 0-50 chars/s = linear, 50+ chars/s = perfect
        if chars_per_second <= 50:
            return chars_per_second / 50.0
        else:
            return 1.0

    @staticmethod
    async def _calculate_cost_efficiency(profile_result: PipelineResult,
                                         user_system_result: PipelineResult) -> float:
        """Calculate cost efficiency (placeholder for real cost calculation)."""
        # In production, this would use actual API costs, token counts, etc.
        # For now, use answer length as a proxy for cost
        profile_length = len(profile_result.final_answer) if profile_result.final_answer else 0
        user_length = len(user_system_result.final_answer) if user_system_result.final_answer else 0

        if user_length == 0:
            return 0.0

        # Simple heuristic: shorter answers are more cost-efficient
        # Normalize based on profile performance
        if profile_length > 0:
            efficiency = profile_length / user_length
            return min(efficiency, 1.0)
        else:
            return 0.5

    async def _calculate_stability_score(self, profile_result: PipelineResult,
                                         user_system_result: PipelineResult) -> float:
        """Calculate stability score based on error rates and consistency."""
        profile_stable = profile_result.success and not profile_result.error
        user_stable = user_system_result.success and not user_system_result.error

        if profile_stable and user_stable:
            return 1.0
        elif user_stable:
            return 0.8  # User system stable but profile had issues
        elif profile_stable:
            return 0.3  # Profile stable but user system had issues
        else:
            return 0.1  # Both unstable

    @property
    def supported_metrics(self) -> List[str]:
        return [
            "latency_score",
            "throughput_efficiency",
            "cost_efficiency",
            "stability_score"
        ]


class SemanticGauger(BaseGauger):
    """Advanced semantic evaluation using LLM-based judgements."""
    def __init__(self, config: GaugerConfig | None = None, llm_client: Any = None):
        super().__init__(config)
        self.llm_client = llm_client  # To be replaced with JudgeEvaluator

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

    @staticmethod
    async def _llm_answer_quality_assessment(
            query: str,
            answer: str,
            sources: List[Document]
    ) -> float:
        """Use LLM to assess answer quality (placeholder implementation)."""
        await asyncio.sleep(0.01)  # simulate LLM call (until we implement it later)

        # Mock implementation
        if not answer or not query:
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

        gaugers = [
            AnswerQualityGauger(config=config),
            RetrievalQualityGauger(config=config),
            PerformanceGauger(config=config),
        ]

        # Only include SemanticGauger if LLM client is available
        # gaugers.append(SemanticGauger(config=config))

        return gaugers

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
        final_answer="Gabagool! Ova heaa!",
        # final_answer="Machine learning is a field of AI where computers learn from data to make predictions or decisions without explicit programming for every task.",
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