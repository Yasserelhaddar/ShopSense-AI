"""
Model evaluation and performance monitoring for the Knowledge Engine.

This module provides comprehensive evaluation capabilities for trained models:
- Performance metrics calculation (accuracy, perplexity, BLEU, ROUGE)
- A/B testing framework for model comparison
- Automated quality assessment
- Production model monitoring

Key Features:
- Shopping-domain specific evaluation metrics
- Conversation quality assessment
- Comparative model evaluation
- Performance trending and alerts
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from shared.logging import get_logger
from core.data import DatasetManager
from config.settings import KnowledgeSettings


logger = get_logger("knowledge-service")
settings = KnowledgeSettings()


@dataclass
class EvaluationMetrics:
    """
    Container for model evaluation metrics.

    Attributes:
        accuracy: Overall response accuracy
        perplexity: Language model perplexity
        bleu_score: BLEU score for text generation quality
        rouge_score: ROUGE score for content overlap
        response_time_ms: Average response time
        shopping_relevance: Domain-specific relevance score
    """
    accuracy: float
    perplexity: float
    bleu_score: float
    rouge_score: float
    response_time_ms: float
    shopping_relevance: float


class ModelEvaluator:
    """
    Evaluates model performance using various metrics and test scenarios.

    This class provides comprehensive evaluation capabilities for
    shopping conversation models, including domain-specific metrics.
    """

    def __init__(self):
        """Initialize the model evaluator."""
        self.dataset_manager = DatasetManager()
        self.evaluation_cache: Dict[str, Dict] = {}

    async def evaluate_model(
        self,
        model_id: str,
        test_dataset: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of a model.

        Args:
            model_id: ID of the model to evaluate
            test_dataset: Dataset to use for evaluation
            metrics: Specific metrics to compute

        Returns:
            Dictionary containing evaluation results and metrics

        Note:
            This method runs various evaluation tests and returns
            comprehensive performance metrics for the model.
        """
        if metrics is None:
            metrics = ["accuracy", "perplexity", "bleu", "rouge", "shopping_relevance"]

        if test_dataset is None:
            test_dataset = "evaluation_dataset"

        logger.info(f"Starting evaluation for model {model_id}")

        # Load test data
        test_conversations = await self._load_test_data(test_dataset)

        # Initialize results
        evaluation_results = {
            "model_id": model_id,
            "test_dataset": test_dataset,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
            "detailed_results": [],
            "summary": {}
        }

        # Run evaluations
        if "accuracy" in metrics:
            accuracy = await self._evaluate_accuracy(model_id, test_conversations)
            evaluation_results["metrics"]["accuracy"] = accuracy

        if "perplexity" in metrics:
            perplexity = await self._evaluate_perplexity(model_id, test_conversations)
            evaluation_results["metrics"]["perplexity"] = perplexity

        if "bleu" in metrics:
            bleu_score = await self._evaluate_bleu(model_id, test_conversations)
            evaluation_results["metrics"]["bleu_score"] = bleu_score

        if "rouge" in metrics:
            rouge_score = await self._evaluate_rouge(model_id, test_conversations)
            evaluation_results["metrics"]["rouge_score"] = rouge_score

        if "shopping_relevance" in metrics:
            relevance = await self._evaluate_shopping_relevance(model_id, test_conversations)
            evaluation_results["metrics"]["shopping_relevance"] = relevance

        # Calculate overall performance score
        overall_score = await self._calculate_overall_score(evaluation_results["metrics"])
        evaluation_results["summary"]["overall_score"] = overall_score

        # Cache results
        self.evaluation_cache[model_id] = evaluation_results

        logger.info(f"Evaluation completed for model {model_id}: {overall_score:.3f} overall score")
        return evaluation_results

    async def compare_models(
        self,
        model_ids: List[str],
        test_dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test dataset.

        Args:
            model_ids: List of model IDs to compare
            test_dataset: Dataset to use for comparison

        Returns:
            Dictionary with comparative results

        Note:
            Runs the same evaluation suite on multiple models
            and provides comparative analysis and rankings.
        """
        logger.info(f"Comparing {len(model_ids)} models")

        comparison_results = {
            "models": model_ids,
            "test_dataset": test_dataset,
            "timestamp": datetime.utcnow().isoformat(),
            "individual_results": {},
            "comparison": {},
            "rankings": {}
        }

        # Evaluate each model
        for model_id in model_ids:
            model_results = await self.evaluate_model(model_id, test_dataset)
            comparison_results["individual_results"][model_id] = model_results

        # Generate comparative analysis
        comparison_results["comparison"] = await self._generate_comparison_analysis(
            comparison_results["individual_results"]
        )

        # Rank models by different metrics
        comparison_results["rankings"] = await self._rank_models(
            comparison_results["individual_results"]
        )

        logger.info("Model comparison completed")
        return comparison_results

    async def monitor_production_model(
        self,
        model_id: str,
        sample_conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Monitor a production model's performance.

        Args:
            model_id: ID of the production model
            sample_conversations: Recent conversations for analysis

        Returns:
            Dictionary with monitoring results and alerts

        Note:
            Analyzes recent model performance and detects
            any degradation or issues that need attention.
        """
        logger.info(f"Monitoring production model {model_id}")

        monitoring_results = {
            "model_id": model_id,
            "timestamp": datetime.utcnow().isoformat(),
            "sample_size": len(sample_conversations),
            "metrics": {},
            "alerts": [],
            "recommendations": []
        }

        # Quick performance check
        response_times = []
        relevance_scores = []

        for conversation in sample_conversations:
            # TODO: Implement actual performance monitoring
            response_times.append(250.0)  # Mock response time
            relevance_scores.append(0.85)  # Mock relevance score

        # Calculate monitoring metrics
        avg_response_time = sum(response_times) / len(response_times)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        monitoring_results["metrics"]["avg_response_time_ms"] = avg_response_time
        monitoring_results["metrics"]["avg_relevance_score"] = avg_relevance

        # Check for alerts
        if avg_response_time > 500:
            monitoring_results["alerts"].append({
                "type": "performance",
                "message": f"High response time: {avg_response_time:.1f}ms",
                "severity": "warning"
            })

        if avg_relevance < 0.7:
            monitoring_results["alerts"].append({
                "type": "quality",
                "message": f"Low relevance score: {avg_relevance:.2f}",
                "severity": "critical"
            })

        logger.info(f"Production monitoring completed: {len(monitoring_results['alerts'])} alerts")
        return monitoring_results

    async def _load_test_data(self, test_dataset: str) -> List[Dict[str, Any]]:
        """
        Load test dataset for evaluation.

        Args:
            test_dataset: Name of the test dataset

        Returns:
            List of test conversations
        """
        try:
            return await self.dataset_manager.load_dataset(test_dataset)
        except FileNotFoundError:
            # Create a small test dataset if none exists
            logger.warning(f"Test dataset {test_dataset} not found, creating sample data")
            return await self._create_sample_test_data()

    async def _create_sample_test_data(self) -> List[Dict[str, Any]]:
        """
        Create sample test data for evaluation.

        Returns:
            List of sample conversations for testing
        """
        sample_conversations = [
            {
                "id": "test_1",
                "messages": [
                    {"role": "user", "content": "I need a laptop for gaming under $1500"},
                    {"role": "assistant", "content": "I recommend the ASUS ROG series with RTX 4060"}
                ],
                "expected_response": "Gaming laptop recommendation with GPU specification"
            },
            {
                "id": "test_2",
                "messages": [
                    {"role": "user", "content": "Compare iPhone 15 and Samsung Galaxy S24"},
                    {"role": "assistant", "content": "Both phones offer excellent performance..."}
                ],
                "expected_response": "Detailed comparison of phone features"
            }
        ]
        return sample_conversations

    async def _evaluate_accuracy(
        self,
        model_id: str,
        test_conversations: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate model accuracy on test conversations.

        Args:
            model_id: Model to evaluate
            test_conversations: Test conversation data

        Returns:
            Accuracy score between 0 and 1
        """
        from core.training import TrainingManager

        training_manager = TrainingManager()

        correct_responses = 0
        total_responses = 0

        for conversation in test_conversations:
            try:
                # Extract user message and expected response
                messages = conversation.get("messages", [])
                if len(messages) < 2:
                    continue

                # Get user messages only (exclude assistant responses)
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                expected_response = conversation.get("expected_response", "")

                if not user_messages:
                    continue

                # Run inference with the model
                result = await training_manager.run_inference(
                    model_id=model_id,
                    messages=user_messages,
                    max_tokens=150,
                    temperature=0.7
                )

                generated_response = result.get("content", "")

                # Simple accuracy check: response should be non-empty and coherent
                # For better accuracy, we'd compare with expected_response using semantic similarity
                if generated_response and len(generated_response.strip()) > 10:
                    # Basic check: response should not be repetitive gibberish
                    words = generated_response.split()
                    unique_words = set(words)
                    if len(unique_words) > len(words) * 0.3:  # At least 30% unique words
                        correct_responses += 1

                total_responses += 1

            except Exception as e:
                logger.warning(f"Error evaluating conversation: {e}")
                continue

        accuracy = correct_responses / total_responses if total_responses > 0 else 0.0
        logger.info(f"Accuracy evaluation: {accuracy:.3f} ({correct_responses}/{total_responses} responses)")
        return accuracy

    async def _evaluate_perplexity(
        self,
        model_id: str,
        test_conversations: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate model perplexity on test data.

        Args:
            model_id: Model to evaluate
            test_conversations: Test conversation data

        Returns:
            Perplexity score (lower is better)
        """
        try:
            import torch
            import math
        except ImportError:
            logger.warning("torch not available, returning default perplexity")
            return 10.0

        from core.training import TrainingManager

        training_manager = TrainingManager()
        total_loss = 0.0
        total_tokens = 0

        for conversation in test_conversations:
            try:
                messages = conversation.get("messages", [])
                if len(messages) < 2:
                    continue

                # Get assistant response to calculate perplexity on
                assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
                if not assistant_messages:
                    continue

                target_text = assistant_messages[0]["content"]

                # This is a simplified perplexity calculation
                # In production, we'd want to use the model's actual forward pass
                # For now, use inference and estimate based on response quality
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                if not user_messages:
                    continue

                result = await training_manager.run_inference(
                    model_id=model_id,
                    messages=user_messages,
                    max_tokens=len(target_text.split()),
                    temperature=0.7
                )

                # Simplified perplexity estimation based on output quality
                # This is not the true perplexity but a proxy measure
                generated_text = result.get("content", "")

                # Calculate a rough similarity score
                gen_words = set(generated_text.lower().split())
                target_words = set(target_text.lower().split())

                if len(target_words) > 0:
                    overlap = len(gen_words & target_words) / len(target_words)
                    # Convert overlap to a perplexity-like score (lower is better)
                    # High overlap = low perplexity, low overlap = high perplexity
                    estimated_loss = -math.log(max(overlap, 0.01))  # Avoid log(0)
                    total_loss += estimated_loss
                    total_tokens += 1

            except Exception as e:
                logger.warning(f"Error calculating perplexity for conversation: {e}")
                continue

        if total_tokens == 0:
            logger.warning("No valid conversations for perplexity calculation")
            return 10.0

        # Calculate average perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        logger.info(f"Perplexity evaluation: {perplexity:.3f} ({total_tokens} conversations)")
        return perplexity

    async def _evaluate_bleu(
        self,
        model_id: str,
        test_conversations: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate BLEU score for text generation quality.

        Args:
            model_id: Model to evaluate
            test_conversations: Test conversation data

        Returns:
            BLEU score between 0 and 1
        """
        try:
            from sacrebleu import corpus_bleu
        except ImportError:
            logger.warning("sacrebleu not installed, returning default score")
            return 0.0

        from core.training import TrainingManager

        training_manager = TrainingManager()
        references = []
        hypotheses = []

        for conversation in test_conversations:
            try:
                messages = conversation.get("messages", [])
                if len(messages) < 2:
                    continue

                # Get user messages and expected response
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                expected_response = conversation.get("expected_response", "")

                # Get assistant response from conversation as reference
                assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
                if not user_messages or not assistant_messages:
                    continue

                reference = assistant_messages[0]["content"]

                # Run inference with the model
                result = await training_manager.run_inference(
                    model_id=model_id,
                    messages=user_messages,
                    max_tokens=150,
                    temperature=0.7
                )

                hypothesis = result.get("content", "").strip()

                if hypothesis and reference:
                    hypotheses.append(hypothesis)
                    references.append([reference])  # BLEU expects list of references

            except Exception as e:
                logger.warning(f"Error evaluating conversation for BLEU: {e}")
                continue

        if not hypotheses or not references:
            logger.warning("No valid conversation pairs for BLEU evaluation")
            return 0.0

        # Calculate corpus BLEU score
        bleu_result = corpus_bleu(hypotheses, references)
        bleu_score = bleu_result.score / 100.0  # Convert to 0-1 scale

        logger.info(f"BLEU score evaluation: {bleu_score:.3f} ({len(hypotheses)} conversations)")
        return bleu_score

    async def _evaluate_rouge(
        self,
        model_id: str,
        test_conversations: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate ROUGE score for content overlap.

        Args:
            model_id: Model to evaluate
            test_conversations: Test conversation data

        Returns:
            ROUGE score between 0 and 1
        """
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            logger.warning("rouge-score not installed, returning default score")
            return 0.0

        from core.training import TrainingManager

        training_manager = TrainingManager()
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for conversation in test_conversations:
            try:
                messages = conversation.get("messages", [])
                if len(messages) < 2:
                    continue

                # Get user messages and assistant response as reference
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]

                if not user_messages or not assistant_messages:
                    continue

                reference = assistant_messages[0]["content"]

                # Run inference with the model
                result = await training_manager.run_inference(
                    model_id=model_id,
                    messages=user_messages,
                    max_tokens=150,
                    temperature=0.7
                )

                hypothesis = result.get("content", "").strip()

                if hypothesis and reference:
                    scores = scorer.score(reference, hypothesis)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)

            except Exception as e:
                logger.warning(f"Error evaluating conversation for ROUGE: {e}")
                continue

        if not rouge1_scores:
            logger.warning("No valid conversation pairs for ROUGE evaluation")
            return 0.0

        # Average ROUGE-L score (most commonly used)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

        logger.info(
            f"ROUGE score evaluation: ROUGE-L={avg_rougeL:.3f}, "
            f"ROUGE-1={sum(rouge1_scores)/len(rouge1_scores):.3f}, "
            f"ROUGE-2={sum(rouge2_scores)/len(rouge2_scores):.3f} "
            f"({len(rougeL_scores)} conversations)"
        )
        return avg_rougeL

    async def _evaluate_shopping_relevance(
        self,
        model_id: str,
        test_conversations: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate shopping domain relevance of responses.

        Args:
            model_id: Model to evaluate
            test_conversations: Test conversation data

        Returns:
            Shopping relevance score between 0 and 1
        """
        from core.training import TrainingManager

        # Shopping-domain keywords and concepts
        shopping_keywords = {
            # Product categories
            'product', 'item', 'laptop', 'phone', 'computer', 'tablet', 'camera',
            'headphones', 'watch', 'tv', 'monitor', 'keyboard', 'mouse', 'speaker',
            'gaming', 'smartphone', 'electronics', 'appliance', 'furniture', 'clothing',

            # Brands
            'apple', 'samsung', 'sony', 'dell', 'hp', 'lenovo', 'asus', 'acer',
            'microsoft', 'google', 'amazon', 'nike', 'adidas', 'lg', 'panasonic',

            # Shopping actions
            'buy', 'purchase', 'order', 'shop', 'browse', 'recommend', 'suggest',
            'compare', 'review', 'rating', 'return', 'refund', 'warranty', 'delivery',

            # Product attributes
            'price', 'cost', 'budget', 'cheap', 'expensive', 'affordable', 'discount',
            'sale', 'deal', 'offer', 'specification', 'feature', 'quality', 'brand',
            'model', 'version', 'size', 'color', 'weight', 'dimension', 'performance',

            # Shopping context
            'store', 'online', 'shipping', 'stock', 'available', 'in-stock',
            'out-of-stock', 'pre-order', 'bestseller', 'popular', 'trending',

            # Technical specs
            'cpu', 'gpu', 'ram', 'storage', 'battery', 'screen', 'display', 'camera',
            'processor', 'memory', 'ssd', 'hdd', 'gb', 'tb', 'inch', 'resolution',
            'ghz', 'cores', 'rtx', 'gtx', 'intel', 'amd', 'nvidia'
        }

        training_manager = TrainingManager()
        total_relevance = 0.0
        total_responses = 0

        for conversation in test_conversations:
            try:
                messages = conversation.get("messages", [])
                if len(messages) < 2:
                    continue

                # Get user messages
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                if not user_messages:
                    continue

                # Run inference with the model
                result = await training_manager.run_inference(
                    model_id=model_id,
                    messages=user_messages,
                    max_tokens=150,
                    temperature=0.7
                )

                generated_response = result.get("content", "").lower()

                if not generated_response:
                    continue

                # Count shopping keywords in response
                words_in_response = set(generated_response.split())
                matching_keywords = shopping_keywords & words_in_response

                # Calculate relevance score
                # If response has at least 10% of keywords, it's highly relevant
                # Scale based on keyword density
                keyword_count = len(matching_keywords)
                word_count = len(generated_response.split())

                if word_count > 0:
                    # Keyword density (keywords per 100 words)
                    density = (keyword_count / word_count) * 100

                    # Score based on density
                    # 0-2%: low relevance (0.0-0.3)
                    # 2-5%: medium relevance (0.3-0.6)
                    # 5-10%: high relevance (0.6-0.9)
                    # 10%+: very high relevance (0.9-1.0)
                    if density >= 10:
                        relevance_score = min(1.0, 0.9 + (density - 10) / 100)
                    elif density >= 5:
                        relevance_score = 0.6 + (density - 5) * 0.06
                    elif density >= 2:
                        relevance_score = 0.3 + (density - 2) * 0.1
                    else:
                        relevance_score = density * 0.15

                    total_relevance += relevance_score
                    total_responses += 1

            except Exception as e:
                logger.warning(f"Error evaluating shopping relevance: {e}")
                continue

        if total_responses == 0:
            logger.warning("No valid responses for shopping relevance evaluation")
            return 0.0

        avg_relevance = total_relevance / total_responses
        logger.info(f"Shopping relevance evaluation: {avg_relevance:.3f} ({total_responses} responses)")
        return avg_relevance

    async def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted overall performance score.

        Args:
            metrics: Dictionary of individual metric scores

        Returns:
            Overall performance score between 0 and 1
        """
        weights = {
            "accuracy": 0.3,
            "shopping_relevance": 0.25,
            "bleu_score": 0.2,
            "rouge_score": 0.15,
            "perplexity": 0.1  # Lower is better, so we'll invert this
        }

        weighted_score = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            if metric in weights:
                weight = weights[metric]

                # Invert perplexity (lower is better)
                if metric == "perplexity":
                    # Convert perplexity to 0-1 scale (assume max reasonable perplexity is 10)
                    normalized_value = max(0, 1 - (value / 10))
                else:
                    normalized_value = value

                weighted_score += weight * normalized_value
                total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    async def _generate_comparison_analysis(
        self,
        individual_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comparative analysis between models.

        Args:
            individual_results: Results for each model

        Returns:
            Dictionary with comparative analysis
        """
        analysis = {
            "best_performing": {},
            "metric_differences": {},
            "recommendations": []
        }

        # Find best performing model for each metric
        for metric in ["accuracy", "bleu_score", "rouge_score", "shopping_relevance"]:
            best_model = None
            best_score = -1

            for model_id, results in individual_results.items():
                if metric in results["metrics"]:
                    score = results["metrics"][metric]
                    if score > best_score:
                        best_score = score
                        best_model = model_id

            if best_model:
                analysis["best_performing"][metric] = {
                    "model": best_model,
                    "score": best_score
                }

        return analysis

    async def _rank_models(
        self,
        individual_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Rank models by different performance metrics.

        Args:
            individual_results: Results for each model

        Returns:
            Dictionary with model rankings for each metric
        """
        rankings = {}

        metrics = ["accuracy", "bleu_score", "rouge_score", "shopping_relevance", "overall_score"]

        for metric in metrics:
            model_scores = []
            for model_id, results in individual_results.items():
                if metric in results.get("summary", {}):
                    score = results["summary"][metric]
                elif metric in results.get("metrics", {}):
                    score = results["metrics"][metric]
                else:
                    continue

                model_scores.append((model_id, score))

            # Sort by score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [model_id for model_id, _ in model_scores]

        return rankings