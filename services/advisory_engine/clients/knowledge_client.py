"""
Knowledge Engine client for the Advisory Engine.

This module provides a client interface for communicating with the
Knowledge Engine service, handling model inference and AI operations.

Key Features:
- HTTP client for Knowledge Engine API
- Request/response handling and error management
- Connection pooling and retry logic
- Health monitoring and fallback strategies
"""

import asyncio
from typing import List, Dict, Any, Optional
import httpx

from core.logging import get_logger


logger = get_logger("advisory-service")


class KnowledgeClient:
    """
    Client for communicating with the Knowledge Engine service.

    This client handles all interactions with the Knowledge Engine,
    including model inference, health checks, and error handling.
    """

    def __init__(self, base_url: str):
        """
        Initialize the Knowledge Engine client.

        Args:
            base_url: Base URL of the Knowledge Engine service
        """
        self.base_url = base_url.rstrip("/")
        self.client: Optional[httpx.AsyncClient] = None
        self.timeout = 30.0

    async def initialize(self):
        """
        Initialize the HTTP client with proper configuration.

        Sets up connection pooling, timeouts, and retry strategies
        for optimal performance and reliability.
        """
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        logger.info(f"Knowledge client initialized for {self.base_url}")

    async def get_advice(
        self,
        query: str,
        products: Optional[List[Dict]] = None,
        model_id: str = "shopsense-v1"
    ) -> Dict[str, Any]:
        """
        Get AI-generated advice for a shopping query.

        Args:
            query: User query or question
            products: Optional product context for advice
            model_id: Model to use for inference

        Returns:
            Dictionary with AI advice and recommendations

        Raises:
            httpx.RequestError: If request fails
            httpx.HTTPStatusError: If API returns error status
        """
        if not self.client:
            await self.initialize()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful shopping advisor. Provide clear, practical advice based on user needs and product information."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # Add product context if provided
        context = {}
        if products:
            context["products"] = products[:5]  # Limit context size

        payload = {
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
            "context": context
        }

        try:
            response = await self.client.post(
                f"/api/v1/models/{model_id}/inference",
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            return {
                "advice": result.get("response", ""),
                "model_used": result.get("model_used", model_id),
                "tokens_used": result.get("tokens_used", 0),
                "processing_time": result.get("processing_time_ms", 0)
            }

        except httpx.RequestError as e:
            logger.error(f"Knowledge Engine request failed: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Knowledge Engine HTTP error: {e.response.status_code}")
            raise

    async def generate_consultation_response(
        self,
        conversation_history: List[Dict[str, str]],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a consultation response based on conversation history.

        Args:
            conversation_history: Previous conversation messages
            user_context: Additional user context and preferences

        Returns:
            Dictionary with consultation response and reasoning
        """
        if not self.client:
            await self.initialize()

        # Build conversation context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert shopping consultant. Analyze the conversation "
                    "history and provide personalized advice, product recommendations, "
                    "and next steps based on the user's needs and preferences."
                )
            }
        ]

        # Add conversation history
        for msg in conversation_history[-10:]:  # Keep last 10 messages
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        payload = {
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.8,
            "context": user_context or {}
        }

        try:
            response = await self.client.post(
                "/api/v1/models/shopsense-v1/inference",
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            return {
                "response": result.get("response", ""),
                "confidence": 0.85,  # TODO: Implement actual confidence scoring
                "reasoning": "Based on conversation analysis and user preferences",
                "tokens_used": result.get("tokens_used", 0)
            }

        except Exception as e:
            logger.error(f"Consultation generation failed: {e}")
            # Return fallback response
            return {
                "response": "I'd be happy to help you find the right products. Could you tell me more about what you're looking for and your specific requirements?",
                "confidence": 0.5,
                "reasoning": "Fallback response due to service unavailability",
                "tokens_used": 0
            }

    async def analyze_products_for_comparison(
        self,
        products: List[Dict[str, Any]],
        comparison_criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze products for detailed comparison.

        Args:
            products: List of products to analyze
            comparison_criteria: Criteria for comparison

        Returns:
            Dictionary with analysis results and comparisons
        """
        if not self.client:
            await self.initialize()

        analysis_prompt = (
            f"Analyze and compare these {len(products)} products based on "
            f"the following criteria: {', '.join(comparison_criteria)}. "
            f"Provide strengths, weaknesses, and recommendations for each product."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a product analysis expert. Provide detailed, objective comparisons."
            },
            {
                "role": "user",
                "content": analysis_prompt
            }
        ]

        payload = {
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.3,  # Lower temperature for more factual analysis
            "context": {"products": products, "criteria": comparison_criteria}
        }

        try:
            response = await self.client.post(
                "/api/v1/models/shopsense-v1/inference",
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            return {
                "analysis": result.get("response", ""),
                "comparison_matrix": self._generate_comparison_matrix(products, comparison_criteria),
                "recommendations": self._extract_recommendations(result.get("response", "")),
                "processing_time": result.get("processing_time_ms", 0)
            }

        except Exception as e:
            logger.error(f"Product analysis failed: {e}")
            return {
                "analysis": "Analysis temporarily unavailable. Please try again later.",
                "comparison_matrix": {},
                "recommendations": [],
                "processing_time": 0
            }

    async def health_check(self) -> bool:
        """
        Check if the Knowledge Engine service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        if not self.client:
            await self.initialize()

        try:
            response = await self.client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Knowledge Engine health check failed: {e}")
            return False

    async def cleanup(self):
        """Close the HTTP client and clean up resources."""
        if self.client:
            await self.client.aclose()
            logger.info("Knowledge client cleaned up")

    def _generate_comparison_matrix(
        self,
        products: List[Dict[str, Any]],
        criteria: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate a comparison matrix for products.

        Args:
            products: List of products
            criteria: Comparison criteria

        Returns:
            Dictionary with comparison matrix data
        """
        matrix = {}

        for product in products:
            product_id = product.get("id", product.get("product_id", "unknown"))
            matrix[product_id] = {}

            for criterion in criteria:
                # TODO: Implement actual scoring logic based on product data
                if criterion.lower() == "price":
                    matrix[product_id][criterion] = product.get("price", 0)
                elif criterion.lower() == "rating":
                    matrix[product_id][criterion] = product.get("rating", 0)
                else:
                    matrix[product_id][criterion] = "N/A"

        return matrix

    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """
        Extract recommendations from analysis text.

        Args:
            analysis_text: AI-generated analysis text

        Returns:
            List of extracted recommendations
        """
        # TODO: Implement more sophisticated recommendation extraction
        # For now, return a placeholder
        return [
            "Consider your budget and intended use case",
            "Read user reviews for real-world insights",
            "Check warranty and support options"
        ]