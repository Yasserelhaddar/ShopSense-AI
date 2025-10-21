"""
Recommendation engine for the Advisory Engine.

This module handles the core recommendation logic, including:
- Product recommendation generation
- User preference analysis
- Product comparison and ranking
- Feedback processing and learning
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from shared.logging import get_logger


logger = get_logger("advisory-service")


class RecommendationEngine:
    """
    Core recommendation engine for generating personalized product recommendations.

    This engine coordinates between the Knowledge and Discovery engines to provide
    intelligent, context-aware product recommendations and advice.
    """

    def __init__(self, knowledge_client, discovery_client):
        """
        Initialize the recommendation engine.

        Args:
            knowledge_client: Client for Knowledge Engine API
            discovery_client: Client for Discovery Engine API
        """
        self.knowledge_client = knowledge_client
        self.discovery_client = discovery_client

    async def initialize(self):
        """Initialize the recommendation engine."""
        logger.info("Recommendation engine initialized")

    async def generate_recommendations(
        self,
        query: str,
        user_preferences: Optional[Dict[str, Any]] = None,
        budget_range: Optional[Dict[str, float]] = None,
        use_cases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate personalized product recommendations.

        Args:
            query: User search query
            user_preferences: User preferences and constraints
            budget_range: Budget constraints
            use_cases: Intended use cases

        Returns:
            Dictionary with recommendations, advice, and insights
        """
        start_time = datetime.utcnow()

        # Build search filters
        filters = {}
        if budget_range:
            if "min" in budget_range:
                filters["min_price"] = budget_range["min"]
            if "max" in budget_range:
                filters["max_price"] = budget_range["max"]

        # Search for products
        products = await self.discovery_client.search_products(
            query=query,
            filters=filters,
            limit=10
        )

        # Get AI advice
        ai_response = await self.knowledge_client.get_advice(
            query=query,
            products=products[:5]
        )

        # Generate follow-up questions
        follow_ups = self._generate_follow_up_questions(query, user_preferences)

        # Generate insights
        insights = self._generate_search_insights(query, products, budget_range)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Transform products to ProductRecommendation format
        recommendations = self._transform_products_to_recommendations(products, query)

        return {
            "products": recommendations,
            "advice": ai_response.get("advice", ""),
            "follow_ups": follow_ups,
            "insights": insights,
            "processing_time_ms": processing_time
        }

    async def compare_products(
        self,
        product_ids: List[str],
        comparison_criteria: List[str],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple products and provide analysis.

        Args:
            product_ids: List of product IDs to compare
            comparison_criteria: Criteria for comparison
            user_preferences: User preferences

        Returns:
            Dictionary with comparison results and analysis
        """
        # Get product details
        products = []
        for product_id in product_ids:
            product = await self.discovery_client.get_product_details(product_id)
            if product:
                products.append(product)

        # Generate AI analysis
        analysis_result = await self.knowledge_client.analyze_products_for_comparison(
            products=products,
            comparison_criteria=comparison_criteria
        )

        return {
            "products": products,
            "matrix": analysis_result.get("comparison_matrix", {}),
            "analysis": analysis_result.get("analysis", ""),
            "recommendation": "Based on the analysis...",  # TODO: Generate specific recommendation
            "strengths_weaknesses": self._analyze_strengths_weaknesses(products)
        }

    async def get_trending_recommendations(self) -> Dict[str, Any]:
        """Get trending product recommendations."""
        # TODO: Implement trending logic
        return {
            "products": [],
            "categories": ["laptops", "smartphones", "headphones"],
            "seasonal": [],
            "updated_at": datetime.utcnow().isoformat()
        }

    async def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback for learning."""
        # TODO: Implement feedback processing
        feedback_id = f"feedback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Processed feedback: {feedback_id}")
        return {"id": feedback_id}

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences and history."""
        # TODO: Implement user preference retrieval
        return {
            "preferences": {},
            "history": [],
            "recent_recommendations": []
        }

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Recommendation engine cleaned up")

    def _generate_follow_up_questions(
        self,
        query: str,
        user_preferences: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate relevant follow-up questions."""
        questions = [
            "What's your budget range for this purchase?",
            "Are there any specific brands you prefer or want to avoid?",
            "How will you primarily use this product?"
        ]

        # Customize based on query
        if "laptop" in query.lower():
            questions.extend([
                "Do you prefer Windows, macOS, or have no preference?",
                "Is portability important, or do you prioritize performance?"
            ])

        return questions[:3]  # Return top 3 questions

    def _generate_search_insights(
        self,
        query: str,
        products: List[Dict[str, Any]],
        budget_range: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Generate insights about the search and market."""
        insights = {
            "market_analysis": "Current market trends show...",
            "price_range": "Prices typically range from...",
            "popular_features": ["Feature 1", "Feature 2"],
            "buying_tips": ["Compare warranties", "Check return policies"]
        }

        if products:
            prices = [p.get("price", 0) for p in products if p.get("price")]
            if prices:
                insights["price_range"] = f"${min(prices):.2f} - ${max(prices):.2f}"

        return insights

    def _analyze_strengths_weaknesses(
        self,
        products: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Analyze product strengths and weaknesses."""
        analysis = {}

        for product in products:
            product_id = product.get("id", "unknown")
            analysis[product_id] = {
                "strengths": ["High rating", "Good value"],
                "weaknesses": ["Limited availability"]
            }

        return analysis

    def _transform_products_to_recommendations(
        self,
        products: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Transform Discovery Engine products to ProductRecommendation format.

        Args:
            products: Raw products from Discovery Engine
            query: User search query for generating recommendations

        Returns:
            List of products in ProductRecommendation format
        """
        recommendations = []

        for product in products:
            # Map Discovery Engine fields to ProductRecommendation fields
            recommendation = {
                "product_id": product.get("id", ""),
                "title": product.get("title", ""),
                "price": product.get("price", 0.0),
                "store": product.get("store", ""),
                "rating": product.get("rating"),
                "image_url": product.get("image_url"),
                "product_url": product.get("product_url"),
                "match_score": product.get("similarity_score", 0.0),
                "why_recommended": self._generate_recommendation_reason(product, query),
                "key_benefits": self._extract_key_benefits(product)
            }

            recommendations.append(recommendation)

        return recommendations

    def _generate_recommendation_reason(
        self,
        product: Dict[str, Any],
        query: str
    ) -> str:
        """
        Generate a reason why this product is recommended.

        Args:
            product: Product details
            query: User search query

        Returns:
            Recommendation reason string
        """
        reasons = []

        # Check similarity score
        similarity = product.get("similarity_score", 0)
        if similarity > 0.7:
            reasons.append("Highly relevant to your search")
        elif similarity > 0.5:
            reasons.append("Good match for your needs")
        else:
            reasons.append("Matches your query")

        # Check rating
        rating = product.get("rating")
        if rating and rating >= 4.5:
            reasons.append(f"excellent {rating:.1f} star rating")
        elif rating and rating >= 4.0:
            reasons.append(f"strong {rating:.1f} star rating")

        # Check price
        price = product.get("price", 0)
        if price > 0:
            if price < 100:
                reasons.append("affordable price point")
            elif price > 1000:
                reasons.append("premium option")

        return ", ".join(reasons) if reasons else "Matches your search criteria"

    def _extract_key_benefits(self, product: Dict[str, Any]) -> List[str]:
        """
        Extract key benefits from product data.

        Args:
            product: Product details

        Returns:
            List of key benefits
        """
        benefits = []

        # Add rating benefit
        rating = product.get("rating")
        if rating and rating >= 4.0:
            benefits.append(f"{rating:.1f}★ customer rating")

        # Add store benefit
        store = product.get("store", "")
        if store:
            benefits.append(f"Available at {store}")

        # Add price benefit (if competitive)
        price = product.get("price", 0)
        if price > 0:
            benefits.append(f"${price:.2f}")

        # Generic benefits if we don't have enough
        if len(benefits) < 2:
            benefits.append("Quality product")

        return benefits[:3]  # Return top 3 benefits