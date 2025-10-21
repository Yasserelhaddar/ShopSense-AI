"""
Advisory Engine API schemas.

This module defines Pydantic models for request and response validation
in the Advisory Engine API. These schemas ensure type safety and provide
automatic documentation for recommendation and consultation endpoints.

Models:
- SearchRequest: AI-powered search parameters
- SearchResponse: Search results with AI advice
- AdviceRequest: Consultation parameters
- AdviceResponse: AI-generated advice
- ComparisonRequest: Product comparison parameters
- ComparisonResponse: Comparison results
- HealthResponse: Service health status
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """
    User preferences for personalized recommendations.

    Attributes:
        priority: User priority factors (price, quality, features, etc.)
        brand_preferences: Preferred brands
        budget_range: Budget constraints
        use_cases: Intended use cases
        style_preferences: Style and design preferences
    """
    priority: List[str] = Field(default_factory=list, description="Priority factors")
    brand_preferences: List[str] = Field(default_factory=list, description="Preferred brands")
    budget_range: Optional[Dict[str, float]] = Field(None, description="Budget constraints")
    use_cases: List[str] = Field(default_factory=list, description="Use cases")
    style_preferences: List[str] = Field(default_factory=list, description="Style preferences")


class ConversationMessage(BaseModel):
    """
    A message in a conversation.

    Attributes:
        role: Message sender role (user, assistant, system)
        content: Message content
        timestamp: When the message was sent
    """
    role: str = Field(..., description="Message sender role")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")


class ProductRecommendation(BaseModel):
    """
    A product recommendation with reasoning.

    Attributes:
        product_id: Product identifier
        title: Product title
        price: Product price
        store: Store name
        rating: Product rating
        image_url: Product image URL
        product_url: Product page URL
        match_score: How well it matches user requirements
        why_recommended: Explanation of why this product is recommended
        key_benefits: Key benefits for the user
    """
    product_id: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    price: float = Field(..., ge=0, description="Product price")
    store: str = Field(..., description="Store name")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Product rating")
    image_url: Optional[str] = Field(None, description="Product image URL")
    product_url: Optional[str] = Field(None, description="Product page URL")
    match_score: float = Field(..., ge=0, le=1, description="Match score")
    why_recommended: str = Field(..., description="Recommendation reasoning")
    key_benefits: List[str] = Field(default_factory=list, description="Key benefits")


class SearchRequest(BaseModel):
    """
    Request model for AI-powered product search.

    Attributes:
        query: User search query
        user_preferences: User preferences and constraints
        budget_range: Budget constraints
        use_cases: Intended use cases
        max_results: Maximum number of results
        include_alternatives: Include alternative suggestions
        model_id: Knowledge Engine model ID to use for inference
    """
    query: str = Field(..., description="Search query")
    user_preferences: Optional[UserPreferences] = Field(None, description="User preferences")
    budget_range: Optional[Dict[str, float]] = Field(None, description="Budget range")
    use_cases: List[str] = Field(default_factory=list, description="Use cases")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results")
    include_alternatives: bool = Field(default=True, description="Include alternatives")
    model_id: Optional[str] = Field(default="shopping_advisor_production_v2", description="Model ID for inference")


class SearchResponse(BaseModel):
    """
    Response model for AI-powered search results.

    Attributes:
        search_results: List of recommended products
        ai_advice: AI-generated advice for the search
        follow_up_questions: Suggested follow-up questions
        search_insights: Insights about the search and market
        total_results: Total number of results found
        processing_time_ms: Search processing time
    """
    search_results: List[ProductRecommendation] = Field(..., description="Search results")
    ai_advice: str = Field(..., description="AI-generated advice")
    follow_up_questions: List[str] = Field(default_factory=list, description="Follow-up questions")
    search_insights: Dict[str, Any] = Field(default_factory=dict, description="Search insights")
    total_results: int = Field(..., ge=0, description="Total results")
    processing_time_ms: float = Field(..., ge=0, description="Processing time")


class AdviceRequest(BaseModel):
    """
    Request model for shopping consultation.

    Attributes:
        conversation_history: Previous conversation messages
        user_context: Additional user context
        specific_questions: Specific questions to address
        consultation_type: Type of consultation needed
        model_id: Knowledge Engine model ID to use for inference
    """
    conversation_history: List[ConversationMessage] = Field(..., description="Conversation history")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context")
    specific_questions: List[str] = Field(default_factory=list, description="Specific questions")
    consultation_type: str = Field(default="general", description="Consultation type")
    model_id: Optional[str] = Field(default="shopping_advisor_production_v2", description="Model ID for inference")


class AdviceResponse(BaseModel):
    """
    Response model for shopping consultation.

    Attributes:
        advice: AI-generated advice
        product_recommendations: Recommended products
        next_steps: Suggested next steps
        confidence_score: Confidence in the advice
        reasoning: Reasoning behind the advice
    """
    advice: str = Field(..., description="AI-generated advice")
    product_recommendations: List[ProductRecommendation] = Field(
        default_factory=list, description="Product recommendations"
    )
    next_steps: List[str] = Field(default_factory=list, description="Next steps")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score")
    reasoning: str = Field(..., description="Reasoning behind advice")


class ComparisonCriteria(BaseModel):
    """
    Criteria for product comparison.

    Attributes:
        factors: Factors to compare (price, features, quality, etc.)
        weights: Importance weights for each factor
        user_priorities: User-specific priorities
    """
    factors: List[str] = Field(..., description="Comparison factors")
    weights: Optional[Dict[str, float]] = Field(None, description="Factor weights")
    user_priorities: List[str] = Field(default_factory=list, description="User priorities")


class ComparisonRequest(BaseModel):
    """
    Request model for product comparison.

    Attributes:
        product_ids: List of product IDs to compare
        comparison_criteria: Comparison criteria and weights
        user_preferences: User preferences for comparison
        include_alternatives: Include alternative suggestions
        model_id: Knowledge Engine model ID to use for inference
    """
    product_ids: List[str] = Field(..., min_items=2, max_items=5, description="Product IDs")
    comparison_criteria: ComparisonCriteria = Field(..., description="Comparison criteria")
    user_preferences: Optional[UserPreferences] = Field(None, description="User preferences")
    include_alternatives: bool = Field(default=False, description="Include alternatives")
    model_id: Optional[str] = Field(default="shopping_advisor_production_v2", description="Model ID for inference")


class ProductComparison(BaseModel):
    """
    Product information for comparison.

    Attributes:
        product: Product details
        scores: Scores for different criteria
        strengths: Product strengths
        weaknesses: Product weaknesses
        overall_score: Overall comparison score
    """
    product: ProductRecommendation = Field(..., description="Product details")
    scores: Dict[str, float] = Field(..., description="Criteria scores")
    strengths: List[str] = Field(..., description="Product strengths")
    weaknesses: List[str] = Field(..., description="Product weaknesses")
    overall_score: float = Field(..., ge=0, le=1, description="Overall score")


class ComparisonResponse(BaseModel):
    """
    Response model for product comparison.

    Attributes:
        products: Products with comparison data
        comparison_matrix: Detailed comparison matrix
        ai_analysis: AI-generated comparison analysis
        recommendation: Final recommendation
        strengths_weaknesses: Summary of strengths and weaknesses
    """
    products: List[ProductComparison] = Field(..., description="Products comparison")
    comparison_matrix: Dict[str, Dict[str, Any]] = Field(..., description="Comparison matrix")
    ai_analysis: str = Field(..., description="AI analysis")
    recommendation: str = Field(..., description="Final recommendation")
    strengths_weaknesses: Dict[str, List[str]] = Field(..., description="Strengths and weaknesses")


class HealthResponse(BaseModel):
    """
    Response model for health check.

    Attributes:
        service: Service health status
        dependencies: Status of dependent services
        overall: Overall service health
        timestamp: Health check timestamp
    """
    service: str = Field(..., description="Service status")
    dependencies: Dict[str, str] = Field(..., description="Dependency statuses")
    overall: str = Field(..., description="Overall health")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")