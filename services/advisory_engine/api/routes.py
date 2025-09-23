"""
Advisory Engine API routes.

This module defines all HTTP endpoints for the Advisory Engine service,
including product search, shopping consultation, and comparison functionality.

Endpoints:
- POST /search: AI-powered product search
- POST /advice: Shopping consultation
- POST /compare: Product comparison
- GET /health: Health check
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from core.logging import get_logger
from api.schemas import (
    SearchRequest,
    SearchResponse,
    AdviceRequest,
    AdviceResponse,
    ComparisonRequest,
    ComparisonResponse,
    HealthResponse
)
from config.settings import AdvisorySettings


# Initialize router and dependencies
router = APIRouter()
logger = get_logger("advisory-service")
settings = AdvisorySettings()


@router.post("/search", response_model=SearchResponse)
async def ai_powered_search(
    request: Request,
    search_request: SearchRequest
):
    """
    Perform AI-powered product search with recommendations.

    Args:
        request: FastAPI request object for accessing app state
        search_request: Search query and user preferences

    Returns:
        SearchResponse: Search results with AI-generated advice

    Raises:
        HTTPException: If search fails

    Note:
        This endpoint combines semantic product search with AI-generated
        recommendations and advice tailored to user preferences.
    """
    try:
        recommendation_engine = request.app.state.recommendation_engine

        # Generate recommendations based on user query and preferences
        recommendations = await recommendation_engine.generate_recommendations(
            query=search_request.query,
            user_preferences=search_request.user_preferences,
            budget_range=search_request.budget_range,
            use_cases=search_request.use_cases
        )

        logger.info(f"Generated {len(recommendations['products'])} recommendations for: {search_request.query}")

        return SearchResponse(
            search_results=recommendations["products"],
            ai_advice=recommendations["advice"],
            follow_up_questions=recommendations["follow_ups"],
            search_insights=recommendations["insights"],
            total_results=len(recommendations["products"]),
            processing_time_ms=recommendations.get("processing_time_ms", 0)
        )

    except Exception as e:
        logger.error(f"AI-powered search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advice", response_model=AdviceResponse)
async def shopping_consultation(
    request: Request,
    advice_request: AdviceRequest
):
    """
    Provide shopping consultation and advice.

    Args:
        request: FastAPI request object
        advice_request: Conversation history and consultation request

    Returns:
        AdviceResponse: AI-generated advice and recommendations

    Raises:
        HTTPException: If consultation fails

    Note:
        This endpoint provides conversational shopping advice based on
        user's conversation history and specific questions or needs.
    """
    try:
        consultation_engine = request.app.state.consultation_engine

        # Generate consultation response
        consultation_result = await consultation_engine.provide_consultation(
            conversation_history=advice_request.conversation_history,
            user_context=advice_request.user_context,
            specific_questions=advice_request.specific_questions
        )

        logger.info("Generated shopping consultation response")

        return AdviceResponse(
            advice=consultation_result["advice"],
            product_recommendations=consultation_result["recommendations"],
            next_steps=consultation_result["next_steps"],
            confidence_score=consultation_result["confidence"],
            reasoning=consultation_result["reasoning"]
        )

    except Exception as e:
        logger.error(f"Shopping consultation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonResponse)
async def compare_products(
    request: Request,
    comparison_request: ComparisonRequest
):
    """
    Compare products and provide detailed analysis.

    Args:
        request: FastAPI request object
        comparison_request: Products to compare and comparison criteria

    Returns:
        ComparisonResponse: Detailed product comparison

    Raises:
        HTTPException: If comparison fails

    Note:
        This endpoint provides detailed product comparisons with
        AI-generated analysis of strengths, weaknesses, and recommendations.
    """
    try:
        recommendation_engine = request.app.state.recommendation_engine

        # Perform product comparison
        comparison_result = await recommendation_engine.compare_products(
            product_ids=comparison_request.product_ids,
            comparison_criteria=comparison_request.comparison_criteria,
            user_preferences=comparison_request.user_preferences
        )

        logger.info(f"Compared {len(comparison_request.product_ids)} products")

        return ComparisonResponse(
            products=comparison_result["products"],
            comparison_matrix=comparison_result["matrix"],
            ai_analysis=comparison_result["analysis"],
            recommendation=comparison_result["recommendation"],
            strengths_weaknesses=comparison_result["strengths_weaknesses"]
        )

    except Exception as e:
        logger.error(f"Product comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/trending")
async def get_trending_recommendations(request: Request):
    """
    Get trending product recommendations.

    Args:
        request: FastAPI request object

    Returns:
        dict: Trending products and categories

    Note:
        This endpoint provides trending product recommendations
        based on current market data and user behavior patterns.
    """
    try:
        recommendation_engine = request.app.state.recommendation_engine

        trending = await recommendation_engine.get_trending_recommendations()

        logger.info("Retrieved trending recommendations")

        return {
            "trending_products": trending["products"],
            "trending_categories": trending["categories"],
            "seasonal_recommendations": trending["seasonal"],
            "updated_at": trending["updated_at"]
        }

    except Exception as e:
        logger.error(f"Failed to get trending recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(
    request: Request,
    feedback_data: Dict[str, Any]
):
    """
    Submit user feedback on recommendations.

    Args:
        request: FastAPI request object
        feedback_data: User feedback and ratings

    Returns:
        dict: Feedback submission confirmation

    Note:
        This endpoint collects user feedback to improve
        recommendation quality and personalization.
    """
    try:
        recommendation_engine = request.app.state.recommendation_engine

        # Process feedback
        feedback_result = await recommendation_engine.process_feedback(feedback_data)

        logger.info("Processed user feedback")

        return {
            "status": "received",
            "feedback_id": feedback_result["id"],
            "message": "Thank you for your feedback!"
        }

    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/preferences")
async def get_user_preferences(
    request: Request,
    user_id: str
):
    """
    Get user preferences and history.

    Args:
        request: FastAPI request object
        user_id: User identifier

    Returns:
        dict: User preferences and recommendation history

    Note:
        This endpoint retrieves user preferences and past
        interactions for personalized recommendations.
    """
    try:
        recommendation_engine = request.app.state.recommendation_engine

        preferences = await recommendation_engine.get_user_preferences(user_id)

        logger.info(f"Retrieved preferences for user {user_id}")

        return {
            "user_id": user_id,
            "preferences": preferences["preferences"],
            "history": preferences["history"],
            "recommendations": preferences["recent_recommendations"]
        }

    except Exception as e:
        logger.error(f"Failed to get user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint for service monitoring.

    Args:
        request: FastAPI request object

    Returns:
        HealthResponse: Service health status and dependencies

    Note:
        This endpoint checks connectivity to dependent services
        and reports overall Advisory service health.
    """
    health_status = {
        "service": "healthy",
        "dependencies": {}
    }

    try:
        knowledge_client = request.app.state.knowledge_client
        discovery_client = request.app.state.discovery_client

        # Check Knowledge Engine
        try:
            await knowledge_client.health_check()
            health_status["dependencies"]["knowledge_engine"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["knowledge_engine"] = f"unhealthy: {str(e)}"

        # Check Discovery Engine
        try:
            await discovery_client.health_check()
            health_status["dependencies"]["discovery_engine"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["discovery_engine"] = f"unhealthy: {str(e)}"

        # Check OpenAI API
        try:
            # TODO: Implement OpenAI health check
            health_status["dependencies"]["openai"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["openai"] = f"unhealthy: {str(e)}"

        # Check Redis cache
        try:
            # TODO: Implement Redis health check
            health_status["dependencies"]["redis"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["redis"] = f"unhealthy: {str(e)}"

        # Determine overall health
        all_dependencies_healthy = all(
            status == "healthy"
            for status in health_status["dependencies"].values()
        )

        health_status["overall"] = "healthy" if all_dependencies_healthy else "degraded"

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_status["service"] = "unhealthy"
        health_status["overall"] = "unhealthy"

    return HealthResponse(**health_status)