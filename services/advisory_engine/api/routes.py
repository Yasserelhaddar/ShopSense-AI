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

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shared.logging import get_logger
from api.schemas import (
    SearchRequest,
    SearchResponse,
    AdviceRequest,
    AdviceResponse,
    ComparisonRequest,
    ComparisonResponse,
    HealthResponse,
    UserPreferences
)
from api.middleware import get_current_user
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

        User activity tracking: If user is authenticated and tracking is enabled,
        search results are automatically saved to their history for personalization.
    """
    try:
        recommendation_engine = request.app.state.recommendation_engine
        clerk_client = request.app.state.clerk_user_client

        # Generate recommendations based on user query and preferences
        recommendations = await recommendation_engine.generate_recommendations(
            query=search_request.query,
            user_preferences=search_request.user_preferences,
            budget_range=search_request.budget_range,
            use_cases=search_request.use_cases
        )

        logger.info(f"Generated {len(recommendations['products'])} recommendations for: {search_request.query}")

        # Auto-track: Save recommendations to user's history (if authenticated and tracking enabled)
        user = get_current_user(request)
        if user and clerk_client:
            tracking_enabled = await clerk_client.is_tracking_enabled(user.user_id)
            if tracking_enabled:
                await recommendation_engine.update_recent_recommendations(
                    user.user_id,
                    recommendations["products"]
                )
                logger.debug(f"Saved search recommendations to history for user {user.user_id}")

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

        User activity tracking: If user is authenticated and tracking is enabled,
        conversation messages are automatically saved to their history for personalization.
    """
    try:
        consultation_engine = request.app.state.consultation_engine
        clerk_client = request.app.state.clerk_user_client

        # Extract user's latest question for tracking
        user_question = None
        if advice_request.conversation_history:
            # Get the last user message from conversation history
            for msg in reversed(advice_request.conversation_history):
                if hasattr(msg, 'role') and msg.role == "user":
                    user_question = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    user_question = msg.get("content")
                    break

        if not user_question and advice_request.specific_questions:
            # Use first specific question if no conversation history
            user_question = advice_request.specific_questions[0]

        # Generate consultation response
        consultation_result = await consultation_engine.provide_consultation(
            conversation_history=advice_request.conversation_history,
            user_context=advice_request.user_context,
            specific_questions=advice_request.specific_questions
        )

        logger.info("Generated shopping consultation response")

        # Auto-track: Save conversation to user's history (if authenticated and tracking enabled)
        user = get_current_user(request)
        if user and clerk_client and user_question:
            tracking_enabled = await clerk_client.is_tracking_enabled(user.user_id)
            if tracking_enabled:
                # Save user's question
                await consultation_engine.save_conversation_message(
                    user.user_id,
                    "user",
                    user_question
                )
                # Save assistant's response
                await consultation_engine.save_conversation_message(
                    user.user_id,
                    "assistant",
                    consultation_result["advice"]
                )
                logger.debug(f"Saved consultation messages to history for user {user.user_id}")

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


# Schema for feedback request
class FeedbackRequest(BaseModel):
    """Feedback submission model."""
    product_id: str = Field(..., description="Product identifier")
    rating: float = Field(..., ge=0, le=5, description="User rating (0-5)")
    feedback: Optional[str] = Field(None, description="Optional feedback text")


@router.post("/feedback")
async def submit_feedback(
    request: Request,
    feedback_request: FeedbackRequest
):
    """
    Submit user feedback on recommendations.

    Requires authentication.

    Args:
        request: FastAPI request object
        feedback_request: User feedback and ratings

    Returns:
        dict: Feedback submission confirmation

    Note:
        This endpoint collects user feedback to improve
        recommendation quality and personalization.
    """
    try:
        # Get authenticated user
        user = get_current_user(request)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to submit feedback"
            )

        recommendation_engine = request.app.state.recommendation_engine

        # Process feedback with user_id
        feedback_result = await recommendation_engine.process_feedback(
            user_id=user.user_id,
            product_id=feedback_request.product_id,
            rating=feedback_request.rating,
            feedback=feedback_request.feedback
        )

        logger.info(f"Processed feedback from user {user.user_id}")

        return {
            "status": "received",
            "feedback_id": feedback_result["id"],
            "message": "Thank you for your feedback!"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/preferences")
async def get_my_preferences(request: Request):
    """
    Get authenticated user's preferences and history.

    Requires authentication.

    Args:
        request: FastAPI request object

    Returns:
        dict: User preferences and recommendation history

    Note:
        This endpoint retrieves the authenticated user's preferences
        and past interactions for personalized recommendations.
    """
    try:
        # Get authenticated user
        user = get_current_user(request)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to access preferences"
            )

        recommendation_engine = request.app.state.recommendation_engine

        preferences = await recommendation_engine.get_user_preferences(user.user_id)

        logger.info(f"Retrieved preferences for user {user.user_id}")

        return {
            "user_id": user.user_id,
            "email": user.email,
            "preferences": preferences["preferences"],
            "history": preferences["history"],
            "recommendations": preferences["recent_recommendations"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/user/preferences")
async def update_my_preferences(
    request: Request,
    preferences: UserPreferences
):
    """
    Update authenticated user's shopping preferences.

    Requires authentication.

    Args:
        request: FastAPI request object
        preferences: User preferences to save

    Returns:
        dict: Update confirmation

    Note:
        Saves user shopping preferences to Clerk metadata for
        future personalized recommendations.
    """
    try:
        # Get authenticated user
        user = get_current_user(request)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to update preferences"
            )

        recommendation_engine = request.app.state.recommendation_engine

        # Convert Pydantic model to dict
        preferences_dict = preferences.model_dump() if hasattr(preferences, 'model_dump') else preferences.dict()

        success = await recommendation_engine.save_user_preferences(
            user.user_id,
            preferences_dict
        )

        if success:
            logger.info(f"Updated preferences for user {user.user_id}")
            return {
                "status": "success",
                "message": "Preferences updated successfully",
                "user_id": user.user_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save preferences")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/user/clear-history")
async def clear_my_history(request: Request):
    """
    Clear authenticated user's activity history.

    Requires authentication.

    Deletes all historical data stored in Redis:
    - Conversation history
    - Product recommendations
    - Feedback history
    - Recently viewed products

    Note: This does NOT delete user preferences stored in Clerk.
    To update preferences (including tracking settings), use PUT /user/preferences.

    Args:
        request: FastAPI request object

    Returns:
        dict: Deletion confirmation with cleared data summary
    """
    try:
        # Get authenticated user
        user = get_current_user(request)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to clear history"
            )

        redis_client = request.app.state.redis_user_client

        if not redis_client:
            raise HTTPException(
                status_code=503,
                detail="History clearing service unavailable"
            )

        # Get summary before clearing
        summary_before = await redis_client.get_user_data_summary(user.user_id)

        # Clear all user data
        success = await redis_client.clear_all_user_data(user.user_id)

        if success:
            logger.info(f"Cleared history for user {user.user_id}")
            return {
                "status": "success",
                "message": "History cleared successfully",
                "user_id": user.user_id,
                "cleared_data": summary_before
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear history")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
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