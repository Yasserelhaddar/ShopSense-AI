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


def transform_product_to_schema(product: dict) -> dict:
    """
    Transform a product from Discovery Engine format to API schema format.

    Args:
        product: Product dict from Discovery/Recommendation Engine
                 (with fields: id, similarity_score, description, etc.)

    Returns:
        dict: Product in API schema format
              (with fields: product_id, match_score, why_recommended, etc.)
    """
    # Get product ID - warn if missing
    product_id = product.get("id") or product.get("product_id") or ""
    if not product_id:
        logger.warning(f"Product missing ID: {product.get('title', 'Unknown')}")

    return {
        "product_id": product_id,
        "title": product.get("title") or "",
        "price": product.get("price") or 0.0,
        "store": product.get("store") or "",
        "rating": product.get("rating"),
        "image_url": product.get("image_url"),
        "product_url": product.get("product_url"),
        "match_score": product.get("similarity_score") or 0.8,
        "why_recommended": product.get("description") or "Recommended for you",
        "key_benefits": product.get("key_features", [])[:3] if product.get("key_features") else []
    }


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

        # Get authenticated user and merge saved preferences
        user = get_current_user(request)
        budget_range = search_request.budget_range
        user_preferences = search_request.user_preferences

        if user and clerk_client:
            # Fetch saved preferences from Clerk
            saved_prefs = await clerk_client.get_user_preferences(user.user_id)

            # Merge budget_range: request takes precedence, fall back to saved
            if not budget_range and saved_prefs.get("budget_range"):
                budget_range = saved_prefs["budget_range"]

            # Merge user_preferences: combine saved with request (request takes precedence)
            if saved_prefs:
                merged_prefs = saved_prefs.copy()
                if user_preferences:
                    merged_prefs.update(user_preferences)
                user_preferences = merged_prefs

            logger.debug(f"Applied saved preferences for user {user.user_id}")

        # Generate recommendations based on user query and preferences
        recommendations = await recommendation_engine.generate_recommendations(
            query=search_request.query,
            user_preferences=user_preferences,
            budget_range=budget_range,
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

        # Transform product results to match schema
        search_results = [transform_product_to_schema(p) for p in recommendations["products"]]

        return SearchResponse(
            search_results=search_results,
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

        # Get authenticated user and merge saved preferences into user_context
        user = get_current_user(request)
        user_context = advice_request.user_context or {}

        if user and clerk_client:
            # Fetch saved preferences from Clerk
            saved_prefs = await clerk_client.get_user_preferences(user.user_id)

            # Merge saved preferences into user_context
            if saved_prefs:
                merged_context = saved_prefs.copy()
                merged_context.update(user_context)
                user_context = merged_context

            logger.debug(f"Applied saved preferences to consultation for user {user.user_id}")

        # Generate consultation response
        consultation_result = await consultation_engine.provide_consultation(
            conversation_history=advice_request.conversation_history,
            user_context=user_context,
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

        # Transform product recommendations to match schema
        product_recommendations = [
            transform_product_to_schema(p) for p in consultation_result["recommendations"]
        ]

        return AdviceResponse(
            advice=consultation_result["advice"],
            product_recommendations=product_recommendations,
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
        clerk_client = request.app.state.clerk_user_client

        # Extract comparison factors from criteria
        comparison_factors = comparison_request.comparison_criteria.factors

        # Convert user preferences to dict if provided
        user_prefs_dict = None
        if comparison_request.user_preferences:
            user_prefs_dict = comparison_request.user_preferences.model_dump() if hasattr(comparison_request.user_preferences, 'model_dump') else comparison_request.user_preferences.dict()

        # Get authenticated user and merge saved preferences
        user = get_current_user(request)
        if user and clerk_client:
            # Fetch saved preferences from Clerk
            saved_prefs = await clerk_client.get_user_preferences(user.user_id)

            # Merge user_preferences: combine saved with request (request takes precedence)
            if saved_prefs:
                merged_prefs = saved_prefs.copy()
                if user_prefs_dict:
                    merged_prefs.update(user_prefs_dict)
                user_prefs_dict = merged_prefs

            logger.debug(f"Applied saved preferences to comparison for user {user.user_id}")

        # Perform product comparison
        comparison_result = await recommendation_engine.compare_products(
            product_ids=comparison_request.product_ids,
            comparison_criteria=comparison_factors,
            user_preferences=user_prefs_dict
        )

        logger.info(f"Compared {len(comparison_request.product_ids)} products")

        # Transform products to ProductComparison format
        product_comparisons = []
        for product in comparison_result["products"]:
            product_id = product.get("id", "")
            strengths_weaknesses = comparison_result["strengths_weaknesses"].get(product_id, {})

            # Transform product to ProductRecommendation schema
            product_rec = transform_product_to_schema(product)

            product_comparisons.append({
                "product": product_rec,
                "scores": {factor: 0.8 for factor in comparison_factors},  # TODO: Calculate actual scores
                "strengths": strengths_weaknesses.get("strengths", ["High quality"]),
                "weaknesses": strengths_weaknesses.get("weaknesses", []),
                "overall_score": 0.8  # TODO: Calculate actual overall score
            })

        # Transform strengths_weaknesses to schema format
        strengths_weaknesses_formatted = {}
        for product_id, analysis in comparison_result["strengths_weaknesses"].items():
            strengths_weaknesses_formatted[product_id] = (
                analysis.get("strengths", []) + ["---"] + analysis.get("weaknesses", [])
            )

        return ComparisonResponse(
            products=product_comparisons,
            comparison_matrix=comparison_result["matrix"],
            ai_analysis=comparison_result["analysis"],
            recommendation=comparison_result["recommendation"],
            strengths_weaknesses=strengths_weaknesses_formatted
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

        # Transform trending products to match expected schema
        trending_products = [transform_product_to_schema(p) for p in trending["products"]]

        return {
            "trending_products": trending_products,
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

        # Transform recent recommendations to match schema
        recent_recommendations = [
            transform_product_to_schema(p) for p in preferences.get("recent_recommendations", [])
        ]

        return {
            "user_id": user.user_id,
            "email": user.email,
            "preferences": preferences["preferences"],
            "history": preferences["history"],
            "recommendations": recent_recommendations
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


# Admin endpoints
from api.middleware import require_admin
from fastapi import Depends


@router.post("/admin/data/collect")
async def admin_trigger_collection(
    request: Request,
    user: Depends = Depends(require_admin),
    sources: List[str] = None,
    categories: Optional[List[str]] = None,
    max_results: int = 100
):
    """
    Admin: Trigger product data collection from Discovery Engine.

    Requires admin role.

    Args:
        request: FastAPI request object
        user: Authenticated admin user (automatically injected)
        sources: List of sources to collect from (e.g., ["amazon", "bestbuy"])
        categories: List of categories to collect (optional)
        max_results: Maximum results per category

    Returns:
        dict: Collection job information
    """
    try:
        discovery_client = request.app.state.discovery_client

        result = await discovery_client.trigger_collection(
            sources=sources or ["amazon"],
            categories=categories,
            max_results=max_results
        )

        logger.info(f"Admin {user.user_id} triggered collection job: {result.get('job_id')}")

        return {
            "status": "success",
            "job": result,
            "admin": user.user_id
        }

    except Exception as e:
        logger.error(f"Admin collection trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/data/collection/{job_id}")
async def admin_get_collection_status(
    request: Request,
    job_id: str,
    user: Depends = Depends(require_admin)
):
    """
    Admin: Get status of a data collection job.

    Requires admin role.

    Args:
        request: FastAPI request object
        job_id: Collection job identifier
        user: Authenticated admin user (automatically injected)

    Returns:
        dict: Job status and progress
    """
    try:
        discovery_client = request.app.state.discovery_client

        status = await discovery_client.get_collection_status(job_id)

        return {
            "job_id": job_id,
            "status": status
        }

    except Exception as e:
        logger.error(f"Failed to get collection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/training/generate-data")
async def admin_generate_training_data(
    request: Request,
    user: Depends = Depends(require_admin),
    num_examples: int = 100,
    domains: Optional[List[str]] = None
):
    """
    Admin: Generate synthetic training data using Knowledge Engine.

    Requires admin role.

    Args:
        request: FastAPI request object
        user: Authenticated admin user (automatically injected)
        num_examples: Number of training examples to generate
        domains: List of domains/topics for the data

    Returns:
        dict: Data generation job information
    """
    try:
        knowledge_client = request.app.state.knowledge_client

        result = await knowledge_client.generate_training_data(
            num_examples=num_examples,
            domains=domains
        )

        logger.info(f"Admin {user.user_id} triggered training data generation: {num_examples} examples")

        return {
            "status": "success",
            "data_generation": result,
            "admin": user.user_id
        }

    except Exception as e:
        logger.error(f"Training data generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/training/start")
async def admin_start_training(
    request: Request,
    user: Depends = Depends(require_admin),
    model_name: str = None,
    base_model: Optional[str] = None,
    training_params: Optional[Dict[str, Any]] = None
):
    """
    Admin: Start model training job on Knowledge Engine.

    Requires admin role.

    Args:
        request: FastAPI request object
        user: Authenticated admin user (automatically injected)
        model_name: Name for the trained model
        base_model: Base model to fine-tune
        training_params: Training parameters

    Returns:
        dict: Training job information
    """
    try:
        knowledge_client = request.app.state.knowledge_client

        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required")

        result = await knowledge_client.start_training(
            model_name=model_name,
            base_model=base_model,
            training_params=training_params
        )

        logger.info(f"Admin {user.user_id} started training job: {result.get('job_id')}")

        return {
            "status": "success",
            "training": result,
            "admin": user.user_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/training/status/{job_id}")
async def admin_get_training_status(
    request: Request,
    job_id: str,
    user: Depends = Depends(require_admin)
):
    """
    Admin: Get status of a training job.

    Requires admin role.

    Args:
        request: FastAPI request object
        job_id: Training job identifier
        user: Authenticated admin user (automatically injected)

    Returns:
        dict: Job status and progress
    """
    try:
        knowledge_client = request.app.state.knowledge_client

        status = await knowledge_client.get_training_status(job_id)

        return {
            "job_id": job_id,
            "status": status
        }

    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/models")
async def admin_list_models(
    request: Request,
    user: Depends = Depends(require_admin)
):
    """
    Admin: List all available models from Knowledge Engine.

    Requires admin role.

    Args:
        request: FastAPI request object
        user: Authenticated admin user (automatically injected)

    Returns:
        dict: List of models with metadata
    """
    try:
        knowledge_client = request.app.state.knowledge_client

        models = await knowledge_client.list_models()

        return {
            "models": models,
            "total": len(models)
        }

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/system/health")
async def admin_system_health(
    request: Request,
    user: Depends = Depends(require_admin)
):
    """
    Admin: Get comprehensive health status of all engines.

    Requires admin role.

    Args:
        request: FastAPI request object
        user: Authenticated admin user (automatically injected)

    Returns:
        dict: Health status of all services
    """
    try:
        knowledge_client = request.app.state.knowledge_client
        discovery_client = request.app.state.discovery_client

        health_status = {
            "advisory_engine": "healthy",
            "knowledge_engine": "unknown",
            "discovery_engine": "unknown"
        }

        # Check Knowledge Engine
        try:
            knowledge_healthy = await knowledge_client.health_check()
            health_status["knowledge_engine"] = "healthy" if knowledge_healthy else "unhealthy"
        except Exception as e:
            health_status["knowledge_engine"] = f"unhealthy: {str(e)}"

        # Check Discovery Engine
        try:
            discovery_healthy = await discovery_client.health_check()
            health_status["discovery_engine"] = "healthy" if discovery_healthy else "unhealthy"
        except Exception as e:
            health_status["discovery_engine"] = f"unhealthy: {str(e)}"

        # Overall status
        all_healthy = all(
            status == "healthy" for status in health_status.values()
        )
        health_status["overall"] = "healthy" if all_healthy else "degraded"

        return health_status

    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/system/stats")
async def admin_system_stats(
    request: Request,
    user: Depends = Depends(require_admin)
):
    """
    Admin: Get system statistics and metadata.

    Requires admin role.

    Args:
        request: FastAPI request object
        user: Authenticated admin user (automatically injected)

    Returns:
        dict: System statistics
    """
    try:
        discovery_client = request.app.state.discovery_client

        # Get categories and stores from Discovery Engine
        categories = await discovery_client.get_categories()
        stores = await discovery_client.get_stores()

        return {
            "categories": categories,
            "stores": stores,
            "total_categories": len(categories),
            "total_stores": len(stores)
        }

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def get_current_user_info(request: Request):
    """
    Debug endpoint to see current user's JWT token metadata.

    This helps verify if admin role is present in the JWT token.
    """
    user = get_current_user(request)
    if not user:
        return {
            "authenticated": False,
            "message": "No user authenticated"
        }

    return {
        "authenticated": True,
        "user_id": user.user_id,
        "email": user.email,
        "metadata": user.metadata,
        "is_admin": user.is_admin
    }


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