"""
Knowledge Engine API routes.

This module defines all HTTP endpoints for the Knowledge Engine service,
including training management, model operations, and inference requests.

Endpoints:
- POST /train: Start training job
- GET /models: List available models
- POST /models/{id}/inference: Model inference
- GET /health: Health check
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from core.logging import get_logger
from api.schemas import (
    TrainingRequest,
    TrainingResponse,
    ModelInfo,
    InferenceRequest,
    InferenceResponse,
    HealthResponse
)
from core.training import TrainingManager
from core.evaluation import ModelEvaluator
from config.settings import KnowledgeSettings


# Initialize router and dependencies
router = APIRouter()
logger = get_logger("knowledge-service")
settings = KnowledgeSettings()
training_manager = TrainingManager()
model_evaluator = ModelEvaluator()


@router.post("/train", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new model training job.

    Args:
        request: Training configuration and parameters
        background_tasks: FastAPI background task manager

    Returns:
        TrainingResponse: Job information and tracking details

    Raises:
        HTTPException: If training cannot be started
    """
    try:
        # Validate training request
        training_config = await training_manager.validate_training_config(request)

        # Start training in background
        job_id = await training_manager.start_training(
            config=training_config,
            background_tasks=background_tasks
        )

        logger.info(f"Training job {job_id} started successfully")

        return TrainingResponse(
            job_id=job_id,
            status="started",
            estimated_duration="4-6 hours",
            wandb_url=f"https://wandb.ai/{settings.wandb_entity}/{settings.wandb_project}/runs/{job_id}"
        )

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List all available models.

    Returns:
        List[ModelInfo]: Available models with metadata

    Raises:
        HTTPException: If models cannot be retrieved
    """
    try:
        models = await training_manager.list_available_models()
        logger.info(f"Retrieved {len(models)} available models")
        return models

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/inference", response_model=InferenceResponse)
async def model_inference(
    model_id: str,
    request: InferenceRequest
):
    """
    Run inference on a specific model.

    Args:
        model_id: ID of the model to use for inference
        request: Inference request with messages and parameters

    Returns:
        InferenceResponse: Model response and metadata

    Raises:
        HTTPException: If inference fails or model not found
    """
    try:
        # Validate model exists
        model = await training_manager.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Run inference
        response = await training_manager.run_inference(
            model_id=model_id,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        logger.info(f"Inference completed for model {model_id}")

        return InferenceResponse(
            response=response.content,
            model_used=model_id,
            tokens_used=response.tokens_used,
            processing_time_ms=response.processing_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/evaluate")
async def evaluate_model(model_id: str):
    """
    Evaluate a model's performance on test data.

    Args:
        model_id: ID of the model to evaluate

    Returns:
        dict: Evaluation metrics and results

    Raises:
        HTTPException: If evaluation fails
    """
    try:
        # Validate model exists
        model = await training_manager.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        # Run evaluation
        evaluation_results = await model_evaluator.evaluate_model(model_id)

        logger.info(f"Evaluation completed for model {model_id}")

        return {
            "model_id": model_id,
            "evaluation_results": evaluation_results,
            "evaluated_at": evaluation_results.get("timestamp")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation failed for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for service monitoring.

    Returns:
        HealthResponse: Service health status and dependencies

    Note:
        This endpoint checks connectivity to external services
        and reports overall service health.
    """
    health_status = {
        "service": "healthy",
        "external_apis": {}
    }

    # Check OpenAI API
    try:
        await training_manager.test_openai_connection()
        health_status["external_apis"]["openai"] = "healthy"
    except Exception as e:
        health_status["external_apis"]["openai"] = f"unhealthy: {str(e)}"

    # Check WandB API
    try:
        await training_manager.test_wandb_connection()
        health_status["external_apis"]["wandb"] = "healthy"
    except Exception as e:
        health_status["external_apis"]["wandb"] = f"unhealthy: {str(e)}"

    # Check model storage
    try:
        await training_manager.test_model_storage()
        health_status["external_apis"]["model_storage"] = "healthy"
    except Exception as e:
        health_status["external_apis"]["model_storage"] = f"unhealthy: {str(e)}"

    # Determine overall health
    all_external_healthy = all(
        status == "healthy"
        for status in health_status["external_apis"].values()
    )

    health_status["overall"] = "healthy" if all_external_healthy else "degraded"

    return HealthResponse(**health_status)