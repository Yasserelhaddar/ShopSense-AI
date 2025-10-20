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

from shared.logging import get_logger
from api.schemas import (
    TrainingRequest,
    TrainingResponse,
    ModelInfo,
    InferenceRequest,
    InferenceResponse,
    HealthResponse,
    DataGenerationRequest,
    DataGenerationResponse,
    TrainingStatusResponse,
    TrainingMetrics
)
from core.training import TrainingManager
from core.data import DatasetManager
from core.evaluation import ModelEvaluator
from config.settings import KnowledgeSettings


# Initialize router and dependencies
router = APIRouter()
logger = get_logger("knowledge-service")
settings = KnowledgeSettings()
training_manager = TrainingManager()
dataset_manager = DatasetManager()
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


@router.post("/data/generate", response_model=DataGenerationResponse)
async def generate_dataset(
    request: DataGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate synthetic training data using OpenAI.

    Args:
        request: Dataset generation configuration
        background_tasks: FastAPI background task manager

    Returns:
        DataGenerationResponse: Job information and cost estimates

    Raises:
        HTTPException: If generation cannot be started

    Note:
        This endpoint estimates OpenAI API costs before starting generation.
        The actual generation happens asynchronously in the background.
    """
    try:
        from datetime import datetime
        from uuid import uuid4

        # Generate job ID
        job_id = f"data_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        # Estimate OpenAI API costs
        # Rough estimate: ~500 tokens per conversation (input + output)
        # GPT-3.5-turbo: $0.0015/1K input tokens, $0.002/1K output tokens
        # Average: ~$0.00175/1K tokens = ~$0.875 per 1000 conversations
        estimated_tokens_per_conversation = 500
        total_tokens = request.number_of_conversations * estimated_tokens_per_conversation
        estimated_cost = (total_tokens / 1000) * 0.00175

        # Estimate duration (roughly 2-3 seconds per conversation with API calls)
        estimated_seconds = request.number_of_conversations * 2.5
        if estimated_seconds < 60:
            estimated_duration = f"{int(estimated_seconds)} seconds"
        elif estimated_seconds < 3600:
            estimated_duration = f"{int(estimated_seconds / 60)} minutes"
        else:
            estimated_duration = f"{estimated_seconds / 3600:.1f} hours"

        # Generate dataset name if not provided
        dataset_name = request.dataset_name or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Schedule generation in background
        background_tasks.add_task(
            dataset_manager.generate_synthetic_dataset,
            dataset_name=dataset_name,
            num_conversations=request.number_of_conversations,
            topics=request.topics or ["general shopping", "product recommendations"],
            difficulty=request.difficulty_level,
            include_product_data=request.include_product_data
        )

        logger.info(
            f"Dataset generation job {job_id} started: "
            f"{request.number_of_conversations} conversations, "
            f"estimated cost ${estimated_cost:.4f}"
        )

        return DataGenerationResponse(
            job_id=job_id,
            status="started",
            estimated_conversations=request.number_of_conversations,
            estimated_cost=round(estimated_cost, 4),
            estimated_duration=estimated_duration,
            dataset_name=dataset_name
        )

    except Exception as e:
        logger.error(f"Failed to start dataset generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """
    Get the current status of a training job.

    Args:
        job_id: Unique identifier for the training job

    Returns:
        TrainingStatusResponse: Job status and progress information

    Raises:
        HTTPException: If job not found or status cannot be retrieved

    Note:
        This endpoint provides real-time progress updates for training jobs,
        including elapsed time, estimated remaining time, and current metrics.
    """
    try:
        from datetime import datetime

        # Check if job exists
        if job_id not in training_manager.active_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Training job {job_id} not found"
            )

        job = training_manager.active_jobs[job_id]

        # Calculate elapsed time
        started_at = job["started_at"]
        elapsed_seconds = (datetime.utcnow() - started_at).total_seconds()

        # Format elapsed time as human-readable
        if elapsed_seconds < 60:
            elapsed_time = f"{int(elapsed_seconds)} seconds"
        elif elapsed_seconds < 3600:
            elapsed_time = f"{int(elapsed_seconds / 60)} minutes"
        else:
            hours = int(elapsed_seconds / 3600)
            minutes = int((elapsed_seconds % 3600) / 60)
            elapsed_time = f"{hours}h {minutes}m"

        # Calculate estimated remaining time
        progress = job.get("progress", 0)
        estimated_remaining = None

        if progress > 0 and progress < 100:
            # Estimate based on current progress
            estimated_total_seconds = (elapsed_seconds / progress) * 100
            remaining_seconds = estimated_total_seconds - elapsed_seconds

            if remaining_seconds < 60:
                estimated_remaining = f"{int(remaining_seconds)} seconds"
            elif remaining_seconds < 3600:
                estimated_remaining = f"{int(remaining_seconds / 60)} minutes"
            else:
                hours = int(remaining_seconds / 3600)
                minutes = int((remaining_seconds % 3600) / 60)
                estimated_remaining = f"{hours}h {minutes}m"

        # Extract training metrics if available
        metrics = None
        if "metrics" in job:
            metrics = TrainingMetrics(**job["metrics"])

        # Get error message if job failed
        error = job.get("error", None)

        logger.info(f"Training job {job_id} status: {job['status']} ({progress}%)")

        return TrainingStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=progress,
            started_at=started_at,
            elapsed_time=elapsed_time,
            estimated_remaining=estimated_remaining,
            metrics=metrics,
            error=error
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status for job {job_id}: {e}")
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

        # Convert Pydantic Message objects to dicts
        messages_dict = [msg.dict() for msg in request.messages]

        # Run inference
        response = await training_manager.run_inference(
            model_id=model_id,
            messages=messages_dict,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        logger.info(f"Inference completed for model {model_id}")

        return InferenceResponse(
            response=response["content"],
            model_used=response.get("model_used", model_id),
            tokens_used=response["tokens_used"],
            processing_time_ms=response["processing_time_ms"]
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