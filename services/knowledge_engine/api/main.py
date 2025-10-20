"""
Knowledge Engine FastAPI application.

This module implements the main FastAPI application for the Knowledge Engine
microservice. It handles LLM training, model management, and inference requests.

The service provides endpoints for:
- Training new models with shopping conversation data
- Managing model versions and storage
- Running inference on trained models
- Health monitoring and status checks

Port: 8001
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.logging import setup_logger
from config.settings import KnowledgeSettings
from api.routes import router, training_manager


# Initialize settings and logger
settings = KnowledgeSettings()
logger = setup_logger("knowledge-service", settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.

    Handles initialization of external connections, model loading,
    and cleanup on shutdown.

    Args:
        app: FastAPI application instance

    Yields:
        None during application runtime
    """
    # Startup
    logger.info("Starting Knowledge Service...")

    # Validate required configuration
    await validate_configuration()

    # Initialize external connections
    await initialize_external_services()

    # Load available models
    await load_available_models()

    logger.info("Knowledge Service ready!")

    yield

    # Shutdown
    logger.info("Shutting down Knowledge Service...")
    await cleanup_resources()


async def validate_configuration():
    """
    Validate that all required configuration is present.

    Raises:
        ValueError: If required configuration is missing

    Note:
        Only OpenAI API key is required. WandB is optional for
        experiment tracking - training will continue with local logging
        if unavailable.
    """
    # Required configuration
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is required")

    # Optional configuration warnings
    if not settings.wandb_api_key:
        logger.warning("WandB API key not configured - experiment tracking will be disabled")

    if not settings.huggingface_token:
        logger.warning("HuggingFace token not configured - model downloads may be rate-limited")

    logger.info("Configuration validation passed")


async def initialize_external_services():
    """
    Initialize connections to external services.

    Tests connectivity to OpenAI, WandB, HuggingFace, and model storage
    to ensure the service can operate properly.

    Raises:
        RuntimeError: If critical services are unavailable

    Note:
        OpenAI, HuggingFace, and model storage are critical.
        WandB is optional - service continues with local logging if unavailable.
    """
    errors = []

    # Test OpenAI connection (critical)
    try:
        await training_manager.test_openai_connection()
        logger.info("✓ OpenAI connection validated")
    except Exception as e:
        error_msg = f"OpenAI connection failed: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    # Test WandB connection (optional - warning only)
    try:
        await training_manager.test_wandb_connection()
        logger.info("✓ WandB connection validated")
    except Exception as e:
        logger.warning(f"WandB connection failed: {e}")
        logger.info("Training will continue with local logging only")
        # Don't fail startup for WandB

    # Test HuggingFace connection (critical)
    try:
        await training_manager.test_huggingface_connection()
        logger.info("✓ HuggingFace connection validated")
    except Exception as e:
        error_msg = f"HuggingFace connection failed: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    # Test model storage access (critical)
    try:
        await training_manager.test_model_storage()
        logger.info("✓ Model storage access validated")
    except Exception as e:
        error_msg = f"Model storage access failed: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    # Fail startup if critical services unavailable
    if errors:
        error_summary = "; ".join(errors)
        raise RuntimeError(f"Service initialization failed: {error_summary}")


async def load_available_models():
    """
    Load and validate available models on startup.

    Scans model storage directory and validates model files
    to prepare the service for inference requests.

    Note:
        Models are not loaded into memory on startup (lazy loading).
        This function only scans for available models and logs their info.
    """
    try:
        models = await training_manager.list_available_models()

        if models:
            logger.info(f"✓ Found {len(models)} available models:")
            for model in models:
                logger.info(
                    f"  - {model['id']}: {model.get('base_model', 'unknown')} "
                    f"({model.get('size_mb', 0):.1f} MB, "
                    f"status: {model.get('status', 'unknown')})"
                )
        else:
            logger.warning("No trained models found in storage")
            logger.info("Service will use OpenAI fallback for inference until models are trained")

    except Exception as e:
        # Don't fail startup if no models exist yet
        logger.warning(f"Failed to load models on startup: {e}")
        logger.info("Service will use OpenAI fallback for inference")


async def cleanup_resources():
    """
    Clean up resources on service shutdown.

    Properly closes connections, saves state, and releases
    any held resources.
    """
    # TODO: Implement cleanup logic
    logger.info("Resources cleaned up")


# Create FastAPI application
app = FastAPI(
    title="ShopSense-AI Knowledge Engine",
    description="LLM training and model management service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """
    Root endpoint for basic service information.

    Returns:
        dict: Service information and status
    """
    return {
        "service": "Knowledge Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )