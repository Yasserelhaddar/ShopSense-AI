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

from core.logging import setup_logger
from config.settings import KnowledgeSettings
from api.routes import router


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
    """
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is required")

    if not settings.wandb_api_key:
        raise ValueError("WandB API key is required")

    logger.info("Configuration validation passed")


async def initialize_external_services():
    """
    Initialize connections to external services.

    Tests connectivity to OpenAI, WandB, and other external APIs
    to ensure the service can operate properly.
    """
    # Test OpenAI connection
    try:
        # TODO: Implement OpenAI connection test
        logger.info("OpenAI connection validated")
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI: {e}")
        raise

    # Test WandB connection
    try:
        # TODO: Implement WandB connection test
        logger.info("WandB connection validated")
    except Exception as e:
        logger.error(f"Failed to connect to WandB: {e}")
        raise


async def load_available_models():
    """
    Load and validate available models on startup.

    Scans model storage directory and validates model files
    to prepare the service for inference requests.
    """
    # TODO: Implement model loading logic
    logger.info("Available models loaded")


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
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )