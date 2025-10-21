"""
Advisory Engine FastAPI application.

This module implements the main FastAPI application for the Advisory Engine
microservice. It handles user-facing recommendations, shopping consultation,
and orchestrates communication with other services.

The service provides endpoints for:
- AI-powered product search and recommendations
- Shopping consultation and advice
- Product comparison and analysis
- User preference learning and personalization

Port: 8003
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.logging import setup_logger
from config.settings import AdvisorySettings
from api.routes import router
from clients.knowledge_client import KnowledgeClient
from clients.discovery_client import DiscoveryClient
from core.recommendations import RecommendationEngine
from core.consultation import ConsultationEngine


# Initialize settings and logger
settings = AdvisorySettings()
logger = setup_logger("advisory-service", settings.log_level)

# Global clients and engines
knowledge_client = None
discovery_client = None
recommendation_engine = None
consultation_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.

    Handles initialization of service clients, recommendation engines,
    caching systems, and cleanup on shutdown.

    Args:
        app: FastAPI application instance

    Yields:
        None during application runtime
    """
    global knowledge_client, discovery_client, recommendation_engine, consultation_engine

    # Startup
    logger.info("Starting Advisory Service...")

    # Validate configuration
    await validate_configuration()

    # Initialize service clients
    knowledge_client = KnowledgeClient(settings.knowledge_service_url)
    await knowledge_client.initialize()

    discovery_client = DiscoveryClient(settings.discovery_service_url)
    await discovery_client.initialize()

    # Initialize recommendation and consultation engines
    recommendation_engine = RecommendationEngine(
        knowledge_client=knowledge_client,
        discovery_client=discovery_client
    )
    await recommendation_engine.initialize()

    consultation_engine = ConsultationEngine(
        knowledge_client=knowledge_client,
        discovery_client=discovery_client
    )
    await consultation_engine.initialize()

    # Make engines and clients available to routes through app state
    app.state.knowledge_client = knowledge_client
    app.state.discovery_client = discovery_client
    app.state.recommendation_engine = recommendation_engine
    app.state.consultation_engine = consultation_engine

    # Test service connections
    await test_service_connections()

    logger.info("Advisory Service ready!")

    yield

    # Shutdown
    logger.info("Shutting down Advisory Service...")
    await cleanup_resources()


async def validate_configuration():
    """
    Validate that all required configuration is present.

    Raises:
        ValueError: If required configuration is missing
    """
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is required")

    if not settings.knowledge_service_url:
        raise ValueError("Knowledge service URL is required")

    if not settings.discovery_service_url:
        raise ValueError("Discovery service URL is required")

    logger.info("Configuration validation passed")


async def test_service_connections():
    """
    Test connections to dependent services.

    Validates connectivity to Knowledge Engine and Discovery Engine
    to ensure the Advisory service can operate properly.
    """
    try:
        # Test Knowledge Engine connection
        await knowledge_client.health_check()
        logger.info("Knowledge Engine connection validated")
    except Exception as e:
        logger.warning(f"Knowledge Engine connection failed: {e}")

    try:
        # Test Discovery Engine connection
        await discovery_client.health_check()
        logger.info("Discovery Engine connection validated")
    except Exception as e:
        logger.warning(f"Discovery Engine connection failed: {e}")


async def cleanup_resources():
    """
    Clean up resources on service shutdown.

    Properly closes service clients, clears caches, and releases
    any held resources.
    """
    global knowledge_client, discovery_client, recommendation_engine, consultation_engine

    if knowledge_client:
        await knowledge_client.cleanup()

    if discovery_client:
        await discovery_client.cleanup()

    if recommendation_engine:
        await recommendation_engine.cleanup()

    if consultation_engine:
        await consultation_engine.cleanup()

    logger.info("Resources cleaned up")


# Create FastAPI application
app = FastAPI(
    title="ShopSense-AI Advisory Engine",
    description="User-facing recommendations and shopping consultation service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(",") if settings.allowed_origins != "*" else ["*"],
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
        "service": "Advisory Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


# App state is set in lifespan function after initialization


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )