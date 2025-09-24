"""
Discovery Engine FastAPI application.

This module implements the main FastAPI application for the Discovery Engine
microservice. It handles product data collection, processing, and search functionality.

The service provides endpoints for:
- Product search with semantic similarity
- Deal discovery and price monitoring
- Data collection from multiple e-commerce platforms
- Vector database management

Port: 8002
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.logging import setup_logger
from config.settings import DiscoverySettings
from api.routes import router
from core.collectors import CollectorManager
from core.processors import EmbeddingProcessor
from core.storage import StorageManager


# Initialize settings and logger
settings = DiscoverySettings()
logger = setup_logger("discovery-service", settings.log_level)

# Global managers
collector_manager = None
embedding_processor = None
storage_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.

    Handles initialization of vector database connections, embedding models,
    data collection services, and cleanup on shutdown.

    Args:
        app: FastAPI application instance

    Yields:
        None during application runtime
    """
    global collector_manager, embedding_processor, storage_manager

    # Startup
    logger.info("Starting Discovery Service...")

    # Validate configuration
    await validate_configuration()

    # Initialize core components
    storage_manager = StorageManager()
    await storage_manager.initialize()

    embedding_processor = EmbeddingProcessor()
    await embedding_processor.initialize()

    collector_manager = CollectorManager(storage_manager, embedding_processor)
    await collector_manager.initialize()

    # Start background tasks
    await start_background_tasks()

    # Make managers available to routes through app state
    app.state.collector_manager = collector_manager
    app.state.embedding_processor = embedding_processor
    app.state.storage_manager = storage_manager

    logger.info("Discovery Service ready!")

    yield

    # Shutdown
    logger.info("Shutting down Discovery Service...")
    await cleanup_resources()


async def validate_configuration():
    """
    Validate that all required configuration is present.

    Raises:
        ValueError: If required configuration is missing
    """
    if not settings.apify_api_key:
        raise ValueError("Apify API key is required")

    if not settings.qdrant_url:
        raise ValueError("Qdrant URL is required")

    logger.info("Configuration validation passed")


async def start_background_tasks():
    """
    Start background data collection and monitoring tasks.

    Initiates scheduled product collection (if hybrid mode), price monitoring,
    and database maintenance tasks based on collection strategy.
    """
    # Start product collection task only if in hybrid mode
    if settings.collection_strategy == "hybrid":
        asyncio.create_task(scheduled_product_collection())
        logger.info(f"Scheduled collection enabled (strategy: {settings.collection_strategy})")
    else:
        logger.info(f"Scheduled collection disabled (strategy: {settings.collection_strategy})")

    # Start price monitoring task
    asyncio.create_task(scheduled_price_monitoring())

    # Start database maintenance task
    asyncio.create_task(scheduled_database_maintenance())

    logger.info("Background tasks started")


async def scheduled_product_collection():
    """
    Scheduled task for continuous product data collection.

    Runs periodic collection of trending products and new arrivals
    from configured e-commerce platforms.
    """
    while True:
        try:
            logger.info("Starting scheduled product collection")
            await collector_manager.collect_trending_products()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in scheduled collection: {e}")
            await asyncio.sleep(300)  # Retry after 5 minutes on error


async def scheduled_price_monitoring():
    """
    Scheduled task for price monitoring and deal detection.

    Monitors price changes and identifies deals across
    tracked products.
    """
    while True:
        try:
            logger.info("Starting scheduled price monitoring")
            await collector_manager.monitor_prices()
            await asyncio.sleep(1800)  # Run every 30 minutes
        except Exception as e:
            logger.error(f"Error in price monitoring: {e}")
            await asyncio.sleep(300)


async def scheduled_database_maintenance():
    """
    Scheduled task for database maintenance.

    Performs cleanup of old data, index optimization,
    and vector database maintenance.
    """
    while True:
        try:
            logger.info("Starting database maintenance")
            await storage_manager.perform_maintenance()
            await asyncio.sleep(86400)  # Run daily
        except Exception as e:
            logger.error(f"Error in database maintenance: {e}")
            await asyncio.sleep(3600)  # Retry after 1 hour


async def cleanup_resources():
    """
    Clean up resources on service shutdown.

    Properly closes database connections, stops background tasks,
    and releases any held resources.
    """
    global collector_manager, embedding_processor, storage_manager

    if collector_manager:
        await collector_manager.cleanup()

    if embedding_processor:
        await embedding_processor.cleanup()

    if storage_manager:
        await storage_manager.cleanup()

    logger.info("Resources cleaned up")


# Create FastAPI application
app = FastAPI(
    title="ShopSense-AI Discovery Engine",
    description="Product data collection and search service",
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
        "service": "Discovery Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


# App state is set in lifespan function after initialization


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )