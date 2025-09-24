"""
Discovery Engine API routes.

This module defines all HTTP endpoints for the Discovery Engine service,
including product search, data collection, and deal monitoring.

Endpoints:
- GET /products/search: Search products with semantic similarity
- GET /products/{id}: Get specific product details
- GET /deals: Get current deals and price drops
- POST /products/collect: Trigger data collection
- GET /health: Health check
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from shared.logging import get_logger
from api.schemas import (
    ProductSearchRequest,
    ProductSearchResponse,
    ProductDetail,
    DealInfo,
    CollectionRequest,
    CollectionResponse,
    HealthResponse
)
from config.settings import DiscoverySettings


# Initialize router and dependencies
router = APIRouter()
logger = get_logger("discovery-service")
settings = DiscoverySettings()


@router.get("/products/search", response_model=ProductSearchResponse)
async def search_products(
    request: Request,
    query: str = Query(..., description="Search query"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of results"),
    min_price: Optional[float] = Query(default=None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(default=None, ge=0, description="Maximum price filter"),
    category: Optional[str] = Query(default=None, description="Product category filter"),
    store: Optional[str] = Query(default=None, description="Store filter"),
    sort_by: str = Query(default="relevance", description="Sort criteria")
):
    """
    Search for products using semantic similarity.

    Args:
        request: FastAPI request object for accessing app state
        query: Search query string
        limit: Maximum number of results to return
        min_price: Minimum price filter
        max_price: Maximum price filter
        category: Product category filter
        store: Store name filter
        sort_by: Sort criteria (relevance, price_asc, price_desc, rating)

    Returns:
        ProductSearchResponse: Search results with products and metadata

    Raises:
        HTTPException: If search fails
    """
    try:
        # Get storage manager from app state
        storage_manager = request.app.state.storage_manager
        embedding_processor = request.app.state.embedding_processor

        # Generate query embedding
        query_embedding = await embedding_processor.generate_embedding(query)

        # Build search filters
        filters = {}
        if min_price is not None:
            filters["min_price"] = min_price
        if max_price is not None:
            filters["max_price"] = max_price
        if category:
            filters["category"] = category
        if store:
            filters["store"] = store

        # Perform vector search
        search_results = await storage_manager.search_products(
            query_embedding=query_embedding,
            limit=limit,
            filters=filters,
            sort_by=sort_by
        )

        logger.info(f"Product search completed: {len(search_results)} results for '{query}'")

        return ProductSearchResponse(
            products=search_results,
            total_results=len(search_results),
            search_time_ms=45,  # TODO: Implement actual timing
            query=query,
            filters_applied=filters
        )

    except Exception as e:
        logger.error(f"Product search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/{product_id}", response_model=ProductDetail)
async def get_product_details(
    request: Request,
    product_id: str
):
    """
    Get detailed information about a specific product.

    Args:
        request: FastAPI request object
        product_id: Unique product identifier

    Returns:
        ProductDetail: Detailed product information

    Raises:
        HTTPException: If product not found or retrieval fails
    """
    try:
        storage_manager = request.app.state.storage_manager

        # Retrieve product details
        product = await storage_manager.get_product_by_id(product_id)

        if not product:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

        logger.info(f"Retrieved product details for {product_id}")

        return ProductDetail(**product)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deals", response_model=List[DealInfo])
async def get_current_deals(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200, description="Number of deals"),
    min_discount: float = Query(default=10.0, ge=0, le=100, description="Minimum discount percentage"),
    category: Optional[str] = Query(default=None, description="Category filter")
):
    """
    Get current deals and price drops.

    Args:
        request: FastAPI request object
        limit: Maximum number of deals to return
        min_discount: Minimum discount percentage
        category: Product category filter

    Returns:
        List[DealInfo]: Current deals and discounts

    Raises:
        HTTPException: If deal retrieval fails
    """
    try:
        storage_manager = request.app.state.storage_manager

        # Build filters
        filters = {"min_discount": min_discount}
        if category:
            filters["category"] = category

        # Get current deals
        deals = await storage_manager.get_current_deals(
            limit=limit,
            filters=filters
        )

        logger.info(f"Retrieved {len(deals)} current deals")

        return deals

    except Exception as e:
        logger.error(f"Failed to retrieve deals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/products/collect", response_model=CollectionResponse)
async def trigger_product_collection(
    request: Request,
    collection_request: CollectionRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger product data collection from specified sources.

    Args:
        request: FastAPI request object
        collection_request: Collection parameters and sources
        background_tasks: Background task manager

    Returns:
        CollectionResponse: Collection job information

    Raises:
        HTTPException: If collection cannot be started
    """
    try:
        collector_manager = request.app.state.collector_manager

        # Validate collection request
        if not collection_request.sources:
            raise HTTPException(status_code=400, detail="At least one source must be specified")

        # Start collection in background
        job_id = await collector_manager.start_collection_job(
            sources=collection_request.sources,
            categories=collection_request.categories,
            priority=collection_request.priority,
            background_tasks=background_tasks
        )

        logger.info(f"Collection job {job_id} started for sources: {collection_request.sources}")

        return CollectionResponse(
            job_id=job_id,
            status="started",
            sources=collection_request.sources,
            estimated_duration="15-30 minutes"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start collection job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/categories")
async def get_product_categories(request: Request):
    """
    Get available product categories.

    Args:
        request: FastAPI request object

    Returns:
        dict: Available categories with counts
    """
    try:
        storage_manager = request.app.state.storage_manager

        categories = await storage_manager.get_available_categories()

        logger.info(f"Retrieved {len(categories)} product categories")

        return {
            "categories": categories,
            "total_categories": len(categories)
        }

    except Exception as e:
        logger.error(f"Failed to retrieve categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products/stores")
async def get_available_stores(request: Request):
    """
    Get available stores/sources.

    Args:
        request: FastAPI request object

    Returns:
        dict: Available stores with statistics
    """
    try:
        storage_manager = request.app.state.storage_manager

        stores = await storage_manager.get_available_stores()

        logger.info(f"Retrieved {len(stores)} available stores")

        return {
            "stores": stores,
            "total_stores": len(stores)
        }

    except Exception as e:
        logger.error(f"Failed to retrieve stores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/status/{job_id}")
async def get_collection_status(
    request: Request,
    job_id: str
):
    """
    Get status of a collection job.

    Args:
        request: FastAPI request object
        job_id: Collection job identifier

    Returns:
        dict: Job status and progress information
    """
    try:
        collector_manager = request.app.state.collector_manager

        status = await collector_manager.get_job_status(job_id)

        if not status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
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
        This endpoint checks connectivity to Qdrant, external APIs,
        and reports overall service health.
    """
    health_status = {
        "service": "healthy",
        "external_apis": {}
    }

    try:
        storage_manager = request.app.state.storage_manager
        collector_manager = request.app.state.collector_manager

        # Check Qdrant connection
        try:
            await storage_manager.test_connection()
            health_status["external_apis"]["qdrant"] = "healthy"
        except Exception as e:
            health_status["external_apis"]["qdrant"] = f"unhealthy: {str(e)}"

        # Check Apify API
        try:
            await collector_manager.test_apify_connection()
            health_status["external_apis"]["apify"] = "healthy"
        except Exception as e:
            health_status["external_apis"]["apify"] = f"unhealthy: {str(e)}"

        # Check Best Buy API
        try:
            await collector_manager.test_bestbuy_connection()
            health_status["external_apis"]["bestbuy"] = "healthy"
        except Exception as e:
            health_status["external_apis"]["bestbuy"] = f"unhealthy: {str(e)}"

        # Determine overall health
        all_external_healthy = all(
            status == "healthy"
            for status in health_status["external_apis"].values()
        )

        health_status["overall"] = "healthy" if all_external_healthy else "degraded"

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_status["service"] = "unhealthy"
        health_status["overall"] = "unhealthy"

    return HealthResponse(**health_status)