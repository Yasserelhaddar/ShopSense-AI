"""
Discovery Engine API schemas.

This module defines Pydantic models for request and response validation
in the Discovery Engine API. These schemas ensure type safety and provide
automatic documentation for product search and collection endpoints.

Models:
- ProductSearchRequest: Product search parameters
- ProductSearchResponse: Search results with metadata
- ProductDetail: Detailed product information
- DealInfo: Deal and discount information
- CollectionRequest: Data collection parameters
- CollectionResponse: Collection job status
- HealthResponse: Service health status
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, HttpUrl


class ProductInfo(BaseModel):
    """
    Basic product information matching core ProductData structure.

    This schema aligns with the ProductData dataclass from collectors.py
    to ensure consistency between core collection and API responses.

    Attributes:
        id: Unique product identifier (e.g., "amazon_B08X123456")
        title: Product title/name
        price: Current price
        currency: Price currency code (default: USD)
        store: Store/source name (e.g., "amazon")
        brand: Product brand (optional)
        category: Product category (optional)
        description: Product description (optional, for detailed responses)
        rating: Average rating (0-5 scale, optional)
        reviews_count: Number of reviews (optional)
        image_url: Product image URL (optional)
        product_url: Link to product page (optional)
        availability: Stock status (in_stock, out_of_stock, unknown)
        similarity_score: Search relevance score (for search results)
    """
    id: str = Field(..., description="Unique product identifier")
    title: str = Field(..., description="Product title")
    price: float = Field(..., ge=0, description="Current price")
    currency: str = Field(default="USD", description="Currency code")
    store: str = Field(..., description="Store name")
    brand: Optional[str] = Field(None, description="Product brand")
    category: Optional[str] = Field(None, description="Product category")
    description: Optional[str] = Field(None, description="Product description")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Average rating")
    reviews_count: Optional[int] = Field(None, ge=0, description="Number of reviews")
    image_url: Optional[HttpUrl] = Field(None, description="Product image URL")
    product_url: Optional[HttpUrl] = Field(None, description="Product page URL")
    availability: str = Field(default="unknown", description="Availability status")
    similarity_score: Optional[float] = Field(None, ge=0, le=1, description="Search relevance")


class ProductDetail(ProductInfo):
    """
    Detailed product information extending basic product info.

    Matches core ProductData structure from collectors.py with additional
    fields for extended product information like key_features and specifications.

    Additional Attributes (beyond ProductInfo):
        key_features: List of key product features (from core ProductData)
        specifications: Technical specifications dict (from core ProductData)
        model: Product model (additional detail field)
        weight: Product weight (additional detail field)
        dimensions: Product dimensions (additional detail field)
        warranty: Warranty information (additional detail field)
        last_updated: When product data was last updated
    """
    key_features: List[str] = Field(default_factory=list, description="Key product features")
    specifications: Dict[str, str] = Field(default_factory=dict, description="Technical specifications")
    model: Optional[str] = Field(None, description="Product model")
    weight: Optional[str] = Field(None, description="Product weight")
    dimensions: Optional[str] = Field(None, description="Product dimensions")
    warranty: Optional[str] = Field(None, description="Warranty information")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class ProductSearchRequest(BaseModel):
    """
    Request model for product search.

    Attributes:
        query: Search query string
        limit: Maximum number of results
        min_price: Minimum price filter
        max_price: Maximum price filter
        category: Category filter
        store: Store filter
        sort_by: Sort criteria
        include_out_of_stock: Include out of stock items
    """
    query: str = Field(..., description="Search query")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price")
    category: Optional[str] = Field(None, description="Category filter")
    store: Optional[str] = Field(None, description="Store filter")
    sort_by: str = Field(default="relevance", description="Sort criteria")
    include_out_of_stock: bool = Field(default=False, description="Include out of stock")


class ProductSearchResponse(BaseModel):
    """
    Response model for product search results.

    Attributes:
        products: List of matching products
        total_results: Total number of results found
        search_time_ms: Search execution time
        query: Original search query
        filters_applied: Filters that were applied
        suggestions: Search suggestions
    """
    products: List[ProductInfo] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total results count")
    search_time_ms: float = Field(..., ge=0, description="Search time")
    query: str = Field(..., description="Search query")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    suggestions: List[str] = Field(default_factory=list, description="Search suggestions")


class DealInfo(BaseModel):
    """
    Information about a deal or discount.

    Attributes:
        product_id: Product identifier
        title: Product title
        original_price: Original price
        current_price: Discounted price
        discount_amount: Discount amount
        discount_percent: Discount percentage
        deal_type: Type of deal
        expires_at: Deal expiration time
        store: Store offering the deal
        image_url: Product image URL
        product_url: Link to deal
    """
    product_id: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    original_price: float = Field(..., gt=0, description="Original price")
    current_price: float = Field(..., gt=0, description="Discounted price")
    discount_amount: float = Field(..., gt=0, description="Discount amount")
    discount_percent: float = Field(..., gt=0, le=100, description="Discount percentage")
    deal_type: str = Field(..., description="Deal type")
    expires_at: Optional[datetime] = Field(None, description="Deal expiration")
    store: str = Field(..., description="Store name")
    image_url: Optional[HttpUrl] = Field(None, description="Product image")
    product_url: Optional[HttpUrl] = Field(None, description="Deal link")


class CollectionRequest(BaseModel):
    """
    Request model for triggering product collection.

    Attributes:
        sources: List of sources to collect from
        categories: Product categories to focus on
        priority: Collection priority level
        max_results: Maximum products per query (uses config default if not specified)
        max_products: Maximum total products to collect across all queries
        force_refresh: Force refresh of existing data
    """
    sources: List[str] = Field(..., description="Collection sources")
    categories: List[str] = Field(default_factory=list, description="Target categories")
    priority: str = Field(default="normal", description="Collection priority")
    max_results: Optional[int] = Field(default=None, ge=1, le=500, description="Max products per query")
    max_products: int = Field(default=1000, ge=1, le=10000, description="Max total products")
    force_refresh: bool = Field(default=False, description="Force data refresh")


class CollectionResponse(BaseModel):
    """
    Response model for collection job creation.

    Attributes:
        job_id: Unique job identifier
        status: Current job status
        sources: Sources being collected from
        estimated_duration: Estimated completion time
        started_at: Job start time
    """
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    sources: List[str] = Field(..., description="Collection sources")
    estimated_duration: str = Field(..., description="Estimated duration")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Start time")


class CollectionStatus(BaseModel):
    """
    Collection job status information.

    Attributes:
        job_id: Job identifier
        status: Current status
        progress: Completion percentage
        products_collected: Number of products collected
        errors_count: Number of errors encountered
        started_at: Job start time
        estimated_completion: Estimated completion time
        last_update: Last status update time
    """
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current status")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    products_collected: int = Field(..., ge=0, description="Products collected")
    errors_count: int = Field(..., ge=0, description="Errors encountered")
    started_at: datetime = Field(..., description="Start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion")
    last_update: datetime = Field(default_factory=datetime.utcnow, description="Last update")


class CategoryInfo(BaseModel):
    """
    Product category information.

    Attributes:
        name: Category name
        product_count: Number of products in category
        subcategories: List of subcategories
        last_updated: When category was last updated
    """
    name: str = Field(..., description="Category name")
    product_count: int = Field(..., ge=0, description="Product count")
    subcategories: List[str] = Field(default_factory=list, description="Subcategories")
    last_updated: Optional[datetime] = Field(None, description="Last update")


class StoreInfo(BaseModel):
    """
    Store/source information.

    Attributes:
        name: Store name
        product_count: Number of products from store
        last_collection: Last collection time
        api_status: API connection status
        collection_frequency: How often data is collected
    """
    name: str = Field(..., description="Store name")
    product_count: int = Field(..., ge=0, description="Product count")
    last_collection: Optional[datetime] = Field(None, description="Last collection")
    api_status: str = Field(..., description="API status")
    collection_frequency: str = Field(..., description="Collection frequency")


class HealthResponse(BaseModel):
    """
    Response model for health check.

    Attributes:
        service: Service health status
        external_apis: Status of external API dependencies
        overall: Overall service health
        database_status: Database connection status
        timestamp: Health check timestamp
    """
    service: str = Field(..., description="Service status")
    external_apis: Dict[str, str] = Field(..., description="External API statuses")
    overall: str = Field(..., description="Overall health")
    database_status: Optional[str] = Field(None, description="Database status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")