"""
Discovery Engine configuration settings.

This module defines all configuration parameters for the Discovery Engine
microservice, including external API credentials, database connections,
and service-specific parameters.

Configuration is loaded from environment variables with the DISCOVERY_ prefix.
"""

import os
from typing import Optional
from pydantic import Field, field_validator, HttpUrl

from shared.config import BaseSettings


class DiscoverySettings(BaseSettings):
    """
    Configuration settings for the Discovery Engine service.

    This class manages all configuration for product data collection,
    vector database operations, and external API integrations.

    Attributes:
        Service Configuration:
            port: Service port number
            max_concurrent_collections: Maximum concurrent collection jobs

        Vector Database:
            qdrant_url: Qdrant database URL
            qdrant_collection: Collection name for products
            qdrant_vector_size: Vector dimension size

        Embedding Model:
            embedding_model_name: Sentence transformer model name
            embedding_vector_size: Embedding vector dimensions

        External APIs:
            apify_api_key: Apify API key for Amazon/multi-store data
            apify_rate_limit: Requests per second limit
            bestbuy_api_key: Best Buy official API key
            bestbuy_base_url: Best Buy API base URL
            rapidapi_key: RapidAPI key for additional stores

        Cache:
            redis_url: Redis cache connection URL
            use_redis: Enable Redis caching vs in-memory

        Collection:
            collection_batch_size: Products per collection batch
            collection_interval_hours: Hours between scheduled collections
            max_collection_retries: Maximum retry attempts for failed collections
    """

    # Service Configuration
    port: int = Field(default=8002, description="Service port number")
    allowed_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins (use '*' for all)"
    )
    max_concurrent_collections: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent collection jobs"
    )

    # Vector Database Configuration
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        description="Qdrant database URL"
    )
    qdrant_collection: str = Field(
        default="products",
        description="Qdrant collection name"
    )
    qdrant_vector_size: int = Field(
        default=384,
        ge=128,
        le=1536,
        description="Vector dimension size"
    )
    qdrant_distance: str = Field(
        default="Cosine",
        description="Distance metric (Cosine, Euclidean, Dot)"
    )

    # Embedding Model Configuration
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    embedding_vector_size: int = Field(
        default=384,
        ge=128,
        le=1536,
        description="Embedding vector dimensions"
    )
    embedding_device: str = Field(
        default="auto",
        description="Device for model inference (cuda, cpu, auto)"
    )

    # External API Configuration - Apify
    apify_api_key: str = Field(..., description="Apify API key")
    apify_base_url: str = Field(
        default="https://api.apify.com/v2",
        description="Apify API base URL"
    )
    apify_actor_id: str = Field(
        default="BG3WDrGdteHgZgbPK",
        description="Apify Actor ID for Amazon Product Scraper"
    )
    apify_rate_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Apify requests per second"
    )

    # External API Configuration - Best Buy
    bestbuy_api_key: Optional[str] = Field(
        default=None,
        description="Best Buy API key"
    )
    bestbuy_base_url: str = Field(
        default="https://api.bestbuy.com/v1",
        description="Best Buy API base URL"
    )

    # External API Configuration - RapidAPI
    rapidapi_key: Optional[str] = Field(
        default=None,
        description="RapidAPI key"
    )
    rapidapi_walmart_url: str = Field(
        default="https://walmart-api.rapidapi.com",
        description="RapidAPI Walmart URL"
    )
    rapidapi_ebay_url: str = Field(
        default="https://ebay-search-result.rapidapi.com",
        description="RapidAPI eBay URL"
    )

    # Cache Configuration (Redis)
    use_redis: bool = Field(
        default=False,
        description="Enable Redis caching (false = in-memory, true = Redis)"
    )
    redis_url: str = Field(
        default="redis://redis:6379",
        description="Redis connection URL"
    )
    redis_host: str = Field(
        default="localhost",
        description="Redis host"
    )
    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port"
    )
    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )

    # Collection Configuration
    collection_strategy: str = Field(
        default="triggered",
        description="Collection strategy: 'triggered' (manual only) or 'hybrid' (manual + scheduled)"
    )
    max_products: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum products to collect globally"
    )
    collection_batch_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Products per collection batch"
    )
    collection_timeout_seconds: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Collection timeout in seconds"
    )
    collection_interval_hours: int = Field(
        default=1,
        ge=1,
        le=24,
        description="Hours between scheduled collections"
    )
    max_collection_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum collection retry attempts"
    )
    default_collection_limit: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Default products per query for collection"
    )
    max_collection_limit: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum products per query for collection"
    )
    trending_queries: str = Field(
        default="laptop deals,gaming headphones,smartphone,wireless earbuds,fitness tracker,coffee maker,bluetooth speaker,tablet",
        description="Comma-separated trending product queries"
    )

    # Search Configuration
    default_search_limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Default search result limit"
    )
    max_search_limit: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum search result limit"
    )
    similarity_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (0.0 to 1.0, results below this are filtered)"
    )
    min_similarity_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for search results"
    )

    # Cache Configuration
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds"
    )
    cache_ttl_seconds: int = Field(
        default=900,  # 15 minutes
        ge=60,
        le=3600,
        description="Cache TTL in seconds"
    )
    search_cache_enabled: bool = Field(
        default=True,
        description="Enable search result caching"
    )

    # Performance Configuration
    max_concurrent_searches: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent search requests"
    )
    search_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Search timeout in seconds"
    )
    http_timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=600,
        description="HTTP client timeout in seconds"
    )
    http_pool_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="HTTP client connection pool size"
    )

    # Data Quality Configuration
    min_product_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum product quality score"
    )
    require_image_url: bool = Field(
        default=False,
        description="Require product image URL"
    )
    require_reviews: bool = Field(
        default=False,
        description="Require product reviews"
    )

    class Config:
        """Pydantic configuration."""
        env_prefix = "DISCOVERY_"
        case_sensitive = False
        # Look for .env file in the config directory first
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        env_file_encoding = 'utf-8'

    @field_validator('qdrant_url')
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """
        Validate Qdrant URL format.

        Args:
            v: Qdrant URL

        Returns:
            Validated URL

        Raises:
            ValueError: If URL format is invalid
        """
        if not v.startswith(('http://', 'https://')):
            raise ValueError('qdrant_url must start with http:// or https://')
        return v

    @field_validator('embedding_vector_size', 'qdrant_vector_size')
    @classmethod
    def validate_vector_sizes_match(cls, v: int) -> int:
        """
        Validate that embedding and Qdrant vector sizes match.

        Args:
            v: Vector size value
            values: Previously validated values

        Returns:
            Validated vector size

        Raises:
            ValueError: If vector sizes don't match
        """
        # Note: Cross-field validation would need to be done with model_validator in Pydantic v2
        # For now, we'll skip this validation
        return v

    @field_validator('embedding_model_name')
    @classmethod
    def validate_embedding_model(cls, v: str) -> str:
        """
        Validate embedding model name.

        Args:
            v: Model name

        Returns:
            Validated model name

        Raises:
            ValueError: If model name is invalid
        """
        supported_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        ]

        if v not in supported_models:
            raise ValueError(f'embedding_model_name must be one of {supported_models}')
        return v

    @field_validator('collection_strategy')
    @classmethod
    def validate_collection_strategy(cls, v: str) -> str:
        """
        Validate collection strategy value.

        Args:
            v: Collection strategy

        Returns:
            Validated collection strategy

        Raises:
            ValueError: If strategy is invalid
        """
        valid_strategies = ["triggered", "hybrid"]
        if v not in valid_strategies:
            raise ValueError(f'collection_strategy must be one of {valid_strategies}')
        return v

    @field_validator('collection_batch_size')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """
        Validate collection batch size is reasonable.

        Args:
            v: Batch size

        Returns:
            Validated batch size

        Raises:
            ValueError: If batch size is unreasonable
        """
        if v > 1000:
            raise ValueError('collection_batch_size should not exceed 1000 for memory efficiency')
        return v

    def get_qdrant_config(self) -> dict:
        """
        Get Qdrant configuration dictionary.

        Returns:
            Dictionary with Qdrant connection parameters
        """
        return {
            "url": self.qdrant_url,
            "collection": self.qdrant_collection,
            "vector_size": self.qdrant_vector_size,
            "distance": "Cosine"  # For similarity search
        }

    def get_embedding_config(self) -> dict:
        """
        Get embedding configuration dictionary.

        Returns:
            Dictionary with embedding model parameters
        """
        return {
            "model_name": self.embedding_model_name,
            "vector_size": self.embedding_vector_size,
            "normalize_embeddings": True
        }

    def get_collection_config(self) -> dict:
        """
        Get collection configuration dictionary.

        Returns:
            Dictionary with collection parameters
        """
        return {
            "batch_size": self.collection_batch_size,
            "interval_hours": self.collection_interval_hours,
            "max_retries": self.max_collection_retries,
            "max_concurrent": self.max_concurrent_collections,
            "default_limit": self.default_collection_limit,
            "max_limit": self.max_collection_limit,
            "trending_queries": self.get_trending_queries()
        }

    def get_trending_queries(self) -> list[str]:
        """
        Parse trending queries from configuration string.

        Returns:
            List of trending query strings
        """
        queries = [q.strip() for q in self.trending_queries.split(",")]
        return [q for q in queries if q]  # Remove empty strings

    def get_api_configs(self) -> dict:
        """
        Get all external API configurations.

        Returns:
            Dictionary with API configurations for each service
        """
        return {
            "apify": {
                "api_key": self.apify_api_key,
                "rate_limit": self.apify_rate_limit
            },
            "bestbuy": {
                "api_key": self.bestbuy_api_key,
                "base_url": self.bestbuy_base_url,
                "enabled": bool(self.bestbuy_api_key)
            },
            "rapidapi": {
                "api_key": self.rapidapi_key,
                "walmart_url": self.rapidapi_walmart_url,
                "ebay_url": self.rapidapi_ebay_url,
                "enabled": bool(self.rapidapi_key)
            }
        }

    def is_production_ready(self) -> bool:
        """
        Check if configuration is ready for production use.

        Returns:
            True if all production requirements are met
        """
        production_requirements = [
            self.apify_api_key,
            self.qdrant_url
        ]

        return all(req for req in production_requirements)