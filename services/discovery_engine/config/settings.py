"""
Discovery Engine configuration settings.

This module defines all configuration parameters for the Discovery Engine
microservice, including external API credentials, database connections,
and service-specific parameters.

Configuration is loaded from environment variables with the DISCOVERY_ prefix.
"""

from typing import Optional
from pydantic import Field, validator, HttpUrl

from core.config import BaseSettings


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
            appify_api_key: Appify API key for Amazon/multi-store data
            appify_base_url: Appify API base URL
            appify_rate_limit: Requests per second limit
            bestbuy_api_key: Best Buy official API key
            bestbuy_base_url: Best Buy API base URL
            rapidapi_key: RapidAPI key for additional stores

        Database:
            postgres_url: PostgreSQL connection URL
            redis_url: Redis cache connection URL

        Collection:
            collection_batch_size: Products per collection batch
            collection_interval_hours: Hours between scheduled collections
            max_collection_retries: Maximum retry attempts for failed collections
    """

    # Service Configuration
    port: int = Field(default=8002, description="Service port number")
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

    # External API Configuration - Appify
    appify_api_key: str = Field(..., description="Appify API key")
    appify_base_url: str = Field(
        default="https://api.appify.com/v1",
        description="Appify API base URL"
    )
    appify_rate_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Appify requests per second"
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

    # Database Configuration
    postgres_url: str = Field(
        default="postgresql://admin:password123@postgres:5432/shopsense",
        description="PostgreSQL connection URL"
    )
    redis_url: str = Field(
        default="redis://redis:6379",
        description="Redis connection URL"
    )

    # Collection Configuration
    collection_batch_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Products per collection batch"
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

    # Cache Configuration
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

    @validator('qdrant_url')
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

    @validator('embedding_vector_size', 'qdrant_vector_size')
    def validate_vector_sizes_match(cls, v: int, values: dict) -> int:
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
        if 'embedding_vector_size' in values and 'qdrant_vector_size' in values:
            if values['embedding_vector_size'] != values['qdrant_vector_size']:
                raise ValueError('embedding_vector_size and qdrant_vector_size must match')
        return v

    @validator('embedding_model_name')
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

    @validator('collection_batch_size')
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
            "max_concurrent": self.max_concurrent_collections
        }

    def get_api_configs(self) -> dict:
        """
        Get all external API configurations.

        Returns:
            Dictionary with API configurations for each service
        """
        return {
            "appify": {
                "api_key": self.appify_api_key,
                "base_url": self.appify_base_url,
                "rate_limit": self.appify_rate_limit
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
            self.appify_api_key,
            self.qdrant_url,
            self.postgres_url
        ]

        return all(req for req in production_requirements)