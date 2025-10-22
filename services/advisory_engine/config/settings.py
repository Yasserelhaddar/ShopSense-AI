"""
Advisory Engine configuration settings.

Configuration is loaded from environment variables with the ADVISORY_ prefix.
"""

from typing import Optional
from pydantic import Field

from shared.config import BaseSettings


class AdvisorySettings(BaseSettings):
    """Configuration settings for the Advisory Engine service."""

    # Service Configuration
    port: int = Field(default=8003, description="Service port number")
    allowed_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins (use '*' for all)"
    )

    # Authentication Configuration (Clerk)
    clerk_publishable_key: Optional[str] = Field(
        default=None,
        description="Clerk publishable key (safe for frontend)"
    )
    clerk_secret_key: Optional[str] = Field(
        default=None,
        description="Clerk secret key (backend only)"
    )
    clerk_issuer: Optional[str] = Field(
        default=None,
        description="Clerk JWT issuer URL"
    )
    auth_enabled: bool = Field(
        default=False,
        description="Enable authentication (requires Clerk keys)"
    )

    # External Service URLs
    knowledge_service_url: str = Field(
        default="http://knowledge-service:8001",
        description="Knowledge Engine service URL"
    )
    discovery_service_url: str = Field(
        default="http://discovery-service:8002",
        description="Discovery Engine service URL"
    )

    # External API Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model for fallback inference"
    )
    openai_max_tokens: int = Field(default=1000, description="OpenAI max tokens")
    openai_temperature: float = Field(default=0.7, description="OpenAI temperature")

    # Cache Configuration
    redis_url: str = Field(
        default="redis://redis:6379",
        description="Redis connection URL"
    )
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl_seconds: int = Field(
        default=900,
        description="Cache TTL in seconds (15 minutes)"
    )

    # Recommendation Configuration
    max_recommendations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum recommendations per request"
    )
    recommendation_timeout_seconds: int = Field(
        default=30,
        description="Recommendation generation timeout"
    )
    default_model_id: str = Field(
        default="shopping_advisor_production_v2",
        description="Default model ID for knowledge engine requests"
    )

    # Search Configuration
    max_search_results: int = Field(default=20, description="Max search results")
    search_timeout_seconds: int = Field(default=30, description="Search timeout")

    # Performance Configuration
    http_timeout_seconds: int = Field(
        default=60,
        description="HTTP timeout for service calls in seconds"
    )
    http_pool_size: int = Field(
        default=100,
        description="HTTP connection pool size"
    )
    http_max_keepalive: int = Field(
        default=5,
        description="HTTP max keepalive connections"
    )
    http_max_connections: int = Field(
        default=10,
        description="HTTP max total connections"
    )
    max_concurrent_calls: int = Field(default=10, description="Max concurrent calls")

    # Retry Configuration
    discovery_retry_attempts: int = Field(default=3, description="Discovery retry attempts")
    knowledge_retry_attempts: int = Field(default=2, description="Knowledge retry attempts")
    retry_backoff_multiplier: int = Field(default=2, description="Retry backoff multiplier")

    class Config:
        """Pydantic configuration."""
        env_prefix = "ADVISORY_"
        case_sensitive = False
        env_file = "config/.env"
        env_file_encoding = "utf-8"