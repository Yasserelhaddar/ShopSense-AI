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

    # Cache Configuration
    redis_url: str = Field(
        default="redis://redis:6379",
        description="Redis connection URL"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        description="Cache TTL in seconds"
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

    class Config:
        """Pydantic configuration."""
        env_prefix = "ADVISORY_"
        case_sensitive = False