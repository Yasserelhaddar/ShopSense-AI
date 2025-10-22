"""
Base configuration utilities for ShopSense-AI microservices.

This module provides base configuration classes and utilities that can be
extended by individual services for their specific configuration needs.

Usage:
    from core.config import BaseSettings

    class ServiceSettings(BaseSettings):
        service_port: int = 8001
        redis_url: str = "redis://localhost:6379"

        class Config:
            env_prefix = "SERVICE_"
"""

from typing import Any, Dict, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class BaseSettings(PydanticBaseSettings):
    """
    Base settings class for all ShopSense-AI services.

    This class provides common configuration patterns and validation
    that should be used across all microservices.

    Attributes:
        debug: Enable debug mode
        environment: Deployment environment (development, production)
        log_level: Logging level for the service
    """

    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"

    class Config:
        """Pydantic configuration for BaseSettings."""
        env_file = ".env"
        case_sensitive = False


    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """
        Validate log level is one of the accepted values.

        Args:
            v: Log level string

        Returns:
            Validated log level

        Raises:
            ValueError: If log level is invalid
        """
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()

    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration dictionary.

        Returns:
            Dictionary with database connection parameters
        """
        return {
            "echo": self.debug,
            "pool_pre_ping": True,
            "pool_recycle": 300,
        }

    def is_production(self) -> bool:
        """
        Check if running in production environment.

        Returns:
            True if environment is production, False otherwise
        """
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """
        Check if running in development environment.

        Returns:
            True if environment is development, False otherwise
        """
        return self.environment.lower() == "development"