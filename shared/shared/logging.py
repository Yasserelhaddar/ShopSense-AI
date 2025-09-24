"""
Simple logging setup for ShopSense-AI microservices.

This module provides a centralized logging configuration that can be used
across all services. It includes structured logging, appropriate formatting,
and log level management.

Usage:
    from core.logging import setup_logger

    logger = setup_logger("service-name")
    logger.info("Service started successfully")
"""

import logging
import sys
from typing import Optional


def setup_logger(
    service_name: str,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger for a microservice.

    Args:
        service_name: Name of the service (e.g., "knowledge-service")
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("discovery-service", "DEBUG")
        >>> logger.info("Starting product collection")
    """
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Default format with service name and structured information
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def get_logger(service_name: str) -> logging.Logger:
    """
    Get an existing logger by service name.

    Args:
        service_name: Name of the service

    Returns:
        Logger instance if exists, otherwise creates a new one
    """
    logger = logging.getLogger(service_name)
    if not logger.handlers:
        return setup_logger(service_name)
    return logger