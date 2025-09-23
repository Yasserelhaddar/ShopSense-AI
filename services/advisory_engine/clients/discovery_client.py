"""
Discovery Engine client for the Advisory Engine.

This module provides a client interface for communicating with the
Discovery Engine service for product search and data retrieval.
"""

import httpx
from typing import List, Dict, Any, Optional

from core.logging import get_logger


logger = get_logger("advisory-service")


class DiscoveryClient:
    """Client for communicating with the Discovery Engine service."""

    def __init__(self, base_url: str):
        """Initialize the Discovery Engine client."""
        self.base_url = base_url.rstrip("/")
        self.client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        """Initialize the HTTP client."""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0
        )
        logger.info(f"Discovery client initialized for {self.base_url}")

    async def search_products(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for products using the Discovery Engine.

        Args:
            query: Search query
            filters: Search filters
            limit: Maximum results

        Returns:
            List of product dictionaries
        """
        if not self.client:
            await self.initialize()

        params = {"query": query, "limit": limit}
        if filters:
            params.update(filters)

        try:
            response = await self.client.get("/api/v1/products/search", params=params)
            response.raise_for_status()
            result = response.json()
            return result.get("products", [])
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            return []

    async def get_product_details(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed product information."""
        if not self.client:
            await self.initialize()

        try:
            response = await self.client.get(f"/api/v1/products/{product_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get product details: {e}")
            return None

    async def health_check(self) -> bool:
        """Check Discovery Engine health."""
        if not self.client:
            await self.initialize()

        try:
            response = await self.client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            await self.client.aclose()