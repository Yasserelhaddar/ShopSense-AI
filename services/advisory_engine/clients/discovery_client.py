"""
Discovery Engine client for the Advisory Engine.

This module provides a client interface for communicating with the
Discovery Engine service for product search and data retrieval.
"""

import httpx
from typing import List, Dict, Any, Optional

from shared.logging import get_logger


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
            response = await self.client.get("/api/v1/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            await self.client.aclose()

    # Admin operations

    async def trigger_collection(
        self,
        sources: List[str],
        categories: Optional[List[str]] = None,
        max_results: int = 100,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Trigger product data collection from specified sources.

        Args:
            sources: List of sources to collect from (e.g., ["amazon", "bestbuy"])
            categories: List of categories to collect (optional)
            max_results: Maximum number of results per category
            priority: Collection priority (normal, high, urgent)

        Returns:
            Collection job information with job_id and status
        """
        if not self.client:
            await self.initialize()

        payload = {
            "sources": sources,
            "categories": categories or [],
            "max_results": max_results,
            "priority": priority
        }

        try:
            response = await self.client.post("/api/v1/products/collect", json=payload)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Collection job started: {result.get('job_id')}")
            return result
        except Exception as e:
            logger.error(f"Failed to trigger collection: {e}")
            raise

    async def get_collection_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a collection job.

        Args:
            job_id: Collection job identifier

        Returns:
            Job status and progress information
        """
        if not self.client:
            await self.initialize()

        try:
            response = await self.client.get(f"/api/v1/collection/status/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get collection status: {e}")
            raise

    async def get_categories(self) -> List[str]:
        """
        Get available product categories.

        Returns:
            List of available categories
        """
        if not self.client:
            await self.initialize()

        try:
            response = await self.client.get("/api/v1/products/categories")
            response.raise_for_status()
            result = response.json()
            return result.get("categories", [])
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []

    async def get_stores(self) -> List[str]:
        """
        Get available stores/sources.

        Returns:
            List of available stores
        """
        if not self.client:
            await self.initialize()

        try:
            response = await self.client.get("/api/v1/products/stores")
            response.raise_for_status()
            result = response.json()
            return result.get("stores", [])
        except Exception as e:
            logger.error(f"Failed to get stores: {e}")
            return []