"""
Product data collection from external APIs and sources.

This module handles data collection from various e-commerce platforms:
- Appify API for Amazon and multi-store data
- Best Buy official API
- RapidAPI services for additional stores
- Web scraping with rate limiting and retry logic

Key Features:
- Rate limiting and retry mechanisms
- Data validation and cleaning
- Concurrent collection with queue management
- Error handling and fallback strategies
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from core.logging import get_logger
from config.settings import DiscoverySettings


logger = get_logger("discovery-service")
settings = DiscoverySettings()


class CollectorManager:
    """
    Manages all product data collection operations.

    Coordinates collection from multiple sources, handles job scheduling,
    and manages rate limiting across different APIs.
    """

    def __init__(self):
        """Initialize the collector manager with API clients."""
        self.appify_collector = AppifyCollector()
        self.bestbuy_collector = BestBuyCollector()
        self.rapidapi_collector = RapidAPICollector()
        self.active_jobs: Dict[str, Dict] = {}

    async def initialize(self):
        """Initialize all collectors and validate API connections."""
        await self.appify_collector.initialize()
        await self.bestbuy_collector.initialize()
        await self.rapidapi_collector.initialize()
        logger.info("CollectorManager initialized successfully")

    async def start_collection_job(
        self,
        sources: List[str],
        categories: List[str],
        priority: str,
        background_tasks
    ) -> str:
        """
        Start a new collection job.

        Args:
            sources: List of sources to collect from
            categories: Product categories to focus on
            priority: Job priority (low, normal, high)
            background_tasks: FastAPI background tasks

        Returns:
            str: Unique job identifier
        """
        job_id = f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        # Initialize job tracking
        self.active_jobs[job_id] = {
            "status": "starting",
            "sources": sources,
            "categories": categories,
            "priority": priority,
            "started_at": datetime.utcnow(),
            "progress": 0,
            "products_collected": 0,
            "errors": []
        }

        # Schedule collection
        background_tasks.add_task(self._run_collection_job, job_id)

        logger.info(f"Collection job {job_id} started for sources: {sources}")
        return job_id

    async def collect_trending_products(self):
        """
        Collect trending products from all sources.

        This method is called by scheduled tasks to collect
        popular and trending products across platforms.
        """
        logger.info("Starting trending products collection")

        trending_queries = [
            "gaming laptop",
            "wireless headphones",
            "smartphone",
            "4K TV",
            "fitness tracker",
            "coffee maker",
            "air fryer",
            "robot vacuum"
        ]

        for query in trending_queries:
            try:
                # Collect from Appify (Amazon)
                products = await self.appify_collector.search_amazon(query, max_results=20)
                await self._process_collected_products(products, "amazon")

                # Rate limiting
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error collecting trending products for '{query}': {e}")

        logger.info("Trending products collection completed")

    async def monitor_prices(self):
        """
        Monitor price changes for tracked products.

        Checks for price drops and identifies deals across
        previously collected products.
        """
        logger.info("Starting price monitoring")

        # TODO: Implement price monitoring logic
        # - Get list of tracked products
        # - Check current prices
        # - Compare with historical data
        # - Identify deals and price drops

        logger.info("Price monitoring completed")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a collection job.

        Args:
            job_id: Job identifier

        Returns:
            Job status dictionary or None if not found
        """
        return self.active_jobs.get(job_id)

    async def test_appify_connection(self):
        """Test connection to Appify API."""
        await self.appify_collector.test_connection()

    async def test_bestbuy_connection(self):
        """Test connection to Best Buy API."""
        await self.bestbuy_collector.test_connection()

    async def cleanup(self):
        """Clean up collector resources."""
        await self.appify_collector.cleanup()
        await self.bestbuy_collector.cleanup()
        await self.rapidapi_collector.cleanup()

    async def _run_collection_job(self, job_id: str):
        """Execute collection job in background."""
        try:
            job = self.active_jobs[job_id]
            job["status"] = "running"

            total_products = 0

            for source in job["sources"]:
                job["status"] = f"collecting_from_{source}"

                if source == "amazon":
                    products = await self._collect_from_amazon(job["categories"])
                elif source == "bestbuy":
                    products = await self._collect_from_bestbuy(job["categories"])
                else:
                    logger.warning(f"Unknown source: {source}")
                    continue

                total_products += len(products)
                job["products_collected"] = total_products

                # Update progress
                progress = (job["sources"].index(source) + 1) / len(job["sources"]) * 100
                job["progress"] = progress

            job["status"] = "completed"
            job["completed_at"] = datetime.utcnow()

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            logger.error(f"Collection job {job_id} failed: {e}")

    async def _collect_from_amazon(self, categories: List[str]) -> List[Dict]:
        """Collect products from Amazon via Appify."""
        all_products = []

        for category in categories:
            products = await self.appify_collector.search_amazon(category, max_results=50)
            all_products.extend(products)
            await asyncio.sleep(1)  # Rate limiting

        return all_products

    async def _collect_from_bestbuy(self, categories: List[str]) -> List[Dict]:
        """Collect products from Best Buy."""
        all_products = []

        for category in categories:
            products = await self.bestbuy_collector.search_products(category, max_results=50)
            all_products.extend(products)
            await asyncio.sleep(1)

        return all_products

    async def _process_collected_products(self, products: List[Dict], source: str):
        """Process and store collected products."""
        # TODO: Implement product processing and storage
        logger.info(f"Processed {len(products)} products from {source}")


class AppifyCollector:
    """Collector for Appify API (Amazon + multi-store data)."""

    def __init__(self):
        """Initialize Appify collector."""
        self.client = None
        self.rate_limiter = asyncio.Semaphore(settings.appify_rate_limit)

    async def initialize(self):
        """Initialize HTTP client with proper configuration."""
        self.client = httpx.AsyncClient(
            base_url=settings.appify_base_url,
            headers={"Authorization": f"Bearer {settings.appify_api_key}"},
            timeout=30.0
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def search_amazon(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search Amazon products via Appify API.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of product dictionaries
        """
        async with self.rate_limiter:
            response = await self.client.post(
                "/scrape/amazon",
                json={
                    "query": query,
                    "max_results": max_results,
                    "include_reviews": True,
                    "include_images": True
                }
            )

            if response.status_code == 200:
                data = response.json()
                return self._clean_amazon_data(data.get("products", []))
            else:
                logger.error(f"Appify API error: {response.status_code}")
                return []

    def _clean_amazon_data(self, raw_products: List[Dict]) -> List[Dict]:
        """Clean and standardize Amazon product data."""
        cleaned_products = []

        for product in raw_products:
            try:
                cleaned = {
                    "id": self._generate_product_id(product),
                    "title": product.get("title", "").strip(),
                    "price": self._parse_price(product.get("price")),
                    "rating": self._parse_rating(product.get("rating")),
                    "reviews_count": self._parse_reviews_count(product.get("reviews_count")),
                    "image_url": product.get("image_url"),
                    "product_url": product.get("product_url"),
                    "store": "amazon",
                    "collected_at": datetime.utcnow().isoformat()
                }

                if cleaned["title"] and cleaned["price"] is not None:
                    cleaned_products.append(cleaned)

            except Exception as e:
                logger.warning(f"Error cleaning product data: {e}")

        return cleaned_products

    async def test_connection(self):
        """Test Appify API connection."""
        if not self.client:
            await self.initialize()

        response = await self.client.get("/health")
        if response.status_code != 200:
            raise Exception(f"Appify API connection failed: {response.status_code}")

    async def cleanup(self):
        """Clean up HTTP client."""
        if self.client:
            await self.client.aclose()

    def _generate_product_id(self, product: Dict) -> str:
        """Generate unique product ID."""
        identifier = f"{product.get('title', '')}{product.get('price', '')}{product.get('store', 'amazon')}"
        return hashlib.md5(identifier.encode()).hexdigest()

    def _parse_price(self, price_str: Any) -> Optional[float]:
        """Parse price string to float."""
        if not price_str:
            return None

        try:
            # Remove currency symbols and extract number
            price_clean = str(price_str).replace("$", "").replace(",", "").strip()
            return float(price_clean)
        except (ValueError, TypeError):
            return None

    def _parse_rating(self, rating_str: Any) -> Optional[float]:
        """Parse rating string to float."""
        if not rating_str:
            return None

        try:
            rating = float(str(rating_str).split()[0])  # Take first number
            return min(max(rating, 0), 5)  # Clamp between 0-5
        except (ValueError, TypeError, IndexError):
            return None

    def _parse_reviews_count(self, reviews_str: Any) -> int:
        """Parse reviews count string to integer."""
        if not reviews_str:
            return 0

        try:
            # Extract number from string like "1,234 reviews"
            reviews_clean = str(reviews_str).replace(",", "").split()[0]
            return int(reviews_clean)
        except (ValueError, TypeError, IndexError):
            return 0


class BestBuyCollector:
    """Collector for Best Buy official API."""

    def __init__(self):
        """Initialize Best Buy collector."""
        self.client = None

    async def initialize(self):
        """Initialize HTTP client."""
        self.client = httpx.AsyncClient(
            base_url=settings.bestbuy_base_url,
            params={"apiKey": settings.bestbuy_api_key},
            timeout=30.0
        )

    async def search_products(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search Best Buy products.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of product dictionaries
        """
        # TODO: Implement Best Buy API integration
        logger.info(f"Searching Best Buy for: {query}")
        return []

    async def test_connection(self):
        """Test Best Buy API connection."""
        # TODO: Implement connection test
        pass

    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            await self.client.aclose()


class RapidAPICollector:
    """Collector for RapidAPI services (Walmart, eBay, etc.)."""

    def __init__(self):
        """Initialize RapidAPI collector."""
        self.client = None

    async def initialize(self):
        """Initialize HTTP client."""
        self.client = httpx.AsyncClient(
            headers={"X-RapidAPI-Key": settings.rapidapi_key},
            timeout=30.0
        )

    async def search_walmart(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search Walmart products via RapidAPI.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of product dictionaries
        """
        # TODO: Implement Walmart API integration
        logger.info(f"Searching Walmart for: {query}")
        return []

    async def test_connection(self):
        """Test RapidAPI connection."""
        # TODO: Implement connection test
        pass

    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            await self.client.aclose()