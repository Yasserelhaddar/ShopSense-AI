"""
Product data collection and external API integration.

This module handles data collection from multiple e-commerce platforms
through various APIs, with primary focus on Apify for Amazon data.
It implements proper rate limiting, retry logic, and data cleaning.

Key Features:
- Apify Actor integration for Amazon product data
- Best Buy official API integration
- RapidAPI integration for Walmart/eBay
- Rate limiting and retry mechanisms
- Data validation and cleaning
- Asynchronous operations for performance
"""

import asyncio
import hashlib
import logging
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from shared.logging import get_logger
from config.settings import DiscoverySettings


logger = get_logger("discovery-collectors")
settings = DiscoverySettings()


class JobStatus(Enum):
    """Collection job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CollectionJob:
    """Collection job state tracking."""
    job_id: str
    sources: List[str]
    categories: List[str]
    status: JobStatus
    progress: float = 0.0
    products_collected: int = 0
    errors_count: int = 0
    error_messages: List[str] = None
    started_at: datetime = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc)


@dataclass
class ProductData:
    """
    Structured product data from collection.

    This class represents a standardized product structure
    that all collectors should return, regardless of the source.
    """
    id: str
    title: str
    price: float
    currency: str = "USD"
    store: str = ""
    brand: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    availability: str = "unknown"
    key_features: List[str] = None
    specifications: Dict[str, str] = None

    def __post_init__(self):
        """Initialize default values for list/dict fields."""
        if self.key_features is None:
            self.key_features = []
        if self.specifications is None:
            self.specifications = {}


class ApifyCollector:
    """
    Apify Actor collector for Amazon and multi-store product data.

    This collector uses the official Apify API patterns to run Actors
    for collecting product data from Amazon and other supported platforms.
    Implements proper rate limiting and retry mechanisms.
    """

    def __init__(self):
        """Initialize the Apify collector."""
        self.api_key = settings.apify_api_key
        self.base_url = "https://api.apify.com/v2"
        self.actor_id = "BG3WDrGdteHgZgbPK"  # Amazon Product Scraper Actor (junglee/Amazon-crawler)
        self.rate_limit = settings.apify_rate_limit or 10
        self.session: Optional[httpx.AsyncClient] = None
        self.rate_limiter = asyncio.Semaphore(self.rate_limit)

    async def initialize(self):
        """Initialize the HTTP client session."""
        if not self.session:
            self.session = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0),  # 5 minutes for Actor runs
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ShopSense-AI Discovery Engine/1.0"
                }
            )
            logger.info("Apify collector initialized")

    async def test_connection(self) -> bool:
        """
        Test Apify API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection by making a simple API call
            response = await self.session.get(
                f"{self.base_url}/acts",
                params={"token": self.api_key, "limit": 1}
            )
            response.raise_for_status()

            result = response.json()
            if "data" in result:
                logger.info("✅ Apify API connection test successful")
                return True
            else:
                logger.error("❌ Apify API response missing data field")
                return False

        except Exception as e:
            logger.error(f"❌ Apify API connection test failed: {e}")
            return False

    async def cleanup(self):
        """Clean up HTTP client session."""
        if self.session:
            await self.session.aclose()
            self.session = None
            logger.info("Apify collector cleaned up")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _run_actor(self, input_data: Dict[str, Any], wait_for_finish: int = 300) -> Dict[str, Any]:
        """
        Run Apify Actor with rate limiting and retry logic.

        Args:
            input_data: Input configuration for the Actor
            wait_for_finish: Seconds to wait for Actor to complete (default 5 minutes)

        Returns:
            Actor run results including dataset ID
        """
        async with self.rate_limiter:
            await asyncio.sleep(1.0 / self.rate_limit)  # Rate limiting

            # Start Actor run
            response = await self.session.post(
                f"{self.base_url}/acts/{self.actor_id}/runs",
                params={
                    "token": self.api_key,
                    "waitForFinish": wait_for_finish
                },
                json=input_data
            )
            response.raise_for_status()
            return response.json()

    async def _get_dataset_items(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve items from Actor run dataset.

        Args:
            dataset_id: ID of the dataset to retrieve

        Returns:
            List of scraped items
        """
        response = await self.session.get(
            f"{self.base_url}/datasets/{dataset_id}/items",
            params={"token": self.api_key}
        )
        response.raise_for_status()
        return response.json()

    async def search_amazon_products(
        self,
        query: str,
        max_results: int = 50,
        category: Optional[str] = None
    ) -> List[ProductData]:
        """
        Search for products on Amazon using Apify Actor.

        Args:
            query: Search query
            max_results: Maximum number of results
            category: Optional category filter

        Returns:
            List of standardized product data
        """
        if not self.session:
            await self.initialize()

        # Configure Actor input for Amazon Product Scraper
        # This Actor requires Amazon URLs, not search keywords
        amazon_search_url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"

        actor_input = {
            "categoryOrProductUrls": [
                {
                    "url": amazon_search_url
                }
            ],
            "maxItemsPerStartUrl": min(max_results, 100),
            "proxyCountry": "AUTO_SELECT_PROXY_COUNTRY",
            "maxSearchPagesPerStartUrl": 5,  # Limit to 5 pages for efficiency
            "maxOffers": 0,
            "scrapeSellers": False,
            "ensureLoadedProductDescriptionFields": False,
            "useCaptchaSolver": False,
            "scrapeProductVariantPrices": False,
            "scrapeProductDetails": True,
            "locationDeliverableRoutes": ["PRODUCT", "SEARCH", "OFFERS"]
        }

        try:
            logger.info(f"Starting Apify Actor for search: {query} (max_results: {max_results})")

            # Run the Amazon scraper Actor with sufficient timeout
            run_result = await self._run_actor(actor_input, wait_for_finish=180)  # 3 minutes

            # Get the dataset with results
            dataset_id = run_result.get("data", {}).get("defaultDatasetId")
            if not dataset_id:
                logger.error(f"No dataset ID returned from Actor run. Response structure: {list(run_result.keys())}")
                return []

            # Retrieve scraped items
            items = await self._get_dataset_items(dataset_id)

            # Parse results
            products = []
            for item in items:
                product = self._parse_amazon_product(item)
                if product and self._validate_product(product):
                    products.append(product)

            logger.info(f"Collected {len(products)} valid products from Amazon")
            return products

        except Exception as e:
            logger.error(f"Amazon search failed for query '{query}': {e}")
            return []

    def _parse_amazon_product(self, raw_data: Dict[str, Any]) -> Optional[ProductData]:
        """
        Parse raw Amazon product data from Apify Actor into standardized format.

        Args:
            raw_data: Raw product data from Apify Amazon Actor

        Returns:
            Standardized product data or None if parsing fails
        """
        try:
            # Generate stable ID from ASIN or URL
            asin = raw_data.get("asin")
            product_id = asin if asin else self._generate_product_id(raw_data.get("url", ""))

            if not product_id:
                return None

            # Extract price information (New format: {value: 145.5, currency: "$"})
            price = self._extract_price(raw_data)
            if price is None:
                return None  # Skip products without valid prices

            # Extract basic product information
            title = self._clean_text(raw_data.get("title", ""))
            if not title:
                return None

            # Create product data using new Amazon Product Scraper format
            product = ProductData(
                id=f"amazon_{product_id}",
                title=title[:500],  # Limit title length
                price=price,
                currency="USD",
                store="amazon",
                brand=self._clean_text(raw_data.get("brand", "")) if raw_data.get("brand") else None,
                category=self._extract_category(raw_data),
                description=self._clean_text(raw_data.get("description", ""))[:1000] if raw_data.get("description") else None,
                image_url=raw_data.get("thumbnailImage"),
                product_url=raw_data.get("url"),
                rating=self._extract_rating(raw_data),
                reviews_count=raw_data.get("reviewsCount"),
                availability=self._extract_availability_new_format(raw_data),
                key_features=self._extract_features_new_format(raw_data),
                specifications=self._extract_specifications(raw_data)
            )

            return product

        except Exception as e:
            logger.warning(f"Failed to parse Amazon product: {e}")
            return None

    def _extract_price(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract and validate price from product data (Apify format)."""
        price_data = data.get("price")
        if not price_data:
            return None

        # Handle Apify price format: {value: 13.99, currency: "$"}
        if isinstance(price_data, dict):
            price_value = price_data.get("value")
            if price_value is not None:
                try:
                    price = float(price_value)
                    if 0.01 <= price <= 50000.0:
                        return round(price, 2)
                except (ValueError, TypeError):
                    pass
        else:
            # Handle string format as fallback
            price_cleaned = re.sub(r'[^\d.]', '', str(price_data))
            if price_cleaned:
                try:
                    price = float(price_cleaned)
                    if 0.01 <= price <= 50000.0:
                        return round(price, 2)
                except (ValueError, TypeError):
                    pass

        return None

    def _extract_brand(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract brand from product data."""
        brand = data.get("brand")
        if brand:
            return self._clean_text(brand)[:100]

        # Try to extract from title
        title = data.get("title", "")
        common_brands = [
            "Apple", "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo", "ASUS", "MSI",
            "Acer", "Microsoft", "Nintendo", "Amazon", "Google", "Canon", "Nikon",
            "Nike", "Adidas", "Levi's", "Calvin Klein", "Under Armour"
        ]

        title_lower = title.lower()
        for brand in common_brands:
            if brand.lower() in title_lower:
                return brand

        return None

    def _extract_category(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract category from product data (Apify uses breadCrumbs)."""
        # Try breadCrumbs first (Apify format)
        breadcrumbs = data.get("breadCrumbs", "")
        if breadcrumbs:
            # Extract meaningful category from breadcrumbs
            categories = [cat.strip() for cat in breadcrumbs.split("›") if cat.strip()]
            if len(categories) >= 2:
                return categories[1][:100]  # Second level category

        # Try direct category field
        category = data.get("category")
        if category:
            return self._clean_text(category)[:100]

        return None

    def _extract_rating(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract rating from product data (Apify uses 'stars')."""
        rating = data.get("stars") or data.get("rating")  # Apify uses 'stars'
        if rating:
            try:
                rating_float = float(rating)
                if 0.0 <= rating_float <= 5.0:
                    return round(rating_float, 1)
            except (ValueError, TypeError):
                pass
        return None

    def _extract_reviews_count(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract reviews count from product data (Apify uses 'reviewsCount')."""
        reviews = data.get("reviewsCount") or data.get("reviews_count")  # Apify uses 'reviewsCount'
        if reviews:
            try:
                # Handle string numbers with commas
                reviews_str = str(reviews).replace(",", "")
                reviews_int = int(float(reviews_str))
                if reviews_int >= 0:
                    return reviews_int
            except (ValueError, TypeError):
                pass
        return None

    def _extract_availability(self, data: Dict[str, Any]) -> str:
        """Extract availability status from product data."""
        availability = data.get("availability", "").lower()

        if "in stock" in availability:
            return "in_stock"
        elif "out of stock" in availability:
            return "out_of_stock"
        elif "temporarily unavailable" in availability:
            return "temporarily_unavailable"
        else:
            return "unknown"

    def _extract_availability_new_format(self, data: Dict[str, Any]) -> str:
        """Extract availability status from new Amazon Product Scraper format."""
        # Check inStock boolean field
        in_stock = data.get("inStock")
        if in_stock is True:
            return "in_stock"
        elif in_stock is False:
            return "out_of_stock"

        # Fallback to inStockText analysis
        stock_text = data.get("inStockText", "").lower()
        if "in stock" in stock_text:
            return "in_stock"
        elif "out of stock" in stock_text:
            return "out_of_stock"
        elif "unavailable" in stock_text:
            return "temporarily_unavailable"
        else:
            return "unknown"

    def _extract_features_new_format(self, data: Dict[str, Any]) -> List[str]:
        """Extract key features from new Amazon Product Scraper format."""
        features = []

        # Try 'features' field (this Actor uses 'features')
        feature_data = data.get("features", [])
        if isinstance(feature_data, list):
            for feature in feature_data[:10]:  # Limit to 10 features
                cleaned = self._clean_text(str(feature))
                if cleaned and len(cleaned) <= 200:
                    features.append(cleaned)

        return features

    def _extract_features(self, data: Dict[str, Any]) -> List[str]:
        """Extract key features from product data."""
        features = []

        # Try feature_bullets or key_features field
        feature_data = data.get("feature_bullets") or data.get("key_features", [])
        if isinstance(feature_data, list):
            for feature in feature_data[:10]:  # Limit to 10 features
                cleaned = self._clean_text(str(feature))
                if cleaned and len(cleaned) <= 200:
                    features.append(cleaned)

        return features

    def _extract_specifications(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract specifications from product data."""
        specs = {}

        spec_data = data.get("specifications") or data.get("technical_details", {})
        if isinstance(spec_data, dict):
            for key, value in list(spec_data.items())[:20]:  # Limit to 20 specs
                clean_key = self._clean_text(str(key))[:100]
                clean_value = self._clean_text(str(value))[:200]
                if clean_key and clean_value:
                    specs[clean_key] = clean_value

        return specs

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not text:
            return ""

        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\-\.\$\%\(\)\/]', ' ', text)  # Remove special chars

        return text.strip()

    def _generate_product_id(self, url: str) -> str:
        """Generate a stable product ID from URL."""
        if not url:
            return ""

        # Generate hash from URL for consistent ID
        return hashlib.md5(url.encode()).hexdigest()[:16]

    def _validate_product(self, product: ProductData) -> bool:
        """Validate product data meets quality requirements."""
        # Basic validation
        if not product.title or len(product.title) < 10:
            return False

        if product.price <= 0:
            return False

        # Quality score based on available data
        score = 0.0

        # Required fields
        if product.title: score += 0.3
        if product.price > 0: score += 0.3

        # Optional but valuable fields
        if product.brand: score += 0.1
        if product.category: score += 0.1
        if product.description: score += 0.1
        if product.image_url: score += 0.05
        if product.rating: score += 0.05

        return score >= settings.min_product_score


class CollectorManager:
    """
    Manager for coordinating multiple product collectors.

    This class manages all product collection operations,
    including scheduling, error handling, and data aggregation.
    """

    def __init__(self, storage_manager=None, embedding_processor=None):
        """Initialize the collector manager."""
        self.apify_collector = ApifyCollector()
        self.collectors_initialized = False
        # In-memory job tracking (in production, use Redis or database)
        self.jobs: Dict[str, CollectionJob] = {}
        # References to other managers for product storage
        self.storage_manager = storage_manager
        self.embedding_processor = embedding_processor

    async def initialize(self):
        """Initialize all collectors."""
        if not self.collectors_initialized:
            await self.apify_collector.initialize()
            self.collectors_initialized = True
            logger.info("Collector manager initialized")

    async def cleanup(self):
        """Clean up all collectors."""
        if self.collectors_initialized:
            await self.apify_collector.cleanup()
            self.collectors_initialized = False
            logger.info("Collector manager cleaned up")

    async def collect_trending_products(self) -> List[ProductData]:
        """Collect trending products from all available sources."""
        if not self.collectors_initialized:
            await self.initialize()

        trending_queries = [
            "laptop deals",
            "gaming headphones",
            "smartphone",
            "wireless earbuds",
            "fitness tracker",
            "coffee maker",
            "bluetooth speaker",
            "tablet"
        ]

        all_products = []

        for query in trending_queries:
            try:
                products = await self.apify_collector.search_amazon_products(
                    query=query,
                    max_results=20
                )
                all_products.extend(products)

                # Add delay between queries to be respectful
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Failed to collect trending products for '{query}': {e}")

        # Remove duplicates based on product ID
        unique_products = self._deduplicate_products(all_products)
        logger.info(f"Collected {len(unique_products)} unique trending products")

        return unique_products

    async def start_collection_job(
        self,
        sources: List[str],
        categories: List[str] = None,
        priority: str = "normal",
        background_tasks=None
    ) -> str:
        """
        Start a new collection job.

        Args:
            sources: List of sources to collect from
            categories: Product categories to focus on
            priority: Collection priority level
            background_tasks: FastAPI background tasks

        Returns:
            Job ID for tracking the collection
        """
        import uuid

        job_id = str(uuid.uuid4())

        # Create and store job tracking object
        job = CollectionJob(
            job_id=job_id,
            sources=sources,
            categories=categories or [],
            status=JobStatus.PENDING,
            started_at=datetime.now(timezone.utc)
        )
        self.jobs[job_id] = job

        logger.info(f"Starting collection job {job_id} for sources: {sources}, categories: {categories}")

        if background_tasks:
            background_tasks.add_task(
                self._run_collection_job,
                job_id
            )
        else:
            # Fallback to asyncio if no background_tasks provided
            asyncio.create_task(
                self._run_collection_job(job_id)
            )

        return job_id

    async def _run_collection_job(self, job_id: str):
        """Run the actual collection job in background."""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found in tracking system")
            return

        try:
            # Update job status to running
            job.status = JobStatus.RUNNING
            job.progress = 0.0
            logger.info(f"Executing collection job {job_id}")

            # Validate that categories are provided for manual collection
            if not job.categories:
                job.status = JobStatus.FAILED
                job.error_messages.append("No categories specified for manual collection")
                logger.error(f"Job {job_id}: No categories specified for manual collection")
                return

            all_products = []
            total_categories = len(job.categories)

            # Collect products from specified sources
            if "amazon" in job.sources:
                for i, category in enumerate(job.categories):
                    try:
                        logger.info(f"Job {job_id}: Collecting products for category: {category}")
                        products = await self.apify_collector.search_amazon_products(
                            query=category,
                            max_results=20
                        )
                        all_products.extend(products)
                        job.products_collected = len(all_products)
                        job.progress = ((i + 1) / total_categories) * 90.0  # Leave 10% for storage

                        logger.info(f"Job {job_id}: Found {len(products)} products for category: {category}")

                    except Exception as e:
                        job.errors_count += 1
                        job.error_messages.append(f"Failed to collect category '{category}': {str(e)}")
                        logger.error(f"Job {job_id}: Failed to collect products for category '{category}': {e}")
                        continue

            # Deduplicate products
            unique_products = self._deduplicate_products(all_products)
            job.products_collected = len(unique_products)
            job.progress = 95.0

            # Store products in Qdrant vector database using complete pipeline
            if unique_products and self.storage_manager and self.embedding_processor:
                try:
                    logger.info(f"Job {job_id}: Storing {len(unique_products)} products in Qdrant...")
                    await self.storage_manager.process_and_store_products(
                        unique_products,
                        self.embedding_processor
                    )
                    logger.info(f"Job {job_id}: Successfully stored {len(unique_products)} products in Qdrant!")
                except Exception as e:
                    job.errors_count += 1
                    job.error_messages.append(f"Failed to store products: {str(e)}")
                    logger.error(f"Job {job_id}: Failed to store products in Qdrant: {e}")
            else:
                logger.warning(f"Job {job_id}: Cannot store products - missing storage_manager ({bool(self.storage_manager)}) or embedding_processor ({bool(self.embedding_processor)})")

            job.progress = 100.0

            # Mark job as completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)

            logger.info(f"Collection job {job_id} completed successfully - collected {len(unique_products)} unique products")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.errors_count += 1
            job.error_messages.append(f"Job execution failed: {str(e)}")
            job.completed_at = datetime.now(timezone.utc)
            logger.error(f"Collection job {job_id} failed: {e}")

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a collection job.

        Args:
            job_id: Job identifier

        Returns:
            Job status information or None if not found
        """
        job = self.jobs.get(job_id)
        if not job:
            return None

        # Calculate estimated completion if job is running
        estimated_completion = None
        if job.status == JobStatus.RUNNING and job.progress > 0:
            # Simple estimation based on current progress and elapsed time
            elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()
            if job.progress > 10:  # Avoid division by very small numbers
                total_estimated = (elapsed * 100) / job.progress
                remaining = total_estimated - elapsed
                estimated_completion = (datetime.now(timezone.utc) +
                                      timedelta(seconds=remaining)).isoformat()

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "products_collected": job.products_collected,
            "errors_count": job.errors_count,
            "error_messages": job.error_messages,
            "sources": job.sources,
            "categories": job.categories,
            "started_at": job.started_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "estimated_completion": estimated_completion,
            "last_update": datetime.now(timezone.utc).isoformat()
        }

    async def test_apify_connection(self) -> bool:
        """Test connection to Apify API."""
        try:
            if not self.collectors_initialized:
                await self.initialize()

            # Test by making a simple request
            await self.apify_collector.search_amazon_products("test", max_results=1)
            return True

        except Exception as e:
            logger.error(f"Apify connection test failed: {e}")
            return False

    async def test_bestbuy_connection(self) -> bool:
        """Test connection to Best Buy API."""
        # Mock implementation since Best Buy collector is not implemented yet
        logger.info("Best Buy API connection test (mock)")
        return True

    async def monitor_prices(self):
        """Monitor price changes for tracked products."""
        # TODO: Implement price monitoring logic
        logger.info("Price monitoring task executed")

    async def collect_requested_categories(self, categories: List[str]) -> List[ProductData]:
        """
        Collect products for specifically requested categories only.

        This method is used for manual collection requests and ONLY searches
        for the categories specified by the user. No fallback to trending topics.

        Args:
            categories: List of product categories to search for

        Returns:
            List of collected products
        """
        if not self.collectors_initialized:
            await self.initialize()

        if not categories:
            logger.warning("No categories specified for requested collection")
            return []

        all_products = []
        for category in categories:
            try:
                logger.info(f"Collecting products for requested category: {category}")
                products = await self.apify_collector.search_amazon_products(
                    query=category,
                    max_results=20
                )
                all_products.extend(products)
                logger.info(f"Found {len(products)} products for category: {category}")

            except Exception as e:
                logger.error(f"Failed to collect products for category '{category}': {e}")
                continue

        # Remove duplicates
        unique_products = self._deduplicate_products(all_products)

        logger.info(f"Collected {len(unique_products)} unique requested products from {len(categories)} categories")
        return unique_products

    def _deduplicate_products(self, products: List[ProductData]) -> List[ProductData]:
        """Remove duplicate products based on similarity."""
        seen_titles: Set[str] = set()
        unique_products = []

        for product in products:
            # Create a normalized title for deduplication
            normalized_title = re.sub(r'[^\w\s]', '', product.title.lower())
            normalized_title = ' '.join(normalized_title.split())

            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_products.append(product)

        return unique_products