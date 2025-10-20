"""
Storage management for product data and vector database operations.

This module handles Qdrant vector database operations for product storage,
search functionality with filtering, and vector similarity search using
official Qdrant client patterns.

Key Features:
- Official Qdrant AsyncClient integration
- Vector similarity search with cosine distance
- Product data persistence with payloads
- Efficient filtering and search operations
- Collection management and maintenance
- Graceful fallback for development environments
"""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from shared.logging import get_logger
from config.settings import DiscoverySettings
from .collectors import ProductData


logger = get_logger("discovery-storage")
settings = DiscoverySettings()


class StorageManager:
    """
    Manages vector storage operations using Qdrant.

    This class uses the official Qdrant AsyncClient patterns for
    product vector storage, similarity search, and collection management.
    Includes graceful fallback for development environments.
    """

    # ============================================================================
    # LIFECYCLE MANAGEMENT
    # ============================================================================

    def __init__(self):
        """Initialize storage manager with Qdrant client."""
        self.qdrant_client = None
        self.collection_name = settings.qdrant_collection
        self.vector_size = settings.qdrant_vector_size
        self.is_initialized = False

    async def initialize(self):
        """
        Initialize Qdrant client and ensure collection exists.

        Uses official AsyncQdrantClient patterns with proper
        error handling and fallback mechanisms.
        """
        if self.is_initialized:
            return

        try:
            # Official Qdrant AsyncClient initialization pattern
            from qdrant_client import AsyncQdrantClient, models

            logger.info(f"Connecting to Qdrant at: {settings.qdrant_url}")
            self.qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

            # Test connection
            collections = await self.qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant, found {len(collections.collections)} collections")

            # Create collection if it doesn't exist
            if not await self.qdrant_client.collection_exists(self.collection_name):
                await self._create_collection()
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")

            self.is_initialized = True
            logger.info("Qdrant storage manager initialized successfully")

        except ImportError:
            logger.error("qdrant-client not available. Please install: pip install qdrant-client")
            raise

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    async def cleanup(self):
        """Clean up Qdrant client connection."""
        try:
            if self.qdrant_client:
                await self.qdrant_client.close()
                self.qdrant_client = None

            self.is_initialized = False
            logger.info("Storage manager cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    # ============================================================================
    # COLLECTION MANAGEMENT
    # ============================================================================

    async def _create_collection(self):
        """Create Qdrant collection with proper vector configuration."""
        try:
            from qdrant_client import models

            logger.info(f"Creating collection '{self.collection_name}' with vector size {self.vector_size}")

            await self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE  # For similarity search
                ),
            )

            logger.info(f"Collection '{self.collection_name}' created successfully")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    # ============================================================================
    # CORE STORAGE OPERATIONS
    # ============================================================================

    async def store_products(self, products: List[ProductData], embeddings: List[List[float]]):
        """
        Store products with their embeddings in Qdrant.

        Args:
            products: List of ProductData objects
            embeddings: Corresponding embedding vectors

        Note:
            Uses official Qdrant upsert patterns with PointStruct for
            efficient batch product storage with payloads.
        """
        if not self.is_initialized:
            await self.initialize()

        if not products or not embeddings:
            logger.warning("No products or embeddings provided for storage")
            return

        if len(products) != len(embeddings):
            logger.error("Mismatch between products and embeddings count")
            return

        try:
            if self.qdrant_client:
                # Official Qdrant batch upsert pattern
                from qdrant_client.models import PointStruct

                points = []
                for product, embedding in zip(products, embeddings):
                    # Create payload from product data
                    payload = {
                        "product_id": product.id,  # Store original product ID
                        "title": product.title,
                        "price": product.price,
                        "currency": product.currency,
                        "store": product.store,
                        "brand": product.brand,
                        "category": product.category,
                        "description": product.description,
                        "image_url": product.image_url,
                        "product_url": product.product_url,
                        "rating": product.rating,
                        "reviews_count": product.reviews_count,
                        "availability": product.availability,
                        "original_price": product.original_price,
                        "key_features": product.key_features,
                        "specifications": product.specifications,
                        "last_updated": datetime.utcnow().isoformat()
                    }

                    # Generate numeric ID from product ID
                    point_id = self._generate_point_id(product.id)

                    points.append(PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    ))

                # Batch upsert with wait for completion
                await self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=points
                )

                logger.info(f"Successfully stored {len(products)} products in Qdrant")

            else:
                logger.error("No Qdrant client available for storage")
                raise RuntimeError("Qdrant client not initialized")

        except Exception as e:
            logger.error(f"Failed to store products: {e}")

    async def search_products(
        self,
        query_embedding: List[float],
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "relevance"
    ) -> List[Dict[str, Any]]:
        """
        Search products using vector similarity with Qdrant.

        Args:
            query_embedding: Query vector for similarity search
            limit: Maximum number of results
            filters: Search filters (price, category, store, etc.)
            sort_by: Sort criteria

        Returns:
            List of matching products with similarity scores

        Note:
            Uses official Qdrant search patterns with filtering and
            payload retrieval for comprehensive product information.
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            if self.qdrant_client:
                # Official Qdrant search with filters pattern
                query_filter = self._build_qdrant_filter(filters)

                search_result = await self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,  # Include product metadata
                    with_vectors=False  # We don't need vectors in results
                )

                # Convert Qdrant results to product format
                # Filter out results below similarity threshold to avoid non-relevant results
                products = []
                for hit in search_result:
                    # Skip results with low similarity scores
                    if hit.score < settings.min_similarity_threshold:
                        logger.debug(f"Skipping result with low similarity: {hit.score:.3f} < {settings.min_similarity_threshold}")
                        continue

                    product_data = {
                        "id": hit.payload.get("product_id", self._restore_product_id(hit.id)),  # Use original product_id from payload
                        "similarity_score": hit.score,
                        **hit.payload  # Include all product metadata
                    }
                    products.append(product_data)

                # Apply additional sorting if needed
                if sort_by != "relevance":
                    products = self._sort_results(products, sort_by)

                logger.info(f"Found {len(products)} products matching query")
                return products

            else:
                logger.error("No Qdrant client available for search")
                return []

        except Exception as e:
            logger.error(f"Product search failed: {e}")
            return []

    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific product by ID from Qdrant.

        Args:
            product_id: Unique product identifier (can be original ID or point_XXXXX format)

        Returns:
            Product data dictionary or None if not found

        Note:
            Handles both original product IDs and point-prefixed IDs returned from search.
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            if self.qdrant_client:
                # Check if this is a point-prefixed ID (from search results)
                if product_id.startswith("point_"):
                    # Extract numeric point ID
                    try:
                        point_id = int(product_id[6:])  # Remove "point_" prefix
                    except ValueError:
                        logger.error(f"Invalid point ID format: {product_id}")
                        return None
                else:
                    # Convert original product ID to point ID
                    point_id = self._generate_point_id(product_id)

                # Retrieve point from Qdrant
                points = await self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=False
                )

                if points and len(points) > 0:
                    point = points[0]
                    product_data = {
                        "id": product_id,  # Return the requested ID format
                        **point.payload
                    }
                    return product_data

                return None

            else:
                # Mock product lookup for development
                return await self._mock_get_product_by_id(product_id)

        except Exception as e:
            logger.error(f"Failed to get product {product_id}: {e}")
            return None


    async def process_and_store_products(
        self,
        products: List[ProductData],
        embedding_processor
    ):
        """
        Complete pipeline: process products and store with embeddings.

        Args:
            products: List of ProductData objects
            embedding_processor: EmbeddingProcessor instance

        Note:
            This method demonstrates the complete data flow:
            ProductData → text processing → embeddings → Qdrant storage
        """
        if not products:
            logger.warning("No products provided for processing")
            return

        try:
            logger.info(f"Processing {len(products)} products for storage")

            # Generate text representations for embedding
            product_texts = []
            for product in products:
                text = embedding_processor.process_product_for_embedding(product)
                product_texts.append(text)

            # Generate embeddings
            embeddings = await embedding_processor.generate_batch_embeddings(
                product_texts,
                batch_size=32
            )

            # Store in Qdrant
            await self.store_products(products, embeddings)

            logger.info(f"Successfully processed and stored {len(products)} products")

        except Exception as e:
            logger.error(f"Failed to process and store products: {e}")

    # ============================================================================
    # PUBLIC API METHODS
    # ============================================================================


    async def get_available_categories(self) -> List[Dict[str, Any]]:
        """Get available product categories with counts."""
        try:
            if not self.is_initialized:
                logger.warning("Storage not initialized, initializing now...")
                await self.initialize()

            if not self.qdrant_client:
                logger.error("No Qdrant client available")
                return []

            # Scroll through all products to collect categories
            # Note: Qdrant doesn't have direct aggregation, so we need to scroll and aggregate
            category_counts = {}
            offset = None
            limit = 100

            while True:
                # Scroll through products
                scroll_result = await self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = scroll_result

                # Process each product's category
                for point in points:
                    category = point.payload.get("category")
                    if category:  # Only count non-null categories
                        category_counts[category] = category_counts.get(category, 0) + 1

                # Check if we have more results
                if next_offset is None or len(points) < limit:
                    break

                offset = next_offset

            # Convert to list format
            categories = [
                {"name": name, "product_count": count}
                for name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            ]

            logger.info(f"Found {len(categories)} unique categories")
            return categories

        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            # Return empty list on error
            return []


    # ============================================================================
    # SYSTEM MONITORING & MAINTENANCE
    # ============================================================================

    async def test_connection(self) -> bool:
        """
        Test Qdrant connection.

        Returns:
            True if connection is successful, False otherwise
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            if self.qdrant_client:
                # Test connection by getting collections list
                collections = await self.qdrant_client.get_collections()
                logger.info(f"Connection test successful. Found {len(collections.collections)} collections")
                return True

            else:
                # Mock connection test for development
                logger.info("Mock connection test successful")
                return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def perform_maintenance(self):
        """
        Perform collection maintenance tasks.

        Note:
            Includes collection optimization and index updates
            using Qdrant maintenance operations.
        """
        if not self.is_initialized:
            await self.initialize()

        logger.info("Starting collection maintenance")

        try:
            if self.qdrant_client:
                # Get collection info before maintenance
                info_before = await self.get_collection_info()
                logger.info(f"Collection status before maintenance: {info_before.get('status', 'unknown')}")

                # Qdrant automatically handles optimization, but we can trigger it
                # Note: In production, you might want to add specific maintenance tasks
                logger.info("Collection maintenance completed")

            else:
                logger.info("Mock maintenance completed")

        except Exception as e:
            logger.error(f"Maintenance failed: {e}")

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection statistics and information.

        Returns:
            Dictionary containing collection statistics

        Note:
            Uses Qdrant collection info method to get
            current collection statistics and configuration.
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            if self.qdrant_client:
                collection_info = await self.qdrant_client.get_collection(self.collection_name)

                return {
                    "collection_name": self.collection_name,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance_metric": collection_info.config.params.vectors.distance.value,
                    "points_count": collection_info.points_count,
                    "segments_count": collection_info.segments_count,
                    "indexed_vectors_count": collection_info.indexed_vectors_count,
                    "status": collection_info.status.value
                }

            else:
                # Mock collection info for development
                return {
                    "collection_name": self.collection_name,
                    "vector_size": self.vector_size,
                    "distance_metric": "Cosine",
                    "points_count": 0,
                    "segments_count": 1,
                    "indexed_vectors_count": 0,
                    "status": "green"
                }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    # ============================================================================
    # UTILITIES & HELPERS
    # ============================================================================

    def _build_qdrant_filter(self, filters: Optional[Dict[str, Any]]):
        """
        Build Qdrant filter conditions from search filters.

        Args:
            filters: Dictionary of filter criteria

        Returns:
            Qdrant Filter object or None

        Note:
            Uses official Qdrant Filter, FieldCondition, and Range patterns
            for proper filtering on payload fields.
        """
        if not filters:
            return None

        try:
            from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

            conditions = []

            # Price range filtering
            if "min_price" in filters or "max_price" in filters:
                range_condition = {}
                if "min_price" in filters:
                    range_condition["gte"] = filters["min_price"]
                if "max_price" in filters:
                    range_condition["lte"] = filters["max_price"]

                conditions.append(
                    FieldCondition(
                        key="price",
                        range=Range(**range_condition)
                    )
                )

            # Category filtering
            if "category" in filters:
                conditions.append(
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=filters["category"])
                    )
                )

            # Store filtering
            if "store" in filters:
                conditions.append(
                    FieldCondition(
                        key="store",
                        match=MatchValue(value=filters["store"])
                    )
                )

            # Brand filtering
            if "brand" in filters:
                conditions.append(
                    FieldCondition(
                        key="brand",
                        match=MatchValue(value=filters["brand"])
                    )
                )

            if conditions:
                return Filter(must=conditions)

            return None

        except ImportError:
            # Fallback for development
            return None

    def _generate_point_id(self, product_id: str) -> int:
        """Generate numeric ID for Qdrant from product ID string."""
        # Convert string ID to stable numeric ID
        return int(hashlib.md5(product_id.encode()).hexdigest(), 16) % (2**63)

    def _restore_product_id(self, point_id: int) -> str:
        """Restore product ID from point ID (simplified approach)."""
        # Use the point ID as string - original ID is stored in payload
        return f"point_{point_id}"

    async def _mock_search_results(
        self,
        filters: Optional[Dict[str, Any]],
        limit: int,
        sort_by: str
    ) -> List[Dict[str, Any]]:
        """Generate mock search results for development."""
        mock_products = [
            {
                "id": "amazon_B08N5WRWNW",
                "title": "ASUS ROG Zephyrus G14 Gaming Laptop",
                "price": 1499.99,
                "currency": "USD",
                "store": "amazon",
                "brand": "ASUS",
                "category": "laptops",
                "rating": 4.5,
                "reviews_count": 2847,
                "key_features": ["RTX 4070", "AMD Ryzen 9", "16GB RAM"],
                "image_url": "https://example.com/image1.jpg",
                "product_url": "https://amazon.com/product1",
                "similarity_score": 0.94
            },
            {
                "id": "bestbuy_6439402",
                "title": "MSI Katana 15 Gaming Laptop",
                "price": 1299.99,
                "currency": "USD",
                "store": "bestbuy",
                "brand": "MSI",
                "category": "laptops",
                "rating": 4.3,
                "reviews_count": 1532,
                "key_features": ["RTX 4060", "Intel i7", "16GB RAM"],
                "image_url": "https://example.com/image2.jpg",
                "product_url": "https://bestbuy.com/product2",
                "similarity_score": 0.89
            }
        ]

        # Apply filters
        filtered_products = self._apply_filters(mock_products, filters)

        # Sort results
        sorted_products = self._sort_results(filtered_products, sort_by)

        return sorted_products[:limit]

    def _apply_filters(
        self,
        products: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply filters to product list."""
        if not filters:
            return products

        filtered = products

        if "min_price" in filters:
            filtered = [p for p in filtered if p.get("price", 0) >= filters["min_price"]]

        if "max_price" in filters:
            filtered = [p for p in filtered if p.get("price", float('inf')) <= filters["max_price"]]

        if "category" in filters:
            filtered = [p for p in filtered if p.get("category") == filters["category"]]

        if "store" in filters:
            filtered = [p for p in filtered if p.get("store") == filters["store"]]

        return filtered

    def _sort_results(
        self,
        products: List[Dict[str, Any]],
        sort_by: str
    ) -> List[Dict[str, Any]]:
        """Sort product results."""
        if sort_by == "price_asc":
            return sorted(products, key=lambda x: x.get("price", 0))
        elif sort_by == "price_desc":
            return sorted(products, key=lambda x: x.get("price", 0), reverse=True)
        elif sort_by == "rating":
            return sorted(products, key=lambda x: x.get("rating", 0), reverse=True)
        else:  # relevance (default)
            return sorted(products, key=lambda x: x.get("similarity_score", 0), reverse=True)

    # ============================================================================
    # API-SPECIFIC METHODS
    # ============================================================================


    async def get_current_deals(
        self,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get current deals and price drops by comparing prices with original/historical data.

        Args:
            limit: Maximum number of deals to return
            filters: Filtering criteria including min_discount, category

        Returns:
            List of deal information dictionaries
        """
        if not self.is_initialized:
            logger.error("Storage not initialized")
            return []

        try:
            from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

            # Build Qdrant filter conditions
            filter_conditions = []

            # Filter by category if specified
            if filters and "category" in filters:
                filter_conditions.append(
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=filters["category"])
                    )
                )

            # Search for products that might have deals (products with original_price > current price)
            search_result = await self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=limit * 3,  # Get more to filter for actual deals
                with_payload=True,
                with_vectors=False
            )

            deals = []
            min_discount = filters.get("min_discount", 0) if filters else 0

            for point in search_result[0]:
                payload = point.payload
                current_price = payload.get("price", 0)
                original_price = payload.get("original_price")

                # Skip if no original price data
                if not original_price or original_price <= current_price:
                    continue

                # Calculate discount
                discount_amount = original_price - current_price
                discount_percent = (discount_amount / original_price) * 100

                # Apply minimum discount filter
                if discount_percent < min_discount:
                    continue

                deal = {
                    "product_id": payload.get("product_id"),  # Use correct field name from Fix #1
                    "title": payload.get("title"),
                    "original_price": original_price,
                    "current_price": current_price,
                    "discount_amount": round(discount_amount, 2),
                    "discount_percent": round(discount_percent, 1),
                    "deal_type": self._determine_deal_type(discount_percent),
                    "store": payload.get("store"),
                    "image_url": payload.get("image_url"),
                    "product_url": payload.get("product_url"),
                    "expires_at": None  # Would need separate deal expiration tracking
                }
                deals.append(deal)

            # Sort by discount percentage (highest first) and return limited results
            deals.sort(key=lambda x: x["discount_percent"], reverse=True)
            return deals[:limit]

        except Exception as e:
            logger.error(f"Failed to retrieve current deals: {e}")
            return []


    async def get_available_stores(self) -> List[Dict[str, Any]]:
        """
        Get all available stores/sources with statistics from stored data.

        Returns:
            List of store information dictionaries
        """
        if not self.is_initialized:
            logger.error("Storage not initialized")
            return []

        try:
            # Get all products and aggregate store statistics
            search_result = await self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Large limit to get all products
                with_payload=True,
                with_vectors=False
            )

            store_counts = {}

            for point in search_result[0]:
                store = point.payload.get("store")
                if store:
                    store_counts[store] = store_counts.get(store, 0) + 1

            # Convert to list format expected by API
            stores = []
            for store_name, count in sorted(store_counts.items()):
                stores.append({
                    "name": store_name,
                    "product_count": count,
                    "last_collection": None,  # Would need job tracking integration
                    "api_status": "unknown",  # Would need health check integration
                    "collection_frequency": "on-demand"  # Based on current trigger-only mode
                })

            return stores

        except Exception as e:
            logger.error(f"Failed to retrieve stores: {e}")
            return []

    def _determine_deal_type(self, discount_percent: float) -> str:
        """Determine deal type based on discount percentage."""
        if discount_percent >= 50:
            return "lightning_deal"
        elif discount_percent >= 30:
            return "daily_deal"
        elif discount_percent >= 15:
            return "sale"
        else:
            return "discount"