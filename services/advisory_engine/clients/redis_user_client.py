"""
Redis client for user historical data management.

This module provides a client for managing user data that grows over time:
- Conversation history
- Product recommendations
- Feedback history
- Recently viewed products

All data is stored in Redis with TTL for automatic cleanup.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json

import redis.asyncio as redis
from shared.logging import get_logger


logger = get_logger("advisory-service")


class RedisUserClient:
    """
    Client for managing user historical data in Redis.

    Uses Redis with AOF persistence to store:
    - Conversation history (last 7 days)
    - Recent recommendations (last 30 days)
    - Feedback history (last 90 days)
    - Recently viewed products (last 30 days)
    """

    def __init__(self, redis_url: str):
        """
        Initialize the Redis client.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
        """
        self.redis_url = redis_url
        self.client: Optional[redis.Redis] = None

        # TTL settings (in seconds)
        self.CONVERSATION_TTL = 7 * 24 * 60 * 60      # 7 days
        self.RECOMMENDATIONS_TTL = 30 * 24 * 60 * 60  # 30 days
        self.FEEDBACK_TTL = 90 * 24 * 60 * 60         # 90 days
        self.VIEWED_TTL = 30 * 24 * 60 * 60           # 30 days

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.client.ping()
            logger.info("Redis user client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self.client = None

    async def cleanup(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("Redis user client connection closed")

    # ==================== Conversation History ====================

    async def add_conversation_message(
        self,
        user_id: str,
        role: str,
        content: str
    ) -> bool:
        """
        Add a message to user's conversation history.

        Stores last 50 messages with 7-day TTL.

        Args:
            user_id: User identifier
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Redis client not available")
            return False

        try:
            key = f"user:{user_id}:conversations"
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add to list (right push)
            await self.client.rpush(key, json.dumps(message))

            # Keep only last 50 messages
            await self.client.ltrim(key, -50, -1)

            # Set TTL
            await self.client.expire(key, self.CONVERSATION_TTL)

            logger.debug(f"Added conversation message for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add conversation message: {e}")
            return False

    async def get_conversation_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get user's recent conversation history.

        Args:
            user_id: User identifier
            limit: Maximum number of messages to return (default: 10)

        Returns:
            List of conversation messages (most recent first)
        """
        if not self.client:
            logger.warning("Redis client not available")
            return []

        try:
            key = f"user:{user_id}:conversations"

            # Get last N messages
            messages = await self.client.lrange(key, -limit, -1)

            # Parse and return (reverse to get most recent first)
            parsed = [json.loads(msg) for msg in messages]
            parsed.reverse()

            return parsed

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    # ==================== Recommendations ====================

    async def save_recommendations(
        self,
        user_id: str,
        recommendations: List[Dict[str, Any]]
    ) -> bool:
        """
        Save user's recent product recommendations.

        Args:
            user_id: User identifier
            recommendations: List of product recommendations

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Redis client not available")
            return False

        try:
            key = f"user:{user_id}:recommendations"

            # Store compact version
            compact_recs = []
            for rec in recommendations[:10]:  # Keep last 10
                compact_recs.append({
                    "product_id": rec.get("product_id"),
                    "title": rec.get("title"),
                    "price": rec.get("price"),
                    "match_score": rec.get("match_score"),
                    "timestamp": datetime.utcnow().isoformat()
                })

            # Add each recommendation
            for rec in compact_recs:
                await self.client.rpush(key, json.dumps(rec))

            # Keep only last 10
            await self.client.ltrim(key, -10, -1)

            # Set TTL
            await self.client.expire(key, self.RECOMMENDATIONS_TTL)

            logger.debug(f"Saved {len(compact_recs)} recommendations for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save recommendations: {e}")
            return False

    async def get_recent_recommendations(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get user's recent recommendations.

        Args:
            user_id: User identifier
            limit: Maximum number of recommendations to return

        Returns:
            List of recent recommendations
        """
        if not self.client:
            logger.warning("Redis client not available")
            return []

        try:
            key = f"user:{user_id}:recommendations"

            # Get last N recommendations
            recs = await self.client.lrange(key, -limit, -1)

            # Parse and return (reverse to get most recent first)
            parsed = [json.loads(rec) for rec in recs]
            parsed.reverse()

            return parsed

        except Exception as e:
            logger.error(f"Failed to get recent recommendations: {e}")
            return []

    # ==================== Feedback ====================

    async def save_feedback(
        self,
        user_id: str,
        product_id: str,
        rating: float,
        feedback: Optional[str] = None
    ) -> bool:
        """
        Save user feedback on product recommendations.

        Args:
            user_id: User identifier
            product_id: Product identifier
            rating: User rating (0-5)
            feedback: Optional feedback text

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Redis client not available")
            return False

        try:
            key = f"user:{user_id}:feedback"

            feedback_entry = {
                "product_id": product_id,
                "rating": rating,
                "feedback": feedback,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add to list
            await self.client.rpush(key, json.dumps(feedback_entry))

            # Keep only last 50 feedback entries
            await self.client.ltrim(key, -50, -1)

            # Set TTL
            await self.client.expire(key, self.FEEDBACK_TTL)

            logger.debug(f"Saved feedback for user {user_id} on product {product_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False

    async def get_feedback_history(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get user's feedback history.

        Args:
            user_id: User identifier
            limit: Maximum number of feedback entries to return

        Returns:
            List of feedback entries
        """
        if not self.client:
            logger.warning("Redis client not available")
            return []

        try:
            key = f"user:{user_id}:feedback"

            # Get last N feedback entries
            feedback = await self.client.lrange(key, -limit, -1)

            # Parse and return (reverse to get most recent first)
            parsed = [json.loads(fb) for fb in feedback]
            parsed.reverse()

            return parsed

        except Exception as e:
            logger.error(f"Failed to get feedback history: {e}")
            return []

    # ==================== Recently Viewed ====================

    async def add_viewed_product(
        self,
        user_id: str,
        product_id: str,
        product_title: str
    ) -> bool:
        """
        Track recently viewed products.

        Args:
            user_id: User identifier
            product_id: Product identifier
            product_title: Product title

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Redis client not available")
            return False

        try:
            key = f"user:{user_id}:viewed"

            viewed_entry = {
                "product_id": product_id,
                "title": product_title,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add to list
            await self.client.rpush(key, json.dumps(viewed_entry))

            # Keep only last 20
            await self.client.ltrim(key, -20, -1)

            # Set TTL
            await self.client.expire(key, self.VIEWED_TTL)

            return True

        except Exception as e:
            logger.error(f"Failed to track viewed product: {e}")
            return False

    async def get_recently_viewed(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get user's recently viewed products.

        Args:
            user_id: User identifier
            limit: Maximum number of products to return

        Returns:
            List of recently viewed products
        """
        if not self.client:
            logger.warning("Redis client not available")
            return []

        try:
            key = f"user:{user_id}:viewed"

            # Get last N viewed products
            viewed = await self.client.lrange(key, -limit, -1)

            # Parse and return (reverse to get most recent first)
            parsed = [json.loads(v) for v in viewed]
            parsed.reverse()

            return parsed

        except Exception as e:
            logger.error(f"Failed to get recently viewed: {e}")
            return []

    # ==================== User Data Summary ====================

    async def get_user_data_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of all user data in Redis.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with counts and recent data
        """
        if not self.client:
            return {
                "conversations_count": 0,
                "recommendations_count": 0,
                "feedback_count": 0,
                "viewed_count": 0
            }

        try:
            # Get counts
            conv_count = await self.client.llen(f"user:{user_id}:conversations")
            recs_count = await self.client.llen(f"user:{user_id}:recommendations")
            feedback_count = await self.client.llen(f"user:{user_id}:feedback")
            viewed_count = await self.client.llen(f"user:{user_id}:viewed")

            return {
                "conversations_count": conv_count,
                "recommendations_count": recs_count,
                "feedback_count": feedback_count,
                "viewed_count": viewed_count
            }

        except Exception as e:
            logger.error(f"Failed to get user data summary: {e}")
            return {
                "conversations_count": 0,
                "recommendations_count": 0,
                "feedback_count": 0,
                "viewed_count": 0
            }

    async def clear_all_user_data(self, user_id: str) -> bool:
        """
        Clear all user historical data from Redis.

        Deletes:
        - Conversation history
        - Recommendations
        - Feedback
        - Recently viewed products

        Args:
            user_id: User identifier

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Redis client not available")
            return False

        try:
            # Delete all user keys
            keys_to_delete = [
                f"user:{user_id}:conversations",
                f"user:{user_id}:recommendations",
                f"user:{user_id}:feedback",
                f"user:{user_id}:viewed"
            ]

            deleted_count = 0
            for key in keys_to_delete:
                result = await self.client.delete(key)
                deleted_count += result

            logger.info(f"Cleared {deleted_count} data keys for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear user data: {e}")
            return False