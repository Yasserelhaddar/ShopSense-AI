"""
Clerk client for lightweight user preferences management.

This module provides a wrapper around the Clerk Backend API for managing
ONLY lightweight user configuration stored in Clerk's public_metadata.

Historical/growing data (conversations, feedback, recommendations) is stored in Redis.
"""

from typing import Dict, Any
from datetime import datetime

from clerk_backend_api import Clerk
from shared.logging import get_logger


logger = get_logger("advisory-service")


class ClerkUserClient:
    """
    Client for managing lightweight user preferences in Clerk's metadata.

    Uses Clerk's public_metadata to store ONLY:
    - Shopping preferences (priority, brands, budget, use_cases, style)
    - Tracking preferences (tracking_enabled flag)
    - Preferences update timestamp

    Total size: ~200-500 bytes per user (safe for metadata limits)

    Historical data (conversations, recommendations, feedback) is in Redis.
    """

    def __init__(self, secret_key: str):
        """
        Initialize the Clerk client.

        Args:
            secret_key: Clerk secret key for backend API
        """
        self.client = Clerk(bearer_auth=secret_key)
        self.initialized = True

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get lightweight user shopping preferences from Clerk metadata.

        Args:
            user_id: Clerk user ID

        Returns:
            Dictionary with shopping preferences only
        """
        try:
            user = self.client.users.get(user_id=user_id)
            metadata = user.public_metadata or {}

            return metadata.get("shopping_preferences", {})

        except Exception as e:
            logger.error(f"Failed to get user preferences for {user_id}: {e}")
            return {}

    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Update lightweight user shopping preferences in Clerk metadata.

        Only stores configuration data (priority, brands, budget, etc.).
        Does NOT store historical data (conversations, feedback, etc.).

        Args:
            user_id: Clerk user ID
            preferences: Shopping preferences to store

        Returns:
            True if successful, False otherwise
        """
        try:
            user = self.client.users.get(user_id=user_id)
            current_metadata = user.public_metadata or {}

            # Update only preferences and timestamp
            current_metadata["shopping_preferences"] = preferences
            current_metadata["preferences_updated_at"] = datetime.utcnow().isoformat()

            self.client.users.update(
                user_id=user_id,
                public_metadata=current_metadata
            )

            logger.info(f"Updated preferences for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update preferences for {user_id}: {e}")
            return False

    async def is_tracking_enabled(self, user_id: str) -> bool:
        """
        Check if activity tracking is enabled for this user.

        Args:
            user_id: Clerk user ID

        Returns:
            True if tracking is enabled (default), False if user opted out
        """
        try:
            user = self.client.users.get(user_id=user_id)
            metadata = user.public_metadata or {}

            # Default to True (opt-out model)
            return metadata.get("tracking_enabled", True)

        except Exception as e:
            logger.error(f"Failed to get tracking preference for {user_id}: {e}")
            # Default to True on error (allow tracking)
            return True

    async def set_tracking_enabled(self, user_id: str, enabled: bool) -> bool:
        """
        Set activity tracking preference for user.

        Args:
            user_id: Clerk user ID
            enabled: True to enable tracking, False to disable

        Returns:
            True if successful, False otherwise
        """
        try:
            user = self.client.users.get(user_id=user_id)
            current_metadata = user.public_metadata or {}

            # Update tracking preference
            current_metadata["tracking_enabled"] = enabled
            current_metadata["preferences_updated_at"] = datetime.utcnow().isoformat()

            self.client.users.update(
                user_id=user_id,
                public_metadata=current_metadata
            )

            logger.info(f"Updated tracking preference for user {user_id}: {enabled}")
            return True

        except Exception as e:
            logger.error(f"Failed to update tracking preference for {user_id}: {e}")
            return False