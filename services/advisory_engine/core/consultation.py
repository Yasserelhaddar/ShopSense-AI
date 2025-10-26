"""
Consultation engine for conversational shopping advice.

This module handles conversational interactions and provides
personalized shopping consultation based on user conversations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from shared.logging import get_logger


logger = get_logger("advisory-service")


class ConsultationEngine:
    """
    Consultation engine for providing conversational shopping advice.

    This engine analyzes conversation history and provides contextual,
    personalized shopping advice and recommendations.
    """

    def __init__(self, knowledge_client, discovery_client, clerk_client=None, redis_client=None):
        """
        Initialize the consultation engine.

        Args:
            knowledge_client: Client for Knowledge Engine API
            discovery_client: Client for Discovery Engine API
            clerk_client: Client for Clerk user metadata (lightweight preferences, optional)
            redis_client: Client for Redis user data (conversation history, optional)
        """
        self.knowledge_client = knowledge_client
        self.discovery_client = discovery_client
        self.clerk_client = clerk_client
        self.redis_client = redis_client

    async def initialize(self):
        """Initialize the consultation engine."""
        logger.info("Consultation engine initialized")

    async def provide_consultation(
        self,
        conversation_history: List[Dict[str, str]],
        user_context: Optional[Dict[str, Any]] = None,
        specific_questions: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Provide shopping consultation based on conversation.

        Args:
            conversation_history: Previous conversation messages
            user_context: Additional user context
            specific_questions: Specific questions to address

        Returns:
            Dictionary with consultation advice and recommendations
        """
        # Convert ConversationMessage objects to dictionaries if needed
        conversation_dicts = []
        for msg in conversation_history:
            if hasattr(msg, 'model_dump'):  # Pydantic v2
                conversation_dicts.append(msg.model_dump())
            elif hasattr(msg, 'dict'):  # Pydantic v1
                conversation_dicts.append(msg.dict())
            else:
                conversation_dicts.append(msg)  # Already a dict

        # Analyze conversation to extract intent and requirements
        intent_analysis = self._analyze_conversation_intent(conversation_dicts)

        # Get relevant product recommendations FIRST
        recommendations = []
        if intent_analysis.get("product_search_needed"):
            search_query = intent_analysis.get("extracted_query", "")
            if search_query:
                products = await self.discovery_client.search_products(
                    query=search_query,
                    limit=5
                )
                recommendations = products

        # Generate consultation response with product context
        # Build enhanced user context with products
        enhanced_context = user_context.copy() if user_context else {}
        if recommendations:
            enhanced_context["available_products"] = recommendations

        consultation_response = await self.knowledge_client.generate_consultation_response(
            conversation_history=conversation_dicts,
            user_context=enhanced_context
        )

        # Generate next steps
        next_steps = self._generate_next_steps(intent_analysis, specific_questions)

        return {
            "advice": consultation_response.get("response", ""),
            "recommendations": recommendations,
            "next_steps": next_steps,
            "confidence": consultation_response.get("confidence", 0.8),
            "reasoning": consultation_response.get("reasoning", "")
        }

    async def save_conversation_message(
        self,
        user_id: str,
        role: str,
        content: str
    ) -> bool:
        """
        Save a conversation message to Redis.

        Args:
            user_id: User identifier
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            True if successful, False otherwise
        """
        if self.redis_client:
            return await self.redis_client.add_conversation_message(
                user_id=user_id,
                role=role,
                content=content
            )
        return False

    async def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get user's conversation history from Redis.

        Args:
            user_id: User identifier
            limit: Maximum number of messages to return

        Returns:
            List of conversation messages
        """
        if self.redis_client:
            return await self.redis_client.get_conversation_history(user_id, limit=limit)
        return []

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Consultation engine cleaned up")

    def _analyze_conversation_intent(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze conversation to extract user intent and requirements.

        Args:
            conversation_history: Conversation messages

        Returns:
            Dictionary with intent analysis results
        """
        if not conversation_history:
            return {"product_search_needed": False}

        # Get the latest user message
        last_user_message = ""
        for msg in reversed(conversation_history):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break

        # Enhanced intent detection with multiple signals
        message_lower = last_user_message.lower()

        # Action keywords (explicit purchase/recommendation intent)
        action_keywords = ["looking for", "need", "want", "buy", "purchase", "recommend", "suggest", "help me find"]
        has_action_intent = any(keyword in message_lower for keyword in action_keywords)

        # Product categories
        product_categories = [
            "laptop", "computer", "phone", "smartphone", "tablet", "headphone", "headset",
            "camera", "monitor", "keyboard", "mouse", "speaker", "tv", "watch", "smartwatch",
            "console", "gaming", "printer", "router", "charger", "cable", "case", "bag"
        ]
        has_product_category = any(category in message_lower for category in product_categories)

        # Question patterns (implicit product queries)
        question_patterns = ["which", "what", "best", "top", "good", "better", "recommend", "should i"]
        has_question_pattern = any(pattern in message_lower for pattern in question_patterns)

        # Determine if product search is needed
        product_search_needed = (
            has_action_intent or
            (has_product_category and has_question_pattern) or
            (has_product_category and "?" in last_user_message)
        )

        return {
            "product_search_needed": product_search_needed,
            "extracted_query": last_user_message if product_search_needed else "",
            "conversation_length": len(conversation_history),
            "user_engaged": len(conversation_history) > 2
        }

    def _generate_next_steps(
        self,
        intent_analysis: Dict[str, Any],
        specific_questions: Optional[List[str]]
    ) -> List[str]:
        """
        Generate suggested next steps for the user.

        Args:
            intent_analysis: Analysis of conversation intent
            specific_questions: Any specific questions to address

        Returns:
            List of suggested next steps
        """
        next_steps = []

        if intent_analysis.get("product_search_needed"):
            next_steps.append("Review the recommended products above")
            next_steps.append("Let me know your budget range for more targeted suggestions")
            next_steps.append("Share any specific features or requirements you have")
        else:
            next_steps.append("Tell me more about what you're looking for")
            next_steps.append("Share your budget range and preferences")

        if specific_questions:
            next_steps.append("I'll address your specific questions in detail")

        return next_steps[:3]  # Return top 3 next steps