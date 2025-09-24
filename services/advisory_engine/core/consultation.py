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

    def __init__(self, knowledge_client, discovery_client):
        """Initialize the consultation engine."""
        self.knowledge_client = knowledge_client
        self.discovery_client = discovery_client

    async def initialize(self):
        """Initialize the consultation engine."""
        logger.info("Consultation engine initialized")

    async def provide_consultation(
        self,
        conversation_history: List[Dict[str, str]],
        user_context: Optional[Dict[str, Any]] = None,
        specific_questions: Optional[List[str]] = None
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
        # Analyze conversation to extract intent and requirements
        intent_analysis = self._analyze_conversation_intent(conversation_history)

        # Generate consultation response
        consultation_response = await self.knowledge_client.generate_consultation_response(
            conversation_history=conversation_history,
            user_context=user_context
        )

        # Get relevant product recommendations
        recommendations = []
        if intent_analysis.get("product_search_needed"):
            search_query = intent_analysis.get("extracted_query", "")
            if search_query:
                products = await self.discovery_client.search_products(
                    query=search_query,
                    limit=3
                )
                recommendations = products

        # Generate next steps
        next_steps = self._generate_next_steps(intent_analysis, specific_questions)

        return {
            "advice": consultation_response.get("response", ""),
            "recommendations": recommendations,
            "next_steps": next_steps,
            "confidence": consultation_response.get("confidence", 0.8),
            "reasoning": consultation_response.get("reasoning", "")
        }

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

        # Simple intent detection (TODO: Implement more sophisticated NLP)
        search_keywords = ["looking for", "need", "want", "buy", "purchase", "recommend"]
        product_search_needed = any(keyword in last_user_message.lower() for keyword in search_keywords)

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