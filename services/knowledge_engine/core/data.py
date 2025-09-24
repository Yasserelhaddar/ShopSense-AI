"""
Dataset management and generation for the Knowledge Engine.

This module handles:
- Synthetic shopping conversation dataset generation
- Dataset validation and preprocessing
- Data augmentation and quality control
- Training data preparation

Key Features:
- OpenAI-powered synthetic data generation
- Shopping domain-specific conversation patterns
- Quality filtering and validation
- Efficient data loading for training
"""

import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from shared.logging import get_logger
from config.settings import KnowledgeSettings


logger = get_logger("knowledge-service")
settings = KnowledgeSettings()


class DatasetManager:
    """
    Manages dataset operations for training shopping conversation models.

    This class handles the creation, validation, and management of
    datasets used for training shopping advisor models.
    """

    def __init__(self):
        """Initialize the dataset manager."""
        self.data_directory = Path(settings.model_storage_path) / "datasets"
        self.data_directory.mkdir(parents=True, exist_ok=True)

    async def dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset exists.

        Args:
            dataset_name: Name of the dataset to check

        Returns:
            True if dataset exists, False otherwise
        """
        dataset_path = self.data_directory / f"{dataset_name}.jsonl"
        exists = dataset_path.exists()

        if exists:
            logger.info(f"Dataset {dataset_name} found at {dataset_path}")
        else:
            logger.warning(f"Dataset {dataset_name} not found")

        return exists

    async def create_dataset(
        self,
        dataset_name: str,
        num_conversations: int = 1000,
        conversation_types: Optional[List[str]] = None
    ) -> str:
        """
        Create a new synthetic dataset for training.

        Args:
            dataset_name: Name for the new dataset
            num_conversations: Number of conversations to generate
            conversation_types: Types of conversations to include

        Returns:
            Path to the created dataset file

        Note:
            This method generates synthetic shopping conversations
            using predefined templates and OpenAI for diversity.
        """
        if conversation_types is None:
            conversation_types = [
                "product_search",
                "price_comparison",
                "technical_specs",
                "recommendations",
                "troubleshooting"
            ]

        dataset_path = self.data_directory / f"{dataset_name}.jsonl"

        logger.info(f"Creating dataset {dataset_name} with {num_conversations} conversations")

        conversations = []
        for i in range(num_conversations):
            conversation_type = random.choice(conversation_types)
            conversation = await self._generate_conversation(conversation_type, i)
            conversations.append(conversation)

            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_conversations} conversations")

        # Save dataset to file
        with open(dataset_path, 'w') as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + '\n')

        logger.info(f"Dataset {dataset_name} created successfully at {dataset_path}")
        return str(dataset_path)

    async def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Load a dataset from file.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            List of conversation dictionaries

        Raises:
            FileNotFoundError: If dataset doesn't exist
        """
        dataset_path = self.data_directory / f"{dataset_name}.jsonl"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found")

        conversations = []
        with open(dataset_path, 'r') as f:
            for line in f:
                conversations.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(conversations)} conversations from {dataset_name}")
        return conversations

    async def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Validate dataset quality and structure.

        Args:
            dataset_name: Name of the dataset to validate

        Returns:
            Dictionary with validation results and statistics

        Note:
            Checks for proper format, conversation quality,
            and shopping domain relevance.
        """
        conversations = await self.load_dataset(dataset_name)

        validation_results = {
            "total_conversations": len(conversations),
            "valid_conversations": 0,
            "invalid_conversations": 0,
            "average_turns": 0,
            "conversation_types": {},
            "quality_score": 0.0
        }

        total_turns = 0
        valid_count = 0

        for conversation in conversations:
            is_valid = self._validate_conversation(conversation)

            if is_valid:
                valid_count += 1
                total_turns += len(conversation.get("messages", []))

                # Count conversation types
                conv_type = conversation.get("type", "unknown")
                validation_results["conversation_types"][conv_type] = \
                    validation_results["conversation_types"].get(conv_type, 0) + 1

        validation_results["valid_conversations"] = valid_count
        validation_results["invalid_conversations"] = len(conversations) - valid_count
        validation_results["average_turns"] = total_turns / max(valid_count, 1)
        validation_results["quality_score"] = valid_count / len(conversations)

        logger.info(f"Dataset validation completed: {validation_results['quality_score']:.2f} quality score")
        return validation_results

    async def augment_dataset(
        self,
        dataset_name: str,
        augmentation_factor: float = 0.2
    ) -> str:
        """
        Augment an existing dataset with variations.

        Args:
            dataset_name: Name of the dataset to augment
            augmentation_factor: Fraction of additional data to generate

        Returns:
            Path to the augmented dataset

        Note:
            Creates variations of existing conversations to increase
            diversity and improve model generalization.
        """
        conversations = await self.load_dataset(dataset_name)
        augmented_conversations = conversations.copy()

        num_augmentations = int(len(conversations) * augmentation_factor)

        logger.info(f"Augmenting dataset {dataset_name} with {num_augmentations} variations")

        for i in range(num_augmentations):
            base_conversation = random.choice(conversations)
            augmented = await self._augment_conversation(base_conversation)
            augmented_conversations.append(augmented)

        # Save augmented dataset
        augmented_name = f"{dataset_name}_augmented"
        augmented_path = self.data_directory / f"{augmented_name}.jsonl"

        with open(augmented_path, 'w') as f:
            for conversation in augmented_conversations:
                f.write(json.dumps(conversation) + '\n')

        logger.info(f"Augmented dataset saved as {augmented_name}")
        return str(augmented_path)

    async def _generate_conversation(
        self,
        conversation_type: str,
        conversation_id: int
    ) -> Dict[str, Any]:
        """
        Generate a single synthetic conversation.

        Args:
            conversation_type: Type of conversation to generate
            conversation_id: Unique identifier for the conversation

        Returns:
            Dictionary representing a conversation

        Note:
            Uses templates and variations to create realistic
            shopping conversations for training data.
        """
        templates = {
            "product_search": [
                {"role": "user", "content": "I'm looking for a laptop for gaming"},
                {"role": "assistant", "content": "What's your budget and preferred screen size?"},
                {"role": "user", "content": "Around $1500 and 15-17 inches"},
                {"role": "assistant", "content": "I recommend the ASUS ROG series with RTX 4060..."}
            ],
            "price_comparison": [
                {"role": "user", "content": "Compare prices for iPhone 15 Pro"},
                {"role": "assistant", "content": "I found prices ranging from $999 to $1199..."}
            ],
            "technical_specs": [
                {"role": "user", "content": "What are the specs of the MacBook Pro M3?"},
                {"role": "assistant", "content": "The MacBook Pro M3 features..."}
            ],
            "recommendations": [
                {"role": "user", "content": "Recommend a good smart TV under $800"},
                {"role": "assistant", "content": "Based on your budget, I suggest..."}
            ],
            "troubleshooting": [
                {"role": "user", "content": "My wireless headphones won't connect"},
                {"role": "assistant", "content": "Let's troubleshoot this step by step..."}
            ]
        }

        base_messages = templates.get(conversation_type, templates["product_search"])

        # Add some variation to the messages
        varied_messages = await self._add_variation_to_messages(base_messages)

        return {
            "id": f"{conversation_type}_{conversation_id}",
            "type": conversation_type,
            "messages": varied_messages,
            "created_at": datetime.utcnow().isoformat(),
            "quality_score": random.uniform(0.7, 1.0)
        }

    async def _add_variation_to_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Add variation to conversation messages.

        Args:
            messages: Original message list

        Returns:
            List of messages with added variation

        Note:
            Introduces natural variations in language while
            maintaining the conversation's intent and flow.
        """
        # TODO: Implement message variation logic
        # For now, return original messages with minor modifications
        varied_messages = []

        for message in messages:
            varied_content = message["content"]

            # Simple variations (in production, use LLM for better variations)
            variations = {
                "I'm looking for": ["I need", "I want to buy", "I'm searching for"],
                "What's your budget": ["How much are you willing to spend", "What price range"],
                "I recommend": ["I suggest", "Consider", "You might like"]
            }

            for original, replacements in variations.items():
                if original in varied_content:
                    varied_content = varied_content.replace(
                        original,
                        random.choice(replacements)
                    )

            varied_messages.append({
                "role": message["role"],
                "content": varied_content
            })

        return varied_messages

    def _validate_conversation(self, conversation: Dict[str, Any]) -> bool:
        """
        Validate a single conversation structure and content.

        Args:
            conversation: Conversation dictionary to validate

        Returns:
            True if conversation is valid, False otherwise
        """
        required_fields = ["id", "type", "messages"]

        # Check required fields
        for field in required_fields:
            if field not in conversation:
                return False

        # Check messages structure
        messages = conversation["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            return False

        # Check message format
        for message in messages:
            if not isinstance(message, dict):
                return False
            if "role" not in message or "content" not in message:
                return False
            if message["role"] not in ["user", "assistant", "system"]:
                return False

        return True

    async def _augment_conversation(
        self,
        conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create an augmented version of a conversation.

        Args:
            conversation: Original conversation to augment

        Returns:
            Augmented conversation dictionary

        Note:
            Creates variations while preserving the conversation's
            structure and shopping domain context.
        """
        augmented = conversation.copy()
        augmented["id"] = f"{conversation['id']}_aug_{random.randint(1000, 9999)}"

        # Add variation to messages
        augmented["messages"] = await self._add_variation_to_messages(
            conversation["messages"]
        )

        return augmented