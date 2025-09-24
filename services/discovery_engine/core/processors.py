"""
Product data processing and embedding generation.

This module handles text embedding generation for semantic search using
sentence-transformers with clean fallback mechanisms for development.

Key Features:
- Official sentence-transformers integration
- Efficient batch processing with configurable sizes
- Clean text preprocessing and normalization
- Graceful fallback for development environments
- Product text combination for optimal embeddings
"""

import asyncio
import hashlib
import re
from typing import List, Dict, Any, Optional

import numpy as np

from shared.logging import get_logger
from config.settings import DiscoverySettings
from .collectors import ProductData


logger = get_logger("discovery-processors")
settings = DiscoverySettings()


class EmbeddingProcessor:
    """
    Sentence transformer-based embedding processor for product data.

    This class uses the official sentence-transformers library to generate
    high-quality embeddings for semantic search. Includes graceful fallback
    for development environments where the model may not be available.
    """

    def __init__(self):
        """Initialize the embedding processor."""
        self.model = None
        self.model_name = settings.embedding_model_name
        self.vector_size = settings.embedding_vector_size
        self.is_initialized = False

    async def initialize(self):
        """
        Initialize the sentence transformer model.

        Uses official sentence-transformers patterns with proper
        error handling and fallback mechanisms.
        """
        if self.is_initialized:
            return

        try:
            # Official sentence-transformers initialization pattern
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Verify vector size matches expected dimensions
            test_embedding = self.model.encode(["test"])
            actual_size = len(test_embedding[0])

            if actual_size != self.vector_size:
                logger.warning(
                    f"Model vector size ({actual_size}) doesn't match configured size ({self.vector_size}). "
                    f"Using actual model size."
                )
                self.vector_size = actual_size

            logger.info(f"Sentence transformer initialized successfully (vector size: {self.vector_size})")
            self.is_initialized = True

        except ImportError:
            logger.warning(
                "sentence-transformers not available. Using deterministic mock embeddings for development."
            )
            self.model = None
            self.is_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize sentence transformer: {e}")
            logger.info("Falling back to mock embeddings")
            self.model = None
            self.is_initialized = True

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using sentence transformers.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector as list of floats

        Note:
            Uses official sentence-transformers encode() method with
            normalization. Falls back to deterministic mock for development.
        """
        if not self.is_initialized:
            await self.initialize()

        if not text or not text.strip():
            return [0.0] * self.vector_size

        # Preprocess text for optimal embedding
        cleaned_text = self._preprocess_text(text)

        try:
            if self.model:
                # Official sentence-transformers pattern
                embedding = self.model.encode(
                    cleaned_text,
                    normalize_embeddings=True,  # Cosine similarity optimization
                    convert_to_numpy=True
                )
                return embedding.tolist()
            else:
                # Deterministic mock for development
                return self._generate_mock_embedding(cleaned_text)

        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            return self._generate_mock_embedding(cleaned_text)

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of normalized embedding vectors

        Note:
            Uses sentence-transformers batch processing for efficiency.
            Processes in smaller batches to manage memory usage.
        """
        if not self.is_initialized:
            await self.initialize()

        if not texts:
            return []

        all_embeddings = []

        # Process in batches for memory efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Preprocess all texts in batch
            cleaned_texts = [self._preprocess_text(text) for text in batch_texts]

            try:
                if self.model:
                    # Official batch processing pattern
                    batch_embeddings = self.model.encode(
                        cleaned_texts,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    all_embeddings.extend(batch_embeddings.tolist())
                else:
                    # Mock batch processing
                    batch_embeddings = [
                        self._generate_mock_embedding(text)
                        for text in cleaned_texts
                    ]
                    all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Error in batch embedding generation: {e}")
                # Fallback to individual processing
                batch_embeddings = [
                    self._generate_mock_embedding(text)
                    for text in cleaned_texts
                ]
                all_embeddings.extend(batch_embeddings)

            # Allow other async tasks to run
            await asyncio.sleep(0)

        logger.info(f"Generated {len(all_embeddings)} embeddings in {len(texts)//batch_size + 1} batches")
        return all_embeddings

    def process_product_for_embedding(self, product: ProductData) -> str:
        """
        Process product data into optimized text for embedding generation.

        Args:
            product: ProductData instance

        Returns:
            Combined and optimized text string for embedding

        Note:
            Combines multiple product fields in order of importance
            for optimal semantic search performance.
        """
        text_parts = []

        # Title (highest importance for search)
        if product.title:
            text_parts.append(product.title)

        # Brand and category (important for filtering and categorization)
        if product.brand:
            text_parts.append(f"Brand: {product.brand}")

        if product.category:
            text_parts.append(f"Category: {product.category}")

        # Description (rich semantic content)
        if product.description:
            # Limit description length for optimal embedding
            desc = product.description[:400]
            text_parts.append(desc)

        # Key features (important product attributes)
        if product.key_features:
            # Take top 5 features for relevance
            features = " ".join(product.key_features[:5])
            text_parts.append(f"Features: {features}")

        # Specifications (technical details)
        if product.specifications:
            spec_parts = []
            for key, value in list(product.specifications.items())[:8]:  # Limit specs
                spec_parts.append(f"{key}: {value}")
            if spec_parts:
                text_parts.append(" ".join(spec_parts))

        # Combine with separator optimized for embeddings
        combined_text = " | ".join(text_parts)
        return combined_text

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for optimal embedding generation.

        Args:
            text: Raw text to preprocess

        Returns:
            Cleaned and normalized text ready for embedding

        Note:
            Applies cleaning optimized for sentence transformer models
            while preserving semantic meaning.
        """
        if not text:
            return ""

        # Basic normalization
        text = text.strip()

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove problematic characters but preserve semantic ones
        text = re.sub(r'[^\w\s\-\.\$\%\(\)\/\+]', ' ', text)

        # Limit length for transformer models (typical 512 token limit)
        max_length = 512
        if len(text) > max_length:
            # Smart truncation at word boundary
            text = text[:max_length]
            last_space = text.rfind(' ')
            if last_space > max_length * 0.8:  # If close to end, truncate at word
                text = text[:last_space]

        return text.strip()

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic mock embedding for development.

        Args:
            text: Input text

        Returns:
            Normalized mock embedding vector

        Note:
            Creates consistent embeddings for the same input text
            using hash-based seeding for reproducibility.
        """
        # Create deterministic seed from text content
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash, 16) % (2**32)

        # Generate deterministic random vector
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, self.vector_size)

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    async def cleanup(self):
        """
        Clean up embedding processor resources.

        Releases model memory and clears caches.
        """
        if self.model:
            # Clear model from memory
            del self.model
            self.model = None

        self.is_initialized = False
        logger.info("Embedding processor cleaned up")