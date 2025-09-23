"""
Knowledge Engine API schemas.

This module defines Pydantic models for request and response validation
in the Knowledge Engine API. These schemas ensure type safety and provide
automatic documentation for the API endpoints.

Models:
- TrainingRequest: Configuration for training jobs
- TrainingResponse: Training job status and information
- ModelInfo: Model metadata and performance metrics
- InferenceRequest: Inference parameters and messages
- InferenceResponse: Inference results and metadata
- HealthResponse: Service health status
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    A single message in a conversation.

    Attributes:
        role: The role of the message sender (user, assistant, system)
        content: The content of the message
    """
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class TrainingConfig(BaseModel):
    """
    Configuration parameters for model training.

    Attributes:
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        max_length: Maximum sequence length
        use_qlora: Whether to use QLoRA for efficient training
    """
    epochs: int = Field(default=3, ge=1, le=10, description="Number of training epochs")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    batch_size: int = Field(default=1, ge=1, le=32, description="Training batch size")
    max_length: int = Field(default=2048, ge=512, le=4096, description="Maximum sequence length")
    use_qlora: bool = Field(default=True, description="Use QLoRA for efficient training")


class TrainingRequest(BaseModel):
    """
    Request model for starting a training job.

    Attributes:
        dataset_name: Name of the dataset to use for training
        base_model: Base model to fine-tune
        training_config: Training configuration parameters
        model_name: Optional name for the resulting model
    """
    dataset_name: str = Field(..., description="Name of the training dataset")
    base_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="Base model to fine-tune"
    )
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    model_name: Optional[str] = Field(None, description="Name for the resulting model")


class TrainingResponse(BaseModel):
    """
    Response model for training job creation.

    Attributes:
        job_id: Unique identifier for the training job
        status: Current status of the job
        estimated_duration: Estimated time to completion
        wandb_url: URL to WandB experiment tracking
    """
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    estimated_duration: str = Field(..., description="Estimated completion time")
    wandb_url: Optional[str] = Field(None, description="WandB experiment URL")


class PerformanceMetrics(BaseModel):
    """
    Model performance metrics.

    Attributes:
        accuracy: Model accuracy score
        perplexity: Model perplexity score
        bleu_score: BLEU score for text generation
        rouge_score: ROUGE score for text generation
    """
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Accuracy score")
    perplexity: Optional[float] = Field(None, gt=0, description="Perplexity score")
    bleu_score: Optional[float] = Field(None, ge=0, le=1, description="BLEU score")
    rouge_score: Optional[float] = Field(None, ge=0, le=1, description="ROUGE score")


class ModelInfo(BaseModel):
    """
    Information about an available model.

    Attributes:
        id: Unique model identifier
        base_model: Base model that was fine-tuned
        created_at: When the model was created
        performance: Performance metrics
        status: Current model status
        size_mb: Model size in megabytes
    """
    id: str = Field(..., description="Unique model identifier")
    base_model: str = Field(..., description="Base model name")
    created_at: datetime = Field(..., description="Model creation timestamp")
    performance: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")
    status: str = Field(..., description="Model status")
    size_mb: Optional[float] = Field(None, gt=0, description="Model size in MB")


class InferenceRequest(BaseModel):
    """
    Request model for model inference.

    Attributes:
        messages: List of conversation messages
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        context: Additional context for inference
    """
    messages: List[Message] = Field(..., description="Conversation messages")
    max_tokens: int = Field(default=500, ge=1, le=2000, description="Maximum tokens")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0, le=1, description="Top-p sampling")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class InferenceResponse(BaseModel):
    """
    Response model for inference results.

    Attributes:
        response: Generated text response
        model_used: ID of the model used
        tokens_used: Number of tokens used
        processing_time_ms: Processing time in milliseconds
        finish_reason: Reason inference stopped
    """
    response: str = Field(..., description="Generated response")
    model_used: str = Field(..., description="Model ID used")
    tokens_used: int = Field(..., ge=0, description="Tokens consumed")
    processing_time_ms: float = Field(..., ge=0, description="Processing time")
    finish_reason: Optional[str] = Field(None, description="Reason inference stopped")


class HealthResponse(BaseModel):
    """
    Response model for health check.

    Attributes:
        service: Service health status
        external_apis: Status of external API dependencies
        overall: Overall service health
        timestamp: Health check timestamp
    """
    service: str = Field(..., description="Service status")
    external_apis: Dict[str, str] = Field(..., description="External API statuses")
    overall: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")