"""
Knowledge Engine configuration settings.

This module defines all configuration parameters for the Knowledge Engine
microservice, including external API credentials, model settings, and
service-specific parameters.

Configuration is loaded from environment variables with the KNOWLEDGE_ prefix.
"""

from typing import Optional
from pydantic import Field, validator

from shared.config import BaseSettings


class KnowledgeSettings(BaseSettings):
    """
    Configuration settings for the Knowledge Engine service.

    This class manages all configuration for LLM training, model management,
    and inference operations including external API credentials and training parameters.

    Attributes:
        Service Configuration:
            port: Service port number
            model_storage_path: Directory for storing trained models

        External APIs:
            openai_api_key: OpenAI API key for data generation and fallback inference
            openai_model: Default OpenAI model for operations
            huggingface_token: Hugging Face token for model downloads
            wandb_api_key: Weights & Biases API key for experiment tracking
            wandb_project: WandB project name
            wandb_entity: WandB entity/team name

        Model Training:
            base_model: Default base model for fine-tuning
            max_length: Maximum sequence length for training
            batch_size: Training batch size
            learning_rate: Learning rate for training
            max_epochs: Maximum training epochs

        Storage:
            aws_access_key_id: AWS access key for S3 storage
            aws_secret_access_key: AWS secret key for S3 storage
            s3_bucket: S3 bucket for model storage
            s3_region: AWS region for S3 bucket
    """

    # Service Configuration
    port: int = Field(default=8001, description="Service port number")
    model_storage_path: str = Field(
        default="/models",
        description="Directory for storing trained models"
    )

    # External API Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-3.5-turbo",
        description="Default OpenAI model"
    )
    openai_training_model: str = Field(
        default="gpt-4",
        description="OpenAI model for training data generation"
    )

    huggingface_token: str = Field(..., description="Hugging Face token")
    base_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="Default base model for fine-tuning"
    )

    wandb_api_key: str = Field(..., description="Weights & Biases API key")
    wandb_project: str = Field(
        default="shopsense-training",
        description="WandB project name"
    )
    wandb_entity: str = Field(
        default="shopsense-team",
        description="WandB entity/team name"
    )

    # Training Configuration
    max_length: int = Field(
        default=2048,
        ge=512,
        le=4096,
        description="Maximum sequence length"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Training batch size"
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0,
        description="Learning rate for training"
    )
    max_epochs: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum training epochs"
    )

    # QLoRA Configuration
    use_qlora: bool = Field(
        default=True,
        description="Use QLoRA for efficient fine-tuning"
    )
    lora_r: int = Field(
        default=16,
        ge=4,
        le=64,
        description="LoRA rank parameter"
    )
    lora_alpha: int = Field(
        default=32,
        ge=8,
        le=128,
        description="LoRA alpha parameter"
    )
    lora_dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="LoRA dropout rate"
    )

    # Storage Configuration
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key for S3 storage"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret key for S3 storage"
    )
    s3_bucket: str = Field(
        default="shopsense-models",
        description="S3 bucket for model storage"
    )
    s3_region: str = Field(
        default="us-east-1",
        description="AWS region for S3 bucket"
    )

    # Performance Configuration
    max_concurrent_trainings: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum concurrent training jobs"
    )
    inference_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Inference timeout in seconds"
    )
    model_cache_size: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of models to keep in memory"
    )

    class Config:
        """Pydantic configuration."""
        env_prefix = "KNOWLEDGE_"
        env_file = "config/.env"
        case_sensitive = False

    @validator('model_storage_path')
    def validate_model_storage_path(cls, v: str) -> str:
        """
        Validate model storage path.

        Args:
            v: Model storage path

        Returns:
            Validated path

        Raises:
            ValueError: If path is invalid
        """
        if not v or v.isspace():
            raise ValueError('model_storage_path cannot be empty')
        return v.strip()

    @validator('wandb_project')
    def validate_wandb_project(cls, v: str) -> str:
        """
        Validate WandB project name.

        Args:
            v: Project name

        Returns:
            Validated project name

        Raises:
            ValueError: If project name is invalid
        """
        if not v or not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('wandb_project must contain only alphanumeric characters, hyphens, and underscores')
        return v

    @validator('base_model')
    def validate_base_model(cls, v: str) -> str:
        """
        Validate base model name.

        Args:
            v: Model name

        Returns:
            Validated model name

        Raises:
            ValueError: If model name is invalid
        """
        supported_models = [
            "gpt2",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "facebook/blenderbot-400M-distill"
        ]

        if v not in supported_models:
            raise ValueError(f'base_model must be one of {supported_models}')
        return v

    @validator('learning_rate')
    def validate_learning_rate(cls, v: float) -> float:
        """
        Validate learning rate is within reasonable bounds.

        Args:
            v: Learning rate

        Returns:
            Validated learning rate

        Raises:
            ValueError: If learning rate is out of bounds
        """
        if not (1e-6 <= v <= 1e-2):
            raise ValueError('learning_rate must be between 1e-6 and 1e-2')
        return v

    def get_model_config(self) -> dict:
        """
        Get model configuration dictionary.

        Returns:
            Dictionary with model configuration parameters
        """
        return {
            "base_model": self.base_model,
            "max_length": self.max_length,
            "use_qlora": self.use_qlora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout
        }

    def get_training_config(self) -> dict:
        """
        Get training configuration dictionary.

        Returns:
            Dictionary with training parameters
        """
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "max_length": self.max_length
        }

    def get_wandb_config(self) -> dict:
        """
        Get WandB configuration dictionary.

        Returns:
            Dictionary with WandB settings
        """
        return {
            "project": self.wandb_project,
            "entity": self.wandb_entity,
            "api_key": self.wandb_api_key
        }

    def get_storage_config(self) -> dict:
        """
        Get storage configuration dictionary.

        Returns:
            Dictionary with storage settings
        """
        config = {
            "local_path": self.model_storage_path,
            "s3_bucket": self.s3_bucket,
            "s3_region": self.s3_region
        }

        if self.aws_access_key_id and self.aws_secret_access_key:
            config.update({
                "aws_access_key_id": self.aws_access_key_id,
                "aws_secret_access_key": self.aws_secret_access_key,
                "use_s3": True
            })
        else:
            config["use_s3"] = False

        return config

    def is_production_ready(self) -> bool:
        """
        Check if configuration is ready for production use.

        Returns:
            True if all production requirements are met
        """
        production_requirements = [
            self.openai_api_key,
            self.huggingface_token,
            self.wandb_api_key,
            self.aws_access_key_id,  # S3 storage for production
            self.aws_secret_access_key
        ]

        return all(req for req in production_requirements)