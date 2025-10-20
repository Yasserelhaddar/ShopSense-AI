"""
Model training and management for the Knowledge Engine.

This module handles the core training functionality including:
- Model fine-tuning with QLoRA for efficiency
- Training job management and monitoring
- Model storage and versioning
- Inference on trained models

Key Features:
- Efficient training with QLoRA (4-bit quantization)
- WandB integration for experiment tracking
- S3/local storage for model persistence
- Training progress monitoring
"""

import asyncio
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

from shared.logging import get_logger
from core.data import DatasetManager
from core.platform import PlatformDetector
from config.settings import KnowledgeSettings


logger = get_logger("knowledge-service")
settings = KnowledgeSettings()

# Initialize platform detector (singleton)
platform_detector = PlatformDetector()


class TrainingManager:
    """
    Manages model training, storage, and inference operations.

    This class coordinates all aspects of model lifecycle including
    training job scheduling, model storage, and inference requests.
    """

    def __init__(self):
        """Initialize the training manager with required dependencies."""
        self.dataset_manager = DatasetManager()
        self.active_jobs: Dict[str, Dict] = {}
        self.model_cache: Dict[str, tuple] = {}  # model_id -> (model, tokenizer, timestamp)
        self.cache_timestamps: List[tuple] = []  # (timestamp, model_id) for LRU

    async def validate_training_config(self, request) -> Dict[str, Any]:
        """
        Validate and prepare training configuration.

        Args:
            request: Training request object

        Returns:
            Dict containing validated and platform-adjusted training configuration

        Raises:
            ValueError: If configuration is invalid
        """
        config = {
            "dataset_name": request.dataset_name,
            "base_model": request.base_model,
            "training_config": request.training_config.dict(),
            "model_name": request.model_name or f"shopsense-{int(time.time())}"
        }

        # Validate dataset exists
        if not await self.dataset_manager.dataset_exists(request.dataset_name):
            raise ValueError(f"Dataset {request.dataset_name} not found")

        # Validate base model
        if not await self._validate_base_model(request.base_model):
            raise ValueError(f"Base model {request.base_model} not supported")

        # Validate and adjust configuration for platform capabilities
        config = platform_detector.validate_and_adjust_config(config)

        logger.info(f"Training configuration validated: {config['model_name']}")
        return config

    async def start_training(
        self,
        config: Dict[str, Any],
        background_tasks
    ) -> str:
        """
        Start a new training job.

        Args:
            config: Validated training configuration
            background_tasks: FastAPI background tasks manager

        Returns:
            str: Unique job ID for tracking

        Raises:
            RuntimeError: If training cannot be started
        """
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        # Initialize job tracking
        self.active_jobs[job_id] = {
            "status": "starting",
            "started_at": datetime.utcnow(),
            "config": config,
            "progress": 0
        }

        # Schedule training in background
        background_tasks.add_task(self._run_training_job, job_id, config)

        logger.info(f"Training job {job_id} scheduled")
        return job_id

    async def _run_training_job(self, job_id: str, config: Dict[str, Any]):
        """
        Execute the training job in background.

        Args:
            job_id: Unique job identifier
            config: Training configuration

        Note:
            This method runs the actual training process including:
            - Dataset preparation
            - Model initialization with QLoRA
            - Training loop with monitoring
            - Model saving and evaluation
        """
        try:
            self.active_jobs[job_id]["status"] = "preparing_data"

            # TODO: Implement dataset preparation
            await self._prepare_training_data(config["dataset_name"])

            self.active_jobs[job_id]["status"] = "initializing_model"

            # TODO: Implement model initialization
            await self._initialize_model(config["base_model"], config)

            self.active_jobs[job_id]["status"] = "training"

            # TODO: Implement training loop
            await self._train_model(job_id, config)

            self.active_jobs[job_id]["status"] = "saving_model"

            # TODO: Implement model saving
            await self._save_model(job_id, config["model_name"])

            self.active_jobs[job_id]["status"] = "completed"
            self.active_jobs[job_id]["completed_at"] = datetime.utcnow()

            logger.info(f"Training job {job_id} completed successfully")

        except Exception as e:
            self.active_jobs[job_id]["status"] = "failed"
            self.active_jobs[job_id]["error"] = str(e)
            logger.error(f"Training job {job_id} failed: {e}")

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with metadata.

        Returns:
            List of model information dictionaries

        Note:
            Scans model storage and returns metadata for each
            available model including performance metrics.
        """
        models = []

        try:
            import json

            # Path to models directory
            models_dir = os.path.join(settings.model_storage_path, "models")

            # Check if models directory exists
            if not os.path.exists(models_dir):
                logger.warning(f"Models directory not found: {models_dir}")
                return models

            # Scan for model directories
            for model_name in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_name)

                # Skip if not a directory
                if not os.path.isdir(model_path):
                    continue

                # Load metadata if available
                metadata_path = os.path.join(model_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                        # Convert ISO timestamp to datetime if needed
                        if isinstance(metadata.get("created_at"), str):
                            from dateutil import parser
                            metadata["created_at"] = parser.isoparse(metadata["created_at"])

                        models.append(metadata)
                        logger.debug(f"Loaded metadata for model {model_name}")

                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {model_name}: {e}")
                else:
                    # Create basic metadata if file doesn't exist
                    logger.warning(f"No metadata.json found for {model_name}")

            logger.info(f"Found {len(models)} available models")
            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model information dictionary or None if not found
        """
        # TODO: Implement model retrieval from storage
        models = await self.list_available_models()
        for model in models:
            if model["id"] == model_id:
                return model
        return None

    async def run_inference(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run inference on a trained model with OpenAI fallback.

        Args:
            model_id: ID of model to use
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary with inference results

        Note:
            This method attempts to use the specified trained model.
            If the model is unavailable or fails, it falls back to OpenAI.
        """
        start_time = time.time()

        try:
            # Load model from cache or filesystem
            model, tokenizer = await self._load_model_for_inference(model_id)

            # Format messages for the model
            prompt = self._format_messages_for_inference(messages)

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            import torch

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode response (remove input prompt)
            input_length = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_length:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

            processing_time = (time.time() - start_time) * 1000

            logger.info(f"Model {model_id} inference completed in {processing_time:.1f}ms")

            return {
                "content": response_text.strip(),
                "tokens_used": len(outputs[0]),
                "processing_time_ms": processing_time,
                "model_used": model_id
            }

        except FileNotFoundError:
            logger.warning(f"Model {model_id} not found, using OpenAI fallback")
            return await self._fallback_openai_inference(messages, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Inference failed for model {model_id}: {e}, using OpenAI fallback")
            # Try OpenAI as final fallback
            return await self._fallback_openai_inference(messages, max_tokens, temperature)

    async def _fallback_openai_inference(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Fallback to OpenAI when custom model unavailable.

        Args:
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary with inference results

        Note:
            Uses OpenAI's GPT models as a fallback when trained
            models are unavailable or fail.
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.openai_api_key)
            start_time = time.time()

            # Add system message for shopping advisor context
            system_message = {
                "role": "system",
                "content": "You are a helpful shopping advisor. Provide product recommendations, answer questions about products, compare options, and help users make informed purchase decisions."
            }

            # Prepare messages with system context
            full_messages = [system_message] + messages

            response = await client.chat.completions.create(
                model=settings.openai_model,  # gpt-3.5-turbo by default
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            processing_time = (time.time() - start_time) * 1000

            logger.info(f"OpenAI fallback inference completed in {processing_time:.1f}ms")

            return {
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "processing_time_ms": processing_time
            }

        except Exception as e:
            logger.error(f"OpenAI fallback inference failed: {e}")
            # Return error response
            return {
                "content": "I apologize, but I'm unable to process your request at this time. Please try again later.",
                "tokens_used": 0,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }

    async def test_openai_connection(self):
        """
        Test connection to OpenAI API.

        Raises:
            Exception: If connection fails
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.openai_api_key)

            # Simple test call - list available models
            models = await client.models.list()
            logger.info("OpenAI connection successful")

        except Exception as e:
            logger.error(f"OpenAI connection failed: {e}")
            raise ConnectionError(f"OpenAI API unavailable: {str(e)}")

    async def test_wandb_connection(self):
        """
        Test connection to WandB.

        Raises:
            Exception: If connection fails

        Note:
            WandB connection is optional. If not configured,
            training will continue with local logging only.
        """
        try:
            import wandb

            # Check if API key is configured
            if not settings.wandb_api_key:
                logger.warning("WandB API key not configured - skipping test")
                return

            # Login to WandB
            wandb.login(key=settings.wandb_api_key, relogin=True)

            # Test API access
            api = wandb.Api()
            _ = api.viewer  # This will fail if not authenticated

            logger.info("WandB connection successful")

        except ImportError:
            logger.warning("WandB not installed - experiment tracking will be disabled")
        except Exception as e:
            logger.warning(f"WandB connection failed: {e}")
            # Don't raise - WandB is optional
            logger.info("Training will continue with local logging only")

    async def test_huggingface_connection(self):
        """
        Test connection to HuggingFace Hub.

        Raises:
            Exception: If connection fails

        Note:
            Tests authentication and ability to download models from
            HuggingFace Hub using the configured token.
        """
        try:
            from huggingface_hub import HfApi
            from transformers import AutoTokenizer

            # Check if HF token is configured
            if not settings.huggingface_token:
                logger.warning("HuggingFace token not configured")
                raise ConnectionError("HuggingFace token not set in settings")

            # Test API access
            api = HfApi(token=settings.huggingface_token)

            # Verify token by getting user info
            user_info = api.whoami()
            logger.info(f"HuggingFace connection successful (user: {user_info.get('name', 'unknown')})")

            # Test model download capability with a small tokenizer
            # Use GPT-2 tokenizer as it's small and widely available
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "gpt2",
                    token=settings.huggingface_token,
                    cache_dir=os.path.join(settings.model_storage_path, "cache")
                )
                logger.info("HuggingFace model download capability verified")
            except Exception as e:
                logger.warning(f"HuggingFace model download test failed: {e}")
                # Don't fail the health check for this - token auth is more important

        except ImportError as e:
            logger.error(f"HuggingFace libraries not installed: {e}")
            raise ConnectionError("HuggingFace libraries (transformers, huggingface_hub) not installed")
        except Exception as e:
            logger.error(f"HuggingFace connection failed: {e}")
            raise ConnectionError(f"HuggingFace Hub unavailable: {str(e)}")

    async def test_model_storage(self):
        """
        Test access to model storage.

        Raises:
            Exception: If storage access fails

        Note:
            Verifies both read and write access to the model storage
            directory. Creates the directory if it doesn't exist.
        """
        try:
            # Ensure storage directory exists
            if not os.path.exists(settings.model_storage_path):
                os.makedirs(settings.model_storage_path, exist_ok=True)
                logger.info(f"Created model storage directory: {settings.model_storage_path}")

            # Test write access
            test_file = os.path.join(settings.model_storage_path, ".write_test")
            with open(test_file, "w") as f:
                f.write("write test")

            # Test read access
            with open(test_file, "r") as f:
                content = f.read()

            # Clean up test file
            os.remove(test_file)

            logger.info(f"Model storage access successful: {settings.model_storage_path}")

        except PermissionError as e:
            logger.error(f"Permission denied for model storage: {e}")
            raise PermissionError(f"Cannot access model storage at {settings.model_storage_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Model storage access failed: {e}")
            raise RuntimeError(f"Model storage unavailable: {str(e)}")

    async def _validate_base_model(self, model_name: str) -> bool:
        """
        Validate that a base model is supported.

        Args:
            model_name: Name of the base model

        Returns:
            True if model is supported, False otherwise
        """
        supported_models = [
            "gpt2",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "microsoft/DialoGPT-medium"
        ]
        return model_name in supported_models

    async def _prepare_training_data(self, dataset_name: str):
        """
        Prepare training data for the specified dataset.

        Args:
            dataset_name: Name of the dataset to prepare

        Returns:
            HuggingFace Dataset object ready for training

        Note:
            Converts conversation JSON data to instruction-tuned format
            compatible with the base model's expected input format.
        """
        logger.info(f"Preparing training data for {dataset_name}")

        try:
            from datasets import Dataset

            # Load conversations from DatasetManager
            conversations = await self.dataset_manager.load_dataset(dataset_name)
            logger.info(f"Loaded {len(conversations)} conversations from {dataset_name}")

            # Format conversations for training
            formatted_data = []
            for i, conv in enumerate(conversations):
                # Convert conversation messages to training text
                text = self._format_conversation_for_training(conv["messages"])
                formatted_data.append({"text": text})

                if (i + 1) % 100 == 0:
                    logger.debug(f"Formatted {i + 1}/{len(conversations)} conversations")

            # Create HuggingFace Dataset
            dataset = Dataset.from_list(formatted_data)

            logger.info(f"Prepared {len(dataset)} training examples")
            return dataset

        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise RuntimeError(f"Data preparation failed: {str(e)}")

    def _format_conversation_for_training(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation messages as training text.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Formatted training text string

        Note:
            Uses Mistral instruction format:
            <s>[INST] {user_message} [/INST] {assistant_response}</s>
        """
        formatted = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"<s>[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted += f" {msg['content']}</s>"
        return formatted

    async def _initialize_model(self, base_model: str, config: Dict):
        """
        Initialize model with platform-optimized configuration.

        Args:
            base_model: Base model name
            config: Full configuration including training_config and platform settings

        Note:
            This method loads the base model from HuggingFace Hub with:
            - Platform-specific device mapping and dtype
            - Optional 4-bit quantization (BitsAndBytes) for memory efficiency
            - LoRA adapters (PEFT) for parameter-efficient fine-tuning
            - Gradient checkpointing for reduced memory usage
        """
        training_config = config["training_config"]
        platform_config = config.get("platform", {})

        logger.info(
            f"Loading base model: {base_model} | QLoRA={training_config['use_qlora']} | "
            f"Device={platform_config.get('device', 'auto')} | "
            f"dtype={platform_config.get('torch_dtype', 'auto')}"
        )

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # QLoRA configuration (4-bit quantization)
            if training_config["use_qlora"]:
                logger.info("Configuring 4-bit quantization with BitsAndBytes")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True  # Nested quantization for extra memory savings
                )
            else:
                bnb_config = None

            # Get platform-specific torch dtype
            torch_dtype_str = platform_config.get("torch_dtype", "float32")
            if torch_dtype_str == "float16":
                torch_dtype = torch.float16
            elif torch_dtype_str == "float32":
                torch_dtype = torch.float32
            elif torch_dtype_str == "auto":
                torch_dtype = "auto"
            else:
                torch_dtype = torch.float32

            device_map = platform_config.get("device_map", "auto")

            # Load base model from HuggingFace Hub
            logger.info(f"Downloading model from HuggingFace Hub: {base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map=device_map,  # Platform-optimized device placement
                trust_remote_code=True,
                token=settings.huggingface_token,
                cache_dir=os.path.join(settings.model_storage_path, "cache"),
                torch_dtype=torch_dtype  # Platform-optimized dtype
            )

            # Load tokenizer
            logger.info(f"Loading tokenizer for {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                token=settings.huggingface_token,
                trust_remote_code=True,
                cache_dir=os.path.join(settings.model_storage_path, "cache")
            )

            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

            # Prepare model for k-bit training (QLoRA)
            if training_config["use_qlora"]:
                logger.info("Preparing model for k-bit training")
                model = prepare_model_for_kbit_training(model)

                # LoRA configuration
                lora_config = LoraConfig(
                    r=settings.lora_r,  # Rank of LoRA matrices
                    lora_alpha=settings.lora_alpha,  # Scaling factor
                    lora_dropout=settings.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
                    inference_mode=False
                )

                logger.info(f"Applying LoRA with r={settings.lora_r}, alpha={settings.lora_alpha}")
                model = get_peft_model(model, lora_config)

                # Log trainable parameters
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_pct = 100 * trainable_params / total_params

                logger.info(
                    f"Model parameters: {trainable_params:,} trainable / {total_params:,} total "
                    f"({trainable_pct:.2f}% trainable)"
                )

            # Store in instance for training
            self.current_model = model
            self.current_tokenizer = tokenizer

            logger.info(f"Model initialization completed successfully")

        except ImportError as e:
            logger.error(f"Missing required library: {e}")
            raise ImportError(
                f"Required library not installed: {e}. "
                "Run: uv sync to install dependencies"
            )
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize model {base_model}: {str(e)}")

    async def _train_model(self, job_id: str, config: Dict[str, Any]):
        """
        Execute the training loop with QLoRA.

        Args:
            job_id: Job identifier for progress tracking
            config: Training configuration

        Note:
            Uses SFTTrainer (Supervised Fine-Tuning Trainer) from TRL library
            with WandB logging for experiment tracking and progress monitoring.
        """
        logger.info(f"Starting training for job {job_id}")

        try:
            from transformers import TrainingArguments, TrainerCallback
            from trl import SFTTrainer
            import wandb

            # Prepare training dataset
            train_dataset = await self._prepare_training_data(config["dataset_name"])

            # Training arguments
            output_dir = os.path.join(settings.model_storage_path, "checkpoints", job_id)
            os.makedirs(output_dir, exist_ok=True)

            # Get platform-optimized configuration
            platform_config = config.get("platform", {})
            use_fp16 = platform_config.get("fp16", False)
            optimizer = platform_config.get("optimizer", "adamw_torch")

            logger.info(
                f"Training configuration: FP16={use_fp16}, "
                f"Optimizer={optimizer}, Device={platform_config.get('device', 'unknown')}"
            )

            # Initialize WandB BEFORE TrainingArguments to control project/entity
            wandb_run = None
            if settings.wandb_api_key:
                try:
                    wandb_run = wandb.init(
                        project=settings.wandb_project,
                        entity=settings.wandb_entity,
                        name=job_id,
                        config={
                            "base_model": config["base_model"],
                            "dataset": config["dataset_name"],
                            **config["training_config"],
                            "platform": platform_config
                        },
                        tags=["shopsense", "qlora", "shopping-advisor"]
                    )
                    self.active_jobs[job_id]["wandb_url"] = wandb_run.url
                    logger.info(f"WandB tracking initialized: {wandb_run.url}")
                except Exception as e:
                    logger.warning(f"WandB initialization failed: {e}, continuing without tracking")

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=config["training_config"]["epochs"],
                per_device_train_batch_size=config["training_config"]["batch_size"],
                gradient_accumulation_steps=4,  # Effective batch size = batch_size * 4
                learning_rate=config["training_config"]["learning_rate"],
                logging_steps=10,
                save_steps=100,
                save_total_limit=3,
                fp16=use_fp16,  # Platform-optimized mixed precision
                warmup_ratio=0.05,
                lr_scheduler_type="cosine",
                optim=optimizer,  # Platform-optimized optimizer
                max_grad_norm=0.3,
                report_to="wandb" if settings.wandb_api_key else "none",
                run_name=job_id,
                logging_first_step=True,
                logging_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=False
            )

            # Custom callback for progress updates
            class ProgressCallback(TrainerCallback):
                def __init__(self, job_id, manager):
                    self.job_id = job_id
                    self.manager = manager

                def on_epoch_end(self, args, state, control, **kwargs):
                    epoch = state.epoch
                    progress = (epoch / args.num_train_epochs) * 100
                    self.manager.active_jobs[self.job_id]["progress"] = progress
                    self.manager.active_jobs[self.job_id]["current_epoch"] = int(epoch)
                    logger.info(f"Job {self.job_id} epoch {int(epoch)} completed: {progress:.1f}%")

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        # Update job metrics
                        if "loss" in logs:
                            self.manager.active_jobs[self.job_id]["train_loss"] = logs["loss"]
                        if "learning_rate" in logs:
                            self.manager.active_jobs[self.job_id]["learning_rate"] = logs["learning_rate"]

                def on_step_end(self, args, state, control, **kwargs):
                    # Update progress based on steps
                    if state.max_steps > 0:
                        progress = (state.global_step / state.max_steps) * 100
                        self.manager.active_jobs[self.job_id]["progress"] = progress

            # Initialize SFT Trainer for conversation fine-tuning
            trainer = SFTTrainer(
                model=self.current_model,
                args=training_args,
                train_dataset=train_dataset,
                processing_class=self.current_tokenizer,  # Renamed from 'tokenizer' in TRL 0.24.0
                formatting_func=lambda x: x["text"]  # Extract text field from dataset
            )

            # Add progress callback
            trainer.add_callback(ProgressCallback(job_id, self))

            # Start training
            logger.info(f"Training started for job {job_id} with {len(train_dataset)} examples")
            trainer.train()

            # Save final model
            final_model_dir = os.path.join(settings.model_storage_path, "checkpoints", job_id, "final")
            logger.info(f"Saving final model to {final_model_dir}")
            trainer.save_model(final_model_dir)
            self.current_tokenizer.save_pretrained(final_model_dir)

            # Finish WandB run
            if wandb_run:
                wandb.finish()

            logger.info(f"Training completed successfully for job {job_id}")

        except ImportError as e:
            logger.error(f"Missing required library for training: {e}")
            raise ImportError(
                f"Required library not installed: {e}. "
                "Run: uv sync to install training dependencies"
            )
        except Exception as e:
            logger.error(f"Training failed for job {job_id}: {e}")
            raise RuntimeError(f"Training failed: {str(e)}")

    async def _save_model(self, job_id: str, model_name: str):
        """
        Save the trained model to storage.

        Args:
            job_id: Job identifier
            model_name: Name for the saved model

        Note:
            Saves model to local filesystem and optionally uploads to S3 for
            production deployment. Includes metadata.json with model information.
        """
        logger.info(f"Saving model {model_name} for job {job_id}")

        try:
            import shutil
            import json

            # Source directory (training checkpoint)
            source_dir = os.path.join(settings.model_storage_path, "checkpoints", job_id, "final")

            # Target directory (named model)
            target_dir = os.path.join(settings.model_storage_path, "models", model_name)
            os.makedirs(target_dir, exist_ok=True)

            # Copy model files
            logger.info(f"Copying model files from {source_dir} to {target_dir}")
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)

            # Calculate model size
            size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(target_dir)
                for filename in filenames
            )
            size_mb = size_bytes / (1024 * 1024)

            # Save metadata
            metadata = {
                "id": model_name,
                "job_id": job_id,
                "created_at": datetime.utcnow().isoformat(),
                "base_model": self.active_jobs[job_id]["config"]["base_model"],
                "dataset": self.active_jobs[job_id]["config"]["dataset_name"],
                "training_config": self.active_jobs[job_id]["config"]["training_config"],
                "status": "ready",
                "size_mb": round(size_mb, 2),
                "performance": {
                    "final_loss": self.active_jobs[job_id].get("train_loss"),
                    "training_epochs": self.active_jobs[job_id]["config"]["training_config"]["epochs"]
                }
            }

            metadata_path = os.path.join(target_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model {model_name} saved successfully ({size_mb:.2f} MB)")

            # Optional: Upload to S3 for production
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                logger.info(f"Uploading model {model_name} to S3")
                await self._upload_to_s3(model_name, target_dir)

        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise RuntimeError(f"Model saving failed: {str(e)}")

    async def _upload_to_s3(self, model_name: str, model_dir: str):
        """
        Upload model to S3 storage.

        Args:
            model_name: Name of the model
            model_dir: Local directory containing the model

        Note:
            Uploads all files in the model directory to S3 bucket
            configured in settings. Preserves directory structure.
        """
        try:
            import boto3
            from botocore.exceptions import ClientError

            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.s3_region
            )

            # Upload all files in model directory
            files_uploaded = 0
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, model_dir)
                    s3_key = f"models/{model_name}/{relative_path}"

                    logger.debug(f"Uploading {relative_path} to S3")
                    s3_client.upload_file(local_path, settings.s3_bucket, s3_key)
                    files_uploaded += 1

            logger.info(f"Model {model_name} uploaded to S3 ({files_uploaded} files)")

        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise RuntimeError(f"S3 upload failed: {str(e)}")
        except ImportError:
            logger.warning("boto3 not installed - skipping S3 upload")

    async def _load_model_for_inference(self, model_id: str) -> tuple:
        """
        Load a trained model for inference with LRU caching.

        Args:
            model_id: ID of the model to load

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            FileNotFoundError: If model not found in storage

        Note:
            Implements LRU cache with configurable size from settings.
            Models are evicted when cache exceeds model_cache_size limit.
        """
        # Check if model is in cache
        if model_id in self.model_cache:
            model, tokenizer, _ = self.model_cache[model_id]
            # Update timestamp for LRU
            timestamp = time.time()
            self.model_cache[model_id] = (model, tokenizer, timestamp)
            logger.info(f"Model {model_id} loaded from cache")
            return model, tokenizer

        # Load model from filesystem
        model_dir = os.path.join(settings.model_storage_path, "models", model_id)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model {model_id} not found in storage")

        logger.info(f"Loading model {model_id} from {model_dir}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)

            # Check if this is a PEFT model
            adapter_config_path = os.path.join(model_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                # Load as PEFT model
                import json
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)

                base_model = adapter_config.get("base_model_name_or_path", "gpt2")
                logger.info(f"Loading PEFT model with base {base_model}")

                # Load base model
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True
                )

                # Load PEFT adapters
                model = PeftModel.from_pretrained(base_model_obj, model_dir)
            else:
                # Load as regular model
                logger.info(f"Loading standard model")
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True
                )

            # Implement LRU eviction if cache is full
            if len(self.model_cache) >= settings.model_cache_size:
                # Find oldest model in cache
                oldest_model_id = min(
                    self.model_cache.keys(),
                    key=lambda k: self.model_cache[k][2]  # timestamp
                )
                logger.info(f"Evicting model {oldest_model_id} from cache (LRU)")
                del self.model_cache[oldest_model_id]

            # Add to cache
            timestamp = time.time()
            self.model_cache[model_id] = (model, tokenizer, timestamp)

            logger.info(f"Model {model_id} loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def _format_messages_for_inference(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation messages for inference.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Formatted prompt string

        Note:
            Uses Mistral instruction format for consistency with training.
            Only includes the final user message with context from previous turns.
        """
        # Build conversation history
        formatted = ""
        for msg in messages[:-1]:  # All except last message
            if msg["role"] == "user":
                formatted += f"<s>[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted += f" {msg['content']}</s>"

        # Add final user message (expecting completion)
        if messages and messages[-1]["role"] == "user":
            formatted += f"<s>[INST] {messages[-1]['content']} [/INST]"

        return formatted