# Knowledge Engine

AI model training, management, and inference service for ShopSense-AI.

## Overview

The Knowledge Engine is responsible for:
- Fine-tuning LLMs using QLoRA on shopping conversation data
- Managing trained models with version control
- Serving AI inference for product recommendations
- Generating synthetic training data using OpenAI
- Tracking experiments with WandB

## Quick Start

### Prerequisites
- Python 3.9+
- UV package manager
- API keys for OpenAI, HuggingFace, and WandB

### Installation

```bash
cd services/knowledge_engine
uv sync --extra dev
```

### Configuration

Create a `.env` file in `config/` with your API keys:

```bash
KNOWLEDGE_OPENAI_API_KEY=sk-your-key
KNOWLEDGE_HUGGINGFACE_TOKEN=hf_your-token
KNOWLEDGE_WANDB_API_KEY=your-wandb-key
```

### Run the Service

```bash
uv run python -m api.main
```

Visit http://localhost:8001/docs for API documentation.

## Documentation

For complete documentation, see [docs/knowledge_engine.md](../../docs/knowledge_engine.md)

## API Endpoints

- `POST /api/v1/train` - Start training job
- `GET /api/v1/models` - List available models
- `POST /api/v1/models/{id}/inference` - Run inference
- `POST /api/v1/data/generate` - Generate training data
- `GET /api/v1/training/status/{job_id}` - Check training status
- `GET /api/v1/health` - Health check

## Development

```bash
# Run tests
uv run pytest

# Run with auto-reload
uv run uvicorn api.main:app --reload --port 8001
```

## Production

For GPU training on Linux with CUDA:

```bash
uv sync --extra gpu
```

For more details, see the [main documentation](../../docs/knowledge_engine.md).