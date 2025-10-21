# Advisory Engine

User-facing recommendations and consultation service for ShopSense-AI.

## Overview

The Advisory Engine is the orchestration layer that:
- Provides AI-powered product search with natural language understanding
- Delivers personalized shopping consultation through conversational AI
- Performs intelligent product comparison across multiple criteria
- Coordinates between Discovery Engine (product data) and Knowledge Engine (AI inference)
- Implements response caching for performance optimization

## Quick Start

### Prerequisites
- Python 3.9+
- UV package manager
- Docker (for Redis caching)
- Running Discovery Engine (port 8002)
- Running Knowledge Engine (port 8001)

### Installation

```bash
cd services/advisory_engine
uv sync
```

### Configuration

Create a `.env` file in `config/` with your API keys:

```bash
ADVISORY_DISCOVERY_URL=http://localhost:8002
ADVISORY_KNOWLEDGE_URL=http://localhost:8001
ADVISORY_OPENAI_API_KEY=your-openai-key
ADVISORY_REDIS_URL=redis://localhost:6379
```

### Start Redis Cache (Optional)

```bash
docker run -p 6379:6379 redis:7-alpine
```

### Run the Service

```bash
uv run python -m api.main
```

Visit http://localhost:8003/docs for API documentation.

## Documentation

For complete documentation, see [docs/advisory_engine.md](../../docs/advisory_engine.md)

## API Endpoints

- `POST /search` - AI-powered product search
- `POST /advice` - Shopping consultation with conversation history
- `POST /compare` - Multi-product comparison
- `GET /recommendations/trending` - Trending product recommendations
- `GET /health` - Health check

## Development

```bash
# Run tests
uv run pytest

# Run with auto-reload
uv run uvicorn api.main:app --reload --port 8003
```

## Architecture

The Advisory Engine acts as an API gateway that:
1. Receives user queries and extracts intent
2. Calls Discovery Engine for semantic product search
3. Calls Knowledge Engine for AI-powered advice
4. Combines and ranks results based on user preferences
5. Caches responses in Redis for performance

For more details, see the [main documentation](../../docs/advisory_engine.md).