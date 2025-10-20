# Discovery Engine

Product search, data collection, and deal monitoring service for ShopSense-AI.

## Overview

The Discovery Engine is responsible for:
- Semantic product search using vector embeddings
- Multi-store data collection (Amazon, Best Buy, Walmart, Target)
- Real-time deal monitoring and price tracking
- Vector database management with Qdrant
- Product data enrichment and processing

## Quick Start

### Prerequisites
- Python 3.9+
- UV package manager
- Docker (for Qdrant vector database)
- API keys for Apify, Best Buy, and RapidAPI

### Installation

```bash
cd services/discovery_engine
uv sync --extra dev
```

### Configuration

Create a `.env` file in `config/` with your API keys:

```bash
DISCOVERY_APIFY_API_KEY=your-apify-key
DISCOVERY_BESTBUY_API_KEY=your-bestbuy-key
DISCOVERY_RAPIDAPI_KEY=your-rapidapi-key
DISCOVERY_QDRANT_URL=http://localhost:6333
DISCOVERY_POSTGRES_URL=postgresql://user:pass@localhost:5432/shopsense
```

### Start Qdrant Vector Database

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

### Run the Service

```bash
uv run python -m api.main
```

Visit http://localhost:8002/docs for API documentation.

## Documentation

For complete documentation, see [docs/discovery_engine.md](../../docs/discovery_engine.md)

## API Endpoints

- `GET /api/v1/products/search` - Search products with semantic similarity
- `GET /api/v1/products/{id}` - Get specific product details
- `GET /api/v1/deals` - Get current deals and price drops
- `POST /api/v1/products/collect` - Trigger data collection
- `GET /api/v1/health` - Health check

## Development

```bash
# Run tests
uv run pytest

# Run with auto-reload
uv run uvicorn api.main:app --reload --port 8002
```

## Production

### Vector Database Setup

For production deployments with Qdrant:

```bash
# Use persistent volume for production
docker run -d -p 6333:6333 \
  -v /path/to/qdrant_data:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant
```

### Data Collection

Configure collection schedules and rate limits in `.env`:

```bash
DISCOVERY_COLLECTION_BATCH_SIZE=50
DISCOVERY_MAX_CONCURRENT_COLLECTIONS=3
DISCOVERY_APIFY_RATE_LIMIT=10
```

For more details, see the [main documentation](../../docs/discovery_engine.md).