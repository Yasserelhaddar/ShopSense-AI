# Advisory Engine Dockerfile
# Optimized for API orchestration and user-facing operations

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies using UV
RUN uv sync --frozen

# Copy application code
COPY . .

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/api/v1/health || exit 1

# Run the application
CMD ["uv", "run", "python", "-m", "api.main"]