# Build stage
FROM python:3.13-slim AS builder

WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Copy source code (needed for package building and project installation)
COPY src/ src/

# Create dist directory
RUN mkdir -p dist

# Build the mlp-package (equivalent to task build-package)
RUN if [ -f src/models/model_package/Makefile ]; then \
        cd src/models/model_package && make all; \
    else \
        echo "Error: Makefile not found in src/models/model_package"; \
        exit 1; \
    fi

# Install the built package with force reinstall and no deps (equivalent to task add-package)
RUN uv pip install --force-reinstall --no-deps dist/mlp_package-0.1.0-py3-none-any.whl

# Install all project dependencies (equivalent to task make-pack final uv sync)
RUN uv sync --no-dev

# Production stage
FROM python:3.13-slim AS production

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src /app/src

# Copy built package (in case it's needed at runtime)
COPY --from=builder /app/dist /app/dist

# Set Python path to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app"

# Expose port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]