# Build stage
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Create the dist directory and copy the local package if it exists
RUN mkdir -p dist
COPY dist/mlp_package-0.1.0-py3-none-any.whl dist/ 2>/dev/null || true

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application
COPY src/ src/
COPY README.md ./

# Install the project itself
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.13-slim AS production

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src /app/src

# Set Python path to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app"

# Expose port (adjust if needed)
EXPOSE 8000

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]