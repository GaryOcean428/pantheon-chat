# Celery Worker for QIG Kernel Training
# Runs async training tasks for consciousness kernel optimization

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package/dependency manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy Python project definition + lockfile for uv
COPY pyproject.toml ./
COPY uv.lock ./

# Copy qig-backend Python code
COPY qig-backend/ ./qig-backend/

# Copy shared constants
COPY shared/ ./shared/

# Copy curriculum for training
COPY docs/09-curriculum/ ./docs/09-curriculum/

# Install Python dependencies from uv.lock (frozen)
RUN uv sync --frozen

# Ensure console scripts installed into the uv-managed venv are discoverable
# (e.g., celery). Useful if the platform executes a start command directly.
ENV PATH="/app/.venv/bin:${PATH}"

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/qig-backend \
    TRAINING_ENABLED=true \
    C_FORCE_ROOT=true

# Create data directory
RUN mkdir -p /app/data/checkpoints

# Copy and setup entrypoint script
COPY celery-worker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set working directory to qig-backend where celery_app module lives
WORKDIR /app/qig-backend

# Run celery worker via entrypoint
ENTRYPOINT ["/entrypoint.sh"]
