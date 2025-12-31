# Celery Beat Scheduler for QIG Kernel Training
# Schedules periodic training tasks (hourly batch, nightly consolidation, shadow sync)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy qig-backend Python code
COPY qig-backend/ ./qig-backend/

# Copy shared constants
COPY shared/ ./shared/

# Copy curriculum for training
COPY docs/09-curriculum/ ./docs/09-curriculum/

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask flask-cors numpy scipy psycopg2-binary \
    celery[redis] redis \
    pypdf openai anthropic \
    safetensors msgpack opt-einsum sqlalchemy

# Install PyTorch (CPU only for training)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/qig-backend \
    TRAINING_ENABLED=true \
    C_FORCE_ROOT=true

# Create data directory for beat schedule persistence
RUN mkdir -p /app/data

# Run celery beat scheduler
CMD ["celery", "-A", "training.celery_app", "beat", "--loglevel=info"]
