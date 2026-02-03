# Pantheon Chat - Full Stack Build
# Node.js 24 + Python 3.13 for QIG Backend + Kernel Training
#
# This builds both the TypeScript frontend/API and includes the Python
# QIG backend with Celery support for async kernel training.
#
# SMART ROUTING: Uses docker-entrypoint.sh to route to correct service
# based on RAILWAY_SERVICE_NAME environment variable.

FROM node:24-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js dependencies first (for caching)
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts && npm rebuild

# Copy source files
COPY . .

# Build TypeScript (frontend + server)
# Set Node memory limit to prevent OOM during build
ENV NODE_OPTIONS="--max-old-space-size=2048"
RUN echo "=== Building TypeScript ===" && \
    npm run build && \
    echo "=== Build output ===" && \
    ls -la dist/ && \
    test -f dist/index.js || (echo "ERROR: dist/index.js not found!" && exit 1)

# Production image
FROM node:24-slim

WORKDIR /app

# Install Python runtime, curl for healthcheck, and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    ca-certificates \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package/dependency manager)
# NOTE: We intentionally avoid pip-based installs in images.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv python install 3.13 --managed-python

ENV UV_PYTHON=3.13 \
    UV_MANAGED_PYTHON=1

# Copy built Node.js files from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

# Copy Python project definition + lockfile for uv
COPY --from=builder /app/pyproject.toml ./
COPY --from=builder /app/uv.lock ./

# Copy Python backend (includes training module)
COPY qig-backend ./qig-backend

# Copy shared constants for Python
COPY shared ./shared

# Copy curriculum files for nightly training consolidation
COPY docs/09-curriculum ./docs/09-curriculum

# Install Python dependencies from uv.lock (frozen)
RUN uv sync --frozen

RUN /app/.venv/bin/python -c "import sys; assert sys.version_info[:2] == (3, 13)"

# Ensure console scripts installed into the uv-managed venv are discoverable
# (e.g., celery, gunicorn, pytest). This also helps if the platform executes
# a start command directly rather than through `uv run`.
ENV PATH="/app/.venv/bin:${PATH}"

# Create data directory for Railway volume mount
RUN mkdir -p /app/data /app/data/checkpoints

# Copy smart entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Set environment variables
ENV NODE_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/qig-backend \
    TRAINING_ENABLED=true \
    CHECKPOINT_DIR=/app/data/checkpoints \
    C_FORCE_ROOT=true

# Expose ports (dynamic based on service)
EXPOSE 5000

# NOTE: Docker HEALTHCHECK removed - Railway manages health checks via service settings
# Celery workers don't expose HTTP endpoints, so Docker HEALTHCHECK causes failures
# For pantheon-chat service, Railway health check path should be set to /api/health

# Use smart entrypoint that routes based on RAILWAY_SERVICE_NAME
# - pantheon-chat → node dist/index.js
# - celery-worker → python celery worker
# - Beat → python celery beat
ENTRYPOINT ["/docker-entrypoint.sh"]
