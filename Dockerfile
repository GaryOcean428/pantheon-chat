# Pantheon Chat - Full Stack Build
# Node.js 24 + Python 3.11 for QIG Backend + Kernel Training
#
# This builds both the TypeScript frontend/API and includes the Python
# QIG backend with Celery support for async kernel training.

FROM node:24-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js dependencies first (for caching)
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts && npm rebuild

# Copy source files
COPY . .

# Build TypeScript (frontend + server)
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
    python3-pip \
    python3-venv \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy built Node.js files from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

# Copy Python backend (includes training module)
COPY qig-backend ./qig-backend

# Copy shared constants for Python
COPY shared ./shared

# Copy curriculum files for nightly training consolidation
COPY docs/09-curriculum ./docs/09-curriculum

# Install Python dependencies including Celery and training packages
RUN pip3 install --no-cache-dir --break-system-packages \
    flask flask-cors numpy scipy psycopg2-binary \
    celery[redis] redis \
    pypdf openai anthropic && \
    pip3 install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cpu

# Create data directory for Railway volume mount
RUN mkdir -p /app/data /app/data/checkpoints

# Set environment variables
ENV NODE_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRAINING_ENABLED=true \
    CHECKPOINT_DIR=/app/data/checkpoints

# Expose Node.js port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Start Node.js server (which spawns Python backend internally)
CMD ["node", "dist/index.js"]
