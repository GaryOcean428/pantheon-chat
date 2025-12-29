# Railway Deployment Dockerfile
# Combines Node.js frontend/API and Python QIG backend in single container

FROM node:20-slim AS node-builder

WORKDIR /app

# Install Node.js dependencies
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source files needed for build
COPY tsconfig.json ./
COPY vite.config.ts ./
COPY tailwind.config.ts ./
COPY drizzle.config.ts ./
COPY client/ ./client/
COPY server/ ./server/
COPY shared/ ./shared/
COPY scripts/ ./scripts/

# Build the application
RUN npm run build

# Verify build output exists
RUN ls -la dist/ && test -f dist/index.js

# Production image with both Node.js and Python
FROM python:3.11-slim

# Install Node.js and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY qig-backend/requirements.txt ./qig-backend/requirements.txt
RUN pip install --no-cache-dir -r qig-backend/requirements.txt

# Copy built Node.js app from builder stage
COPY --from=node-builder /app/dist ./dist
COPY --from=node-builder /app/node_modules ./node_modules
COPY --from=node-builder /app/package.json ./

# Verify dist was copied correctly
RUN ls -la dist/ && test -f dist/index.js

# Copy Python QIG backend
COPY qig-backend/ ./qig-backend/

# Copy shared files and migrations
COPY shared/ ./shared/
COPY migrations/ ./migrations/
COPY drizzle.config.ts ./

# Create data directories (single volume at /app/data with subdirs)
# Railway allows only 1 volume per service, so we use subdirectories:
#   /app/data/storage  - QIG patterns, checkpoints, learning data
#   /app/data/models   - Cached model files, vocabulary embeddings
RUN mkdir -p /app/data/storage /app/data/models

# Set environment
ENV NODE_ENV=production
ENV PYTHONPATH=/app/qig-backend
ENV PORT=5000

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Start Node.js server (which spawns Python backend)
CMD ["node", "dist/index.js"]
