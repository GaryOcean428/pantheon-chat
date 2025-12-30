# Pantheon Chat - Full Stack Build
# Node.js 24 + Python 3.11 for QIG Backend

FROM node:24-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js dependencies
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts && npm rebuild

# Copy source
COPY . .

# Build TypeScript
RUN echo "=== Building TypeScript ===" && \
    npm run build && \
    echo "=== Build output ===" && \
    ls -la dist/ && \
    test -f dist/index.js

# Production image
FROM node:24-slim

WORKDIR /app

# Install Python runtime and curl for healthcheck
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built Node.js files
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

# Copy Python backend
COPY qig-backend ./qig-backend

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages -r qig-backend/requirements.txt && \
    pip3 install --no-cache-dir --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu

# Create data directory for volume
RUN mkdir -p /app/data

# Expose ports
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Start command
CMD ["node", "dist/index.js"]
