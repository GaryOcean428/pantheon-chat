#!/bin/bash
# Celery Worker Entrypoint
# Runs both health check server and Celery worker for Railway deployment

set -e

cd /app/qig-backend

UV_CMD=${UV_PATH:-uv}
PROJECT_ROOT="/app"

echo "[celery-worker] Starting health check server on port ${PORT:-8080}..."
"$UV_CMD" run --project "$PROJECT_ROOT" python health_worker.py &
HEALTH_PID=$!

echo "[celery-worker] Starting Celery worker..."
"$UV_CMD" run --project "$PROJECT_ROOT" python -m celery -A training.celery_app worker --loglevel=info --concurrency=2 -Q training_fast,training_batch,training_slow &
CELERY_PID=$!

# Function to handle shutdown
shutdown() {
    echo "[celery-worker] Shutting down..."
    kill -TERM $CELERY_PID 2>/dev/null || true
    kill -TERM $HEALTH_PID 2>/dev/null || true
    wait $CELERY_PID
    wait $HEALTH_PID
    exit 0
}

trap shutdown SIGTERM SIGINT

echo "[celery-worker] Health server PID: $HEALTH_PID"
echo "[celery-worker] Celery worker PID: $CELERY_PID"
echo "[celery-worker] All processes started successfully"

# Wait for both processes
wait -n

# If either process exits, shutdown both
shutdown
