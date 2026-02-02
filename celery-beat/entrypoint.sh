#!/bin/bash
# Celery Beat Entrypoint
# Runs both health check server and Celery beat scheduler for Railway deployment

set -e

cd /app/qig-backend

UV_CMD=${UV_PATH:-uv}
PROJECT_ROOT="/app"

echo "[celery-beat] Starting health check server on port ${PORT:-8080}..."
"$UV_CMD" run --project "$PROJECT_ROOT" python health_beat.py &
HEALTH_PID=$!

echo "[celery-beat] Starting Celery beat scheduler..."
"$UV_CMD" run --project "$PROJECT_ROOT" celery -A training.celery_app beat --loglevel=info &
BEAT_PID=$!

# Function to handle shutdown
shutdown() {
    echo "[celery-beat] Shutting down..."
    kill -TERM $BEAT_PID 2>/dev/null || true
    kill -TERM $HEALTH_PID 2>/dev/null || true
    wait $BEAT_PID
    wait $HEALTH_PID
    exit 0
}

trap shutdown SIGTERM SIGINT

echo "[celery-beat] Health server PID: $HEALTH_PID"
echo "[celery-beat] Celery beat PID: $BEAT_PID"
echo "[celery-beat] All processes started successfully"

# Wait for both processes
wait -n

# If either process exits, shutdown both
shutdown
