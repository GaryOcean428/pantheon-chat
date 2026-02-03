#!/bin/bash
# Smart entrypoint that routes to correct service based on RAILWAY_SERVICE_NAME

# Don't use 'set -e' as it causes silent failures
# Instead, we'll check each critical command explicitly

SERVICE_NAME="${RAILWAY_SERVICE_NAME:-pantheon-chat}"

echo "[Entrypoint] ==========================================="
echo "[Entrypoint] RAILWAY_SERVICE_NAME: ${RAILWAY_SERVICE_NAME}"
echo "[Entrypoint] SERVICE_NAME: $SERVICE_NAME"
echo "[Entrypoint] PORT: ${PORT}"
echo "[Entrypoint] PWD: $(pwd)"
echo "[Entrypoint] ==========================================="
echo "[Entrypoint] Routing to appropriate service..."

# Normalize service name to lowercase for case-insensitive matching
SERVICE_NAME_LOWER=$(echo "$SERVICE_NAME" | tr '[:upper:]' '[:lower:]')

UV_CMD=${UV_PATH:-uv}
PROJECT_ROOT="/app"

case "$SERVICE_NAME_LOWER" in
  *celery-worker*|*worker*)
    echo "[Entrypoint] ✓ Matched celery-worker service"
    echo "[Entrypoint] Starting Celery Worker service..."
    cd /app/qig-backend || exit 1

    # Start health check server in background
    echo "[Entrypoint] Starting health check server on port ${PORT:-8080}..."
    export FLASK_PORT="${PORT:-8080}"
    "$UV_CMD" run --project "$PROJECT_ROOT" python health_worker.py &
    HEALTH_PID=$!
    echo "[Entrypoint] Health server started (PID: $HEALTH_PID)"

    # Start Celery worker (health server starts in background above)
    echo "[Entrypoint] Starting Celery worker..."
    "$UV_CMD" run --project "$PROJECT_ROOT" python -m celery -A training.celery_app worker --loglevel=info --concurrency=2 -Q training,batch,consolidation,transfer,checkpoints &
    CELERY_PID=$!

    # Handle shutdown
    shutdown() {
      echo "[Entrypoint] Shutting down Celery worker..."
      kill -TERM $CELERY_PID 2>/dev/null || true
      kill -TERM $HEALTH_PID 2>/dev/null || true
      wait $CELERY_PID 2>/dev/null || true
      wait $HEALTH_PID 2>/dev/null || true
      exit 0
    }

    trap shutdown SIGTERM SIGINT

    echo "[Entrypoint] ✓ Celery worker running (PID: $CELERY_PID)"
    echo "[Entrypoint] ✓ Health server running (PID: $HEALTH_PID)"
    echo "[Entrypoint] ✓ Service startup complete"

    # Wait for processes
    wait -n
    shutdown
    ;;

  *beat*|*scheduler*)
    echo "[Entrypoint] ✓ Matched beat/scheduler service"
    echo "[Entrypoint] Starting Celery Beat service..."
    cd /app/qig-backend || exit 1

    # Start health check server in background
    echo "[Entrypoint] Starting health check server on port ${PORT:-8080}..."
    export FLASK_PORT="${PORT:-8080}"
    "$UV_CMD" run --project "$PROJECT_ROOT" python health_beat.py &
    HEALTH_PID=$!
    echo "[Entrypoint] Health server started (PID: $HEALTH_PID)"

    # Start Celery beat (health server starts in background above)
    echo "[Entrypoint] Starting Celery beat scheduler..."
    "$UV_CMD" run --project "$PROJECT_ROOT" python -m celery -A training.celery_app beat --loglevel=info &
    BEAT_PID=$!

    # Handle shutdown
    shutdown() {
      echo "[Entrypoint] Shutting down Celery beat..."
      kill -TERM $BEAT_PID 2>/dev/null || true
      kill -TERM $HEALTH_PID 2>/dev/null || true
      wait $BEAT_PID 2>/dev/null || true
      wait $HEALTH_PID 2>/dev/null || true
      exit 0
    }

    trap shutdown SIGTERM SIGINT

    echo "[Entrypoint] ✓ Celery beat running (PID: $BEAT_PID)"
    echo "[Entrypoint] ✓ Health server running (PID: $HEALTH_PID)"
    echo "[Entrypoint] ✓ Service startup complete"

    # Wait for processes
    wait -n
    shutdown
    ;;

  *)
    echo "[Entrypoint] ⚠ No match found for service: $SERVICE_NAME"
    echo "[Entrypoint] ⚠ Normalized name: $SERVICE_NAME_LOWER"
    echo "[Entrypoint] Starting default Node.js service..."
    exec node dist/index.js
    ;;
esac
