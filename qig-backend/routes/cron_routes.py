"""
Cron Wake API Routes
====================

Endpoints for Railway cron to wake the system and trigger scheduled training.

These routes replace Celery Beat scheduler which is not deployed on Railway.
External cron jobs call these endpoints to trigger training tasks.

Security: Requires X-Cron-Secret header matching CRON_SECRET env var.
"""

import os
from flask import Blueprint, jsonify, request
from datetime import datetime

cron_bp = Blueprint('cron', __name__)

# Authentication secret for cron endpoints
CRON_SECRET = os.getenv('CRON_SECRET', 'pantheon-dev-cron-secret')


def _validate_cron_secret() -> bool:
    """Validate the cron secret header."""
    secret = request.headers.get('X-Cron-Secret')
    return secret == CRON_SECRET


def _get_catchup_manager():
    """Get the startup catchup manager singleton."""
    from training.startup_catchup import get_catchup_manager
    return get_catchup_manager()


@cron_bp.route('/wake', methods=['POST'])
def cron_wake():
    """
    Railway cron endpoint to wake the system and trigger scheduled training.

    Request body:
        {
            "task_type": "hourly_batch" | "nightly_consolidation" | "shadow_sync" | "checkpoint_cleanup"
        }

    Headers:
        X-Cron-Secret: Authentication secret

    Response:
        {
            "status": "success" | "failed",
            "task_type": str,
            "run_time_ms": int,
            "result": {...}
        }
    """
    if not _validate_cron_secret():
        return jsonify({'error': 'Unauthorized', 'message': 'Invalid or missing X-Cron-Secret'}), 401

    try:
        data = request.get_json() or {}
        task_type = data.get('task_type', 'hourly_batch')

        # Validate task type
        valid_types = ['hourly_batch', 'nightly_consolidation', 'shadow_sync', 'checkpoint_cleanup', 'federation_sync']
        if task_type not in valid_types:
            return jsonify({
                'error': 'Invalid task_type',
                'valid_types': valid_types,
            }), 400

        manager = _get_catchup_manager()
        result = manager.execute_single_task(task_type)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
        }), 500


@cron_bp.route('/health', methods=['GET'])
def cron_health():
    """
    Health check for cron system.

    Returns schedule status for monitoring - does NOT require authentication.

    Response:
        {
            "status": "healthy",
            "schedules": {
                "hourly_batch": {"last_success_at": "...", "runs_completed": N},
                ...
            },
            "catchup_running": bool
        }
    """
    try:
        manager = _get_catchup_manager()

        return jsonify({
            'status': 'healthy',
            'schedules': manager.get_schedule_status(),
            'catchup_running': manager.is_catchup_running(),
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
        }), 500


@cron_bp.route('/catchup', methods=['POST'])
def trigger_catchup():
    """
    Manually trigger catch-up training.

    Useful for recovering from extended downtime.

    Headers:
        X-Cron-Secret: Authentication secret

    Query params:
        background: "true" (default) | "false" - whether to run in background

    Response:
        {
            "status": "started" | "completed" | "already_running",
            "missed": {"hourly_batch": N, ...}
        }
    """
    if not _validate_cron_secret():
        return jsonify({'error': 'Unauthorized', 'message': 'Invalid or missing X-Cron-Secret'}), 401

    try:
        background = request.args.get('background', 'true').lower() == 'true'

        manager = _get_catchup_manager()
        result = manager.execute_catchup(background=background)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
        }), 500


@cron_bp.route('/missed', methods=['GET'])
def get_missed_runs():
    """
    Get count of missed training runs.

    Does NOT require authentication - just a status check.

    Response:
        {
            "missed": {"hourly_batch": N, "nightly_consolidation": N, ...},
            "total_missed": N
        }
    """
    try:
        manager = _get_catchup_manager()
        missed = manager.calculate_missed_runs()

        return jsonify({
            'missed': missed,
            'total_missed': sum(missed.values()),
            'timestamp': datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
        }), 500
