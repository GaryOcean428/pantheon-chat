"""
Celery Application Configuration
================================

Configures Celery for async training tasks with:
- Redis broker (from Railway)
- Beat scheduler for hourly/nightly tasks
- Task routing and retry policies
"""

import os
from celery import Celery
from celery.schedules import crontab


# Get Redis URL from environment
REDIS_URL = os.getenv(
    "CELERY_BROKER_URL",
    os.getenv("REDIS_URL", "redis://localhost:6379")
)

# Create Celery app
celery_app = Celery(
    "qig_training",
    broker=REDIS_URL,
    backend=os.getenv("CELERY_RESULT_BACKEND", REDIS_URL),
    include=["training.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Result backend
    result_expires=3600,  # 1 hour

    # Retry policy
    task_default_retry_delay=30,  # seconds
    task_max_retries=3,

    # Concurrency (CPU-only, keep low)
    worker_concurrency=2,

    # Task routing
    task_routes={
        "training.tasks.train_from_outcome_task": {"queue": "training"},
        "training.tasks.train_hourly_batch_task": {"queue": "batch"},
        "training.tasks.train_nightly_consolidation_task": {"queue": "consolidation"},
        "training.tasks.knowledge_transfer_task": {"queue": "transfer"},
        "training.tasks.save_checkpoint_task": {"queue": "checkpoints"},
    },

    # Beat scheduler configuration
    beat_schedule={
        # Hourly batch training - process accumulated data
        "hourly-batch-training": {
            "task": "training.tasks.train_hourly_batch_all",
            "schedule": crontab(minute=0),  # Every hour at :00
            "options": {"queue": "batch"},
        },

        # Nightly consolidation - 3 AM UTC
        "nightly-consolidation": {
            "task": "training.tasks.train_nightly_consolidation_all",
            "schedule": crontab(hour=3, minute=0),
            "options": {"queue": "consolidation"},
        },

        # Shadow sync - every 4 hours
        "shadow-sync": {
            "task": "training.tasks.sync_all_shadows",
            "schedule": crontab(minute=30, hour="*/4"),
            "options": {"queue": "transfer"},
        },

        # Checkpoint cleanup - daily at 4 AM UTC
        "checkpoint-cleanup": {
            "task": "training.tasks.cleanup_old_checkpoints",
            "schedule": crontab(hour=4, minute=0),
            "options": {"queue": "checkpoints"},
        },
    },

    # Beat scheduler persistence
    beat_scheduler="celery.beat:PersistentScheduler",
    beat_schedule_filename="/app/data/celerybeat-schedule",
)


# Task base class with default settings
class TrainingTask(celery_app.Task):
    """Base class for training tasks with shared configuration."""

    # Retry settings
    autoretry_for = (Exception,)
    retry_backoff = True
    retry_backoff_max = 600  # 10 minutes
    retry_jitter = True

    # Soft/hard time limits (training can be slow)
    soft_time_limit = 300  # 5 minutes
    time_limit = 600  # 10 minutes

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log failures for debugging."""
        print(f"[Celery] Task {self.name} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        """Log successes for monitoring."""
        print(f"[Celery] Task {self.name} completed successfully")
        super().on_success(retval, task_id, args, kwargs)


# Export task base class
celery_app.Task = TrainingTask


def is_training_enabled() -> bool:
    """Check if training is enabled via environment variable."""
    return os.getenv("TRAINING_ENABLED", "true").lower() == "true"


def get_checkpoint_dir() -> str:
    """Get checkpoint directory, defaulting to Railway volume."""
    return os.getenv(
        "CHECKPOINT_DIR",
        os.path.join(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/app/data"), "checkpoints")
    )
