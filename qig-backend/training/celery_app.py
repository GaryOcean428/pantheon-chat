"""
Celery Application Configuration for Kernel Training
====================================================

This module configures Celery for async training tasks with Redis as broker.

The training system uses three Railway services:
1. Main web service (Flask + TypeScript API)
2. Celery worker service (processes async training tasks)
3. Celery Beat service (triggers periodic tasks)

All services connect via shared Redis instance for task queue.
"""

import os
from celery import Celery
from celery.schedules import crontab
from datetime import timedelta

# Redis connection from Railway environment
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Initialize Celery app
celery_app = Celery(
    'pantheon_training',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        'training.tasks',  # Import tasks module
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Fetch one task at a time (important for long-running training)
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks
    task_acks_late=True,  # Acknowledge task after completion (not before)
    task_reject_on_worker_lost=True,  # Requeue task if worker crashes
    
    # Task result settings
    result_expires=3600,  # Results expire after 1 hour
    result_extended=True,  # Store additional metadata
    
    # Task routing
    task_routes={
        'training.tasks.train_from_outcome_task': {'queue': 'training_fast'},
        'training.tasks.train_hourly_batch_task': {'queue': 'training_batch'},
        'training.tasks.train_nightly_consolidation_task': {'queue': 'training_slow'},
        'training.tasks.knowledge_transfer_task': {'queue': 'training_slow'},
        'training.tasks.save_checkpoint_task': {'queue': 'training_fast'},
    },
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes soft limit (sends exception)
    task_time_limit=600,  # 10 minutes hard limit (kills task)
    
    # Beat scheduler settings
    beat_schedule={
        # Hourly batch training (every hour)
        'hourly-batch-training': {
            'task': 'training.tasks.train_hourly_batch_all',
            'schedule': crontab(minute=0),  # Every hour at :00
            'options': {'queue': 'training_batch'},
        },
        
        # Nightly consolidation (3 AM UTC)
        'nightly-consolidation': {
            'task': 'training.tasks.train_nightly_consolidation_all',
            'schedule': crontab(hour=3, minute=0),  # 3:00 AM UTC daily
            'options': {'queue': 'training_slow'},
        },
        
        # Shadow synchronization (every 4 hours)
        'shadow-sync': {
            'task': 'training.tasks.sync_all_shadows',
            'schedule': crontab(minute=0, hour='*/4'),  # Every 4 hours
            'options': {'queue': 'training_slow'},
        },
        
        # Checkpoint cleanup (daily at 4 AM UTC)
        'checkpoint-cleanup': {
            'task': 'training.tasks.cleanup_old_checkpoints',
            'schedule': crontab(hour=4, minute=0),  # 4:00 AM UTC daily
            'options': {'queue': 'training_fast'},
        },
    },
)

# Task retry configuration
celery_app.conf.task_annotations = {
    'training.tasks.train_from_outcome_task': {
        'max_retries': 3,
        'default_retry_delay': 60,  # 1 minute
    },
    'training.tasks.train_hourly_batch_task': {
        'max_retries': 2,
        'default_retry_delay': 300,  # 5 minutes
    },
    'training.tasks.train_nightly_consolidation_task': {
        'max_retries': 1,
        'default_retry_delay': 600,  # 10 minutes
    },
}


# Helper to check if Celery is available
def is_celery_available() -> bool:
    """
    Check if Celery worker is available and responsive.
    
    Returns:
        True if Celery is available, False otherwise
    """
    try:
        # Try to inspect active workers
        inspect = celery_app.control.inspect()
        workers = inspect.active()
        return workers is not None and len(workers) > 0
    except Exception:
        return False


# Helper to get task result
def get_task_result(task_id: str, timeout: float = 5.0):
    """
    Get result of an async task by ID.
    
    Args:
        task_id: Celery task ID
        timeout: Max seconds to wait for result
        
    Returns:
        Task result or None if not ready
    """
    try:
        from celery.result import AsyncResult
        result = AsyncResult(task_id, app=celery_app)
        
        if result.ready():
            return result.get(timeout=timeout)
        else:
            return None
    except Exception as e:
        print(f"[Celery] Error getting task result: {e}")
        return None


__all__ = [
    'celery_app',
    'is_celery_available',
    'get_task_result',
]
