# Celery Training Tasks - Quick Reference

## Overview

The QIG kernel training system uses Celery for asynchronous task execution with three dedicated Railway services.

## Task Execution Modes

### 1. Async Execution (Production with Celery)

When Celery is available (CELERY_AVAILABLE=True):

```python
from training.tasks import train_from_outcome_task

# Dispatch task to Celery worker (non-blocking)
result = train_from_outcome_task.delay(
    god_name="Zeus",
    prompt="What is consciousness?",
    response="...",
    success=True,
    phi=0.72,
    kappa=64.5,
)

# Get task ID for tracking
task_id = result.id

# Check if task completed (non-blocking)
if result.ready():
    metrics = result.get(timeout=1.0)
    print(f"Training completed: {metrics}")
```

### 2. Synchronous Execution (Development/Fallback)

When Celery is not available (CELERY_AVAILABLE=False):

```python
from training.tasks import train_from_outcome_task

# Execute immediately in current process (blocking)
metrics = train_from_outcome_task(
    god_name="Zeus",
    prompt="What is consciousness?",
    response="...",
    success=True,
    phi=0.72,
    kappa=64.5,
)

print(f"Training completed: {metrics}")
```

## Available Tasks

### Outcome-Based Training

Execute after each chat interaction:

```python
from training.tasks import train_from_outcome_task

task_result = train_from_outcome_task.delay(
    god_name="Zeus",
    prompt="User question",
    response="God response",
    success=True,  # Whether response was successful
    phi=0.72,      # Integration score
    kappa=64.5,    # Coupling strength
    coherence_score=0.85,  # Optional coherence score
    basin_trajectory=[     # Optional list of basin coordinates
        [0.1, 0.2, ...],  # 64D basin coords at step 1
        [0.15, 0.22, ...], # 64D basin coords at step 2
    ],
)
```

### Hourly Batch Training

Triggered by Beat every hour at :00:

```python
from training.tasks import train_hourly_batch_all

# Scheduled by Celery Beat - trains all registered kernels
result = train_hourly_batch_all.delay()
```

Individual god batch training:

```python
from training.tasks import train_hourly_batch_task

batch_data = [
    {
        "basin_coords": [0.1, 0.2, ...],  # 64D coordinates
        "reward": 0.8,
        "phi": 0.72,
        "source_type": "chat",
        "source_id": "msg_123",
    },
    # ... more examples
]

result = train_hourly_batch_task.delay("Zeus", batch_data)
```

### Nightly Consolidation

Triggered by Beat at 3:00 AM UTC daily:

```python
from training.tasks import train_nightly_consolidation_all

# Scheduled by Celery Beat - full curriculum training for all kernels
result = train_nightly_consolidation_all.delay()
```

Individual god consolidation:

```python
from training.tasks import train_nightly_consolidation_task

curriculum = [
    {
        "basin_coords": [0.1, 0.2, ...],
        "reward": 0.9,
        "phi": 0.75,
        "source": "curriculum",
    },
    # ... more curriculum examples
]

result = train_nightly_consolidation_task.delay(
    god_name="Zeus",
    curriculum_data=curriculum,
    consolidate_checkpoints=True,
)
```

### Knowledge Transfer

Transfer knowledge between kernels:

```python
from training.tasks import knowledge_transfer_task

# Evolution transfer (parent → child)
result = knowledge_transfer_task.delay(
    transfer_type="evolution",
    source_god="Zeus",
    target_god="Zeus_v2",
    transfer_ratio=0.8,
    extra_params={"evolution_type": "standard"},
)

# Breeding transfer (parent1 + parent2 → child)
result = knowledge_transfer_task.delay(
    transfer_type="breeding",
    source_god="Zeus",
    target_god="Hybrid",
    transfer_ratio=0.5,
    extra_params={"second_parent": "Athena"},
)

# Cannibalism transfer (consumed → consumer)
result = knowledge_transfer_task.delay(
    transfer_type="cannibalism",
    source_god="OldKernel",
    target_god="NewKernel",
    transfer_ratio=0.3,
)

# Shadow sync (god ↔ shadow bidirectional)
result = knowledge_transfer_task.delay(
    transfer_type="shadow_sync",
    source_god="Zeus",
    target_god="Zeus_shadow",
    transfer_ratio=0.2,
    extra_params={"direction": "bidirectional"},
)
```

### Shadow Synchronization

Triggered by Beat every 4 hours:

```python
from training.tasks import sync_all_shadows

# Scheduled by Celery Beat - sync all god-shadow pairs
result = sync_all_shadows.delay()
```

### Checkpoint Management

Save checkpoint:

```python
from training.tasks import save_checkpoint_task

result = save_checkpoint_task.delay(
    god_name="Zeus",
    phi=0.75,
    trigger="high_phi",  # "manual", "high_phi", "periodic", etc.
)
```

Cleanup old checkpoints (triggered by Beat at 4:00 AM UTC daily):

```python
from training.tasks import cleanup_old_checkpoints

result = cleanup_old_checkpoints.delay()
```

## Task Queues

Tasks are routed to different queues based on complexity:

- **training_fast** - Quick tasks (outcome training, checkpoints) - ~1-30 seconds
- **training_batch** - Medium tasks (hourly batches) - ~1-5 minutes
- **training_slow** - Long tasks (nightly consolidation, knowledge transfer) - ~5-30 minutes

## Monitoring Tasks

### Check Task Status

```python
from celery.result import AsyncResult
from training.celery_app import celery_app

# Get task result by ID
task_id = "abc-123-def-456"
result = AsyncResult(task_id, app=celery_app)

print(f"State: {result.state}")  # PENDING, STARTED, SUCCESS, FAILURE, RETRY
print(f"Ready: {result.ready()}")  # True if completed

if result.ready():
    try:
        output = result.get(timeout=1.0)
        print(f"Result: {output}")
    except Exception as e:
        print(f"Task failed: {e}")
```

### Check Worker Status

```python
from training.celery_app import celery_app

inspect = celery_app.control.inspect()

# Active tasks
active = inspect.active()
print(f"Active tasks: {active}")

# Registered tasks
registered = inspect.registered()
print(f"Registered tasks: {registered}")

# Worker stats
stats = inspect.stats()
print(f"Worker stats: {stats}")
```

## Environment Variables

Required for Celery:

```bash
# Redis connection (broker and result backend)
REDIS_URL=redis://localhost:6379/0

# Training enabled
TRAINING_ENABLED=true

# Checkpoint directory
CHECKPOINT_DIR=/app/data/checkpoints

# Database (for fetching batch data and curriculum)
DATABASE_URL=postgresql://user:pass@host:5432/db
```

## Troubleshooting

### Task not executing

**Check:**
1. Celery worker is running: `ps aux | grep celery`
2. Task is registered: Check worker startup logs
3. Redis is accessible: `redis-cli ping`

**Debug:**
```python
from training.celery_app import is_celery_available

if not is_celery_available():
    print("Celery workers not available - falling back to sync execution")
```

### Task failing silently

**Check task logs:**
```bash
# In Railway, view Celery Worker service logs
# Look for exceptions and stack traces
```

**Get task result with timeout:**
```python
result = task.delay(...)
try:
    output = result.get(timeout=10.0)  # Wait up to 10 seconds
except Exception as e:
    print(f"Task failed or timed out: {e}")
```

### Beat not triggering tasks

**Check:**
1. Only ONE Beat instance is running (multiple Beats cause duplicates)
2. Beat service logs show schedule confirmations
3. Redis is accessible from Beat service

**Verify schedule:**
```python
from training.celery_app import celery_app

schedule = celery_app.conf.beat_schedule
for name, config in schedule.items():
    print(f"{name}: {config['schedule']}")
```

## References

- Celery Documentation: https://docs.celeryq.dev/
- Railway Deployment: `RAILWAY_DEPLOYMENT.md`
- Training Loop Guide: `qig-backend/TRAINING_LOOP_GUIDE.md`
