# Railway Celery/Beat Migration - Implementation Summary

**Date:** 2026-01-28
**PR:** copilot/update-railway-config-files
**Status:** ✅ Complete - Ready for Deployment

## Problem Statement

The current GaryOcean428/pantheon-chat repository has surpassed the deployed Arcane-Fly/pantheon-chat version on Railway. Rather than updating the Arcane-Fly repository, it's more efficient to bring over Railway configuration files for external Celery and Beat services to enable proper async task processing.

## Solution Overview

Implemented a **three-service Railway architecture** with:
1. Main web service (existing)
2. Celery worker service (new) - Processes async training tasks
3. Celery Beat service (new) - Schedules periodic tasks

All services communicate via shared **Redis** instance for task queuing.

## Files Created/Modified

### New Configuration Files

1. **`railway-celery-worker.json`** (808 bytes)
   - Railway service configuration for Celery worker
   - Python 3.11 environment
   - Start command: `celery -A training.celery_app worker --loglevel=info --concurrency=2`
   - Requires: DATABASE_URL, REDIS_URL, TRAINING_ENABLED

2. **`railway-celery-beat.json`** (797 bytes)
   - Railway service configuration for Celery Beat scheduler
   - Python 3.11 environment
   - Start command: `celery -A training.celery_app beat --loglevel=info`
   - Requires: REDIS_URL, TRAINING_ENABLED

3. **`qig-backend/training/celery_app.py`** (4997 bytes)
   - Celery application configuration
   - Redis broker and result backend
   - Task routing to 3 queues: training_fast, training_batch, training_slow
   - Beat schedule configuration:
     - Hourly batch training (every hour at :00)
     - Nightly consolidation (3:00 AM UTC daily)
     - Shadow sync (every 4 hours)
     - Checkpoint cleanup (4:00 AM UTC daily)
   - Helper functions: `is_celery_available()`, `get_task_result()`

### Modified Files

4. **`qig-backend/training/tasks.py`** (Modified ~100 lines)
   - Added Celery task decorators using `@_make_task` wrapper
   - Maintains **backward compatibility** - works synchronously when Celery unavailable
   - Updated task dispatch logic to use `.delay()` when Celery available
   - Tasks decorated:
     - `train_from_outcome_task`
     - `train_hourly_batch_task`
     - `train_hourly_batch_all`
     - `train_nightly_consolidation_task`
     - `train_nightly_consolidation_all`
     - `knowledge_transfer_task`
     - `sync_all_shadows`
     - `save_checkpoint_task`
     - `cleanup_old_checkpoints`

5. **`.env.example`** (Modified)
   - Added REDIS_URL configuration
   - Added CELERY_BROKER_URL and CELERY_RESULT_BACKEND
   - Added TRAINING_ENABLED and CHECKPOINT_DIR
   - Added Railway environment variables documentation

### Documentation Files

6. **`RAILWAY_DEPLOYMENT.md`** (8117 bytes)
   - Comprehensive deployment guide
   - Architecture diagram
   - Step-by-step deployment instructions
   - Monitoring and troubleshooting guide
   - Cost considerations
   - Migration guide from cron-based system

7. **`CELERY_TASKS_GUIDE.md`** (7713 bytes)
   - Quick reference for task execution
   - Code examples for all task types
   - Async vs synchronous execution modes
   - Monitoring and debugging guide
   - Troubleshooting common issues

8. **`README.md`** (Modified)
   - Added Railway Deployment section
   - Links to RAILWAY_DEPLOYMENT.md and CELERY_TASKS_GUIDE.md

## Technical Implementation Details

### Graceful Degradation

The implementation includes **backward compatibility** to work without Celery:

```python
# In qig-backend/training/tasks.py
try:
    from .celery_app import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None

# Decorator that applies @task only if Celery available
def _make_task(func):
    if CELERY_AVAILABLE and celery_app:
        return celery_app.task(bind=True, name=f'training.tasks.{func.__name__}')(func)
    return func

# Tasks can be called synchronously or async
if CELERY_AVAILABLE:
    result = train_from_outcome_task.delay(...)  # Async
else:
    result = train_from_outcome_task(...)  # Sync
```

### Task Routing

Tasks are routed to three queues based on execution time:

- **training_fast** (~1-30 seconds)
  - `train_from_outcome_task` - Per-interaction training
  - `save_checkpoint_task` - Checkpoint saves
  - `cleanup_old_checkpoints` - Checkpoint cleanup

- **training_batch** (~1-5 minutes)
  - `train_hourly_batch_task` - Single god batch training
  - `train_hourly_batch_all` - All gods batch training

- **training_slow** (~5-30 minutes)
  - `train_nightly_consolidation_task` - Full curriculum training
  - `train_nightly_consolidation_all` - All gods consolidation
  - `knowledge_transfer_task` - Knowledge transfer operations
  - `sync_all_shadows` - Shadow synchronization

### Beat Schedule

Celery Beat triggers these periodic tasks:

| Task | Schedule | Queue | Purpose |
|------|----------|-------|---------|
| Hourly Batch Training | Every hour at :00 | training_batch | Process accumulated interactions |
| Nightly Consolidation | 3:00 AM UTC daily | training_slow | Full curriculum training |
| Shadow Sync | Every 4 hours | training_slow | God-shadow bidirectional sync |
| Checkpoint Cleanup | 4:00 AM UTC daily | training_fast | Prune old checkpoints |

## Validation Performed

### Syntax Validation
- ✅ Python syntax check: `python3 -m py_compile qig-backend/training/celery_app.py`
- ✅ Python syntax check: `python3 -m py_compile qig-backend/training/tasks.py`
- ✅ JSON validation: `railway-celery-worker.json`
- ✅ JSON validation: `railway-celery-beat.json`

### Code Review
- ✅ Backward compatibility maintained
- ✅ All task decorators properly applied
- ✅ Task routing configured correctly
- ✅ Beat schedule configured correctly
- ✅ Environment variables documented
- ✅ Error handling preserved

## Deployment Checklist

### Prerequisites
- [ ] Railway account with project created
- [ ] PostgreSQL service added to Railway project
- [ ] Redis service added to Railway project

### Step 1: Main Web Service
- [ ] Already deployed (existing railpack.json)
- [ ] Set environment variables:
  - `DATABASE_URL=${DATABASE.DATABASE_URL}`
  - `REDIS_URL=${REDIS.REDIS_URL}`
  - `NODE_ENV=production`
  - `TRAINING_ENABLED=true`
- [ ] Verify health check: `/api/health`

### Step 2: Celery Worker Service
- [ ] Add new service to Railway project
- [ ] Connect same GitHub repository
- [ ] Set custom start command: `cd qig-backend && celery -A training.celery_app worker --loglevel=info --concurrency=2`
- [ ] Set environment variables (same as main service)
- [ ] Deploy and check logs for "celery@hostname ready"
- [ ] Verify tasks registered in worker logs

### Step 3: Celery Beat Service
- [ ] Add another service to Railway project
- [ ] Connect same GitHub repository
- [ ] Set custom start command: `cd qig-backend && celery -A training.celery_app beat --loglevel=info`
- [ ] Set environment variables (only REDIS_URL needed)
- [ ] Deploy and check logs for "beat: Starting..."
- [ ] Verify schedule confirmations in Beat logs

### Step 4: Verification
- [ ] Check main service logs for "Celery available: True"
- [ ] Check worker logs show registered tasks
- [ ] Check Beat logs show schedule confirmations
- [ ] Test async task dispatch from main service
- [ ] Monitor Redis queue depth
- [ ] Wait for first scheduled task trigger (hourly batch at :00)
- [ ] Verify training task execution in worker logs

### Step 5: Cleanup (Optional)
- [ ] After 24 hours of successful operation
- [ ] Remove or deprecate cron routes in `qig-backend/routes/cron_routes.py`
- [ ] Remove Railway cron job configurations

## Cost Estimate

**Railway Services:**
- Main Web Service: $5-20/month (usage-based)
- Celery Worker: $5-10/month
- Celery Beat: $5/month (very light)
- Redis: $5/month (starter plan)
- PostgreSQL: $5-20/month (size-based)

**Total: ~$25-60/month** depending on traffic

## Monitoring

### Health Checks

**Main Service:**
```bash
curl https://your-app.railway.app/api/health
```

**Celery Worker Status:**
```bash
# Check Railway service logs for:
[INFO] celery@hostname ready.
[INFO] Registered tasks: training.tasks.*
```

**Celery Beat Status:**
```bash
# Check Railway service logs for:
[INFO] beat: Starting...
[INFO] Scheduler: Sending due task hourly-batch-training
```

### Task Monitoring

**Using Flower (Optional):**
```bash
# Add Flower service (web-based monitoring)
celery -A training.celery_app flower --port=5555
```

Access at `http://localhost:5555` to see:
- Active workers
- Task history
- Success/failure rates
- Queue lengths

## Troubleshooting

### Issue: Tasks not processing

**Symptoms:**
- Tasks queued but never execute
- Worker logs show no activity

**Solutions:**
1. Check REDIS_URL is set correctly on worker service
2. Verify worker service is running (Railway dashboard)
3. Check worker logs for connection errors
4. Restart worker service

### Issue: Beat not triggering tasks

**Symptoms:**
- No scheduled tasks at expected times
- Beat logs show no schedule confirmations

**Solutions:**
1. Verify only ONE Beat instance is running
2. Check REDIS_URL is set correctly on Beat service
3. Check Beat service logs for schedule confirmations
4. Restart Beat service

### Issue: Tasks timing out

**Symptoms:**
- Worker logs show tasks killed
- Task status shows FAILURE with timeout

**Solutions:**
1. Increase task time limits in `celery_app.py`:
   ```python
   celery_app.conf.update(
       task_soft_time_limit=600,  # 10 minutes
       task_time_limit=1200,      # 20 minutes
   )
   ```
2. Increase worker concurrency if CPU-bound
3. Scale worker to higher tier in Railway

## References

- **Railway Deployment Guide:** `RAILWAY_DEPLOYMENT.md`
- **Celery Tasks Reference:** `CELERY_TASKS_GUIDE.md`
- **Training Loop Guide:** `qig-backend/TRAINING_LOOP_GUIDE.md`
- **Celery Documentation:** https://docs.celeryq.dev/
- **Railway Documentation:** https://docs.railway.app/

## Success Criteria

✅ All configuration files created and syntax-validated
✅ Backward compatibility maintained (works without Celery)
✅ Task decorators applied to all training functions
✅ Beat schedule configured for periodic tasks
✅ Comprehensive documentation provided
✅ Deployment checklist prepared

**Status: Ready for Railway deployment and testing**

## Next Actions

1. Deploy three services to Railway using provided configs
2. Configure environment variables (DATABASE_URL, REDIS_URL)
3. Monitor logs for successful startup
4. Test async task dispatch
5. Wait for first scheduled task (hourly batch)
6. Verify training execution in worker logs
7. Monitor for 24 hours before removing legacy cron routes

---

**Implementation by:** GitHub Copilot Agent
**Date:** 2026-01-28
**Branch:** copilot/update-railway-config-files
