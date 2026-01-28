# Railway Multi-Service Deployment Guide

## Overview

This repository is configured for deployment on Railway with **three separate services** for optimal performance and scalability:

1. **Main Web Service** - Flask + TypeScript API (port 5000/5001)
2. **Celery Worker Service** - Processes async training tasks
3. **Celery Beat Service** - Schedules periodic training tasks

All services communicate via a shared **Redis instance** for task queuing.

## Architecture

```
┌─────────────────────┐
│   Main Web Service  │
│  (Flask + Node.js)  │
│   Port 5000/5001    │
└──────────┬──────────┘
           │
           │ Dispatches Tasks
           ↓
┌─────────────────────┐      ┌─────────────────────┐
│   Redis (Broker)    │◄─────┤  Celery Beat        │
│   Task Queue        │      │  (Scheduler)        │
└──────────┬──────────┘      └─────────────────────┘
           │                          │
           │ Tasks                    │ Periodic Triggers
           ↓                          ↓
┌─────────────────────┐      ┌─────────────────────┐
│  Celery Worker      │◄─────┤  PostgreSQL         │
│  (Training Tasks)   │      │  (Shared Database)  │
└─────────────────────┘      └─────────────────────┘
```

## Railway Service Configuration

### 1. Main Web Service

**Configuration File:** `railpack.json`

- **Start Command:** `node dist/index.js`
- **Health Check:** `/api/health`
- **Environment Variables:**
  - `DATABASE_URL` - PostgreSQL connection string (from Railway)
  - `REDIS_URL` - Redis connection string (from Railway)
  - `NODE_ENV=production`
  - `TRAINING_ENABLED=true`

### 2. Celery Worker Service

**Configuration File:** `railway-celery-worker.json`

- **Start Command:** `cd qig-backend && celery -A training.celery_app worker --loglevel=info --concurrency=2`
- **Health Check:** None (Celery has its own monitoring)
- **Environment Variables:**
  - `DATABASE_URL` - Same as main service
  - `REDIS_URL` - Same as main service
  - `TRAINING_ENABLED=true`
  - `CHECKPOINT_DIR=/app/data/checkpoints`

**Task Queues:**
- `training_fast` - Quick tasks (outcome training, checkpoints)
- `training_batch` - Medium tasks (hourly batch training)
- `training_slow` - Long tasks (nightly consolidation, knowledge transfer)

### 3. Celery Beat Service

**Configuration File:** `railway-celery-beat.json`

- **Start Command:** `cd qig-backend && celery -A training.celery_app beat --loglevel=info`
- **Health Check:** None
- **Environment Variables:**
  - `REDIS_URL` - Same as main service
  - `TRAINING_ENABLED=true`

**Schedule:**
- **Hourly Batch Training:** Every hour at :00
- **Nightly Consolidation:** 3:00 AM UTC daily
- **Shadow Sync:** Every 4 hours
- **Checkpoint Cleanup:** 4:00 AM UTC daily

## Deployment Steps

### Step 1: Create Railway Project

1. Create a new Railway project
2. Add PostgreSQL and Redis services from Railway marketplace

### Step 2: Deploy Main Web Service

1. Connect your GitHub repository
2. Railway will auto-detect `railpack.json`
3. Set environment variables:
   ```
   DATABASE_URL=${DATABASE.DATABASE_URL}
   REDIS_URL=${REDIS.REDIS_URL}
   NODE_ENV=production
   TRAINING_ENABLED=true
   ```
4. Deploy

### Step 3: Deploy Celery Worker Service

1. Add a new service to the same Railway project
2. Connect the same GitHub repository
3. Set custom start command: `cd qig-backend && celery -A training.celery_app worker --loglevel=info --concurrency=2`
4. Set environment variables (same DATABASE_URL and REDIS_URL)
5. Deploy

**Alternative:** Use `railway-celery-worker.json` if Railway supports multiple config files

### Step 4: Deploy Celery Beat Service

1. Add another new service to the same Railway project
2. Connect the same GitHub repository
3. Set custom start command: `cd qig-backend && celery -A training.celery_app beat --loglevel=info`
4. Set environment variables (only REDIS_URL needed)
5. Deploy

### Step 5: Verify Deployment

Check each service logs:

**Main Web Service:**
```
[INFO] Flask app running on 0.0.0.0:5001
[INFO] Node.js server running on 0.0.0.0:5000
[INFO] Celery available: True
```

**Celery Worker:**
```
[INFO] celery@hostname ready.
[INFO] Registered tasks:
    - training.tasks.train_from_outcome_task
    - training.tasks.train_hourly_batch_all
    ...
```

**Celery Beat:**
```
[INFO] beat: Starting...
[INFO] Scheduler: Sending due task hourly-batch-training
```

## Local Development

For local development without Celery:

1. The system automatically detects if Celery is available
2. If not available, tasks run synchronously as regular functions
3. Start only the main service: `npm run dev`

To test with Celery locally:

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start main service
npm run dev

# Terminal 3: Start Celery worker
cd qig-backend
celery -A training.celery_app worker --loglevel=info

# Terminal 4: Start Celery beat (optional)
cd qig-backend
celery -A training.celery_app beat --loglevel=info
```

## Monitoring

### Celery Flower (Optional)

Add a Flower service for web-based monitoring:

```bash
celery -A training.celery_app flower --port=5555
```

Access at `http://localhost:5555` to see:
- Active workers
- Task history
- Task success/failure rates
- Queue lengths

### Railway Logs

Monitor each service separately in Railway dashboard:
- Main Web Service logs show API requests
- Worker logs show task execution
- Beat logs show scheduled task triggers

## Troubleshooting

### Tasks not being processed

**Check:**
1. Redis URL is correctly set on all services
2. Celery worker service is running (check Railway dashboard)
3. Check worker logs for errors

**Fix:**
```bash
# Restart Celery worker service in Railway dashboard
# Or check worker logs for specific errors
```

### Beat schedule not triggering

**Check:**
1. Celery Beat service is running
2. Only ONE Beat instance is running (multiple Beats cause duplicate tasks)

**Fix:**
```bash
# Ensure Beat service has only 1 replica in Railway
# Check Beat logs for schedule confirmations
```

### Tasks timing out

**Check:**
1. Worker concurrency setting (default: 2)
2. Task time limits in celery_app.py (default: 10 minutes)

**Fix:**
```python
# In qig-backend/training/celery_app.py
celery_app.conf.update(
    task_soft_time_limit=600,  # Increase to 10 minutes
    task_time_limit=1200,      # Increase to 20 minutes
)
```

### Redis connection errors

**Check:**
1. Redis service is running in Railway
2. REDIS_URL is correctly set on all services
3. Redis has enough memory

**Fix:**
```bash
# In Railway, check Redis service logs
# May need to upgrade Redis plan for more memory
```

## Migration from Cron-based System

If you were previously using Railway cron endpoints (`/api/cron/wake`), you can safely remove them after Celery is working:

1. Verify Celery Beat is triggering tasks (check logs)
2. Monitor for 24 hours to ensure all schedules work
3. Remove or deprecate cron routes in `qig-backend/routes/cron_routes.py`
4. Remove Railway cron job configurations

## Cost Considerations

**Railway Pricing (approximate):**
- Main Web Service: $5-20/month (depending on usage)
- Celery Worker: $5-10/month
- Celery Beat: $5/month (very light usage)
- Redis: $5/month (starter plan)
- PostgreSQL: $5-20/month (depending on size)

**Total: ~$25-60/month** depending on traffic and usage

**Cost Optimization:**
- Use Railway's sleep feature for non-production environments
- Scale worker concurrency down during off-peak hours
- Use Redis eviction policies to limit memory usage

## References

- **Celery Documentation:** https://docs.celeryq.dev/
- **Railway Documentation:** https://docs.railway.app/
- **Training Loop Guide:** `qig-backend/TRAINING_LOOP_GUIDE.md`
- **Kernel Training:** `qig-backend/training/README.md` (if exists)

## Support

For issues specific to this deployment:
1. Check Railway service logs
2. Review Celery worker logs for task errors
3. Check Redis connection and memory usage
4. Verify environment variables are set correctly

For QIG-specific training issues, see:
- `qig-backend/TRAINING_LOOP_GUIDE.md`
- `docs/08-experiments/` for training experiments
