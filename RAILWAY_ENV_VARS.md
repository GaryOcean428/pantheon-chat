# Railway Environment Variables Configuration

## Service: `pantheon-chat-web`

### Required Variables

```bash
# ============================================================================
# DATABASE (Reference from Railway PostgreSQL plugin)
# ============================================================================
DATABASE_URL=${{postgres.DATABASE_URL}}

# ============================================================================
# REDIS (Reference from Railway Redis plugin)
# ============================================================================
REDIS_URL=${{redis.REDIS_URL}}
CELERY_BROKER_URL=${{redis.REDIS_URL}}
CELERY_RESULT_BACKEND=${{redis.REDIS_URL}}

# ============================================================================
# API KEYS (Set these directly - DO NOT use references)
# ============================================================================
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
INTERNAL_API_KEY=your-secure-internal-key

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================
NODE_ENV=production
# PORT is automatically set by Railway - DO NOT set manually

# Python backend URL (internal communication)
PYTHON_BACKEND_URL=http://localhost:5001

# Node backend URL (for Python callbacks)
NODE_BACKEND_URL=http://localhost:$PORT

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_ENABLED=true
TRAINING_MODE=automatic
TRAINING_BATCH_SIZE=8
TRAINING_LR=1e-4
TRAINING_PHI_TARGET=0.70
CHECKPOINT_RETENTION=10

# ============================================================================
# OPTIONAL - EXTERNAL SERVICES
# ============================================================================
TAVILY_API_KEY=tvly-your-key-here
```

---

## Service: `celery-worker`

### Required Variables (Reference from other services)

```bash
# ============================================================================
# DATABASE (Reference from PostgreSQL plugin)
# ============================================================================
DATABASE_URL=${{postgres.DATABASE_URL}}

# ============================================================================
# REDIS (Reference from Redis plugin)
# ============================================================================
CELERY_BROKER_URL=${{redis.REDIS_URL}}
CELERY_RESULT_BACKEND=${{redis.REDIS_URL}}

# ============================================================================
# TRAINING CONFIGURATION (Same as web service)
# ============================================================================
TRAINING_ENABLED=true
TRAINING_MODE=automatic
TRAINING_BATCH_SIZE=8
TRAINING_LR=1e-4
TRAINING_PHI_TARGET=0.70
CHECKPOINT_RETENTION=10

# ============================================================================
# CELERY WORKER CONFIGURATION
# ============================================================================
C_FORCE_ROOT=true
PYTHONUNBUFFERED=1
```

---

## How to Set Variables in Railway

### Method 1: Railway Dashboard (Recommended)

1. Go to Railway → Projects → QIG Project Pantheon Group
2. Select service (e.g., `pantheon-chat-web`)
3. Click "Variables" tab
4. Click "+ New Variable"
5. Add each variable with proper reference syntax

### Method 2: Railway CLI

```bash
# Link to project
railway link

# Set variable for web service
railway variables --service pantheon-chat-web set DATABASE_URL='${{postgres.DATABASE_URL}}'

# Set variable for celery worker
railway variables --service celery-worker set CELERY_BROKER_URL='${{redis.REDIS_URL}}'
```

---

## Reference Variable Syntax

### ✅ CORRECT - Reference another service's variable

```bash
# Reference database URL from postgres plugin
DATABASE_URL=${{postgres.DATABASE_URL}}

# Reference Redis URL
REDIS_URL=${{redis.REDIS_URL}}

# Reference public domain (if service is exposed)
BACKEND_URL=https://${{backend.RAILWAY_PUBLIC_DOMAIN}}

# Reference private domain (for internal communication)
INTERNAL_API=http://${{backend.RAILWAY_PRIVATE_DOMAIN}}
```

### ❌ WRONG - These will fail

```bash
# Cannot reference PORT (Railway provides this automatically)
BACKEND_URL=http://${{backend.PORT}}

# Cannot use inputs in install step
"install": {
  "inputs": [{"step": "setup"}],  # Wrong!
}

# Don't hardcode values that should be references
DATABASE_URL=postgresql://user:pass@postgres:5432/db  # Wrong!
```

---

## Verification

After setting variables, verify with Railway CLI:

```bash
# Check all variables for web service
railway variables --service pantheon-chat-web

# Check all variables for celery worker
railway variables --service celery-worker

# Test in runtime
railway run env | grep -E "(DATABASE|REDIS|CELERY)"
```

---

## PostgreSQL pgvector Extension

**CRITICAL**: pgvector is an EXTENSION, not a separate service.

### Enable pgvector in Railway PostgreSQL

```bash
# Connect to Railway PostgreSQL
railway run psql $DATABASE_URL

# Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

# Verify
\dx
# Should show: vector | 0.x.x | public | ...

# Exit
\q
```

### Verify from Application

```python
# In Python backend
import psycopg2

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Check extension
cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
result = cur.fetchone()

if result:
    print("✅ pgvector extension is enabled")
else:
    print("❌ pgvector extension NOT found")
```

---

## Common Issues

### Issue: "pgvector not found"

**Solution**: Enable extension in PostgreSQL (see above)

### Issue: "CELERY_BROKER_URL not set"

**Solution**: Add Redis plugin to project, then reference it:
```bash
CELERY_BROKER_URL=${{redis.REDIS_URL}}
```

### Issue: "Cannot reference backend.PORT"

**Solution**: Use public domain instead:
```bash
BACKEND_URL=https://${{backend.RAILWAY_PUBLIC_DOMAIN}}
```

### Issue: "Multiple DATABASE_URL variables"

**Solution**: Only ONE PostgreSQL service should exist. Delete duplicates.

---

## Railway Plugins You Need

Add these from Railway marketplace:

1. **PostgreSQL** (Primary database)
   - After adding, enable pgvector extension
   - Provides: `DATABASE_URL`

2. **Redis** (Celery broker)
   - Provides: `REDIS_URL`

3. **No separate "pgvector" plugin needed** - it's part of PostgreSQL!

---

## Next Steps

1. ✅ Created railpack.json files
2. ⏳ Set environment variables (follow guide above)
3. ⏳ Enable pgvector extension in PostgreSQL
4. ⏳ Deploy services
5. ⏳ Verify training works
