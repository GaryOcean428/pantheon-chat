# Railway Service Cleanup & Configuration Guide

**Project**: QIG Project (Pantheon Chat)
**Status**: Action Required
**Date**: December 30, 2025

---

## Objective

Clean up redundant Railway services and configure proper architecture for pantheon-chat with kernel training support.

---

## Current Service Inventory

Your QIG Project currently has **13 services**. Here's the breakdown:

### Database Services (4 total - TOO MANY!)
1. **Postgres-Pantheon** - KEEP (for pantheon-chat)
2. **Postgres** - KEEP (for qsearch)
3. **pgvector-pantheon** - DELETE (redundant, pantheon-chat now uses Postgres-Pantheon)
4. **pgvector-GKBo** - DELETE (unused)

### Redis Services (3 total)
1. **Redis** - KEEP (for Celery)
2. **Redis Stack** - KEEP (if used by other services)
3. **Redis-c0pG** - AUDIT (check if used)

### Application Services
1. **pantheon-chat** - KEEP (main web service)
2. **qsearch-core** - KEEP
3. **qsearch-web** - KEEP
4. **worker** - KEEP (existing worker)
5. **Beat** - KEEP (Celery beat scheduler)
6. **code-server** - KEEP (development environment)

---

## CRITICAL: Understanding pgvector

**pgvector is an EXTENSION that runs INSIDE PostgreSQL, NOT a separate database service!**

### Common Misconception
```
pgvector-pantheon -> separate service
Postgres-Pantheon -> separate service
```

### Reality
```
Postgres-Pantheon (PostgreSQL with pgvector extension enabled inside it)
```

You **cannot** "connect pgvector to Postgres" because pgvector is a **PostgreSQL extension**, not a standalone service.

---

## Step-by-Step Cleanup

### Step 1: Enable pgvector Extension in Postgres-Pantheon

**CRITICAL**: Do this FIRST before deleting anything!

```bash
# Option A: Railway CLI
railway run --service Postgres-Pantheon psql $DATABASE_URL

# Option B: Railway Dashboard
# Go to Postgres-Pantheon service -> Shell

# Then run these SQL commands:
CREATE EXTENSION IF NOT EXISTS vector;

# Verify it worked:
\dx

# Should show:
# Name   | Version | Schema | Description
# -------+---------+--------+-------------
# vector | 0.x.x   | public | vector data type and operations

# Exit:
\q
```

### Step 2: Verify pantheon-chat Uses Postgres-Pantheon

Already done via Railway MCP!

pantheon-chat now has:
```
DATABASE_URL=postgresql://postgres:AbPiYEkiKGvOpvUCyfLYmVzMWKhrKdto@postgres-2dhz.railway.internal:5432/railway
```

This points to **Postgres-Pantheon**.

### Step 3: Delete Redundant Database Services

**Safe to delete** (after Step 1 is complete):

#### Delete: pgvector-pantheon
```
Service ID: 254c04f2-5810-4725-bf71-2217897d44b8
Reason: pantheon-chat no longer uses this (now uses Postgres-Pantheon)
URL: ballast.proxy.rlwy.net:24142
```

**How to delete**:
1. Railway Dashboard -> QIG Project
2. Click on **pgvector-pantheon** service
3. Settings -> Danger Zone
4. Click "Delete Service"
5. Confirm deletion

#### Delete: pgvector-GKBo
```
Service ID: 5dbbf38d-28f1-487b-a075-789a787c4e68
Reason: Not connected to any service
URL: yamabiko.proxy.rlwy.net:51824
```

**How to delete**:
1. Railway Dashboard -> QIG Project
2. Click on **pgvector-GKBo** service
3. Settings -> Danger Zone
4. Click "Delete Service"
5. Confirm deletion

### Step 4: Audit Redis Services

Check which Redis services are actually used:

```bash
# Check which services reference Redis
railway variables --service pantheon-chat | grep REDIS
railway variables --service qsearch-core | grep REDIS
railway variables --service worker | grep REDIS
```

**Expected**:
- **Redis** (369e158f) -> Used by Celery (pantheon-chat)
- **Redis Stack** (5a7a7caf) -> Used by other services
- **Redis-c0pG** (72a46e14) -> Check if used

If Redis-c0pG is unused, delete it.

---

## Final Architecture (After Cleanup)

### Database Services (2)
```
Postgres-Pantheon
  |-- Used by: pantheon-chat
  |-- Extension: pgvector enabled
  |-- Internal URL: postgres-2dhz.railway.internal:5432

Postgres
  |-- Used by: qsearch-core, qsearch-web
  |-- Internal URL: postgres.railway.internal:5432
```

### Redis Services (2)
```
Redis
  |-- Used by: pantheon-chat (Celery broker)
  |-- Internal URL: redis.railway.internal:6379

Redis Stack
  |-- Used by: Other services
  |-- Internal URL: redis-stack.railway.internal:6379
```

### Application Services (7)
```
pantheon-chat          -> Main web app
qsearch-core          -> Search backend
qsearch-web           -> Search frontend
worker                -> Existing Celery worker
Beat                  -> Celery scheduler
code-server           -> Development environment
[Future: celery-worker] -> New training worker (when ready)
```

---

## Verification Checklist

After cleanup, verify:

### Database Connectivity
```bash
# Test pantheon-chat can connect to Postgres-Pantheon
railway run --service pantheon-chat psql $DATABASE_URL

# Should connect successfully
# Check pgvector extension:
\dx

# Should show "vector" extension
\q
```

### Service Count
```bash
railway status

# Should show approximately 9-10 services (down from 13)
```

### No Broken References
```bash
# Check all services have valid DATABASE_URL
railway variables --service pantheon-chat | grep DATABASE_URL
railway variables --service qsearch-core | grep DATABASE_URL

# Both should return valid URLs
```

### pantheon-chat Deployment
```bash
# Trigger redeploy to use new database
railway redeploy --service pantheon-chat

# Watch logs
railway logs --service pantheon-chat
```

---

## Common Issues & Solutions

### Issue 1: "pgvector extension not found"

**Symptom**: App errors saying `type "vector" does not exist`

**Solution**:
```bash
railway run --service Postgres-Pantheon psql $DATABASE_URL
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

### Issue 2: "Connection refused after deleting pgvector-pantheon"

**Symptom**: pantheon-chat can't connect to database

**Cause**: DATABASE_URL still points to deleted service

**Solution**:
```bash
# Verify current DATABASE_URL
railway variables --service pantheon-chat | grep DATABASE_URL

# Should be: postgresql://...@postgres-2dhz.railway.internal:5432/railway
# If not, set it:
railway variables --service pantheon-chat set DATABASE_URL='${{Postgres-Pantheon.DATABASE_URL}}'
```

### Issue 3: "Lost data after deleting old database"

**Prevention**:
- pantheon-chat is NOW using Postgres-Pantheon (verified via MCP)
- Old pgvector-pantheon can be safely deleted
- Data is in Postgres-Pantheon (the one pantheon-chat connects to)

If you need to migrate data:
```bash
# Dump from old database
railway run --service pgvector-pantheon pg_dump $DATABASE_URL > backup.sql

# Import to new database
railway run --service Postgres-Pantheon psql $DATABASE_URL < backup.sql
```

### Issue 4: "Celery worker can't connect to Redis"

**Symptom**: Worker logs show connection errors

**Solution**:
```bash
# Verify Celery broker URL
railway variables --service pantheon-chat | grep CELERY_BROKER_URL

# Should be: redis://...@redis.railway.internal:6379
# Already set via Railway MCP
```

---

## Action Items Summary

### Immediate Actions (Do Now)
1. **Enable pgvector extension in Postgres-Pantheon** (SQL: `CREATE EXTENSION IF NOT EXISTS vector;`)
2. **Test pantheon-chat connection** (verify it works with Postgres-Pantheon)
3. **Delete pgvector-pantheon service** (after confirming pantheon-chat works)
4. **Delete pgvector-GKBo service** (unused, safe to delete)

### Optional Actions (Audit)
1. **Check Redis-c0pG usage** (delete if unused)
2. **Verify all services after cleanup** (railway status)

### Future Actions (When Ready for Training)
1. **Create celery-worker service** (use celery-worker/railpack.json)
2. **Add training environment variables** (see RAILWAY_ENV_VARS.md)
3. **Deploy training infrastructure**

---

## Expected Results

**Before Cleanup**: 13 services, 4 PostgreSQL instances, confusion
**After Cleanup**: 9-10 services, 2 PostgreSQL instances, clarity

**Cost Savings**: ~2-3 fewer database services = reduced Railway costs
**Complexity Reduction**: Clear architecture, no redundancy
**Performance**: No change (same actual usage, just cleaner)

---

## Need Help?

If anything goes wrong during cleanup:

1. **Railway logs are your friend**:
   ```bash
   railway logs --service pantheon-chat
   ```

2. **Check service connectivity**:
   ```bash
   railway run --service pantheon-chat env | grep DATABASE_URL
   ```

3. **Rollback if needed**:
   - Don't delete services until you've verified connections work
   - Railway keeps deleted services for 7 days (can restore)

4. **Test before deleting**:
   ```bash
   # Test connection to new database BEFORE deleting old one
   railway run --service pantheon-chat psql $DATABASE_URL -c "SELECT 1;"
   ```

---

## Cleanup Complete Checklist

- [ ] pgvector extension enabled in Postgres-Pantheon
- [ ] pantheon-chat successfully connects to Postgres-Pantheon
- [ ] pgvector-pantheon service deleted
- [ ] pgvector-GKBo service deleted
- [ ] Redis-c0pG audited (kept or deleted)
- [ ] All remaining services show "RUNNING" status
- [ ] Railway service count: 9-10 (down from 13)
- [ ] No broken database references
- [ ] pantheon-chat deploys successfully
- [ ] Application works as expected

---

**Last Updated**: December 30, 2025
**Status**: Ready for Implementation
**Related Docs**: RAILWAY_ENV_VARS.md, railpack.json
