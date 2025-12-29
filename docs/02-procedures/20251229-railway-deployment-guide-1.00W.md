---
id: ISMS-PROC-020
title: Railway Deployment Guide
filename: 20251229-railway-deployment-guide-1.00W.md
classification: Internal
owner: GaryOcean477
version: 1.00
status: Working
function: "Deployment procedures for Railway platform with database sync"
created: 2025-12-29
last_reviewed: 2025-12-29
next_review: 2026-06-29
category: Procedure
supersedes: null
---

# Railway Deployment Guide

## Overview

This guide covers deploying Pantheon-Chat to Railway with:
- PostgreSQL database (Railway Postgres plugin)
- Database synchronization with Replit instance
- Persistent volumes for model cache and data

## Prerequisites

1. Railway account (https://railway.app)
2. Railway CLI installed: `npm install -g @railway/cli`
3. Existing Replit deployment (for sync)

## Quick Start

### 1. Create Railway Project

```bash
# Login to Railway
railway login

# Initialize project in repo
railway init

# Link to existing project or create new
railway link
```

### 2. Add PostgreSQL Database

In Railway Dashboard:
1. Click "New" → "Database" → "PostgreSQL"
2. Railway automatically sets `DATABASE_URL` environment variable

Or via CLI:
```bash
railway add --plugin postgresql
```

### 3. Configure Environment Variables

In Railway Dashboard → Variables:

```env
# Required
NODE_ENV=production
PORT=5000

# Database (auto-set by Railway Postgres plugin)
# DATABASE_URL=postgresql://...

# Sync Configuration (for Replit ↔ Railway sync)
SYNC_API_KEY=your-secure-sync-key-here
REPLIT_SYNC_URL=https://your-replit-instance.repl.co

# Federation (optional)
FEDERATION_MODE=standalone
# CENTRAL_NODE_URL=wss://central-node:8765

# Optional: Redis
# REDIS_URL=redis://...
```

### 4. Deploy

```bash
# Deploy using Nixpacks (automatic)
railway up

# Or deploy with Docker
railway up --dockerfile Dockerfile.railway
```

## Database Sync Between Replit and Railway

### Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Replit         │         │  Railway        │
│  (Primary)      │◄───────►│  (Secondary)    │
│                 │  Sync   │                 │
│  PostgreSQL     │  API    │  PostgreSQL     │
│  (Neon)         │         │  (Railway PG)   │
└─────────────────┘         └─────────────────┘
```

### Setting Up Sync

1. **Generate Sync API Key** (use same key on both instances):
   ```bash
   openssl rand -hex 32
   ```

2. **Configure Replit** (set in Replit Secrets):
   ```
   SYNC_API_KEY=your-generated-key
   ```

3. **Configure Railway** (set in Railway Variables):
   ```
   SYNC_API_KEY=your-generated-key
   REPLIT_SYNC_URL=https://your-replit-instance.repl.co
   ```

### Sync API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sync/status` | GET | Get sync status and table counts |
| `/api/sync/export/vocabulary` | GET | Export vocabulary data |
| `/api/sync/import/vocabulary` | POST | Import vocabulary data |
| `/api/sync/export/conversations` | GET | Export conversations |
| `/api/sync/trigger` | POST | Trigger sync from source |

### Manual Sync

```bash
# Check sync status
curl -H "x-sync-api-key: YOUR_KEY" \
  https://your-railway-instance.up.railway.app/api/sync/status

# Trigger sync from Replit to Railway
curl -X POST \
  -H "x-sync-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"sourceUrl": "https://your-replit-instance.repl.co", "tables": ["vocabulary"]}' \
  https://your-railway-instance.up.railway.app/api/sync/trigger
```

### Automated Sync (Cron)

Add to Railway with cron trigger or use external service:
```bash
# Every 6 hours
0 */6 * * * curl -X POST -H "x-sync-api-key: $SYNC_API_KEY" ...
```

## Volume Configuration

Railway allows **1 volume per service**. Configure via the Dashboard UI.

### Step-by-Step Volume Setup

1. **Go to Railway Dashboard** → Select your project → Select your service

2. **Navigate to Settings** → Scroll to **Volumes** section

3. **Create the Volume**
   - Click **+ New Volume**
   - Name: `pantheon-data`
   - Mount Path: `/app/data`

4. **Deploy** - Railway auto-mounts the volume on next deploy

### Volume Structure

The single volume contains subdirectories:
```
/app/data/
├── storage/   # QIG patterns, checkpoints, learning data
└── models/    # Cached model files, vocabulary embeddings
```

### Verify Volume

```bash
# SSH into Railway container
railway ssh

# Check volume is mounted with subdirectories
ls -la /app/data
ls -la /app/data/storage /app/data/models

# Test write access
echo "test" > /app/data/storage/test.txt && cat /app/data/storage/test.txt
```

### Volume Notes

- **Limit**: 1 volume per service (use subdirectories for organization)
- **Persistence**: Data survives deploys and restarts
- **Size**: 100GB free per service, $0.10/GB beyond
- **Backup**: Railway provides automatic backups

## Migrations

Run database migrations after first deploy:

```bash
# Via Railway CLI
railway run npm run db:push

# Or in Railway shell
npm run db:push
```

## Monitoring

### Health Check
```
GET /api/health
```

### Logs
```bash
railway logs --follow
```

### Metrics
Railway provides built-in metrics in the dashboard.

## Troubleshooting

### Database Connection Issues

1. Check `DATABASE_URL` is set:
   ```bash
   railway variables
   ```

2. Verify PostgreSQL service is running in Railway dashboard

3. Check logs for connection errors:
   ```bash
   railway logs | grep -i "database\|postgres"
   ```

### Python Backend Not Starting

1. Check Python dependencies installed:
   ```bash
   railway run pip list
   ```

2. Verify `PYTHONPATH` is set correctly

### Sync Failures

1. Verify `SYNC_API_KEY` matches on both instances
2. Check network connectivity between instances
3. Review sync endpoint logs

## File Structure

```
├── railway.toml           # Railway configuration
├── nixpacks.toml          # Nixpacks build config
├── Dockerfile.railway     # Railway-optimized Dockerfile
├── docker-compose.railway.yml  # Docker Compose for Railway
└── server/
    └── db-sync-api.ts     # Database sync endpoints
```

## Environment Variable Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `NODE_ENV` | Yes | Set to `production` |
| `PORT` | Yes | Server port (default: 5000) |
| `SYNC_API_KEY` | For sync | Shared secret for sync API |
| `REPLIT_SYNC_URL` | For sync | Replit instance URL |
| `REDIS_URL` | No | Redis connection string |
| `FEDERATION_MODE` | No | `standalone`, `central`, or `edge` |

## Security Notes

1. **SYNC_API_KEY**: Use a strong, randomly generated key
2. **Network**: Railway provides automatic HTTPS
3. **Database**: Railway Postgres is accessible only within project by default
4. **Secrets**: Use Railway's encrypted variables for sensitive data

## Related Documents

- [Replit Deployment Guide](20251212-replit-deployment-guide-1.00W.md)
- [Railway + Replit Deployment](20251208-deployment-railway-replit-1.00F.md)
- [Federation Architecture](../03-technical/federation-architecture.md)
