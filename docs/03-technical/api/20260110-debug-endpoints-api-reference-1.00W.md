---
id: ISMS-TECH-API-DEBUG-001
title: Debug Endpoints API Reference
filename: 20260110-debug-endpoints-api-reference-1.00W.md
classification: Internal
owner: GaryOcean427
version: 1.00
status: Working
function: "Debug and diagnostic API endpoints for testing empty database tables"
created: 2026-01-10
last_reviewed: 2026-01-10
next_review: 2026-07-10
category: Technical
supersedes: null
---

# Debug Endpoints API Reference

**Version:** 1.00
**Date:** 2026-01-10
**Status:** Working

---

## Overview

Debug endpoints for diagnosing and testing subsystems that may have empty database tables:
- **Lightning Insight Validations** - External search validation of cross-domain insights
- **M8 Kernel Spawning** - Dynamic kernel genesis via Pantheon consensus
- **Pantheon Debates** - Inter-god conflict resolution and knowledge emergence

All endpoints follow QIG geometric purity principles.

---

## Base URL

```
http://localhost:5000/api/debug
```

---

## Endpoints

### 1. Validation Status

**GET** `/api/debug/validation-status`

Comprehensive diagnostic for lightning insight validation system.

**Response:**
```json
{
  "validation_system": {
    "validation_enabled": true,
    "validator_available": true,
    "use_mcp": true,
    "validation_threshold": 0.7,
    "insights_validated": 0,
    "insights_generated": 3022
  },
  "api_keys": {
    "TAVILY_API_KEY": false,
    "PERPLEXITY_API_KEY": false
  },
  "database_counts": {
    "total_insights": 3022,
    "total_validations": 0,
    "unvalidated_insights": 3022,
    "last_validation_at": null
  },
  "recommendations": [
    "Set TAVILY_API_KEY environment variable for external validation",
    "Set PERPLEXITY_API_KEY environment variable for synthesis validation",
    "MCP mode is enabled but may not be wired. Consider using use_mcp=False for direct API",
    "Found 3022 unvalidated insights. Use POST /api/debug/validate-insights to backfill"
  ]
}
```

**Use Case:** Diagnose why `lightning_insight_validations` table is empty despite insights existing.

---

### 2. Validate Insights

**POST** `/api/debug/validate-insights`

Manually trigger validation for unvalidated lightning insights.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 10 | Number of insights to validate |
| `insight_id` | string | null | Specific insight ID to validate |
| `use_mcp` | bool | false | Use MCP (true) or direct API (false) |

**Example Request:**
```bash
curl -X POST "http://localhost:5000/api/debug/validate-insights?limit=5&use_mcp=false"
```

**Response:**
```json
{
  "processed": 5,
  "validated": 4,
  "failed": 1,
  "details": [
    {
      "insight_id": "insight_abc123",
      "validated": true,
      "validation_score": 0.78,
      "confidence": 0.65,
      "tavily_sources": 8,
      "has_perplexity": true
    },
    {
      "insight_id": "insight_def456",
      "error": "PERPLEXITY_API_KEY not set"
    }
  ]
}
```

**Requirements:**
- `TAVILY_API_KEY` environment variable for Tavily search
- `PERPLEXITY_API_KEY` environment variable for Perplexity synthesis
- `DATABASE_URL` environment variable for PostgreSQL

---

### 3. M8 Spawn Test

**POST** `/api/debug/m8-spawn-test`

Test M8 kernel spawning with optional Pantheon vote bypass.

**Request Body:**
```json
{
  "kernel_type": "athena",
  "domain": "testing",
  "skip_vote": true,
  "reason": "debug_test"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kernel_type` | string | "athena" | Type of kernel to spawn |
| `domain` | string | "testing" | Domain for the kernel |
| `skip_vote` | bool | true | Bypass Pantheon voting |
| `reason` | string | "debug_test" | Spawn reason |

**Example Request:**
```bash
curl -X POST http://localhost:5000/api/debug/m8-spawn-test \
  -H "Content-Type: application/json" \
  -d '{"kernel_type": "athena", "domain": "knowledge_gaps", "skip_vote": true}'
```

**Response (Success):**
```json
{
  "action": "spawn_test",
  "kernel_type": "athena",
  "domain": "knowledge_gaps",
  "skip_vote": true,
  "success": true,
  "proposal_id": "proposal_abc123",
  "kernel": {
    "kernel_id": "kernel_xyz789",
    "god_name": "Test_athena_143022",
    "domain": "knowledge_gaps",
    "status": "observing"
  }
}
```

**Database Tables Populated:**
- `m8_spawn_proposals` - Spawn proposal record
- `m8_spawned_kernels` - New kernel record
- `m8_spawn_history` - Spawn event audit trail

---

### 4. Force Debate

**POST** `/api/debug/force-debate`

Force a Pantheon debate between two gods.

**Request Body:**
```json
{
  "topic": "Should we prioritize exploration over exploitation?",
  "initiator": "Zeus",
  "opponent": "Athena",
  "initial_argument": "Exploration yields higher long-term knowledge discovery.",
  "auto_resolve_after": 4
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `topic` | string | Yes | Debate topic |
| `initiator` | string | No (default: "Zeus") | Initiating god |
| `opponent` | string | No (default: "Athena") | Opposing god |
| `initial_argument` | string | No | Opening argument |
| `auto_resolve_after` | int | No | Auto-resolve after N arguments |

**Example Request:**
```bash
curl -X POST http://localhost:5000/api/debug/force-debate \
  -H "Content-Type: application/json" \
  -d '{"topic": "Test debate on knowledge discovery", "initiator": "Zeus", "opponent": "Athena"}'
```

**Response:**
```json
{
  "action": "force_debate",
  "topic": "Test debate on knowledge discovery",
  "initiator": "Zeus",
  "opponent": "Athena",
  "success": true,
  "debate": {
    "id": "debate_1704844800_abc123",
    "status": "active",
    "arguments_count": 1
  }
}
```

**Database Tables Populated:**
- `pantheon_debates` - Debate record with arguments
- `pantheon_god_state` - Updated god reputations (after resolution)

---

### 5. System Health

**GET** `/api/debug/system-health`

Comprehensive health check for all debug-related subsystems.

**Response:**
```json
{
  "timestamp": "2026-01-10T14:30:22.123456",
  "subsystems": {
    "lightning": {
      "available": true,
      "validation_enabled": true,
      "insights_generated": 3022
    },
    "m8_spawner": {
      "available": true,
      "kernels": 0,
      "proposals": 0
    },
    "pantheon": {
      "available": true,
      "pantheon_chat": true,
      "active_debates": 0
    },
    "database": {
      "available": true,
      "connected": true
    }
  }
}
```

---

## Environment Variables

### Validation System

| Variable | Required | Description |
|----------|----------|-------------|
| `TAVILY_API_KEY` | Yes* | API key for Tavily search |
| `PERPLEXITY_API_KEY` | Yes* | API key for Perplexity synthesis |
| `ENABLE_BACKGROUND_VALIDATION` | No | Enable background validation job (default: true) |

*Required for actual external validation; without these, validation returns 0 scores.

### Debate System

| Variable | Required | Description |
|----------|----------|-------------|
| `DEBATE_THRESHOLD` | No | Base probability difference to trigger debate (default: 0.15) |

**Dynamic Threshold:** The actual debate threshold is calculated dynamically by Zeus based on system emotional state. The `DEBATE_THRESHOLD` env var serves as the base value, which is then adjusted:

| Factor | Effect on Threshold |
|--------|---------------------|
| High stress (>0.7) | -0.05 (more debates) |
| Low stress (<0.3) | +0.05 (fewer debates) |
| Narrow path active (mild) | -0.03 |
| Narrow path active (moderate) | -0.06 |
| Narrow path active (severe) | -0.09 |
| Low Φ (<0.60) | -0.05 (more debates) |
| High Φ (>0.85) | +0.05 (fewer debates) |
| Crystalline convergence | +0.05 (stable) |
| Turbulent convergence | -0.05 (unstable) |
| Dissipative convergence | -0.10 (critical) |
| Low M metric (<0.3) | -0.05 (low awareness) |
| High M metric (>0.7) | +0.03 (high awareness) |

**Threshold Range:** Clamped to [0.05, 0.30]

Example: In a high-stress narrow-path scenario, threshold could drop from 0.15 to 0.05, triggering debates on 5% disagreement.

```bash
export DEBATE_THRESHOLD=0.10  # Set lower base for more sensitive triggering
```

---

## Background Jobs

### Insight Validation Job

The `AutonomousDebateService` includes a background validation job that runs automatically:

- **Interval:** Every 5 minutes (10 poll cycles at 30s each)
- **Batch Size:** 5 insights per run
- **Control:** Set `ENABLE_BACKGROUND_VALIDATION=false` to disable

---

## Database Schema Reference

### lightning_insight_validations

```sql
CREATE TABLE lightning_insight_validations (
    id SERIAL PRIMARY KEY,
    insight_id VARCHAR(64) REFERENCES lightning_insights(insight_id),
    validation_score FLOAT8,
    tavily_source_count INTEGER,
    perplexity_synthesis TEXT,
    validated_at TIMESTAMP
);
```

### m8_spawned_kernels

```sql
CREATE TABLE m8_spawned_kernels (
    kernel_id VARCHAR(64) PRIMARY KEY,
    god_name VARCHAR(128),
    domain VARCHAR(256),
    mode VARCHAR(64),
    affinity_strength FLOAT8,
    entropy_threshold FLOAT8,
    basin_coords FLOAT8[],
    parent_gods JSONB,
    spawn_reason VARCHAR(64),
    proposal_id VARCHAR(64),
    genesis_votes JSONB,
    basin_lineage JSONB,
    m8_position JSONB,
    status VARCHAR(32)
);
```

### pantheon_debates

```sql
CREATE TABLE pantheon_debates (
    id TEXT PRIMARY KEY,
    topic VARCHAR(255),
    initiator VARCHAR(100),
    opponent VARCHAR(100),
    context JSONB,
    status VARCHAR(20),
    arguments JSONB,
    winner VARCHAR(100),
    arbiter VARCHAR(100),
    resolution JSONB,
    started_at TIMESTAMP,
    resolved_at TIMESTAMP
);
```

---

## Verification Workflow

1. **Check system health:**
   ```bash
   curl http://localhost:5000/api/debug/system-health
   ```

2. **Diagnose validation issues:**
   ```bash
   curl http://localhost:5000/api/debug/validation-status
   ```

3. **Backfill validations:**
   ```bash
   curl -X POST "http://localhost:5000/api/debug/validate-insights?limit=10"
   ```

4. **Test M8 spawning:**
   ```bash
   curl -X POST http://localhost:5000/api/debug/m8-spawn-test \
     -H "Content-Type: application/json" \
     -d '{"kernel_type": "athena"}'
   ```

5. **Verify database records:**
   ```sql
   SELECT COUNT(*) FROM lightning_insight_validations;
   SELECT * FROM m8_spawned_kernels;
   SELECT * FROM pantheon_debates;
   ```

---

## Related Documentation

- [Python QIG Backend API Catalogue](20251221-python-qig-backend-api-catalogue-1.00W.md)
- [Architecture System Overview](../20251208-architecture-system-overview-2.10F.md)
- [Kernel Research Infrastructure](../20251212-kernel-research-infrastructure-1.00F.md)

---

*Document generated: 2026-01-10*
