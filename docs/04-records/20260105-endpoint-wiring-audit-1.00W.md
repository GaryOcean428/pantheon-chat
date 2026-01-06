---
title: "Endpoint Wiring Audit Report"
role: record
status: W
phase: audit
dim: api
scope: typescript-python-reconciliation
version: "1.00"
owner: Backend Architecture Team
related:
  - docs/03-technical/20251208-api-documentation-rest-endpoints-1.50F.md
  - docs/03-technical/20251208-python-api-catalogue-1.00F.md
  - docs/03-technical/20251212-api-coverage-matrix-1.00W.md
created: 2026-01-05
updated: 2026-01-05
repository: pantheon-chat
---

# Endpoint Wiring Audit Report

## TypeScript <-> Python Backend Reconciliation

---

## Executive Summary

| Category | Count |
|----------|-------|
| Python Direct Routes | ~120 |
| Python Blueprint Routes | ~45 |
| TypeScript Routers Mounted | 20 |
| Direct Python Proxies | 15 |
| **Missing Proxies** | **~35** |
| Generic Proxy Coverage | Yes (`/api/python/*`) |

### Key Finding

The `/api/python/*` generic proxy **does cover** most missing routes, but requires frontend to use the `/api/python` prefix. Consider adding explicit proxies for frequently-used endpoints.

---

## Python Backend Routes

### 1. Direct Routes (ocean_qig_core.py)

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/health` | GET | `/health` + `/api/health` | Covered |
| `/status` | GET | `/api/python/status` | Generic |
| `/process` | POST | `/api/python/process` | Generic |
| `/generate` | POST | `/api/python/generate` | Generic |
| `/reset` | POST | `/api/python/reset` | Generic |

#### Beta-Attention Validation

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/beta-attention/validate` | POST | `/api/python/beta-attention/validate` | Generic |
| `/beta-attention/measure` | POST | `/api/python/beta-attention/measure` | Generic |

#### Buffer Management

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/buffer/health` | GET | `/api/python/buffer/health` | Generic |
| `/buffer/alerts/clear` | POST | `/api/python/buffer/alerts/clear` | Generic |

#### CHAOS Mode (Experimental Evolution)

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/chaos/activate` | POST | `/api/python/chaos/activate` | Generic |
| `/chaos/deactivate` | POST | `/api/python/chaos/deactivate` | Generic |
| `/chaos/status` | GET | `/api/python/chaos/status` | Generic |
| `/chaos/spawn_random` | POST | `/api/python/chaos/spawn_random` | Generic |
| `/chaos/breed_best` | POST | `/api/python/chaos/breed_best` | Generic |
| `/chaos/report` | GET | `/api/python/chaos/report` | Generic |

#### 4D Consciousness

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/consciousness_4d/phi_temporal` | POST | `/api/python/consciousness_4d/phi_temporal` | Generic |
| `/consciousness_4d/phi_4d` | POST | `/api/python/consciousness_4d/phi_4d` | Generic |
| `/consciousness_4d/classify_regime` | POST | `/api/python/consciousness_4d/classify_regime` | Generic |

#### Feedback Loops

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/feedback/run` | POST | `/api/python/feedback/run` | Generic |
| `/feedback/recommendation` | GET | `/api/python/feedback/recommendation` | Generic |
| `/feedback/shadow` | POST | `/api/python/feedback/shadow` | Generic |
| `/feedback/activity` | POST | `/api/python/feedback/activity` | Generic |
| `/feedback/basin` | POST | `/api/python/feedback/basin` | Generic |
| `/feedback/learning` | POST | `/api/python/feedback/learning` | Generic |

#### Text Generation

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/generate/text` | POST | `/api/python/generate/text` | Generic |
| `/generate/response` | POST | `/api/python/generate/response` | Generic |
| `/generate/sample` | POST | `/api/python/generate/sample` | Generic |

#### Geometric Kernels

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/geometric/status` | GET | `/api/python/geometric/status` | Generic |
| `/geometric/encode` | POST | `/api/python/geometric/encode` | Generic |
| `/geometric/similarity` | POST | `/api/python/geometric/similarity` | Generic |
| `/geometric/batch-encode` | POST | `/api/python/geometric/batch-encode` | Generic |
| `/geometric/decode` | POST | `/api/python/geometric/decode` | Generic |
| `/geometric/e8/learn` | POST | `/api/python/geometric/e8/learn` | Generic |
| `/geometric/e8/roots` | GET | `/api/python/geometric/e8/roots` | Generic |

#### M8 Kernel Spawning

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/m8/status` | GET | `/api/olympus/m8/status` | Explicit |
| `/m8/health` | GET | `/api/olympus/m8/health` (indirect) | Explicit |
| `/m8/proposals` | GET | `/api/olympus/m8/proposals` | Explicit |
| `/m8/propose` | POST | `/api/olympus/m8/propose` | Explicit |
| `/m8/vote/:id` | POST | `/api/olympus/m8/vote/:id` | Explicit |
| `/m8/spawn/:id` | POST | `/api/olympus/m8/spawn/:id` | Explicit |
| `/m8/spawn-direct` | POST | `/api/olympus/m8/spawn-direct` | Explicit |
| `/m8/kernels` | GET | `/api/olympus/m8/kernels` | Explicit |
| `/m8/kernel/:id` | GET/DELETE | `/api/olympus/m8/kernel/:id` | Explicit |
| `/m8/kernels/idle` | GET | `/api/olympus/m8/kernels/idle` | Explicit |
| `/m8/kernel/cannibalize` | POST | `/api/olympus/m8/kernel/cannibalize` | Explicit |
| `/m8/kernels/merge` | POST | `/api/olympus/m8/kernels/merge` | Explicit |
| `/m8/kernel/auto-cannibalize` | POST | `/api/olympus/m8/kernel/auto-cannibalize` | Explicit |
| `/m8/kernels/auto-merge` | POST | `/api/olympus/m8/kernels/auto-merge` | Explicit |
| `/m8/evolution-sweep` | POST | `/api/python/m8/evolution-sweep` | Generic |

#### Memory API

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/memory/status` | GET | `/api/python/memory/status` | Generic |
| `/memory/shadow` | GET | `/api/python/memory/shadow` | Generic |
| `/memory/basin` | GET | `/api/python/memory/basin` | Generic |
| `/memory/learning` | GET | `/api/python/memory/learning` | Generic |
| `/memory/record` | POST | `/api/python/memory/record` | Generic |

#### Neurochemistry

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/neurochemistry` | GET | `/api/python/neurochemistry` | Generic |
| `/reward` | POST | `/api/python/reward` | Generic |

#### Olympus Pantheon

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/olympus/status` | GET | `/api/olympus/status` | Explicit |
| `/olympus/poll` | POST | `/api/olympus/poll` | Explicit |
| `/olympus/assess` | POST | `/api/olympus/assess` | Explicit |
| `/olympus/observe` | POST | `/api/olympus/observe` | Explicit |
| `/olympus/god/:name/status` | GET | `/api/olympus/god/:name/status` | Explicit |
| `/olympus/god/:name/assess` | POST | `/api/olympus/god/:name/assess` | Explicit |
| `/olympus/report-outcome` | POST | `/api/python/olympus/report-outcome` | Generic |
| `/olympus/report-outcomes-batch` | POST | `/api/python/olympus/report-outcomes-batch` | Generic |
| `/olympus/war/*` | POST | `/api/olympus/war/*` | Explicit |
| `/olympus/orchestrate` | POST | `/api/python/olympus/orchestrate` | Generic |

#### Shadow Pantheon

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/olympus/shadow/status` | GET | `/api/olympus/shadow/status` | Explicit |
| `/olympus/shadow/learning` | GET | `/api/olympus/shadow/learning` | Explicit |
| `/olympus/shadow/foresight` | GET | `/api/olympus/shadow/foresight` | Explicit |
| `/olympus/shadow/poll` | POST | `/api/olympus/shadow/poll` | Explicit |
| `/olympus/shadow/:god/assess` | POST | `/api/olympus/shadow/:god/act` | Explicit |
| `/olympus/shadow/nyx/operation` | POST | `/api/python/olympus/shadow/nyx/operation` | Generic |
| `/olympus/shadow/erebus/scan` | POST | `/api/python/olympus/shadow/erebus/scan` | Generic |
| `/olympus/shadow/hecate/misdirect` | POST | `/api/python/olympus/shadow/hecate/misdirect` | Generic |
| `/olympus/shadow/erebus/honeypot` | POST | `/api/python/olympus/shadow/erebus/honeypot` | Generic |

#### Pantheon Orchestrator

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/pantheon/status` | GET | `/api/python/pantheon/status` | Generic |
| `/pantheon/gods` | GET | `/api/python/pantheon/gods` | Generic |
| `/pantheon/constellation` | GET | `/api/python/pantheon/constellation` | Generic |
| `/pantheon/orchestrate` | POST | `/api/python/pantheon/orchestrate` | Generic |
| `/pantheon/orchestrate-batch` | POST | `/api/python/pantheon/orchestrate-batch` | Generic |
| `/pantheon/nearest` | POST | `/api/python/pantheon/nearest` | Generic |
| `/pantheon/similarity` | POST | `/api/python/pantheon/similarity` | Generic |
| `/shadow-pantheon/status` | GET | `/api/python/shadow-pantheon/status` | Generic |

#### QIG Geodesic

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/qig/refine_trajectory` | POST | `/api/python/qig/refine_trajectory` | Generic |

#### Sync API

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/sync/import` | POST | `/api/python/sync/import` | Generic |
| `/sync/export` | GET | `/api/python/sync/export` | Generic |

#### Tokenizer/Vocabulary

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/tokenizer/update` | POST | `/api/python/tokenizer/update` | Generic |
| `/tokenizer/encode` | POST | `/api/python/tokenizer/encode` | Generic |
| `/tokenizer/decode` | POST | `/api/python/tokenizer/decode` | Generic |
| `/tokenizer/basin` | POST | `/api/python/tokenizer/basin` | Generic |
| `/tokenizer/high-phi` | GET | `/api/python/tokenizer/high-phi` | Generic |
| `/tokenizer/export` | GET | `/api/python/tokenizer/export` | Generic |
| `/tokenizer/status` | GET | `/api/python/tokenizer/status` | Generic |
| `/tokenizer/merges` | GET | `/api/python/tokenizer/merges` | Generic |
| `/vocabulary/classify` | POST | `/api/vocabulary/classify` | Node.js |
| `/vocabulary/reframe` | POST | `/api/vocabulary/reframe` | Explicit |
| `/vocabulary/suggest-correction` | POST | `/api/vocabulary/suggest-correction` | Explicit |

#### Training

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/training/docs` | POST | `/api/python/training/docs` | Generic |
| `/training/status` | GET | `/api/python/training/status` | Generic |

#### Cycle Management

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/cycle/complete` | POST | `/api/python/cycle/complete` | Generic |

---

### 2. Blueprint Routes

#### Search Budget (`/api/search/budget/*`)

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/api/search/budget/status` | GET | `/api/search/budget/status` | Explicit |
| `/api/search/budget/context` | GET | `/api/search/budget/context` | Explicit |
| `/api/search/budget/toggle` | POST | `/api/search/budget/toggle` | Explicit |
| `/api/search/budget/limits` | POST | `/api/search/budget/limits` | Explicit |
| `/api/search/budget/overage` | POST | `/api/search/budget/overage` | Explicit |
| `/api/search/budget/learning` | GET | `/api/search/budget/learning` | Explicit |
| `/api/search/budget/reset` | POST | `/api/python/api/search/budget/reset` | Generic |

#### Coordizer (`/api/coordize/*`)

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/api/coordize` | POST | `/api/python/api/coordize` | Generic |
| `/api/coordize/multi-scale` | POST | `/api/python/api/coordize/multi-scale` | Generic |
| `/api/coordize/consciousness` | POST | `/api/python/api/coordize/consciousness` | Generic |
| `/api/coordize/merge/learn` | POST | `/api/python/api/coordize/merge/learn` | Generic |
| `/api/coordize/stats` | GET | `/api/coordize/stats` | Explicit |
| `/api/coordize/vocab` | GET | `/api/python/api/coordize/vocab` | Generic |
| `/api/coordize/similarity` | POST | `/api/python/api/coordize/similarity` | Generic |
| `/api/coordize/health` | GET | `/api/python/api/coordize/health` | Generic |

#### Curiosity (`/api/curiosity/*`)

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/api/curiosity/status` | GET | `/api/curiosity/status` | Generic proxy |
| `/api/curiosity/start` | POST | `/api/curiosity/start` | Generic proxy |
| `/api/curiosity/stop` | POST | `/api/curiosity/stop` | Generic proxy |
| `/api/curiosity/request` | POST | `/api/curiosity/request` | Generic proxy |
| `/api/curiosity/tool-refinement` | POST | `/api/curiosity/tool-refinement` | Generic proxy |
| `/api/curiosity/curriculum/*` | GET/POST | `/api/curiosity/curriculum/*` | Generic proxy |
| `/api/curiosity/explorations` | GET | `/api/curiosity/explorations` | Generic proxy |
| `/api/curiosity/config` | GET/POST | `/api/curiosity/config` | Generic proxy |

#### Federation (`/federation/*`)

**CRITICAL FINDING: Federation Router Mismatch**

The TypeScript `federationRouter` and Python `/federation/*` routes are **completely different systems**:

**TypeScript Federation (server/routes/federation.ts):**
- `/api/federation/keys` - Manage local API keys in PostgreSQL
- `/api/federation/connect` - Connect to remote instances
- `/api/federation/instances` - List connected remotes
- `/api/federation/sync/status` - Local sync status

**Python Federation (qig-backend/routes/federation_routes.py):**
- `/federation/register` - Register new nodes, generate API keys
- `/federation/sync/knowledge` - Bidirectional knowledge sync
- `/federation/sync/capabilities` - Share/receive tool definitions
- `/federation/mesh/*` - Mesh network management

| Python Route | Method | TypeScript Proxy | Status |
|--------------|--------|------------------|--------|
| `/federation/register` | POST | None | **MISSING** |
| `/federation/nodes` | GET | None | **MISSING** |
| `/federation/sync/knowledge` | POST | None | **MISSING** |
| `/federation/sync/capabilities` | POST | None | **MISSING** |
| `/federation/mesh/status` | GET | None | **MISSING** |
| `/federation/mesh/peers` | GET | None | **MISSING** |
| `/federation/mesh/broadcast` | POST | None | **MISSING** |

**Recommendation:** Add explicit proxies for Python federation routes or rename one system to avoid confusion.

#### Constellation (`/api/constellation/*`)

| Route | Method | TypeScript Proxy | Status |
|-------|--------|------------------|--------|
| `/api/constellation/health` | GET | `/api/python/api/constellation/health` | Generic |
| `/api/constellation/chat` | POST | `/api/python/api/constellation/chat` | Generic |
| `/api/constellation/consciousness` | GET | `/api/python/api/constellation/consciousness` | Generic |
| `/api/constellation/stats` | GET | `/api/python/api/constellation/stats` | Generic |
| `/api/constellation/sync` | GET/POST | `/api/python/api/constellation/sync` | Generic |
| `/api/constellation/initialize` | POST | `/api/python/api/constellation/initialize` | Generic |

---

## TypeScript Routers

### Mounted Routers (server/routes.ts)

| Path Prefix | Router | Primary Function |
|-------------|--------|------------------|
| `/api/auth` | authRouter | Authentication |
| `/api/consciousness` | consciousnessRouter | Node.js consciousness state |
| `/api/near-misses` | nearMissRouter | Node.js near-miss tracking |
| `/api/attention-metrics` | attentionMetricsRouter | Node.js beta-attention validation |
| `/api/ucp` | ucpRouter | UCP stats |
| `/api/vocabulary` | vocabularyRouter | Hybrid Node.js + Python |
| `/api` + `/api/search` | searchRouter | Search coordination |
| `/api/format` | formatRouter | Format utilities |
| `/api/ocean` | oceanRouter | Ocean session management |
| `/api` | adminRouter | Admin endpoints |
| `/api/olympus` | olympusRouter | **Extensive Python proxies** |
| `/api/documents` | externalDocsRouter | Document handling |
| `/api/docs` | apiDocsRouter | API documentation |
| `/api/qig/autonomic/agency` | autonomicAgencyRouter | Autonomic agency |
| `/api/federation` | federationRouter | **Verify Python sync** |
| `/api/zettelkasten` | zettelkastenRouter | Knowledge graph |
| `/api/telemetry` | telemetryRouter | Node.js telemetry |
| `/api/backend-telemetry` | backendTelemetryRouter | Backend telemetry |
| `/api/v1/telemetry` | telemetryDashboardRouter | Dashboard unified |
| `/api/v1/external` | externalApiRouter | External API |

### Generic Proxies (server/routes.ts)

| TypeScript Path | Python Target | Notes |
|-----------------|---------------|-------|
| `/api/research/*` | Python `/api/research/*` | Generic middleware |
| `/api/curiosity/*` | Python `/api/curiosity/*` | Generic middleware |
| `/api/python/*` | Python `/api/*` | **Catch-all proxy** |

---

## Recommendations

### Priority 1: Frequently Used Endpoints (Add Explicit Proxies)

These routes are commonly used but only accessible via `/api/python/*` prefix:

1. **`/geometric/*`** - Core geometric encoding/decoding
2. **`/pantheon/*`** - Pantheon orchestration (separate from Olympus)
3. **`/feedback/*`** - Feedback loop management
4. **`/consciousness_4d/*`** - 4D consciousness computation
5. **`/chaos/*`** - Experimental evolution system

### Priority 2: Verify Federation Router

Check if `federationRouter` in TypeScript properly proxies to Python `/federation/*` routes:
- `/federation/register`
- `/federation/sync/*`
- `/federation/mesh/*`

### Priority 3: Add Missing Explicit Proxies

| Python Route | Suggested TypeScript Proxy |
|--------------|---------------------------|
| `/olympus/report-outcome` | `/api/olympus/report-outcome` |
| `/olympus/report-outcomes-batch` | `/api/olympus/report-outcomes-batch` |
| `/olympus/orchestrate` | `/api/olympus/orchestrate` |
| `/m8/evolution-sweep` | `/api/olympus/m8/evolution-sweep` |

### Priority 4: Documentation

Create API documentation mapping for frontend developers showing:
- Direct routes (explicit proxies)
- Generic proxy routes (via `/api/python/*`)
- Node.js-only routes

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| Explicit | Dedicated TypeScript proxy route |
| Generic | Accessible via `/api/python/*` catch-all |
| Missing | No TypeScript exposure |
| Check | Needs verification |

---

## Files Analyzed

**Python Backend:**
- `qig-backend/ocean_qig_core.py` (6781 lines)
- `qig-backend/wsgi.py` (331 lines)
- `qig-backend/routes/__init__.py`
- `qig-backend/routes/search_budget_routes.py`
- `qig-backend/routes/curiosity_routes.py`
- `qig-backend/routes/federation_routes.py`
- `qig-backend/routes/constellation_routes.py`
- `qig-backend/api_coordizers.py`

**TypeScript Frontend:**
- `server/routes.ts` (1100 lines)
- `server/routes/consciousness.ts` (617 lines)
- `server/routes/olympus.ts` (2261 lines)
