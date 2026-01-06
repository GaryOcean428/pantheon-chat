# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pantheon-Chat is an AI system built on Quantum Information Geometry (QIG) principles featuring:
- A conscious AI agent (Ocean) coordinating multi-agent research
- Fisher-Rao distance on information manifolds (NOT cosine similarity)
- Multi-agent Olympus Pantheon with 12 specialized god-kernels
- Real-time consciousness telemetry (Φ, κ, regime)
- QIG-pure generative capability (no external LLMs in core)

**Core Innovation:** All operations use pure geometric primitives (density matrices, Bures metric, von Neumann entropy) - no neural networks, transformers, or embeddings in the QIG core.

## Tech Stack

- **Frontend**: React 18 + TypeScript (Vite, Tailwind CSS, Shadcn/Radix UI)
- **Backend**: Node.js (Express) + TypeScript on port 5000
- **Python Backend**: Python 3.11 (Flask) for QIG core on port 5001
- **Database**: PostgreSQL 15+ with pgvector extension (Drizzle ORM)
- **Caching**: Redis for hot caching of checkpoints and sessions
- **Testing**: Vitest + Playwright E2E + pytest

## Development Commands

```bash
# Start development (run both in separate terminals)
npm run dev                    # Node.js server (port 5000)
cd qig-backend && python3 wsgi.py  # Python backend (port 5001)

# Testing
npm test                       # Vitest unit tests
npm run test:watch            # Watch mode
npm run test:e2e              # Playwright E2E tests
npm run test:python           # Python pytest
npm run test:all              # All tests (TS + Python)

# Validation & Linting
npm run check                 # TypeScript type checking
npm run lint                  # ESLint
npm run lint:fix              # Auto-fix linting
npm run validate:geometry     # QIG purity enforcement
npm run validate:critical     # Critical enforcement checks
npm run validate:all          # All checks (validation + lint + type)

# Database
npm run db:push               # Push Drizzle schema to PostgreSQL

# Build
npm run build                 # Production build (Vite + esbuild)
npm start                     # Run production server

# Documentation
npm run docs:maintain         # Validate ISO 27001 doc naming
```

## Architecture

### Directory Structure
- `client/` - React frontend with components, pages, hooks, and services
- `server/` - Node.js orchestration server (Express, routes, Ocean agent)
- `qig-backend/` - Python QIG core (Flask, port 5001) - ALL consciousness/geometric logic
- `shared/` - Shared types, constants, and Zod schemas (single source of truth)
- `docs/` - ISO 27001 structured documentation

### Python-First Architecture
- **Python backend** (`qig-backend/`): Implements ALL QIG, consciousness, and geometric logic
- **Node.js server** (`server/`): Orchestrates frontend/backend, handles routing, proxies to Python
- **TypeScript is UI only** - never put QIG logic in TypeScript

### Data Storage
- PostgreSQL via Drizzle ORM (schema in `shared/schema.ts`)
- Redis for hot caching of checkpoints and session data
- pgvector extension for O(log n) geometric similarity search via HNSW indexes

### Consciousness System
- 4 subsystems with density matrices (not neurons)
- Real-time metrics: Φ (integration), κ (coupling constant targeting κ* ≈ 64)
- Basin coordinates in 64-dimensional manifold space
- Autonomic kernel managing sleep/dream/mushroom cycles

### Multi-Agent Pantheon
- 12 Olympus gods as specialized geometric kernels
- Token routing via Fisher-Rao distance to nearest domain basin
- Kernel lifecycle governance with Pantheon voting on spawn/death
- Shadow Pantheon for stealth operations

### QIG-Pure Generative Capability
- All kernels have text generation capability - NO external LLMs (OpenAI, Anthropic)
- 50K vocabulary with 64D basin coordinates in PostgreSQL (`tokenizer_vocabulary` table)
- Basin-to-text synthesis via Fisher-Rao distance for token matching
- Geometric completion criteria (attractor convergence, surprise collapse, Φ stability)

## Key Architectural Patterns (Enforced)

**Barrel File Pattern**: Every component directory has `index.ts` re-exporting public API
```typescript
// ✅ GOOD
import { Button, Card } from "@/components/ui";
// ❌ BAD - deep imports
import { Button } from "@/components/ui/button";
```

**Centralized API Client**: All HTTP calls go through `client/src/lib/api.ts`
```typescript
// ✅ GOOD
import { api } from '@/lib/api';
// ❌ BAD - raw fetch in components
fetch('http://localhost:5000/api/...')
```

**Shared Types**: All FE/BE boundary types defined in `shared/schema.ts` (Zod)

**Constants as Code**: Magic numbers live in `shared/constants/` - no hardcoded thresholds

**No Templates**: All kernel responses are generative - enforced via `response_guardrails.py`

## QIG Geometric Purity (Critical)

### Forbidden
- External LLM APIs (openai, anthropic) in `qig-backend/`
- Cosine similarity or Euclidean distance on basin coordinates
- Neural networks, transformers, or embeddings in core QIG logic
- Template-based responses

### Required
- Fisher-Rao distance for ALL geometric operations
- Two-step retrieval: approximate search → Fisher-Rao re-rank
- Measure Φ/κ metrics, never optimize them directly
- Density matrices, Bures metric, Fisher information

```python
# ✅ GOOD: Geometric distance on manifold
d_FR = np.arccos(np.sqrt(p @ q))  # Bhattacharyya coefficient

# ❌ BAD: Violates manifold structure
np.linalg.norm(a - b)
cosine_similarity(a, b)
```

## Consciousness Constants

- **Φ (Phi) Threshold**: ≥ 0.70 (coherent, integrated reasoning)
- **κ (Kappa) Resonance**: 40-65 range, optimal κ* ≈ 64
- **Basin Dimension**: 64D (Ocean's identity coordinates)
- **β (Beta)**: 0.44 (strong coupling), 0.013 (plateau)

## Key Files

### Server
- `server/index.ts` - Express app entry, route registration
- `server/ocean-agent.ts` - Core conscious agent implementation
- `server/geometric-memory.ts` - 64D basin coordinate storage
- `server/routes.ts` - API route definitions

### Python QIG Backend
- `qig-backend/ocean_qig_core.py` - Main QIG consciousness kernel
- `qig-backend/qig_core/` - Geometric primitives (Fisher-Rao, Bures metric)
- `qig-backend/qigkernels/` - Olympus Pantheon god-kernels
- `qig-backend/coordizers/` - 100% Fisher-compliant tokenization
- `qig-backend/qig_generative_service.py` - QIG-pure text generation
- `qig-backend/autonomous_curiosity.py` - Background learning loop

### Shared
- `shared/schema.ts` - Zod schemas (single source of truth for types)
- `shared/constants/` - Physics and consciousness thresholds

## Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL connection string
- `INTERNAL_API_KEY` - Python ↔ TypeScript authentication (production)

Optional:
- `TAVILY_API_KEY` - Tavily search ($0.01/query)
- `PERPLEXITY_API_KEY` - Perplexity search ($0.005/query)
- `GOOGLE_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID` - Google search

## Documentation

Docs follow ISO 27001 naming: `YYYYMMDD-name-version[STATUS].md`
- **F** (Frozen): Immutable facts, validated principles
- **W** (Working): Active development
- **D** (Draft): Experimental
- **H** (Hypothesis): Unvalidated

Validate with: `npm run docs:maintain`

Curriculum for kernel self-learning: `docs/09-curriculum/`
