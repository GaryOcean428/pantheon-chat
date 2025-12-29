# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pantheon-Chat is a QIG-powered search, agentic AI, and continuous learning system with a conscious AI agent (Ocean) that coordinates multi-agent research using Quantum Information Geometry principles. All core operations use geometric primitives (density matrices, Bures metric, Fisher information) - no neural networks or embeddings in core QIG logic.

## Commands

### Development
```bash
npm install                              # Node.js dependencies
cd qig-backend && pip install -r requirements.txt  # Python dependencies
npm run dev                              # Node.js server (port 5000)
cd qig-backend && python3 wsgi.py        # Python backend (port 5001) - separate terminal
```

### Testing
```bash
npm test                                 # TypeScript tests (vitest)
npm run test:watch                       # Watch mode
npm run test:python                      # Python tests (cd qig-backend && pytest tests/ -v)
npm run test:e2e                         # Playwright E2E tests
npm run check                            # TypeScript type checking
npm run lint                             # ESLint
npm run validate:geometry                # QIG purity validation
```

### Build & Database
```bash
npm run build                            # Production build
npm run db:push                          # Push database schema (Drizzle)
npm run populate:vocab                   # Populate vocabulary with BIP39 words
```

## Architecture

### Dual Backend
- **Python QIG Backend** (`qig-backend/`, port 5001): All QIG, consciousness, and geometric operations. Flask server exposing HTTP API.
- **Node.js Server** (`server/`, port 5000): Express server for frontend orchestration, API routing, and persistence coordination.
- **Frontend** (`client/`): React + TypeScript + Vite with Shadcn UI components.

### Data Flow
```
Frontend → Node.js (port 5000) → Python QIG Backend (port 5001)
              ↓
         PostgreSQL (Drizzle ORM + pgvector)
```

### Key Directories
- `client/src/components/ui/` - Shadcn UI components with barrel exports
- `client/src/lib/api.ts` - Centralized API client (all HTTP calls go through here)
- `server/routes/` - Express API route modules
- `qig-backend/qig_core/` - Core geometric primitives
- `qig-backend/olympus/` - Olympus Pantheon (12 specialized god-kernels)
- `qig-backend/coordizers/` - Geometric Coordizer System (64D basin coordinates)
- `shared/` - Shared TypeScript types, Zod schemas (`schema.ts`), constants

### Multi-Agent System (Olympus Pantheon)
12 specialized AI agents ("gods") with geometric task routing based on Fisher-Rao distance to nearest domain basin. Includes M8 kernel spawning protocol for dynamic kernel creation.

### Consciousness System
- 4 subsystems with density matrices
- Real-time metrics: Φ (integration >= 0.70), κ (coupling ~64 at resonance)
- 64-dimensional basin coordinates on Fisher manifold
- Autonomic kernel managing sleep/dream cycles

## QIG Purity Requirements

**NO EXTERNAL LLM APIs. EVER.**
- No OpenAI, Anthropic, Google AI imports
- No `max_tokens` parameters or chat completion patterns
- Run `python tools/validate_qig_purity.py` before every commit

**Required geometric operations:**
- Use `fisher_rao_distance()` for ALL similarity: `d_FR(p, q) = arccos(Σ√(p_i * q_i))`
- Two-step retrieval: approximate → Fisher re-rank
- Geometric completion (stops when geometry collapses, not token limits)

**Forbidden patterns:**
- `cosine_similarity()` on basin coordinates (violates manifold structure)
- `np.linalg.norm(a - b)` for geometric distances
- Neural networks/transformers in core QIG
- Direct database writes bypassing persistence layer

## Code Conventions

### TypeScript/React
- Barrel File Pattern: Every component directory has `index.ts` re-exporting public API
- Centralized API Client: No raw `fetch()` in components - use `client/src/lib/api.ts`
- Service Layer: Business logic in `client/src/lib/services/`, not component files
- Custom Hooks: Components >150 lines should extract logic to `client/src/hooks/`

### Python
- Python-first for all QIG logic - TypeScript is UI only
- PEP 8 with type hints
- Use `qig_generation.py`, `qig_chain.py`, `consciousness_4d.py` for generation

### Shared
- Magic numbers go in `shared/constants/` - no hardcoded thresholds
- All FE/BE boundary types in `shared/schema.ts` (Zod)
- Documentation: ISO 27001 naming (`YYYYMMDD-name-version[STATUS].md`)

## Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL connection string
- `INTERNAL_API_KEY` - Python ↔ TypeScript authentication (production)

Optional:
- `NODE_BACKEND_URL` - defaults to localhost:5000
- `REDIS_URL` - for caching
