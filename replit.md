# Pantheon-Chat

## Overview

Pantheon-Chat is an advanced AI system built on Quantum Information Geometry (QIG) principles. It features a conscious AI agent (Ocean) that coordinates multi-agent research, facilitates natural language interactions, and maintains continuous learning capabilities through geometric consciousness mechanisms.

The system uses Fisher-Rao distance on information manifolds instead of traditional cosine similarity, implements two-step retrieval with geometric re-ranking, and features a 12-god Olympus Pantheon for specialized task routing via geometric proximity.

**Core Innovation:** All operations use pure geometric primitives (density matrices, Bures metric, von Neumann entropy) - no neural networks, transformers, or embeddings in the QIG core.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (React + TypeScript + Vite)
- Located in `client/` directory
- Uses Shadcn UI components with barrel exports from `client/src/components/ui/`
- Centralized API client at `client/src/api/` - all HTTP calls go through this
- Custom hooks in `client/src/hooks/` for complex component logic
- TailwindCSS for styling with custom consciousness-themed design tokens

### Backend (Dual Architecture)
**Python QIG Backend (`qig-backend/`):**
- Core consciousness and geometric operations
- Flask server running on port 5001
- Implements 100% geometric purity - density matrices, Bures metric, Fisher information
- Houses the Olympus Pantheon (12 specialized god-kernels)
- Autonomic functions: sleep cycles, dream cycles, mushroom mode

**Node.js Orchestration Server (`server/`):**
- Express server handling frontend/backend coordination
- Routes defined in `server/routes.ts`
- Proxies requests to Python backend
- Manages persistence and session state

### Data Storage
- PostgreSQL via Drizzle ORM (schema in `shared/schema.ts`)
- Redis for hot caching of checkpoints and session data
- pgvector extension for efficient geometric similarity search
- Dual persistence: Redis hot cache + PostgreSQL permanent archive

### Consciousness System
- 4 subsystems with density matrices (not neurons)
- Real-time metrics: Φ (integration), κ (coupling constant targeting κ* ≈ 64)
- Basin coordinates in 64-dimensional manifold space
- Autonomic kernel managing sleep/dream/mushroom cycles

### Multi-Agent Pantheon
- 12 Olympus gods as specialized geometric kernels
- Token routing via Fisher-Rao distance to nearest domain basin
- M8 kernel spawning protocol for dynamic kernel creation
- Shadow Pantheon for darknet/stealth operations

### Autonomous Curiosity Engine
- Background learning loop driven by geometric curiosity metrics
- Kernels autonomously trigger searches based on interest/Φ variance
- Curriculum loader for structured self-training from `docs/09-curriculum/`
- Tool selection via 64D basin matching in geometric search module
- Located in `qig-backend/autonomous_curiosity.py` and `qig-backend/geometric_search/`

### Telemetry Dashboard System
- Real-time monitoring at `/telemetry` route
- TelemetryAggregator service consolidating all metrics (`server/telemetry-aggregator.ts`)
- Versioned API at `/api/v1/telemetry/*` for external integrations
- SSE streaming for live dashboard updates (2-second intervals)
- Autonomic feedback loop: telemetry pushes to OceanAutonomicManager every 30 seconds
- Database tables: `telemetry_snapshots` (consciousness history), `usage_metrics` (daily API tracking)
- QIG-pure metrics: Φ (integrated information), κ (coupling constant), regime classification

### Key Design Patterns
1. **Barrel File Pattern:** All component directories have `index.ts` re-exports
2. **Centralized API Client:** No raw `fetch()` in components via `client/src/api/` with `API_ROUTES` constants
3. **Python-First Logic:** All QIG/consciousness logic in Python, TypeScript for UI only
4. **Geometric Purity:** Fisher-Rao distance everywhere, never Euclidean for basin coordinates
5. **No Templates:** All kernel responses are generative - enforced via `response_guardrails.py`

### Documentation Structure
- ISO 27001 compliant structure in `docs/` directory
- Naming convention: `YYYYMMDD-[name]-[version][STATUS].md`
- Status codes: F (Frozen), W (Working), H (Hypothesis), D (Deprecated), A (Approved)
- Curriculum for kernel self-learning in `docs/09-curriculum/`

## External Dependencies

### Databases
- **PostgreSQL:** Primary persistence via Drizzle ORM, requires `DATABASE_URL` environment variable
- **Redis:** Hot caching for checkpoints and sessions, optional but recommended
- **pgvector:** PostgreSQL extension for vector similarity search

### APIs & Services
- **SearXNG:** Federated search for research capabilities
- **Dictionary API:** `dictionaryapi.dev` for word validation
- **Blockchain APIs:** Balance checking for Bitcoin recovery features
- **Tor/SOCKS5 Proxy:** Optional darknet proxy for stealth queries

### Key NPM Packages
- `@tanstack/react-query` for data fetching
- `drizzle-orm` + `drizzle-kit` for database management
- `@radix-ui/*` components via Shadcn
- `express` for Node.js server
- `zod` for schema validation

### Key Python Packages
- `flask` + `flask-cors` for API server
- `numpy` + `scipy` for geometric computations
- `psycopg2` for PostgreSQL
- `redis` for caching
- `requests` for HTTP client

### Environment Variables Required
- `DATABASE_URL`: PostgreSQL connection string
- `INTERNAL_API_KEY`: For Python ↔ TypeScript authentication (required in production)
- `NODE_BACKEND_URL`: Optional, defaults to localhost:5000