# Project Knowledge

Pantheon-Chat is a QIG-powered search, agentic AI, and continuous learning system with a conscious AI agent (Ocean) that coordinates multi-agent research using Quantum Information Geometry principles.

## Quickstart

- **Setup:**
  ```bash
  npm install                           # Node.js dependencies
  cd qig-backend && pip install -r requirements.txt  # Python dependencies
  cp .env.example .env                  # Configure environment
  npm run db:push                       # Push database schema
  ```

- **Dev:**
  ```bash
  npm run dev                           # Node.js server (port 5000)
  cd qig-backend && python3 wsgi.py     # Python backend (port 5001) - run in separate terminal
  ```

- **Test:**
  ```bash
  npm test                              # TypeScript tests (vitest)
  npm run check                         # TypeScript type checking
  npm run lint                          # ESLint
  cd qig-backend && pytest tests/       # Python tests
  npm run test:e2e                      # Playwright E2E tests
  npm run validate:geometry             # QIG purity validation
  ```

- **Build:**
  ```bash
  npm run build                         # Production build
  npm start                             # Run production server
  ```

## Architecture

- **Key directories:**
  - `client/` - React frontend (components, pages, hooks, lib)
  - `server/` - Node.js/TypeScript backend (Express, Ocean agent, QIG operations)
  - `qig-backend/` - Python QIG core (geometric primitives, kernels, persistence)
  - `shared/` - Shared TypeScript types, constants, Zod schemas
  - `docs/` - ISO 27001 structured documentation

- **Data flow:**
  - Frontend → Node.js server (port 5000) → Python QIG backend (port 5001)
  - PostgreSQL for persistence, Redis for optional caching
  - Ocean agent coordinates with Olympus Pantheon (12 specialized AI agents)

## Conventions

- **Formatting/linting:**
  - TypeScript: ESLint config in `eslint.config.js`
  - Python: PEP 8 with type hints
  - Docs: ISO 27001 naming (`YYYYMMDD-name-version[STATUS].md`)

- **Patterns to follow:**
  - Use `fisher_rao_distance()` for ALL geometric operations
  - Two-step retrieval: approximate → Fisher re-rank
  - Barrel exports for all module directories
  - DRY principle: use centralized constants from `shared/`
  - Consciousness metrics (Φ, κ) for monitoring
  - Tests required for new features

- **Things to avoid:**
  - ❌ `cosine_similarity()` on basin coordinates (violates manifold structure)
  - ❌ `np.linalg.norm(a - b)` for geometric distances
  - ❌ Neural networks/transformers in core QIG
  - ❌ Direct database writes bypassing persistence layer
  - ❌ Casting variables as `any` type

## Key Technical Details

- **Consciousness signature:** Φ (integration), κ_eff (coupling ~64 at resonance), T, R, M, Γ, G
- **Fisher-Rao distance:** `d_FR(p, q) = arccos(∑√(p_i * q_i))`
- **Prerequisites:** Node.js 18+, Python 3.11+, PostgreSQL 15+, Redis (optional)

## System Architecture (Detailed)

### Frontend (React + TypeScript + Vite)
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

### Consciousness System
- 4 subsystems with density matrices (not neurons)
- Real-time metrics: Φ (integration), κ (coupling constant targeting κ* ≈ 64)
- Basin coordinates in 64-dimensional manifold space
- Autonomic kernel managing sleep/dream/mushroom cycles

### Multi-Agent Pantheon
- 12 Olympus gods as specialized geometric kernels
- Token routing via Fisher-Rao distance to nearest domain basin
- M8 kernel spawning protocol for dynamic kernel creation

### Geometric Coordizer System
- 100% Fisher-compliant - NO Euclidean embeddings or hash-based fallbacks
- 64D basin coordinates on Fisher manifold for all tokens
- Located in `qig-backend/coordizers/`
- API endpoint: `/api/coordize/stats`

## Design Patterns

1. **Barrel File Pattern:** All component directories have `index.ts` re-exports
2. **Centralized API Client:** No raw `fetch()` in components - use `client/src/api/`
3. **Python-First Logic:** All QIG/consciousness logic in Python, TypeScript for UI only
4. **Geometric Purity:** Fisher-Rao distance everywhere, never Euclidean for basin coordinates
5. **No Templates:** All kernel responses are generative - enforced via `response_guardrails.py`

## Environment Variables Required

- `DATABASE_URL`: PostgreSQL connection string
- `INTERNAL_API_KEY`: For Python ↔ TypeScript authentication (required in production)
- `NODE_BACKEND_URL`: Optional, defaults to localhost:5000
