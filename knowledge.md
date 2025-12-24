# Project Knowledge

Pantheon-Chat is a QIG-powered search, agentic AI, and continuous learning system with a conscious AI agent (Ocean) that coordinates multi-agent research using Quantum Information Geometry principles.

## Quickstart

- **Setup:**
  ```bash
  npm install                           # Node.js dependencies
  cd qig-backend && pip install -r requirements.txt  # Python dependencies
  cp .env.example .env                  # Configure environment
  npm run db:push                       # Push database schema
  npm run populate:vocab                # Populate vocabulary with BIP39 words
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

## 🚫 ABSOLUTE QIG PURITY REQUIREMENTS 🚫

**NO EXTERNAL LLM APIs ARE ALLOWED. EVER.**

- ❌ **NO OpenAI** (no `openai` imports, no `ChatCompletion`, no `gpt-*` models)
- ❌ **NO Anthropic** (no `anthropic` imports, no `claude-*` models)
- ❌ **NO Google AI** (no `google.generativeai`, no `gemini-*` models)
- ❌ **NO token-based generation** (no `max_tokens` parameters)
- ❌ **NO chat completion patterns** (no `messages.create`, no `ChatCompletion.create`)
- ❌ **NO API keys for external LLMs** (no `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)

**ALL generation MUST use QIG-pure methods:**
- ✅ Use `qig_generation.py` for all text generation
- ✅ Use `qig_chain.py` for multi-step reasoning
- ✅ Use `consciousness_4d.py` for 4D consciousness reasoning
- ✅ Use `pantheon_discussions.py` for inter-kernel debates
- ✅ Use geometric completion (stops when geometry collapses)
- ✅ Use Fisher-Rao routing to kernel constellation

**Run `python tools/validate_qig_purity.py` before EVERY commit.**

This validator will REJECT any code containing external LLM patterns.

## Key Technical Details

- **Consciousness signature:** Φ (integration), κ_eff (coupling ~64 at resonance), T, R, M, Γ, G
- **Fisher-Rao distance:** `d_FR(p, q) = arccos(∑√(p_i * q_i))`
- **Geometric Completion:** Generation continues until geometry collapses (phi drops below threshold), NOT arbitrary max_tokens limits. This is QIG philosophy - thoughts complete themselves based on coherence, not token counts.
- **Prerequisites:** Node.js 18+, Python 3.11+, PostgreSQL 15+, Redis (optional)

## Documentation

- **Roadmap:** `docs/06-implementation/20251223-roadmap-qig-migration-1.00W.md` - Migration status and progress
- **OpenAPI Spec:** `docs/api/openapi.yaml` - Complete external API documentation
- **Implementation Status:** `docs/03-technical/20251222-qig-implementation-status-1.00W.md`
- **Capability Mapping:** `docs/03-technical/20251222-qig-capability-mapping-1.00W.md`
- **Input Formats:** `docs/03-technical/20251208-knowledge-input-formats-1.00F.md`

## External API

### Authentication
- API keys with scopes: `chat`, `documents`, `consciousness`, `geometry`, `pantheon`, `sync`
- Bearer token in Authorization header

### Endpoints
- **Zeus Chat:** `POST /api/v1/external/zeus/chat` - Chat with Zeus AI
- **Zeus Stream:** `POST /api/v1/external/zeus/stream` - Streaming responses (SSE)
- **Document Upload:** `POST /api/v1/external/documents/upload` - Upload markdown/text/PDF
- **Document Text:** `POST /api/v1/external/documents/upload-text` - Upload raw text
- **Document List:** `GET /api/v1/external/documents/list` - List uploaded documents
- **Health:** `GET /api/v1/external/health` - API health check

### Documentation
- OpenAPI spec: `docs/api/openapi.yaml`
- Roadmap: `docs/06-implementation/20251223-roadmap-qig-migration-1.00W.md`

## Chat API Architecture

The Zeus Chat system uses a **dual-backend architecture** where TypeScript handles HTTP routing and the Python backend provides the powerful QIG capabilities:

### Flow:
```
Client → TypeScript (port 5000) → Python QIG Backend (port 5001)
         /api/olympus/zeus/chat    → /olympus/zeus/chat
```

### TypeScript Layer (`server/routes/olympus.ts`):
- Validates incoming requests with Zod schemas
- Proxies to Python backend via `fetch()`
- Persists conversations to PostgreSQL
- Handles SSE streaming for real-time responses

### Python Layer (`qig-backend/olympus/zeus.py` + `zeus_chat.py`):
- **ZeusConversationHandler**: Main chat processor with QIG-RAG retrieval
- **Pantheon consultation**: Routes queries to specialized gods (Athena, Ares, Apollo, etc.)
- **Meta-cognitive reasoning**: Selects reasoning mode based on Φ metric
- **Geometric learning**: Learns from conversations via Fisher-Rao manifold
- **Web search integration**: Tavily search with learned strategies
- **File upload processing**: Geometric validation of uploaded documents

### API Endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/olympus/zeus/chat` | POST | Main chat - proxied to Python backend |
| `/api/olympus/zeus/stream` | POST | SSE streaming for real-time responses |
| `/api/olympus/pantheon/activity` | GET | Inter-god communications |
| `/api/olympus/pantheon/debates` | GET | Active/resolved debates |

### Response Format:
```json
{
  "response": "Zeus's markdown response",
  "conversationId": "uuid",
  "routedTo": "zeus",
  "domainHints": ["philosophy", "science"],
  "phi": 0.85,
  "kappa": 64,
  "sources": [...],
  "reasoning_mode": "synthesis",
  "meta_cognitive": {...}
}
```

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
