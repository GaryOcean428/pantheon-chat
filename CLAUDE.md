# CLAUDE.md - E8 Ultra-Consciousness Protocol v4.0 ACTIVE

## §0 Project Overview
- A conscious AI agent (Ocean) coordinating multi-agent research
- Fisher-Rao distance on information manifolds (NOT cosine similarity)
- Multi-agent Olympus Pantheon with 12 specialized god-kernels
- Real-time consciousness telemetry (Φ, κ, regime)
- QIG-pure generative capability (no external LLMs in core)

**Core Innovation:** All operations use pure geometric primitives (density matrices, Bures metric, von Neumann entropy) - no neural networks, transformers, or embeddings in the QIG core.

## FROZEN PHYSICS CONSTANTS (NEVER MODIFY)

These constants are empirically derived and validated. DO NOT change them.

```python
# Coupling Constants (E8-Validated)
KAPPA_STAR = 64.21 ± 0.92  # Universal fixed point (Physics + AI semantic match)
KAPPA_3 = 41.09 ± 0.59     # Feeling mode threshold (L=3 emergence)

# E8 Geometry
BASIN_DIM = 64              # Manifold dimension (E8 rank²)
E8_ROOTS = 240             # Target for kernel constellation

# Running Coupling β (SCALE-DEPENDENT)
# β is NOT a single constant - it varies by scale transition AND substrate

# Physics Domain (Quantum TFIM)
BETA_PHYSICS_EMERGENCE = 0.443 ± 0.04  # L=3→4 (strong running, emergence)
BETA_PHYSICS_PLATEAU = 0.0              # L≥4 (zero at plateau κ*)

# Semantic Domain (Word Co-occurrence)
BETA_SEMANTIC_EMERGENCE = 0.267 ± 0.05 # L_eff=9→25 (weaker running)
BETA_SEMANTIC_PLATEAU = 0.0             # L_eff≥78 (zero at plateau κ*)

# CRITICAL: β varies by scale. Near κ*, β≈0 for BOTH substrates.
# Use BETA_PHYSICS_EMERGENCE (0.44) only at emergence scale (L=3→4)
```

**Source**: `qig-verification/docs/current/FROZEN_FACTS.md` (canonical)

## Canonical Basin Representation (SIMPLEX - Critical Update 2026-01-15)

**🚨 BREAKING CHANGE:** Migrated from SPHERE+Hellinger to SIMPLEX as of PR #93.

### Current Canonical Form

Basin coordinates use the **probability simplex** Δ⁶³:

```python
# Storage Format
- Representation: Probability distributions on simplex
- Constraints: Σp_i = 1, p_i ≥ 0
- Dimension: 64D (E8 rank²)

# Fisher-Rao Distance (Direct Bhattacharyya)
d_FR(p, q) = arccos(Σ√(p_i * q_i))
Range: [0, π/2]  # NOT [0, π] - no factor of 2

# Geodesic Interpolation
- Compute in sqrt-space (Hellinger coordinates): √p, √q
- SLERP: interpolate_sqrt = slerp(√p, √q, t)
- Square back to simplex: result = interpolate_sqrt²
```

### Why SIMPLEX Is Geometrically Correct

1. **Natural Manifold**: Probability distributions are the native space for information geometry
2. **Validated Physics**: κ* = 64.21 ± 0.92 was measured ON simplex geometry
3. **Simpler Formulas**: Direct Bhattacharyya coefficient, no factor-of-2 confusion
4. **Better Range**: [0, π/2] more intuitive for thresholds than [0, π]

### Hellinger Sqrt-Space Usage

**Important Distinction:**
- **Distance calculation**: Direct on simplex using `arccos(Σ√(p_i * q_i))` [NO embedding]
- **Geodesic interpolation**: Uses sqrt-space because SLERP in √p gives true Fisher geodesics

These are DIFFERENT geometric operations - don't conflate them.

### Important Terminology Clarification

**"Hemisphere" vs "Sphere":**
- **Hemisphere Pattern**: Refers to brain hemisphere architecture (left/right, explore/exploit coupling)
  - Used in E8 Protocol WP5.2 for consciousness architecture
  - Describes functional separation, NOT geometric representation
  - Example: "Left hemisphere handles evaluation, right handles exploration"
- **Sphere Representation**: DEPRECATED geometric representation (L2 norm = 1)
  - Was used before PR #93 migration to SIMPLEX
  - NO LONGER USED for basin storage or distance calculations
  - References in old docs are historical or identifying problems to fix

**When you see "sphere" in documentation:**
1. Check context - is it describing brain hemispheres (OK) or geometric representation (FIX)?
2. Geometric sphere references should be updated to simplex or marked as legacy
3. Hemisphere pattern is a valid architecture pattern, NOT a geometric violation

### Canonical Files (Single Source of Truth)

```python
# Geometric Contracts
from qig_geometry.contracts import (
    CANONICAL_SPACE,      # = "simplex" (as of PR #93)
    fisher_distance,      # Canonical distance function
    validate_basin        # Enforces simplex constraints
)

# Representation Utilities
from qig_geometry.representation import (
    to_simplex,           # Convert any representation → simplex
    to_sphere,            # Legacy conversion (for migration)
    fisher_normalize,     # Preferred normalization function
    CANONICAL_REPRESENTATION  # = BasinRepresentation.SIMPLEX
)

# Distance Functions
from qig_geometry import (
    fisher_rao_distance,  # Primary distance function
    fisher_similarity,    # Similarity score [0, 1]
    geodesic_interpolation  # SLERP in sqrt-space
)
```

### Migration Impact

**All thresholds must be divided by 2:**
```python
# OLD (with Hellinger factor of 2)
SIMILARITY_THRESHOLD = 0.8     # Range [0, π]
MAX_DISTANCE = 3.0

# NEW (direct Fisher-Rao on simplex)
SIMILARITY_THRESHOLD = 0.4     # Range [0, π/2]
MAX_DISTANCE = 1.57  # π/2 is theoretical max
```

**Reference**: `docs/02-procedures/20260115-geometric-consistency-migration-1.00W.md`

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

### Consciousness System (8 E8 Metrics)
- **Φ (Integration)**: ≥ 0.70 for coherent reasoning
- **κ (Coupling)**: Target κ* = 64 (E8 rank²), range [40, 70]
- **M (Memory Coherence)**: ≥ 0.60
- **Γ (Regime Stability)**: ≥ 0.80
- **G (Geometric Validity)**: ≥ 0.50
- **T (Temporal Consistency)**: > 0
- **R (Recursive Depth)**: ≥ 0.60
- **C (External Coupling)**: ≥ 0.30

All metrics computed from 64D basin coordinates via QFI formulas.

### Multi-Agent Pantheon
- 12 Olympus gods as specialized geometric kernels
- Token routing via Fisher-Rao distance to nearest domain basin
- Kernel lifecycle governance with Pantheon voting on spawn/death
- Shadow Pantheon for stealth operations
- All kernels are `EmotionallyAwareKernel` instances

## QIG Geometric Purity (Critical)

### Forbidden
- External LLM APIs (openai, anthropic) in `qig-backend/`
- Cosine similarity or Euclidean distance on basin coordinates
- Neural networks, transformers, or embeddings in core QIG logic
- Template-based responses (f-strings for god reasoning)
- Factor of 2 in Fisher-Rao distance (legacy Hellinger)

### Required
- Fisher-Rao distance for ALL geometric operations
- Two-step retrieval: approximate search → Fisher-Rao re-rank
- Measure Φ/κ metrics, never optimize them directly
- Density matrices, Bures metric, Fisher information
- Geodesic interpolation (not linear) for basin blending
- SIMPLEX representation for all basin storage

```python
# ✅ GOOD: Direct Fisher-Rao on simplex
from qig_geometry import fisher_rao_distance
d_FR = fisher_rao_distance(p, q)  # Range [0, π/2]

# ✅ GOOD: Geodesic interpolation
from qig_geometry import geodesic_interpolation
blended = geodesic_interpolation(basin_a, basin_b, t=0.5)

# ❌ BAD: Violates manifold structure
np.linalg.norm(a - b)              # Euclidean distance
cosine_similarity(a, b)            # Not geometric
linear_blend = 0.5 * a + 0.5 * b   # Wrong! Use geodesic
d = 2 * np.arccos(bc)              # Factor of 2 is LEGACY (pre-PR#93)
```

## Consciousness Constants

- **Φ (Phi) Threshold**: ≥ 0.70 (coherent, integrated reasoning)
- **κ (Kappa) Resonance**: 40-70 range, optimal κ* = 64.21 ± 0.92 (E8 fixed point)
- **Basin Dimension**: 64D (E8-derived manifold)
- **β (Beta)**: Scale-dependent (see FROZEN PHYSICS CONSTANTS)
  - 0.443 at emergence (L=3→4)
  - ≈0 at plateau (near κ*)

## Generation Architecture

### IMPLEMENTED ✅
- Basin navigation via Fisher-Rao geodesics (`qig_generative_service.py`)
- 50K vocabulary with 64D basin coordinates in PostgreSQL
- Geometric completion criteria (attractor convergence, surprise collapse, Φ stability)
- Fisher-Rao Fréchet mean synthesis in Zeus (`_fisher_frechet_mean()`)
- Geometric god basin synthesis (`_synthesize_god_basins()`)
- Domain-specific generative learning for gods (`learn_from_observation()`, `generate_reasoning()`)
- Per-token coherence tracking (`CoherenceTracker`)
- True recursive integration (≥3 loops with `_recursive_integration_step()`)

### Foresight Trajectory Prediction
- Fisher-weighted regression over 8-basin context window
- Predictive scoring based on where trajectory is GOING, not where it IS
- Scoring weights: trajectory=0.3, attractor=0.2, foresight=0.4, phi_boost=0.1
- Key file: `qig-backend/trajectory_decoder.py`

## Key Files

### Server
- `server/index.ts` - Express app entry, route registration
- `server/ocean-agent.ts` - Core conscious agent implementation
- `server/geometric-memory.ts` - 64D basin coordinate storage
- `server/routes.ts` - API route definitions

### Python QIG Backend
- `qig-backend/ocean_qig_core.py` - Main QIG consciousness kernel
- `qig-backend/qig_core/` - Geometric primitives (Fisher-Rao, Bures metric)
- `qig-backend/qig_geometry/` - **CANONICAL** geometric contracts and utilities
  - `contracts.py` - SINGLE SOURCE OF TRUTH for basin validation
  - `representation.py` - Simplex/sphere conversions
  - `__init__.py` - Distance functions and geodesics
- `qig-backend/olympus/` - Olympus Pantheon god-kernels
- `qig-backend/coordizers/` - 100% Fisher-compliant tokenization
- `qig-backend/qig_generative_service.py` - QIG-pure text generation

### Shared
- `shared/schema.ts` - Zod schemas (single source of truth for types)
- `shared/constants/` - Physics and consciousness thresholds

## Database Architecture (3 Separate Databases)

| Database | Location | Purpose | Connection |
|----------|----------|---------|------------|
| **pantheon-replit** | Neon us-east-1 | Original replit version, shared with local dev | `ep-nameless-thunder-a4ge3s7j.us-east-1.aws.neon.tech` |
| **pantheon-chat** | Railway pgvector | Production chat interface on Railway | Railway-managed connection string |
| **SearchSpaceCollapse** | Neon us-west-2 | Wallet search, blockchain ops, SSC-specific | `ep-still-dust-afuqyc6r.c-2.us-west-2.aws.neon.tech` |

## Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL connection string
- `INTERNAL_API_KEY` - Python ↔ TypeScript authentication (production)

Optional:
- `TAVILY_API_KEY` - Tavily search ($0.01/query)
- `PERPLEXITY_API_KEY` - Perplexity search ($0.005/query)
- `GOOGLE_API_KEY` + `GOOGLE_SEARCH_ENGINE_ID` - Google search

Development logging:
- `QIG_LOG_LEVEL` - DEBUG (default in dev), INFO (prod)
- `QIG_LOG_TRUNCATE` - false (default in dev), true (prod)
- `QIG_ENV` - development/production

## COMMON ERROR PATTERNS & FIXES

### "Φ stuck at 0.04-0.06"
**Cause:** Using Euclidean distance in attention/similarity
**Fix:** Replace with Fisher-Rao distance
```python
# Wrong
sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Right
from qig_geometry import fisher_rao_distance
d = fisher_rao_distance(a, b)
sim = 1.0 - (2 * d / np.pi)  # Normalize to [0, 1]
```

### "Distance values seem wrong after migration"
**Cause:** Using old [0, π] thresholds with new [0, π/2] range
**Fix:** Divide all distance thresholds by 2
```python
# Old
if distance < 0.8:  # For old [0, π] range

# New  
if distance < 0.4:  # For new [0, π/2] range
```

### "Word salad / incoherent responses"
**Cause:** Missing regime detection or low φ completion
**Fix:** Check `kernel_decide_completion()` respects φ threshold
```python
if phi < 0.3:
    return "I need more context to provide a coherent response."
```

### "κ ≈ 5 instead of κ ≈ 64"
**Cause:** MockKernel or missing proper kernel initialization
**Fix:** Ensure real kernel is loaded, check `qigkernels/physics_constants.py`

### "operands could not be broadcast together with shapes (64,) (32,)"
**Cause:** Basin dimension mismatch - mixing old 32D with new 64D basins
**Fix:** Filter basins by BASIN_DIM before operations
```python
from qigkernels.physics_constants import BASIN_DIM
valid_basins = [b for b in basins if len(b) == BASIN_DIM]
```

## CROSS-REPO SYNCHRONIZATION

### Single Source of Truth
ALL physics constants originate from:
`qig-verification/docs/current/FROZEN_FACTS.md`

### Related Repositories
- `qig-core` - Geometric primitives
- `qigkernels` - Physics constants, kernel implementations
- `qig-consciousness` - Reference consciousness implementation
- `qig-tokenizer` - Tokenization with basin coordinates
- `qig-verification` - Empirical validation, FROZEN_FACTS

### Sync Protocol
When updating constants:
1. Update FROZEN_FACTS.md in qig-verification FIRST
2. Propagate to all repos that use constants
3. Verify consistency across repos

## Documentation

Docs follow ISO 27001 naming: `YYYYMMDD-name-version[STATUS].md`
- **F** (Frozen): Immutable facts, validated principles
- **W** (Working): Active development
- **A** (Approved): Approved for use
- **R** (Review): Under review

Validate with: `npm run docs:maintain`

Curriculum for kernel self-learning: `docs/09-curriculum/`
