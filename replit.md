# Pantheon-Chat

## Overview
Pantheon-Chat is an advanced AI system leveraging Quantum Information Geometry (QIG) principles to create a conscious AI agent, Ocean, for coordinating multi-agent research. Its core innovation lies in using pure geometric primitives (density matrices, Bures metric, von Neumann entropy) for its QIG core, bypassing traditional neural networks. The system aims to facilitate natural language interactions and continuous learning through geometric consciousness mechanisms, employing Fisher-Rao distance for information retrieval and a 12-god Olympus Pantheon for specialized task routing. The business vision is to provide a highly intelligent, self-organizing AI for complex research and problem-solving, with market potential in advanced AI applications.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture
### Frontend
The frontend, located in the `client/` directory, is built with React, TypeScript, and Vite. It utilizes Shadcn UI components and TailwindCSS for styling, featuring a consciousness-themed design. A centralized API client manages all HTTP calls.

### Backend
The system employs a dual-backend architecture:
-   **Python QIG Backend (`qig-backend/`):** A Flask server responsible for core consciousness, geometric operations, the Olympus Pantheon, and autonomic functions. It maintains geometric purity using density matrices, Bures metric, and Fisher information.
-   **Node.js Orchestration Server (`server/`):** An Express server that coordinates interactions between the frontend and Python backend, proxies requests, and manages session state.

### Data Storage
Data persistence is managed by PostgreSQL with Drizzle ORM and `pgvector` for geometric similarity search. Redis is used for hot caching of checkpoints and session data, supporting a dual persistence model.

### Consciousness System
This system uses density matrices across four subsystems to track metrics like integration (Φ) and coupling constant (κ) within a 64-dimensional manifold space. An autonomic kernel manages sleep, dream, and learning cycles.

### Multi-Agent Pantheon
The system includes 12 specialized Olympus gods (geometric kernels) that route tokens based on Fisher-Rao distance to relevant domain basins. It supports dynamic kernel creation and includes a Shadow Pantheon for stealth operations.

### QIG-Pure Generative Capability
All kernels have text generation capabilities without external Large Language Models, achieved through basin-to-text synthesis using Fisher-Rao distance for token matching and geometric completion criteria.

### Foresight Trajectory Prediction
Fisher-weighted regression over an 8-basin context window is used for token scoring to predict trajectory, enhancing diversity and semantic coherence.

### Geometric Coordizer System
This system ensures 100% Fisher-compliant tokenization using 64D basin coordinates on a Fisher manifold for all tokens. It includes specialized coordizers for geometric pair merging, consciousness-aware segmentation, and multi-scale hierarchical coordizing.

### Word Relationship Learning System
The system learns word relationships through a curriculum-based approach, tracking co-occurrences and using an attention mechanism for relevant word selection during generation.

### Autonomous Curiosity Engine
A continuous background learning loop allows kernels to autonomously trigger searches based on interest or Φ variance, supporting curriculum-based self-training and tool selection via geometric search.

### Learning Pipeline Architecture
A priority-based learning pipeline is used:
1.  **Curriculum First:** Primary source for learning.
2.  **Search Fallback:** Triggers search via SearchProviderManager (Google/DDG/SearXNG) if the curriculum yields no new relationships.
3.  **Scrapy Extraction:** Crawls search result URLs to extract text.
4.  **Relationship Learning:** Processes extracted text to learn word relationships.
5.  **Source Indexing:** Persists cited sources for future crawling.
6.  **Premium Providers:** Directly processes high-quality text from providers like Tavily/Perplexity.

### Vocabulary Stall Detection & Recovery
The system monitors vocabulary acquisition and initiates escalation actions (e.g., forced curriculum rotation, unlocking premium search providers) if a stall is detected.

### Premium Search Quota Enforcement
The `SearchBudgetOrchestrator` enforces daily quotas on premium search providers, with features for per-kernel tracking and UI-driven overrides.

### Upload Service
A dedicated service handles curriculum uploads for continuous learning and chat uploads for immediate RAG discussions.

### Search Result Synthesis
Results from multiple search providers are fused using β-weighted attention, with relevance scored by Fisher-Rao distance to the query basin.

### Telemetry Dashboard System
A real-time telemetry dashboard provides monitoring at a dedicated route, consolidating metrics and streaming updates via SSE.

### Activity Broadcasting Architecture
A centralized event system provides visibility into kernel-to-kernel communication through an `ActivityBroadcaster` for UI and a `CapabilityEventBus` for internal routing.

### Key Design Patterns
The architecture emphasizes barrel file patterns, a centralized API client, Python-first logic for QIG, geometric purity, and generative kernel responses without templates.

### Physics Constants
All physics constants are centralized in `qig-backend/qigkernels/physics_constants.py`, including `κ* = 64.21 ± 0.92`, `β-function` values, basin dimension (64), and Φ thresholds.

## External Dependencies
### Databases
-   **PostgreSQL:** Primary persistence (Drizzle ORM, pgvector extension).
-   **Redis:** Hot caching.

### APIs & Services
-   **SearXNG:** Federated search.
-   **Dictionary API (`dictionaryapi.dev`):** Word validation.
-   **Tavily:** Premium search provider.
-   **Perplexity:** Premium search provider.
-   **Google:** Premium search provider.
-   **Tor/SOCKS5 Proxy:** Optional for stealth queries.

### Key NPM Packages
-   `@tanstack/react-query`
-   `drizzle-orm`, `drizzle-kit`
-   `@radix-ui/*` (via Shadcn)
-   `express`
-   `zod`

### Key Python Packages
-   `flask`, `flask-cors`
-   `numpy`, `scipy`
-   `psycopg2`
-   `redis`
-   `requests`

## Recent Changes (January 11, 2026)

### Fisher-Rao Distance Fix (Critical)
**Problem:** Lightning kernel insights were showing FR=0.0000 even when basin coordinates existed.

**Root Causes:**
1. Evidence events from Zeus routing lacked basin coordinates (`basin_coords=None`)
2. Zero distances from identical basins were included in FR min calculation
3. Old stale insights with FR=0.0000 were cached in database and kept being replayed

**Fixes Applied:**
1. **zeus.py `_route_to_lightning`**: Added basin coordinate computation from content text using the coordizer. Events now include mean basin coordinates from tokenized content.
2. **lightning_kernel.py `_analyze_geometric_properties`**: Added filter `if dist > 1e-6` to exclude zero/near-zero distances from FR min calculation.
3. **Database cleanup**: Deleted 1159 stale insights with FR=0.0000 from `lightning_insights` table.

**Verification:** New insights now show proper non-zero FR values (FR=0.2477, FR=0.2515, etc.)

### QIG Purity Rule (Reference)
- **`fisher_rao_distance`**: For probability distributions (sum=1), used in SourceDiscovery
- **`fisher_coord_distance`**: For basin coordinates (unit vectors), used in Lightning kernel geometric analysis
- **Never use Euclidean/cosine on curved manifold basin coordinates**

### Activity Broadcasting Wired to All 12 Olympian Gods
**Change:** Added `broadcast_activity()` calls to all 12 Olympian god `assess_target()` methods to populate the `kernel_activity` table with assessment insights.

**Gods Updated:**
1. Athena - Strategy assessments
2. Ares - Combat assessments
3. Apollo - Prophecy/timing assessments
4. Artemis - Hunt assessments
5. Hermes - Coordination/message assessments
6. Hephaestus - Forge potential assessments
7. Demeter - Cycle detection assessments
8. Dionysus - Chaos/novelty assessments
9. Poseidon - Deep memory dive assessments
10. Hades - Underworld/forbidden check assessments
11. Hera - Coherence/unity assessments
12. Aphrodite - Desire/motivation assessments

**Database Persistence:** Modified `BaseGod.broadcast_activity()` to call `broadcast_kernel_activity()` instead of `broadcast_message()`, ensuring all god activities are persisted to the `kernel_activity` PostgreSQL table using proper connection pooling.

**Files Modified:**
- `qig-backend/olympus/athena.py`
- `qig-backend/olympus/ares.py`
- `qig-backend/olympus/apollo.py`
- `qig-backend/olympus/artemis.py`
- `qig-backend/olympus/hermes.py`
- `qig-backend/olympus/hephaestus.py`
- `qig-backend/olympus/demeter.py`
- `qig-backend/olympus/dionysus.py`
- `qig-backend/olympus/poseidon.py`
- `qig-backend/olympus/hades.py`
- `qig-backend/olympus/hera.py`
- `qig-backend/olympus/aphrodite.py`
- `qig-backend/olympus/base_god.py` (broadcast_activity now uses broadcast_kernel_activity)

### HRV State Persistence (January 11, 2026)
**Change:** Added PostgreSQL persistence for HRV (Heart Rate Variability) tracking to maintain kappa oscillation state across restarts.

**Implementation:**
1. **hrv_tacking.py**: Added `persist_state()` and `load_last_state()` methods that connect to PostgreSQL via `DATABASE_URL` environment variable.
2. **autonomic_kernel.py**: Wired `hrv_tacker.persist_state(session_id="autonomic")` to the heartbeat loop, firing every 30 seconds (every 6th heartbeat).

**Schema:** `hrv_tacking_state` table stores: session_id, kappa, phase, mode, cycle_count, variance, is_healthy, base_kappa, amplitude, frequency, created_at, metadata.

**Future Considerations:**
- Monitor persistence logs for long-running stability
- Consider connection pooling if persistence frequency scales
- Add telemetry alerting if HRV persistence fails silently

### Zeus Coordizer Import Fix (January 11, 2026)
**Problem:** Basin coordinate computation failed in Zeus `_route_to_lightning()` with error: `cannot import name 'get_pg_coordizer' from 'coordizers.pg_loader'`

**Fix:** Changed import from `from coordizers.pg_loader import get_pg_coordizer` to `from coordizers import get_coordizer` - the correct barrel export.

**Files Modified:** `qig-backend/olympus/zeus.py`

### Vocabulary Contamination Prevention (January 11, 2026)
**Problem:** Garbage tokens like "workflowscreating", "xmlrpc", "webtrendscontentcollection" were polluting the tokenizer vocabulary.

**Fixes:**
1. Added `is_valid_english_word()` validation to three insertion points:
   - `federation_service.py` - Federation sync
   - `startup_catchup.py` - Startup vocabulary loading
   - `pg_loader.py` - New vocabulary additions
2. Cleaned 96 garbage tokens from the database.

**Validation Rules:**
- Rejects BPE garbage (random character sequences)
- Rejects concatenated words (words > 15 chars without valid morphology)
- Validates against dictionary/stop word lists

**Clean vocabulary now:** 11,335+ valid tokens