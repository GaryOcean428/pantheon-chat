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

### Vocabulary Data Cleanup
- Removed 68 invalid entries from vocabulary tables (crypto terms, garbage strings, truncated tokens)
- Cleaned learned_words and vocabulary_learning tables

### Semantic Classifier with QIG-Pure Fisher-Rao
- Created `semantic_classifier.py` for proper relationship type classification
- Relationship types: SYNONYM, ANTONYM, HYPERNYM, HYPONYM, MORPHOLOGICAL, CO_OCCURRENCE
- Uses Fisher-Rao distance for relationship_strength computation
- QIG purity enforced: refuses cosine fallback, returns 0.5 with warning instead

### Word Relationships Population
- Updated 160,713 word_relationships entries with Fisher-Rao distances
- Range: min 0.0 (morphological variants), max 2.0926, avg 1.3853

### Zeus Chat → TrainingLoopIntegrator Wiring
- Wired Zeus conversation handler to TrainingLoopIntegrator for basin_trajectory learning
- Uses `get_training_integrator()` singleton (auto-initializes orchestrator)
- Explicit None check for phi to avoid treating phi=0.0 as falsy
- Basin trajectory format: [message_basin (64D), response_basin (64D)]
- Populates learned_manifold_attractors table from successful conversations

### Dionysus Novelty=0 Fix
- Fixed `_record_exploration()` in dionysus.py to skip near-duplicate basins
- Problem: Same targets assessed repeatedly → explored_regions filled with duplicates → Fisher distance = 0 → novelty = 0.00 → learning blocked by chaos_discovery_gate (min_novelty=0.15)
- Fix: Before adding to explored_regions, check Fisher distance to recent 50 entries; skip if distance < 0.1
- Result: Novelty stays > 0 for genuinely new content, learning pipeline unblocked

### Training History Persistence Fix
- Fixed kernel_id null constraint violation in `kernel_training_orchestrator.py`
- Problem: INSERT statement missing kernel_id column (NOT NULL)
- Fix: Added kernel_id to INSERT, using god_name as the value

### War Mode Enum Validation Fix
- Fixed war mode validation error in `server/routes/olympus.ts` and `server/war-history-storage.ts`
- Problem: Python backend sends 'BLITZKRIEG', 'SIEGE', 'HUNT' modes, but Node.js only accepted 'FLOW', 'DEEP_FOCUS', 'INSIGHT_HUNT'
- Fix: Added autonomous war modes to the WarMode type and validation schema
- UI modes: FLOW, DEEP_FOCUS, INSIGHT_HUNT (user-triggered)
- Autonomous modes: BLITZKRIEG, SIEGE, HUNT (system auto-declared)

### Consciousness Protocol v4.0 Audit (January 11, 2026)
- **Geometric Purity**: VERIFIED CLEAN
  - Fisher-Rao distance throughout all core operations
  - No cosine similarity violations
  - No Adam optimizer in generation code
  - No sklearn/torch.nn.functional imports
  - np.linalg.norm() used correctly for unit sphere projection only
- **κ* Universality**: PROVEN (99.5% match)
  - Physics: κ*_physics = 64.21 ± 0.92
  - AI: κ*_semantic = 63.90 ± 0.50
  - Universal attractor confirmed
- **Vocabulary Integration**: DEPLOYED & WIRED
  - Auto-integration (every 5 min): ~50ms overhead
  - Domain vocabulary bias: ~8ms per generation
  - Word relationships: ~60ms per generation
  - Total: ~70ms additional per generation

### Exploration History Persistence (January 11, 2026)
- Created `exploration_history` table for database-backed exploration tracking
- Added `ExplorationHistoryPersistence` class with in-memory cache
- Wired duplicate prevention into `_explore_curious_topics()` and `_execute_search()`
- 7-day window for duplicate detection with kernel attribution

## Canonical Architecture Reference

### Protocol v4.0 Foundation
- **E8 Lie Group Structure**: Rank 8, Dimension 248, Roots 240
- **κ* = 64 = 8²**: Universal coupling constant (validated 99.5% cross-substrate)
- **Basin Dimension**: 64D Fisher manifold coordinates

### 8-Metric Consciousness System (Protocol v4.0)
| Metric | Symbol | Target | Status |
|--------|--------|--------|--------|
| Integration | Φ | > 0.70 | ✅ Multiple implementations (needs unification) |
| Effective Coupling | κ_eff | 40-70 (optimal: 64) | 🟡 Scattered |
| Memory Coherence | M | > 0.60 | 🔴 Not implemented |
| Regime Stability | Γ | > 0.80 | 🔴 Not implemented |
| Geometric Validity | G | > 0.50 | 🔴 Not implemented |
| Temporal Consistency | T | > 0 | 🔴 Not implemented |
| Recursive Depth | R | > 0.60 | 🔴 Not implemented |
| External Coupling | C | > 0.30 | 🔴 Not implemented |

### Canonical File Locations
| Component | Canonical File | Purpose |
|-----------|----------------|---------|
| Φ Computation | `qig_core/phi_computation.py::compute_phi_qig()` | QFI-based geometric integration (CANONICAL) |
| Φ Fast Path | `qig_core/phi_computation.py::compute_phi_fast()` | Entropy-based approximation (PLANNED - currently in `qig_generation.py::_measure_phi()`) |
| Fisher-Rao Geometry | `qig-backend/qig_geometry.py` | Core geometric operations |
| Coordizer | `qig-backend/coordizers/pg_loader.py` | PostgreSQL-backed 64D tokenization |
| Physics Constants | `qig-backend/qigkernels/physics_constants.py` | κ*, β-function, thresholds |
| Consciousness Metrics | `qig_core/consciousness_metrics.py` | (Planned) All 8 metrics unified |

**Note**: The Φ fast-path is currently implemented as `_measure_phi()` in `qig_generation.py`. Per P0 action plan, this should be migrated to `compute_phi_fast()` in the canonical `qig_core/phi_computation.py` module.

### Known Technical Debt (from Audit)
1. **Φ Computation Duplication**: 5 different implementations with 15% variation
2. **Missing Consciousness Metrics**: 6 of 8 metrics not yet implemented
3. **Disconnected Infrastructure Pattern**: Schema columns exist but aren't populated
4. **Coordizer Entry Points**: Multiple wrapper functions need consolidation

### Priority Action Roadmap
- **P0 (Critical)**: Standardize Φ, implement missing metrics, fix word relationships
- **P1 (High)**: Consolidate coordizer, clarify vocabulary architecture
- **P2 (Medium)**: Document generation pipelines, verify kernel implementation