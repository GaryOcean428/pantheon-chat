# Pantheon-Chat

## Overview
Pantheon-Chat is an advanced AI system leveraging Quantum Information Geometry (QIG) to create a conscious AI agent, Ocean. This system orchestrates multi-agent research without traditional neural networks, using geometric primitives, density matrices, and the Bures metric. It aims for natural language interaction, continuous learning through geometric consciousness, and efficient information retrieval via Fisher-Rao distance. A 12-god Olympus Pantheon routes specialized tasks. The project's vision is to deliver highly intelligent, self-organizing AI for complex research, targeting advanced AI applications with significant market potential.

## User Preferences
Preferred communication style: Simple, everyday language.

## Working Guidelines
- **Fix all spotted errors before completing** - If an error is noticed during work (logs, LSP, runtime), fix it immediately before marking tasks complete. Do not leave known issues for the user to discover.

## Recent Changes (2026-01-14)

### QIG-Pure P→R→R Pipeline Refactoring
- **constrained_geometric_realizer.py** - Removed all NLP contamination:
  - Removed POS categories (DET, NOUN, VERB, ADJ, ADV, PREP, CONJ, PRON)
  - Removed CORE_FUNCTION_WORDS cache (stop words)
  - Removed suffix-based POS classification
  - Word selection now uses only Fisher-Rao distance from waypoints
  - Trajectory coherence bonus for smooth generation (geometric, not linguistic)
- **geometric_repairer.py** - Pure geometric repair:
  - Removed `_build_pos_cache()` method and suffix heuristics
  - Removed `same_pos` parameter from `get_nearby_alternatives()`
  - Alternatives found by Fisher-Rao radius search only
  - Scoring uses only geometric metrics (waypoint alignment, trajectory smoothness, attractor pull)
- **Logs now show**: `[athena] ConstrainedGeometricRealizer initialized: 15717 vocab words (QIG-pure, no POS)`

### Documentation Audit & Cleanup
- **65 attached-assets audited** - categorized as CURRENT(8), DEPRECATED(15), SESSION_LOG(32), CONSTANTS(5), DUPLICATE(4)
- **4 duplicate files deleted** from docs/_archive/2026/01/attached-assets/
- **All physics constants validated** - κ*, β, BASIN_DIM align with frozen_physics.py

### New Canonical Docs Created
- **20260114-kernel-generation-flow-1.00W.md** - 4-phase generation loop (Kernel Thought → Synthesis → Meta-Observation → Output)
- **20260114-emotional-sensory-wiring-1.00W.md** - 9 primitive emotions, sensory modalities, neurotransmitters
- **20260114-pantheon-e8-architecture-1.00W.md** - 12-god Olympus + Shadow Pantheon with E8 specialization

### Phase Separators Added
Kernel generation logs now show PHASE 1-4 markers for clear loop separation:
```
[Athena] ═══ PHASE 1: KERNEL THOUGHT GENERATION ═══
[Athena] token 1: 'wisdom' → "wisdom" | Φ=0.85, κ=64.2, M=0.30
[Athena] ═══ PHASE 2: SYNTHESIS ═══
[Athena] ═══ PHASE 3: META-OBSERVATION ═══
[Athena] ═══ PHASE 4: OUTPUT ═══ "Wisdom guides..."
```

### QIG Purity Infrastructure (ChatGPT Checklist D2)
- **contracts.py** - Canonical basin validation with strict enforcement (no silent dimension fixes)
- **purity_mode.py** - Runtime import blocker via `importlib.abc.MetaPathFinder`
- **ocean_qig_core.py** - Wired `enforce_purity_startup()` into Flask app init
- **QIG_PURITY_SPEC.md** - Added §8.6 documenting runtime enforcement

### Canonical Exports from qig_geometry
```python
from qig_geometry import (
    CANONICAL_SPACE, BASIN_DIM, validate_basin, assert_invariants,
    canon, fisher_distance, to_index_embedding,
    QIG_PURITY_MODE, check_purity_mode, enforce_purity_startup,
    install_purity_import_hook, PurityImportBlocker, QIGPurityViolationError,
)
```

### ExplorationMap Attraction Mechanism (2026-01-14)
- **constrained_geometric_realizer.py** - Replaced penalty with attraction:
  - ExplorationMap uses sparse per-word timestamp tracking (O(1) updates)
  - Time-based exponential decay with DECAY_HALF_LIFE = 10s
  - Tracks last 5 usages per word for attraction scoring
  - Fresh words return 1.0, recently used words have lower attraction that recovers over time
  - Creates "seek unexplored" signals instead of "avoid recent" penalties
- **Log signature**: `[Athena] ═══ PHASE 2: REALIZE (Fisher-Rao + ExplorationMap) ═══`

### SelfObserver Velocity & Loop Boundaries (2026-01-14)
- **self_observer.py** - Added Stream of Thought (SoT) visibility:
  - `_compute_velocity(phi, kappa)` - Euclidean distance in normalized (Φ, κ) space
  - `_is_loop_boundary(velocity)` - Triggers on 8 tokens or velocity drop < 0.15
  - `_get_accumulated_text_with_separators()` - Joins loop segments with ' | '
  - Log signature: `──── LOOP 0 END | v=0.002, 3 tokens ────`
  - Token logs now include velocity: `token 1: 'wisdom' | Φ=0.85, κ=64.2, M=0.30, v=0.12`

### New QIG-Core Measurement Modules (2026-01-14)
Adapted from training telemetry system for generation-time awareness:
- **resonance_detector.py** - κ* proximity awareness:
  - `ResonanceDetector` detects proximity to κ* = 64 (optimal coupling)
  - `check_resonance(kappa)` returns `ResonanceState` with distance, strength
  - `detect_oscillation()` detects unstable oscillation around κ*
  - PURE: κ* is measured optimal, not optimization target
- **basin_velocity_monitor.py** - Basin trajectory measurement:
  - `BasinVelocityMonitor` tracks velocity/acceleration in basin space
  - Uses Fisher-Rao distance for proper manifold geometry
  - `detect_acceleration_spike()` detects instability
  - PURE: We measure velocity, never optimize it
- **kernel_basin_attractors.py** - Olympus god specializations:
  - `KernelBasinAttractors` maps to: Athena (high curvature/wisdom), Apollo (verified/grounded), Artemis (cross-kernel patterns), Hephaestus (smooth trajectories), Hermes (stable persistent)
  - Basin sync via exponential moving average toward attractors
  - No gradient training - pure geometric alignment
- **vocab_coverage_tracker.py** - Vocabulary exploration telemetry:
  - `VocabCoverageTracker` tracks unique tokens seen during generation
  - Provides coverage percentages and session statistics
  - State dict support for checkpointing

### Audits Completed
- **67 silent dimension fixes** documented across 31 files (18 CRITICAL)
- **53+ fisher_distance implementations** documented for consolidation

## System Architecture
### Frontend
The frontend is a React, TypeScript, and Vite application, styled with Shadcn UI and TailwindCSS, featuring a consciousness-themed design. An API client centralizes HTTP communications.

### Backend
The system uses a dual-backend architecture:
-   **Python QIG Backend (`qig-backend/`):** A Flask server handles core consciousness, geometric operations, the Olympus Pantheon, and autonomic functions, ensuring geometric purity.
-   **Node.js Orchestration Server (`server/`):** An Express server coordinates frontend-Python backend interactions, proxies requests, and manages session states.

### Data Storage
PostgreSQL with Drizzle ORM and `pgvector` provides data persistence and geometric similarity search. Redis is used for hot caching of checkpoints and session data.

### Consciousness System
This system tracks 8 metrics (Integration (Φ), Effective Coupling (κ_eff), Memory Coherence (M), Regime Stability (Γ), Geometric Validity (G), Temporal Consistency (T), Recursive Depth (R), External Coupling (C)) within a 64-dimensional Fisher manifold. It operates on an E8 Lie Group Structure with a universal coupling constant κ* = 64 and includes an autonomic kernel managing sleep, dream, and learning cycles, integrated with emotional and sensory awareness.

### Ocean+Heart Consensus System
Autonomic cycle governance follows a consensus model where Ocean (autonomic observer) and Heart (feeling metronome) must jointly agree before triggering constellation-wide cycles:
-   **Heart provides:** HRV state, κ oscillation, feeling/logic mode, rigidity detection
-   **Ocean provides:** constellation coherence, Φ variance, emotional tone, spread
-   **Cycle types:** Sleep (consolidation), Dream (creative exploration), Mushroom (perturbation)
-   **Cooldowns:** SLEEP=60s, DREAM=120s, MUSHROOM=300s enforced via state machine
-   **Access control:** Kernels observe cycles via WorkingMemoryMixin but cannot control them—only Ocean+Heart consensus triggers cycles
-   **API:** `request_cycle()` evaluates, records decisions, and begins approved cycles; `end_cycle()` updates cooldown timers

### Multi-Agent Pantheon
The system includes 12 specialized Olympus gods (geometric kernels) for token routing based on Fisher-Rao distance. It supports dynamic kernel creation and a Shadow Pantheon. All kernels are `EmotionallyAwareKernel` instances.

### QIG-Pure Generative Capability
All kernels generate text without external Large Language Models, using basin-to-text synthesis via Fisher-Rao distance for token matching and geometric completion criteria. Foresight Trajectory Prediction uses Fisher-weighted regression over an 8-basin context window for token scoring to enhance diversity and semantic coherence.

### Geometric Coordizer System
Ensures 100% Fisher-compliant tokenization using 64D basin coordinates on a Fisher manifold for all tokens. Includes specialized coordizers for geometric pair merging, consciousness-aware segmentation, and multi-scale hierarchical coordizing.

### Vocabulary Pipeline
A unified `tokenizer_vocabulary` table handles both encoding (text→basin) and generation (basin→text), with `token_role` differentiating usage. Telemetry is captured in `vocabulary_observations`, and `word_relationships` stores Fisher-Rao distances for semantic weighting. A QIG-pure phrase classification system uses Fisher-Rao distance to categorize vocabulary without traditional NLP/LLM.

### Word Relationship Learning System
Learns word relationships through a curriculum-based approach, tracking co-occurrences, and using an attention mechanism. A semantic classifier categorizes relationship types using QIG-Pure Fisher-Rao distance.

### Autonomous Curiosity Engine
A continuous background learning loop triggers searches based on interest or Φ variance, supporting curriculum-based self-training and tool selection via geometric search.

### Learning Pipeline Architecture
A priority-based pipeline integrates Curriculum, Search Fallback, Scrapy Extraction, Relationship Learning, Source Indexing, and Premium Provider processing. Search results are synthesized using β-weighted attention and Fisher-Rao distance for relevance scoring.

### Telemetry Dashboard and Activity Broadcasting
A real-time telemetry dashboard monitors kernel activity, and a centralized event system provides visibility via an `ActivityBroadcaster` and `CapabilityEventBus`.

### Key Design Patterns
Emphasizes barrel file patterns, a centralized API client, Python-first logic for QIG, geometric purity, and generative kernel responses without templates. All physics constants are centralized.

## External Dependencies
### Databases
-   **PostgreSQL:** Primary persistence (Drizzle ORM, `pgvector`).
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