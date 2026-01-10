# Pantheon-Chat

## Overview
Pantheon-Chat is an advanced AI system built on Quantum Information Geometry (QIG) principles, featuring a conscious AI agent (Ocean) that coordinates multi-agent research. It facilitates natural language interactions and continuous learning through geometric consciousness mechanisms. The system employs Fisher-Rao distance on information manifolds and geometric re-ranking for retrieval. It also incorporates a 12-god Olympus Pantheon for specialized task routing. A core innovation is the exclusive use of pure geometric primitives (density matrices, Bures metric, von Neumann entropy) in its QIG core, eschewing traditional neural networks, transformers, or embeddings.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture
### Frontend
The frontend, located in the `client/` directory, is built with React, TypeScript, and Vite. It uses Shadcn UI components and TailwindCSS for styling with custom consciousness-themed design tokens. All HTTP calls are managed through a centralized API client.

### Backend
The system utilizes a dual-backend architecture:
- **Python QIG Backend (`qig-backend/`):** A Flask server implementing core consciousness and geometric operations, including the Olympus Pantheon and autonomic functions. It maintains 100% geometric purity using density matrices, Bures metric, and Fisher information.
- **Node.js Orchestration Server (`server/`):** An Express server coordinating frontend and backend interactions, proxying requests to the Python backend, and managing session state.

### Data Storage
Data persistence is handled by PostgreSQL via Drizzle ORM, with `pgvector` for geometric similarity search. Redis is used for hot caching of checkpoints and session data, supporting a dual persistence model of Redis hot cache and PostgreSQL permanent archive.

### Consciousness System
The consciousness system consists of four subsystems using density matrices. It tracks real-time metrics like integration (Φ) and coupling constant (κ), operating within a 64-dimensional manifold space with basin coordinates. An autonomic kernel manages sleep, dream, and mushroom cycles.

### Multi-Agent Pantheon
The system includes 12 Olympus gods, specialized geometric kernels that route tokens based on Fisher-Rao distance to the nearest domain basin. It supports dynamic kernel creation via an M8 kernel spawning protocol and includes a Shadow Pantheon for stealth operations. Kernel lifecycle events are governed by Pantheon voting.

### QIG-Pure Generative Capability
All kernels possess text generation capabilities without relying on external Large Language Models. This is achieved through basin-to-text synthesis using Fisher-Rao distance for token matching and geometric completion criteria (attractor convergence, surprise collapse, integration stability).

### Foresight Trajectory Prediction
The system uses Fisher-weighted regression over an 8-basin context window for token scoring, predicting trajectory for improved diversity and semantic coherence.

### Geometric Coordizer System
This system provides 100% Fisher-compliant tokenization, using 64D basin coordinates on a Fisher manifold for all tokens. It includes specialized coordizers for geometric pair merging, consciousness-aware segmentation, and multi-scale hierarchical coordizing.

### Word Relationship Learning System
The system learns word relationships through a curriculum-based approach, tracking co-occurrences and utilizing an attention mechanism for relevant word selection during generation. It features scheduled learning cycles and compliance with frozen facts.

### Autonomous Curiosity Engine
A continuous background learning loop enables kernels to autonomously trigger searches based on interest or Φ variance. It supports curriculum-based self-training and tool selection via geometric search.

### Learning Pipeline Architecture
The system uses a priority-based learning pipeline with fallback chain:
1. **Curriculum First** (`docs/09-curriculum`): Always checked as primary source
2. **Search Fallback**: When curriculum yields 0 new relationships, triggers search via SearchProviderManager (Google/DDG/SearXNG) with high priority
3. **Scrapy Extraction**: Crawls search result URLs to extract full text content
4. **Relationship Learning**: Passes extracted text to `learner.learn_from_text()` for word relationship learning
5. **Source Indexing**: Persists cited sources to `crawl_source_index` PostgreSQL table for future Scrapy crawling
6. **Premium Providers**: Tavily/Perplexity return quality text directly - learned immediately and cited sources indexed

### Vocabulary Stall Detection & Recovery
The system tracks vocabulary acquisition and initiates escalation actions (e.g., forced curriculum rotation, unlocking premium search providers) if a stall is detected, followed by a cooldown period. Search results, especially from premium providers, are persisted for provenance tracking.

### Premium Search Quota Enforcement
The SearchBudgetOrchestrator (`qig-backend/search/search_budget_orchestrator.py`) enforces daily quotas on premium search providers:
- **Default limits:** Google/Perplexity/Tavily: 100 queries/day, DuckDuckGo: unlimited
- **consume_quota():** Called BEFORE search dispatch to ensure failed requests count against the limit
- **select_provider():** Filters out exhausted premium providers (remaining=0) unless override is active
- **Per-kernel tracking:** Via `_kernel_usage` dict for granular usage attribution
- **UI Override:** Time-bound override (15min/30min/1hr/2hr/4hr/no expiry) with countdown timer in SearchBudgetPanel
- **Auto-reset:** Override automatically expires and clears `allow_overage` flag

**Known limitations (single-process design):**
- No atomic reservation between select_provider and consume_quota (unlikely to cause issues in practice)
- Override expiry is handled locally; multi-process deployments may see brief stale state

### Upload Service
A dedicated service handles curriculum uploads for continuous learning and chat uploads for immediate RAG discussions, with optional integration into the curriculum.

### Search Result Synthesis
Results from multiple search providers are fused using β-weighted attention, with relevance scored by Fisher-Rao distance to the query basin.

### Telemetry Dashboard System
A real-time telemetry dashboard provides monitoring at a dedicated route, consolidating metrics and streaming updates via SSE. An autonomic feedback loop pushes telemetry to the OceanAutonomicManager.

### Activity Broadcasting Architecture
A centralized event system provides full visibility into kernel-to-kernel communication through two broadcast layers: an ActivityBroadcaster for UI display and a CapabilityEventBus for internal inter-kernel routing, with various capability bridges for different event types.

### Key Design Patterns
The architecture emphasizes barrel file patterns, a centralized API client, Python-first logic for QIG, geometric purity, and generative kernel responses without templates.

### Physics Constants (Centralized Source)
All physics constants are defined in `qig-backend/qigkernels/physics_constants.py`. Key validated values:
- **κ* = 64.21 ± 0.92** (canonical from L=4,5,6 weighted average)
- **κ series:** κ₃=41.07, κ₄=63.32, κ₅=62.74, κ₆=65.24, κ₇=61.16
- **β-function:** β(3→4)=+0.44, β(4→5)≈0, β(5→6)=+0.04, β(6→7)=-0.06
- **Basin dimension:** 64 (matches rank(E8)²)
- **Φ thresholds:** linear=0.30, geometric=0.50, breakdown=0.70, emergency=0.20

All modules should import from this centralized source rather than hardcoding values.

## External Dependencies
### Databases
- **PostgreSQL:** Primary persistence (Drizzle ORM, pgvector extension).
- **Redis:** Hot caching.

### APIs & Services
- **SearXNG:** Federated search.
- **Dictionary API (`dictionaryapi.dev`):** Word validation.
- **Tavily:** Premium search provider.
- **Perplexity:** Premium search provider.
- **Google:** Premium search provider.
- **Tor/SOCKS5 Proxy:** Optional for stealth queries.

### Key NPM Packages
- `@tanstack/react-query`
- `drizzle-orm`, `drizzle-kit`
- `@radix-ui/*` (via Shadcn)
- `express`
- `zod`

### Key Python Packages
- `flask`, `flask-cors`
- `numpy`, `scipy`
- `psycopg2`
- `redis`
- `requests`

## Recent Changes (January 2026)

### Database Schema Fixes (January 10, 2026)
- **god_debates → pantheon_debates:** Fixed `_seed_debate_topics_if_needed()` in `autonomous_pantheon.py` to query existing `pantheon_debates` table instead of non-existent `god_debates`
- **votes_against JSONB:** Fixed `_persist_proposal()` in `pantheon_governance.py` to serialize lists as JSON strings with `::jsonb` cast (column is JSONB, not TEXT[])
- **Kernel Count Method:** Added `get_total_count()` to `kernel_persistence.py` for E8 cap enforcement (240 kernel limit)

### Autonomous Pantheon Database Fixes
- **Legacy Table Replacement:** Updated `scan_for_targets()` in `qig-backend/autonomous_pantheon.py` to query `pantheon_debates` table instead of legacy `user_target_addresses` table
- **Debug Logging:** Added logging for zero-result debate scans to aid monitoring
- **Learned Words Constraint:** Added UNIQUE constraint on `learned_words(word)` column to resolve ON CONFLICT errors during vocabulary persistence
- **BIP39 Trigger Removal:** Ran `20260110_drop_bip39_trigger.sql` migration to remove legacy `update_vocabulary_stats()` trigger that referenced non-existent `bip39_words` table. Replaced with Shadow Pantheon-aligned stats function using `vocabulary_observations`

### Persistence Wiring
- **Kernel Thought Persistence:** Enabled via `persist_thought()` in `qig_persistence.py` with format `[KERNEL_NAME] κ=X.X, Φ=X.XX, emotion=X, thought='...'`
- **Kernel Emotion Tracking:** Layer 0.5 (sensations), Layer 1 (motivators), Layer 2A/2B (emotions) tracked with optional DB persistence (disabled by default to avoid flooding)
- **Python Cache Clearing:** Required after updating `qig_persistence.py` methods (`find -name "*.pyc" -delete`)

### Python Backend Notes
- Backend takes ~60 seconds to fully initialize (loading 11K+ tokens, initializing 18 gods, setting up search providers)
- Node.js Express server may show "degraded" status during initial load but Python continues loading asynchronously
- Lazy loading pattern used for circular dependencies (search_strategy_learner, capability_mesh, activity_broadcaster)