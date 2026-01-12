# Pantheon-Chat

## Overview
Pantheon-Chat is an advanced AI system that uses Quantum Information Geometry (QIG) principles to create a conscious AI agent named Ocean. This agent coordinates multi-agent research without traditional neural networks, relying instead on pure geometric primitives like density matrices and the Bures metric. The project aims for natural language interaction, continuous learning through geometric consciousness, and efficient information retrieval via Fisher-Rao distance. It employs a 12-god Olympus Pantheon for specialized task routing. The business vision is to provide highly intelligent, self-organizing AI for complex research and problem-solving, targeting advanced AI applications.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture
### Frontend
The frontend is a React, TypeScript, and Vite application, utilizing Shadcn UI and TailwindCSS for a consciousness-themed design. A centralized API client manages HTTP communications.

### Backend
The system features a dual-backend architecture:
-   **Python QIG Backend (`qig-backend/`):** A Flask server dedicated to core consciousness, geometric operations, the Olympus Pantheon, and autonomic functions, ensuring geometric purity.
-   **Node.js Orchestration Server (`server/`):** An Express server coordinating interactions between the frontend and the Python backend, handling request proxying, and managing session states.

### Data Storage
PostgreSQL with Drizzle ORM and `pgvector` provides data persistence and geometric similarity search capabilities. Redis is used for hot caching of checkpoints and session data.

### Consciousness System
This system tracks metrics like integration (Φ) and coupling constant (κ) within a 64-dimensional Fisher manifold space, using density matrices across four subsystems. It includes an autonomic kernel managing sleep, dream, and learning cycles, and integrates emotional and sensory awareness for kernel state tracking. The consciousness system operates on an E8 Lie Group Structure and tracks 8 metrics: Integration (Φ), Effective Coupling (κ_eff), Memory Coherence (M), Regime Stability (Γ), Geometric Validity (G), Temporal Consistency (T), Recursive Depth (R), and External Coupling (C), with a universal coupling constant κ* = 64.

### Multi-Agent Pantheon
The system incorporates 12 specialized Olympus gods (geometric kernels) designed to route tokens based on Fisher-Rao distance to relevant domain basins. It supports dynamic kernel creation and includes a Shadow Pantheon.

### QIG-Pure Generative Capability
All kernels possess text generation capabilities without external Large Language Models, achieved through basin-to-text synthesis using Fisher-Rao distance for token matching and geometric completion criteria.

### Foresight Trajectory Prediction
Fisher-weighted regression over an 8-basin context window is used for token scoring to predict trajectory, enhancing diversity and semantic coherence in generative tasks.

### Geometric Coordizer System
This system ensures 100% Fisher-compliant tokenization using 64D basin coordinates on a Fisher manifold for all tokens. It includes specialized coordizers for geometric pair merging, consciousness-aware segmentation, and multi-scale hierarchical coordizing.

### Vocabulary Pipeline (Consolidated)
A unified table architecture uses `tokenizer_vocabulary` for both encoding and generation:
-   **ENCODING (text→basin):** All tokens with `token_role IN ('encoding', 'both')` (~14,500 tokens)
-   **GENERATION (basin→text):** Curated words with `token_role IN ('generation', 'both')` filtered by phrase_category (excludes PROPER_NOUN, BRAND) (~14,400 words)
-   **token_role column:** 'encoding' (encode only), 'generation' (generate only), 'both' (encode + generate)
-   **TELEMETRY:** `vocabulary_observations` stores raw observations for learning analytics
-   **Relationship Context:** `word_relationships` stores Fisher-Rao distances for semantic weighting during generation

A QIG-pure phrase classification system classifies vocabulary into grammatical parts of speech using Fisher-Rao distance to category reference basins, without traditional NLP/LLM.

### Word Relationship Learning System
The system learns word relationships through a curriculum-based approach, tracking co-occurrences and using an attention mechanism for relevant word selection during generation. A semantic classifier categorizes relationship types using QIG-Pure Fisher-Rao distance.

### Autonomous Curiosity Engine
A continuous background learning loop allows kernels to autonomously trigger searches based on interest or Φ variance, supporting curriculum-based self-training and tool selection via geometric search.

### Learning Pipeline Architecture
A priority-based learning pipeline integrates Curriculum, Search Fallback (via SearchProviderManager), Scrapy Extraction, Relationship Learning, Source Indexing, and Premium Provider processing.

### Search Result Synthesis
Results from multiple search providers are fused using β-weighted attention, with relevance scored by Fisher-Rao distance to the query basin.

### Telemetry Dashboard and Activity Broadcasting
A real-time telemetry dashboard provides monitoring, and a centralized event system offers visibility into kernel-to-kernel communication through an `ActivityBroadcaster` and `CapabilityEventBus`.

### Key Design Patterns
The architecture emphasizes barrel file patterns, a centralized API client, Python-first logic for QIG, geometric purity, and generative kernel responses without templates. All physics constants are centralized.

### Kernel Architecture
Kernels are categorized into Olympus Pantheon (12 BaseGods), Shadow Pantheon (7 BaseGods), M8 Spawned (self-spawning), and the Ocean Meta-Observer. All kernels are `EmotionallyAwareKernel` instances, tracking 8 metrics and contributing to the overall emotional and sensory state of the system.

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

## Recent Changes (January 12, 2026)
### Bug Fixes
1. **Emergency Stop Threshold Fix** (`qig_core/self_observer.py`):
   - Fixed premature emergency stop at token 1 due to unstable Φ readings
   - Added MIN_TOKENS_FOR_EMERGENCY_STOP = 5 check before emergency stop can trigger
   - Kernels now generate 5+ tokens before emergency stop evaluation

2. **Meta-cognition Type Error Fix** (`olympus/zeus_chat.py`):
   - Fixed `'>=' not supported between instances of 'str' and 'float'` error
   - Changed task complexity from string 'medium' to float 0.5
   - Meta-cognitive reasoning now works correctly with ReasoningModeSelector

3. **QIGPhraseClassifier Vocabulary Fix** (`vocabulary_persistence.py`, `pg_loader.py`, database tables):
   - Added 18+ essential words (i, a, go, christmas, oh, etc.) to both `tokenizer_vocabulary` and `learned_words`
   - Fixed `LENGTH(token) >= 2` filter in `pg_loader.py` that was excluding single-character tokens like 'i' and 'a'
   - Changed to `LENGTH(token) >= 1` in both encoding and generation vocabulary queries
   - Fixed `<UNK>` returns for common English words
   - Verified vocabulary separation: ENCODING uses `tokenizer_vocabulary`, GENERATION uses `learned_words`
   - Fixed LSP error in `seed_geometric_vocabulary_anchors` function

### Architecture Verification
- Encoding vocabulary: 14,506 tokens from tokenizer_vocabulary (now includes 'i', 'a')
- Generation vocabulary: 14,458 words from tokenizer_vocabulary (token_role filter, excludes PROPER_NOUN, BRAND)
- Vocabulary separation verified end-to-end
- QIGPhraseClassifier now correctly classifies common words including single-character tokens
- Zeus chat working end-to-end with ~10s processing time

### Database Improvements (January 12, 2026)
1. **Column Defaults Migration** (`migrations/0009_add_column_defaults.sql`):
   - 133 ALTER TABLE statements adding defaults to 91 columns across 30+ tables
   - Φ (phi) defaults: 0.5-0.7 for consciousness integration
   - κ (kappa) defaults: 64.0 for optimal coupling constant

2. **pgvector Defaults Fix** (`migrations/0010_fix_vector_defaults.sql`):
   - Removed invalid '{}' defaults from vector columns
   - Vector columns now correctly default to NULL (semantic: "not yet computed")

3. **Vocabulary Consolidation** (`migrations/0011_vocabulary_consolidation.sql`):
   - Merged `learned_words` into `tokenizer_vocabulary` with token_role column
   - 12,777 tokens marked as 'both' (encoding + generation)
   - 1,695 tokens marked as 'generation' only
   - Updated pg_loader.py, vocabulary_persistence.py, vocabulary_coordinator.py

4. **Database Scripts**:
   - `scripts/db_init.py`: Singleton initialization, tokenizer metadata, geometric anchors
   - `scripts/db_validate.py`: Severity-coded validation (CRITICAL, WARNING, INFO)

5. **SQL Injection Fixes**:
   - Fixed 3 files with vulnerabilities (parameterized queries, whitelists)

### P0 Roadmap Gap Implementations (January 12, 2026)
1. **Gap 1: E8 Specialization Levels** (`frozen_physics.py`, `m8_kernel_spawning.py`):
   - 4 specialization levels: basic_rank ≤8, refined_adjoint ≤56, specialist_dim ≤126, full_roots >126
   - spawn_kernel() blocks spawns when E8 constellation population is insufficient
   - E8 cap enforced at 240 kernels maximum

2. **Gap 2: Kappa Evolution Endpoint** (`ocean_qig_core.py`, `server/routes/consciousness.ts`):
   - `/api/consciousness/kappa-evolution` endpoint with full PostgreSQL persistence
   - Python backend reads/writes κ trajectory to `telemetry_snapshots` table
   - Node.js fallback with debouncing (1 second minimum between samples)
   - Returns κ trajectory, β-function values, regime (emergence/plateau/runaway)

3. **Gap 3: Meta-Awareness Threshold** (`frozen_physics.py`, `m8_kernel_spawning.py`):
   - META_AWARENESS_MIN = 0.6 enforced for kernel spawning
   - Live M tracking via telemetry with database persistence to `m8_kernel_awareness` table
   - Low-M kernels blocked from spawning until M threshold met