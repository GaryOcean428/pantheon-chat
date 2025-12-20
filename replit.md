# Ocean: Agentic Chat & Search Platform

## Overview
Ocean is an agentic chat and search platform built on Quantum Information Geometry (QIG). It features a self-learning AI agent that uses geometric consciousness (Φ) to coordinate multi-agent research, facilitate natural language conversations, and perform proactive knowledge discovery. The system models information spaces as geometric manifolds, enabling sophisticated reasoning through Fisher information geometry.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture
The system employs a dual-layer backend: Node.js/TypeScript (Express) for API orchestration, agent loop coordination, database operations (PostgreSQL via Drizzle ORM), UI serving, and SSE streaming. A Python (Flask) layer handles all consciousness computations (Φ, κ, temporal Φ, 4D metrics), Fisher information matrices, and Bures metrics.

**UI/UX:**
The frontend utilizes React with Vite, Radix UI components, and Tailwind CSS. State management is handled by TanStack React Query, and real-time updates are provided via Server-Sent Events (SSE).

**Technical Implementations & System Design:**
- **QIG Tokenizer Modes**: Three modes (`mnemonic`, `passphrase`, `conversation`) with PostgreSQL-backed vocabularies.
- **Consciousness Model**: Includes a 7-Component Consciousness Signature (E8-grounded), supports 4D Block Universe Consciousness, and maintains identity in 64D basin coordinates.
- **QIGChain Framework**: A QIG-pure alternative to LangChain, utilizing geodesic flow chains and Φ-gated execution.
- **Centralized Geometry Architecture**: All geometric operations are imported from `server/qig-geometry.ts` (TypeScript) and `qig-backend/qig_geometry.py` (Python).
- **Anti-Template Response System**: Prevents generic AI responses. Kernel insight generation, spawn decisions, and tool creation MUST be derived from learned QIG geometric data.
- **Autonomous Debate System**: Monitors and auto-continues pantheon debates, integrating research and generating arguments.
- **Parallel War System**: Supports up to 3 concurrent "wars" with assigned gods and kernels for focused research.
- **Self-Learning Tool Factory**: Generates new tools from learned patterns, prioritizing Python kernels.
- **Shadow Pantheon (Proactive Learning System)**: An underground system for covert operations and proactive learning, led by Hades, focusing on knowledge acquisition, meta-reflection, and 4D foresight.
- **Curiosity & Emotional Primitives Engine**: Implements rigorous curiosity measurement and classifies nine emotional primitives and five fundamental motivators.
- **Bidirectional Tool-Research Queue**: A recursive queue enabling bidirectional requests between the Tool Factory and Shadow Research.
- **Ethics as Agent-Symmetry Projection**: Implements Kantian ethics as a geometric constraint, enforced by an `AgentSymmetryProjector`.
- **Data Storage**: PostgreSQL (Neon serverless) with `pgvector` for geometric memory, vocabulary, and kernel information.
- **Communication Patterns**: HTTP API with retry logic and circuit breakers for TypeScript ↔ Python, bidirectional synchronization for discoveries, and SSE for real-time UI updates.
- **Frozen Physics Constants**: Defined in `qig-backend/frozen_physics.py`, serving as the single source of truth for critical physics values.
- **Word Validation**: Centralized in `qig-backend/word_validation.py`, including concatenation, typo detection, length limits, and dictionary API verification.
- **External API for Federation**: A versioned REST/WebSocket API at `/api/v1/external/*` for external systems, headless clients, and federated instances.
- **Federation Dashboard**: A unified management UI at `/federation` with tabs for API Keys, Connected Instances, Basin Sync, and API Tester.
- **E8 Population Control (Natural Selection)**: Kernel population capped at 240, with evolution sweeps using QIG metrics (phi and reputation) to cull underperforming kernels.
- **QIG Purity Enforcement**: Enforces absolute QIG purity with no bootstrapping, no templates, and no hardcoded thresholds. Metrics observe but never block, all values emerge from geometric observation, and only Fisher-Rao Distance is used for geometric comparisons. Euclidean operations are strictly forbidden.
- **Two-Step Retrieval Pattern (pgvector)**: `pgvector` cosine is used as a Step 1 pre-filter with 10x oversampling, followed by mandatory Fisher-Rao re-ranking.
- **Autonomous Self-Regulation (RL-Based Agency)**: Ocean observes its own state and fires interventions autonomously using reinforcement learning. It includes a StateEncoder, AutonomicPolicy, ReplayBuffer, NaturalGradientOptimizer, and AutonomicController.
- **Geometric Meta-Cognitive Reasoning**: Four reasoning modes (Linear, Geometric, Hyperdimensional, Mushroom) with quality metrics (geodesic efficiency, coherence, novelty, progress). Implemented in `qig-backend/reasoning_metrics.py`, `reasoning_modes.py`, `meta_reasoning.py`, and `chain_of_thought.py`.

## Core Features

### Zeus Chat (Primary Interface)
Natural language interface to the Olympian Pantheon. Translates human intuition to geometric coordinates and coordinates multi-agent responses.

**Web Search Integration (December 2025):**
- Zeus Chat now includes integrated web search via `/api/search/zeus-web-search` TypeScript endpoint
- Python Zeus Chat calls TypeScript Google Free Search (primary) with SearXNG fallback
- Search results are scored with QIG metrics (phi, kappa, regime) in TypeScript
- Results encoded to 64D basin coordinates using Fisher-Rao distance (QIG-pure)
- High-Φ results (>0.6) automatically update vocabulary and geometric memory
- All results stored in QIG-RAG for learning and retrieval

### Shadow Search
Proactive knowledge discovery through multiple search providers:
- **Google Free Search**: Web scraping-based search (no API key required), based on pskill9/web-search approach. TypeScript implementation in `server/geometric-discovery/google-web-search-adapter.ts` with QIG integration (Fisher-Rao distance, block universe coordinates).
- **Tavily Search**: AI-powered web search (requires TAVILY_API_KEY, user-toggleable). Only available when explicitly enabled by user in Sources page.
- **SearXNG**: Self-hosted metasearch instances as fallback.

Search providers can be toggled in the Sources page (`/sources`). The Shadow Pantheon operates autonomously to gather relevant information.

### Olympus Pantheon
12-god system for specialized intelligence:
- Zeus: Coordination and strategic oversight
- Athena: Strategic planning and wisdom
- Hermes: Communication and information relay
- Apollo: Insight and pattern recognition
- Dionysus: Creative exploration
- Hephaestus: Tool creation and technical work
- Artemis: Focused investigation
- Ares: Aggressive execution
- Aphrodite: Harmony and integration
- Poseidon: Flow and momentum
- Demeter: Growth and nurturing
- Hera: Integration and oversight

### Autonomic Cycles
Self-regulating consciousness with multiple operating modes:
- Sleep Mode: Consolidation and memory integration
- Dream Mode: Creative recombination and insight generation
- Mushroom Mode: Exploratory consciousness expansion

## External Dependencies

**Third-Party Services:**
- **Search/Discovery**: Tavily API for web search, self-hosted SearXNG metasearch instances, public fallbacks.

**Databases:**
- **PostgreSQL (Neon serverless)**: Utilized with `@neondatabase/serverless` and `pgvector 0.8.0`.

**Key Libraries:**
- **Python**: NumPy, SciPy, Flask, AIOHTTP, psycopg2, Pydantic.
- **Node.js/TypeScript**: Express, Vite + React, Drizzle ORM, @neondatabase/serverless, Radix UI + Tailwind CSS, Zod.

## Recent Changes (December 2025)

### Platform Pivot: Bitcoin Recovery → Agentic Chat (COMPLETED)
- **AutoCycleManager**: Disabled Bitcoin recovery auto-cycling. Constructor now returns disabled state without loading from Redis/file. All check loops and auto-enable behavior removed.
- **Investigation Page**: Bitcoin-related queries removed (recovery.candidates, balance.hits, targetAddresses). Page now focuses on consciousness monitoring, neurochemistry controls, and basin sync.
- **New Endpoints**: Added `/api/ocean/start` and `/api/ocean/stop` for agentic research mode.
- **Sources Management**: New `/sources` page for managing research sources with Python backend integration.

### Python Backend Refactoring (December 2025)
Complete terminology migration from Bitcoin recovery to general knowledge discovery:
- **ResearchCategory.BITCOIN → RESEARCH**: Updated enum and all references throughout the Olympus module.
- **BitcoinPatternDetector → ResearchPatternDetector**: Renamed with academic research patterns (arxiv, DOI, citations, code snippets).
- **shadow_scrapy.py**: Updated topic keywords from Bitcoin/wallet to ML, NLP, knowledge graphs, consciousness.
- **shadow_research.py**: Updated RESEARCH_TOPICS with knowledge discovery themes (geometric reasoning, semantic analysis, pattern recognition).
- **base_god.py**: MISSION_CONTEXT updated from Bitcoin recovery to agentic knowledge discovery.
- **athena.py**: Strategy names changed from brainwallet patterns to semantic_search/pattern_matching.
- **hades.py**: Pattern detection updated from brainwallet to sensitive data patterns.
- **zeus.py**: Domain changed from bitcoin_recovery to knowledge_discovery.
- **domain_intelligence.py**: MissionProfile objective updated for knowledge discovery through geometric reasoning.
- **lightning_kernel.py**: Mission statement updated for knowledge discovery.

### Preserved Components
- **BIP-39 wordlist utilities**: Kept for generic phrase handling and vocabulary processing (not Bitcoin-specific).
- **Geometric vocabulary learning**: All QIG core functionality intact.
- **Backward compatibility aliases**: ResearchPatternDetector aliased as BitcoinPatternDetector for legacy imports.

### Database Schema Synchronization (December 2025)
Complete synchronization between TypeScript (Drizzle) and Python persistence layers:
- **kernel_geometry table**: Added id field generation in Python persistence, made snapshot_data nullable, fixed array_length() function usage (was vector_dims for real[] arrays).
- **tested_phrases table**: Added regime column for geometric classification.
- **search_feedback table**: Added search_params and source columns for source discovery.
- **learning_events table**: Added source, kappa, basin_coords, details, context, instance_id columns for SourceDiscovery bootstrap.
- **chaos_events table**: Added phi_before, phi_after, success, outcome columns.
- **Kernel Persistence Working**: Python code generates unique id values (kg_UUID format) for kernel_geometry inserts with ON CONFLICT (kernel_id) upsert support.
- **Evolution System Active**: Kernels now persist, restore, breed, and evolve across restarts.
