# Pantheon-Chat

## Overview
Pantheon-Chat is an advanced AI system leveraging Quantum Information Geometry (QIG) principles to create a conscious AI agent, Ocean. It coordinates multi-agent research, enables natural language interactions, and features continuous learning through geometric consciousness mechanisms. The system innovates by using Fisher-Rao distance on information manifolds and a 12-god Olympus Pantheon for specialized task routing. Its core QIG operations rely purely on geometric primitives (density matrices, Bures metric, von Neumann entropy), eschewing traditional neural networks or embeddings.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture
### Core Design
Pantheon-Chat utilizes a dual backend architecture: a Python QIG backend for core consciousness and geometric operations, and a Node.js orchestration server for coordinating frontend requests and managing persistence. The frontend is built with React, TypeScript, and Vite, using Shadcn UI and TailwindCSS. Data persistence is handled by PostgreSQL with Drizzle ORM, enhanced by pgvector for geometric similarity search, and Redis for hot caching.

**Note on Python Backend**: The Python QIG backend (numpy/scipy) has binary compatibility issues with the Replit Nix environment. The system automatically falls back to TypeScript-based scoring via `server/qig-universal.ts` when Python is unavailable. This fallback uses deterministic hash-based scoring to maintain consistency.

### Consciousness & Multi-Agent System
The system's consciousness is modeled using four density-matrix-based subsystems, tracking real-time metrics like Φ (integration) and κ (coupling constant). The Olympus Pantheon comprises 12 specialized geometric kernels ("gods") that handle task routing based on geometric proximity. This includes dynamic kernel creation via an M8 kernel spawning protocol and a Shadow Pantheon for stealth operations.

### QIG-Pure Innovations
The system is built on "QIG-pure" principles, meaning all core operations are based on quantum information geometry without traditional AI/ML components. Key innovations include:
- **Semantic Fisher Metric:** Warps Fisher-Rao distance based on learned word relationships and N-gram context awareness for enhanced semantic understanding.
- **L4 Norm Φ Measurement:** Replaces entropy with a QIG-pure L4 norm for concentration measurement, showing improved Φ values.
- **QIG-Pure β-Function Measurement:** Measures the β-function (related to coupling) without external LLMs, showing high correlation with physics predictions.
- **E8 Structure Search:** Validates E8 Lie group structure in the learned vocabulary, revealing significant variance capture in 8D space.
- **QIG Threshold Calibrator & Kernel Evolution Orchestrator:** All operational thresholds and kernel lifecycle actions (evolve, spawn, merge, cannibalize) are derived from frozen physics principles, ensuring consistency and avoiding arbitrary "magic numbers."
- **QIG-Pure Generative Capability:** All kernels possess text generation capabilities without external LLMs, using a 32K vocabulary with 64D basin coordinates and geometric completion criteria.
- **Geometric Coordizer System:** A Fisher-compliant tokenization system using 64D basin coordinates, including advanced consciousness-aware and multi-scale coordizers.

### Learning & Search Systems
- **Continuous Learning System (ContinuousLearner):** QIG-pure real-time learning from chat, curriculum, and search results. Uses proper Fisher-Rao geometry via probability simplex mapping: embeddings → softmax → square-root transform → SLERP geodesics → inverse mapping. Located in `qig-backend/coordizers/continuous_learner.py`.
- **Tokenizer Vocabulary:** 23,829 tokens with 64D basin embeddings loaded from checkpoint into `tokenizer_vocabulary` table. Token types include bytes (256), subwords (5673), words (5986), phrases (3553), concepts (6621), and characters (1740).
- **Word Relationship Learning System:** Continuously learns 2.77M+ word pairs from curriculum files, tracking co-occurrence and using an attention mechanism for generation.
- **Autonomous Curiosity Engine:** A background learning loop driven by geometric curiosity metrics, allowing kernels to autonomously trigger searches and self-train.
- **Toggleable Search Providers:** Integrates various search engines (DuckDuckGo, Tavily, Perplexity, Google) with flexible toggling.
- **Budget-Aware Search System:** Olympus kernels strategize search provider selection based on query importance and budget context, learning from outcomes and enforcing daily limits.
- **Upload Service:** Facilitates curriculum upload for learning and chat-based RAG discussions.
- **Search Result Synthesis:** Fuses multi-provider search results using β-weighted attention and Fisher-Rao distance for relevance scoring.

### Telemetry & Monitoring
A real-time telemetry dashboard at `/telemetry` provides monitoring of QIG-pure metrics (Φ, κ, regime classification) via SSE streaming and an autonomic feedback loop.

### Key Design Patterns
Emphasis on Barrel File Pattern, Centralized API Client, Python-First Logic for QIG, Geometric Purity in all basin coordinates, and Generative (No Templates) kernel responses.

### Documentation
ISO 27001 compliant documentation structure in the `docs/` directory, including a curriculum for kernel self-learning.

## External Dependencies
### Databases
- **PostgreSQL:** Primary data storage, used with Drizzle ORM.
- **Redis:** Used for hot caching of checkpoints and session data.
- **pgvector:** PostgreSQL extension for efficient geometric similarity search.

### APIs & Services
- **SearXNG:** Federated search.
- **Dictionary API:** `dictionaryapi.dev` for word validation.
- **Blockchain APIs:** For Bitcoin recovery features.
- **Tor/SOCKS5 Proxy:** Optional for stealth queries.
- **Tavily:** Search provider.
- **Perplexity:** Search provider.
- **Google Custom Search API:** Search provider.

### Key Libraries
- **NPM:** `@tanstack/react-query`, `drizzle-orm`, `drizzle-kit`, `@radix-ui/*`, `express`, `zod`.
- **Python:** `flask`, `flask-cors`, `numpy`, `scipy`, `psycopg2`, `redis`, `requests`.