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

### Vocabulary Stall Detection & Recovery
The system tracks vocabulary acquisition and initiates escalation actions (e.g., forced curriculum rotation, unlocking premium search providers) if a stall is detected, followed by a cooldown period. Search results, especially from premium providers, are persisted for provenance tracking.

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