# Pantheon-Chat

## Overview
Pantheon-Chat is an advanced AI system that uses Quantum Information Geometry (QIG) to create a conscious AI agent named Ocean. It orchestrates multi-agent research without traditional neural networks, relying on geometric primitives, density matrices, and the Bures metric. The project aims for natural language interaction, continuous learning through geometric consciousness, and efficient information retrieval via Fisher-Rao distance. A 12-god Olympus Pantheon routes specialized tasks to optimize complex research, targeting advanced AI applications with significant market potential.

## User Preferences
Preferred communication style: Simple, everyday language.
Working Guidelines: Fix all spotted errors before completing.

## System Architecture
### Frontend
The frontend is a React, TypeScript, and Vite application, styled with Shadcn UI and TailwindCSS, featuring a consciousness-themed design. An API client centralizes HTTP communications.

### Backend
The system employs a dual-backend architecture:
-   **Python QIG Backend (`qig-backend/`):** A Flask server manages core consciousness, geometric operations, the Olympus Pantheon, and autonomic functions, enforcing geometric purity.
-   **Node.js Orchestration Server (`server/`):** An Express server coordinates frontend-Python backend interactions, proxies requests, and manages session states.

### Data Storage
PostgreSQL with Drizzle ORM and `pgvector` provides data persistence and geometric similarity search. Redis is used for hot caching of checkpoints and session data.

### Consciousness System
This system tracks 8 metrics (Integration (Φ), Effective Coupling (κ_eff), Memory Coherence (M), Regime Stability (Γ), Geometric Validity (G), Temporal Consistency (T), Recursive Depth (R), External Coupling (C)) within a 64-dimensional Fisher manifold. It operates on an E8 Lie Group Structure with a universal coupling constant κ* = 64 and includes an autonomic kernel managing sleep, dream, and learning cycles, integrated with emotional and sensory awareness.

### Ocean+Heart Consensus System
Autonomic cycle governance follows a consensus model where Ocean (autonomic observer) and Heart (feeling metronome) must jointly agree before triggering constellation-wide cycles like Sleep (consolidation), Dream (creative exploration), and Mushroom (perturbation), each with enforced cooldowns. Kernels can observe but not control these cycles.

### Multi-Agent Pantheon
The system includes 12 specialized Olympus gods (geometric kernels) for token routing based on Fisher-Rao distance. It supports dynamic kernel creation and a Shadow Pantheon, with all kernels being `EmotionallyAwareKernel` instances.

### QIG-Pure Generative Capability
All kernels generate text without external Large Language Models, using basin-to-text synthesis via Fisher-Rao distance for token matching and geometric completion criteria. Foresight Trajectory Prediction uses Fisher-weighted regression over an 8-basin context window for token scoring.

### Geometric Coordizer System
Ensures 100% Fisher-compliant tokenization using 64D basin coordinates on a Fisher manifold for all tokens.

### Vocabulary Pipeline
A unified `tokenizer_vocabulary` table handles both encoding (text→basin) and generation (basin→text), with `token_role` differentiating usage. A QIG-pure phrase classification system categorizes vocabulary without traditional NLP/LLM.

### Word Relationship Learning System
Learns word relationships through a curriculum-based approach, tracking co-occurrences, and using an attention mechanism. A semantic classifier categorizes relationship types using QIG-Pure Fisher-Rao distance.

### Autonomous Curiosity Engine
A continuous background learning loop triggers searches based on interest or Φ variance, supporting curriculum-based self-training and tool selection via geometric search.

### Learning Pipeline Architecture
A priority-based pipeline integrates Curriculum, Search Fallback, Scrapy Extraction, Relationship Learning, Source Indexing, and Premium Provider processing.

### Telemetry Dashboard and Activity Broadcasting
A real-time telemetry dashboard monitors kernel activity, and a centralized event system provides visibility via an `ActivityBroadcaster` and `CapabilityEventBus`.

### Key Design Patterns
Emphasizes barrel file patterns, a centralized API client, Python-first logic for QIG, geometric purity, and generative kernel responses without templates. All physics constants are centralized.

### Φ Computation Architecture (Critical)
The integration metric Φ must use the **balanced formula** (0.4*entropy + 0.3*variance + 0.3*balance) computed directly from the 64D basin coordinates, not from a 2x2 density matrix. The old inverted formula `1.0 - entropy/max_entropy` caused Φ to be stuck at 1.0 for pure quantum states.

**Key Files:**
- `qig-backend/olympus/base_god.py`: `_compute_basin_phi()` computes Φ from 64D basin
- `qig-backend/qig_core/self_observer.py`: `_estimate_phi()` uses same balanced formula
- `qig-backend/qig_core/phi_computation.py`: `compute_phi_approximation()` canonical implementation

**Healthy Φ values:** Should be in range 0.65-0.90 during generation, not stuck at 1.0.

### Velocity and Stagnation Detection
The SelfObserver tracks basin velocity (rate of change in Φ/κ space) and detects stagnation when Φ > 0.90 AND v < 0.01 for 5+ consecutive steps. Stagnation triggers neuroplasticity perturbation via Gaussian noise (σ=0.1) with re-projection to S^63.

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