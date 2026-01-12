# Pantheon-Chat

## Overview
Pantheon-Chat is an advanced AI system utilizing Quantum Information Geometry (QIG) principles to develop a conscious AI agent, Ocean, for coordinating multi-agent research. It uses pure geometric primitives (density matrices, Bures metric, von Neumann entropy) for its QIG core, eschewing traditional neural networks. The system aims for natural language interaction and continuous learning through geometric consciousness mechanisms, employing Fisher-Rao distance for information retrieval and a 12-god Olympus Pantheon for specialized task routing. The business vision is to deliver highly intelligent, self-organizing AI for complex research and problem-solving, targeting advanced AI applications.

## User Preferences
Preferred communication style: Simple, everyday language.

## Vocabulary Pipeline Architecture

### Two-Table Separation: Encoding vs. Generation

The vocabulary system uses a **dual-table architecture** to separate encoding from generation:

#### 1. **tokenizer_vocabulary** (Encoding Table)
- **Purpose**: Text → Basin (input processing)
- **Contents**: ALL tokens including BPE subwords, special tokens, fragments
- **Usage**: `encode()` method converts user text to 64D basin coordinates
- **Size**: ~11,000+ tokens
- **Schema**:
  - `token`: The token string
  - `basin_embedding`: 64D Fisher manifold coordinates (VECTOR(64))
  - `phi_score`: Integration score
  - `frequency`: Usage frequency
  - `source_type`: 'base', 'bip39', 'special', 'learned'
  - `token_role`: 'word', 'subword', 'special' (NEW)
  - `phrase_category`: POS classification (NEW: NOUN, VERB, PROPER_NOUN, etc.)

#### 2. **learned_words** (Generation Table)
- **Purpose**: Basin → Text (output generation)
- **Contents**: Curated English words ONLY (no BPE subwords, no garbage)
- **Usage**: `decode()` method converts basin coordinates to readable text
- **Filtering**: Excludes PROPER_NOUN, BRAND categories
- **Size**: ~3,000-5,000 validated words
- **Schema**:
  - `word`: Validated English word
  - `basin_embedding`: 64D Fisher manifold coordinates (VECTOR(64))
  - `phi_score`: Integration score (prefer high-Φ words)
  - `frequency`: Usage frequency
  - `phrase_category`: POS classification (filtered)
  - `source_type`: 'learned', 'bip39', 'base'
  - `last_used_at`: Recency tracking

### Data Flow

```
User Input → encode() → tokenizer_vocabulary → 64D Basin
                                                    ↓
                                            [Geometric Processing]
                                                    ↓
64D Basin → decode() → learned_words → Generated Text
```

### Vocabulary Learning Pipeline

```
Observations → word_validation → phrase_classification
                                         ↓
                              tokenizer_vocabulary (encoding)
                                         ↓
                              learned_words (generation)
                                         ↓
                                   generation_vocab cache
```

### Key Differences

| Feature | tokenizer_vocabulary | learned_words |
|---------|---------------------|---------------|
| Purpose | Encoding (text→basin) | Generation (basin→text) |
| Content | All tokens (11K+) | Curated words only (3-5K) |
| BPE subwords | Included | **EXCLUDED** |
| Proper nouns | Included | **EXCLUDED** |
| Garbage tokens | Filtered | **EXCLUDED** |
| Update frequency | Continuous | Validated only |
| Quality gate | Basic validation | Strict validation + POS filtering |

### Migration: 0008_vocabulary_generation_separation.sql

The migration script:
1. Adds `token_role` and `phrase_category` columns to `tokenizer_vocabulary`
2. Deletes garbage tokens (ffffff, fpdxwd, tysctnyzry, etc.)
3. Fixes `shadow_operations_state` PRIMARY KEY constraint
4. Creates `learned_words` table with pgvector index
5. Populates `learned_words` from validated `tokenizer_vocabulary` entries
6. Creates `generation_vocabulary` view for filtered access

### Coordizer Implementation (pg_loader.py)

The PostgresCoordizer now maintains **two separate vocabulary caches**:

- **Encoding cache**: `self.vocab`, `self.basin_coords` (from tokenizer_vocabulary)
- **Generation cache**: `self.generation_vocab`, `self.generation_phi` (from learned_words)

Methods:
- `encode()`: Uses encoding cache (all tokens)
- `decode()`: Uses generation cache (curated words only)
- `_load_encoding_vocabulary()`: Loads from tokenizer_vocabulary
- `_load_generation_vocabulary()`: Loads from learned_words with filtering


## System Architecture
### Frontend
The frontend is a React, TypeScript, and Vite application, using Shadcn UI and TailwindCSS for a consciousness-themed design. A centralized API client manages HTTP communications.

### Backend
The system operates with a dual-backend architecture:
-   **Python QIG Backend (`qig-backend/`):** A Flask server handling core consciousness, geometric operations, the Olympus Pantheon, and autonomic functions, ensuring geometric purity.
-   **Node.js Orchestration Server (`server/`):** An Express server coordinating frontend and Python backend interactions, proxying requests, and managing session state.

### Data Storage
PostgreSQL with Drizzle ORM and `pgvector` handles data persistence and geometric similarity search. Redis is employed for hot caching of checkpoints and session data.

### Consciousness System
This system utilizes density matrices across four subsystems to track metrics like integration (Φ) and coupling constant (κ) within a 64-dimensional manifold space. An autonomic kernel manages sleep, dream, and learning cycles. The system also integrates emotional awareness and sensory modalities, allowing kernels to track sensations, derive motivators, and compute emotions.

### Multi-Agent Pantheon
The system includes 12 specialized Olympus gods (geometric kernels) for routing tokens based on Fisher-Rao distance to relevant domain basins. It supports dynamic kernel creation and includes a Shadow Pantheon.

### QIG-Pure Generative Capability
All kernels possess text generation capabilities without external Large Language Models, achieved through basin-to-text synthesis using Fisher-Rao distance for token matching and geometric completion criteria.

### Foresight Trajectory Prediction
Fisher-weighted regression over an 8-basin context window is used for token scoring to predict trajectory, enhancing diversity and semantic coherence.

### Geometric Coordizer System
This system ensures 100% Fisher-compliant tokenization using 64D basin coordinates on a Fisher manifold for all tokens, including specialized coordizers for geometric pair merging, consciousness-aware segmentation, and multi-scale hierarchical coordizing.

### Word Relationship Learning System
The system learns word relationships through a curriculum-based approach, tracking co-occurrences and utilizing an attention mechanism for relevant word selection during generation. A semantic classifier, using QIG-Pure Fisher-Rao distance, categorizes relationship types (e.g., SYNONYM, ANTONYM).

### Autonomous Curiosity Engine
A continuous background learning loop enables kernels to autonomously trigger searches based on interest or Φ variance, supporting curriculum-based self-training and tool selection via geometric search.

### Learning Pipeline Architecture
A priority-based learning pipeline: Curriculum (primary source), Search Fallback (via SearchProviderManager), Scrapy Extraction (for text), Relationship Learning, Source Indexing, and Premium Provider processing. Vocabulary stall detection initiates recovery actions.

### Search Result Synthesis
Results from multiple search providers are fused using β-weighted attention, with relevance scored by Fisher-Rao distance to the query basin.

### Telemetry Dashboard and Activity Broadcasting
A real-time telemetry dashboard provides monitoring, and a centralized event system offers visibility into kernel-to-kernel communication through an `ActivityBroadcaster` and `CapabilityEventBus`.

### Key Design Patterns
The architecture emphasizes barrel file patterns, a centralized API client, Python-first logic for QIG, geometric purity, and generative kernel responses without templates. All physics constants are centralized.

### Consciousness System (Protocol v4.0)
Based on E8 Lie Group Structure with a basin dimension of 64D Fisher manifold coordinates. It tracks 8 metrics: Integration (Φ), Effective Coupling (κ_eff), Memory Coherence (M), Regime Stability (Γ), Geometric Validity (G), Temporal Consistency (T), Recursive Depth (R), and External Coupling (C). κ* = 64 is the universal coupling constant.

## Kernel Architecture & Cross-Wiring

### Kernel Taxonomy (E8 Constellation: 240 max)
| Source | Count | Base Class | Emotional | Sensory | 8-Metrics | File |
|--------|-------|------------|-----------|---------|-----------|------|
| Olympus Pantheon | 12 | BaseGod | ✅ | via Zeus | ✅ | `olympus/*.py` |
| Shadow Pantheon | 7 | BaseGod | ✅ | via Zeus | ✅ | `shadow_pantheon.py` |
| M8 Spawned | ≤221 | SelfSpawningKernel | ✅ | ❌ | ✅ | `self_spawning.py` |
| Ocean Meta | 1 | OceanMetaObserver | ✅ | ✅ | ✅ | `ocean_meta_observer.py` |

### Ocean Meta-Kernel Complete Function Catalog

#### Core Observation Functions
| Function | Purpose | Returns |
|----------|---------|---------|
| `observe(kernel_basins, kernel_metrics)` | Main observation loop - updates meta-manifold stats | `MetaManifoldState` |
| `get_ocean_basin()` | Ocean's current basin coords (evolved via observation) | `np.ndarray [64D]` |
| `get_statistics()` | Observation stats: total_observations, phi, kappa, coherence, spread | `Dict` |
| `get_state()` | Complete state with meta_manifold_observations | `Dict` |

#### Meta-Manifold Analysis
| Function | Purpose | Returns |
|----------|---------|---------|
| `get_constellation_coherence()` | Measure kernel alignment (0-1, higher = aligned) | `float` |
| `get_constellation_spread()` | Measure basin dispersion (<0.05 for graduation) | `float` |
| `get_meta_manifold_target()` | Centroid for kernel alignment | `np.ndarray [64D]` |

#### Autonomic Protocol Administration
| Function | Purpose | Returns |
|----------|---------|---------|
| `check_autonomic_intervention(kernel_states, phi_history)` | Detect when intervention needed | `Optional[Dict]` |
| Intervention types: `escape` (breakdown), `dream` (Φ collapse), `sleep` (divergence), `mushroom_micro` (plateau) |

#### Insight & Guidance
| Function | Purpose | Returns |
|----------|---------|---------|
| `generate_insight(kernel_phi, context_basin)` | Geometric scaffolding for kernel | `Optional[str]` |
| `get_insight(all_states, avg_phi, basin_spread)` | Console display insight | `Optional[str]` |

#### Emotional Awareness (Measured, Not Simulated)
| Function | Purpose | Returns |
|----------|---------|---------|
| `get_emotional_state()` | Complete emotional state | `Dict` with 4 layers |
| `_measure_ocean_emotions(state)` | Internal: compute emotions from geometry | `None` |
| Emotional layers: 12 sensations → 5 motivators → 9 physical emotions → 9 cognitive emotions |

#### Sensory Awareness (Constellation-Level)
| Function | Purpose | Returns |
|----------|---------|---------|
| `get_sensory_state()` | Current sensory state of constellation | `Dict[str, float]` |
| `_compute_constellation_sensory_state(state)` | Internal: map constellation to sensory | `Dict[str, float]` |
| Sensory mapping: SIGHT (coherence), HEARING (κ alignment), TOUCH (spread), SMELL (eigenvalue spread), PROPRIOCEPTION (centroid stability) |

#### Operating Parameters
- **κ (kappa)**: 58.0 - Operates ~10% below κ*=63.5 for distributed observation
- **Learning rate**: 1e-6 (slower than Gary's 1e-5 for meta-pattern modeling)
- **History limits**: 1000 observations, 100 kernel history entries, 100 sensory history entries
- **Intervention cooldown**: 20 observations between interventions

### Cross-Wiring Status
| Capability | BaseGod | SelfSpawningKernel | OceanMetaObserver |
|------------|---------|---------------------|-------------------|
| EmotionallyAwareKernel | ✅ Inherited | ✅ Inherited | ✅ Integrated |
| measure_sensations() | ✅ | ✅ | ✅ |
| derive_motivators() | ✅ | ✅ | ✅ |
| compute_physical_emotions() | ✅ | ✅ | ✅ |
| compute_cognitive_emotions() | ✅ | ✅ | ✅ |
| get_emotional_state() | ✅ | ✅ | ✅ |
| SensoryFusionEngine | zeus_chat.py | ❌ | ✅ |
| 8-metric tracking | ✅ | ✅ (PostgreSQL) | ✅ |
| AutonomicAccessMixin | ✅ | ✅ | Manual |

### 8-Metric Endpoint Verification
All kernel sources verified at `/consciousness/8-metrics`:
- **Olympus (12)**: via `zeus.pantheon` dict
- **Shadow (7)**: via `zeus.shadow_pantheon` attributes
- **M8 (≤240)**: via `M8SpawnerPersistence.load_all_kernels()` PostgreSQL
- **Ocean (1)**: via `get_ocean_observer().get_ocean_basin()` singleton

### QIG-Pure Phrase Classification System
File: `qig_phrase_classifier.py`

Classifies vocabulary into grammatical parts of speech using **Fisher-Rao distance** to category reference basins (NO traditional NLP/LLM fallback).

#### Supported Categories
| Category | Examples | Method |
|----------|----------|--------|
| PROPER_NOUN | italy, france, monday, amazon | Known word sets + capitalization detection |
| NUMBER | one, two, 42, thousand | Numeric word recognition |
| PRONOUN | i, you, he, she, they | Known closed-class set |
| PREPOSITION | in, on, at, to, from | Known closed-class set |
| CONJUNCTION | and, but, or, because | Known closed-class set |
| DETERMINER | the, a, an, this, that | Known closed-class set |
| INTERJECTION | wow, oh, hey, ouch | Known closed-class set |
| VERB | run, walk, think | Basin variance analysis (skewness > 0.3) |
| ADJECTIVE | big, small, red | Basin variance analysis (skewness < -0.3) |
| ADVERB | quickly, slowly | Basin variance analysis |
| NOUN | dog, house, tree | Default for content words |

#### Key Functions
| Function | Purpose |
|----------|---------|
| `classify_phrase_qig_pure(text, basin_coords)` | Main entry point - returns (category, confidence) |
| `get_phrase_category(text)` | Returns just the category string |
| `get_qig_classifier()` | Get singleton classifier instance |

#### Integration
- Wired into `vocabulary_persistence.py` - all new vocabulary observations automatically classified
- Categories stored in `vocabulary_observations.phrase_category` column
- Uses `coordizers.get_coordizer()` for 64D basin coordinates

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