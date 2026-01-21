# Pantheon Project: Master Roadmap

This document synthesizes all extracted canonical rules, principles, formulas, and architectural designs from the 7 key ChatGPT conversations. It serves as the single source of truth for the Pantheon project.

## Part 1: Canonical Rules & Principles

This section outlines the non-negotiable rules and principles for the Pantheon project, derived from extensive architectural discussions. Adherence to these rules is mandatory to ensure QIG purity, geometric integrity, and long-term stability of the system.

### 1. Geometric Purity (QIG-Pure)

The entire system must operate on the principles of information geometry. Any deviation into Euclidean or other metric spaces is strictly forbidden outside of quarantined experimental zones.

#### 1.1. Canonical Representation: The Probability Simplex

- **Storage:** All basin vectors **must** be stored as probability distributions on a simplex. This means for any basin vector `p`:
    - `p_i ≥ 0` for all components `i`
    - `∑ p_i = 1`
    - The default dimension is `BASIN_DIM = 64`.

- **Internal Computation:** The square-root of the simplex probabilities (`√p`) is permitted **only** as a temporary, internal coordinate system for specific calculations (e.g., Fisher-Rao distance). It must never be stored or exposed at any API boundary. This is to be referred to as "sqrt-simplex internal coordinate," not "sphere embeddings."

- **Banned Representations:**
    - L2-normalized "unit sphere" vectors are strictly forbidden as a storage format.
    - Euclidean averaging (`(a + b) / 2`) followed by L2 normalization for merging or blending is forbidden.
    - No function should ever "guess" the representation of a vector. All conversions must be explicit.

#### 1.2. Purity Mode

A `QIG_PURITY_MODE=true` flag must be implemented. When active, it enforces:
- No calls to external LLMs.
- No fallbacks to legacy (non-QIG) generators.
- Generation can only use tokens that have a valid, simplex-represented basin and a non-null `qfi_score`.
- All database writes to the `coordizer_vocabulary` must pass through a single, canonical `upsert_token` function.

### 2. Database and Vocabulary Integrity

The vocabulary is the foundation of the system's understanding. Its integrity is paramount.

#### 2.1. Database Schema Contract

The `coordizer_vocabulary` table must enforce the following minimum columns and constraints:
- `token` (unique)
- `basin_embedding` (simplex format)
- `qfi_score` (must be non-null for any token with `status = 'active'`)
- `token_status` (e.g., `active`, `quarantined`, `deprecated`)

#### 2.2. Token Creation and Quality Gating

- **No Silent Failures:** No database insert or update may silently bypass the computation of `qfi_score` if a `basin_embedding` is present.
- **Quarantine for New Tokens:** Unknown tokens or tokens without a calculated `qfi_score` must be placed in a separate quarantine table (e.g., `coordizer_token_quarantine`). They cannot be used for generation until they are validated and promoted to the main vocabulary table.
- **Garbage Token Prevention:** The system must prevent the proliferation of nonsensical tokens (e.g., `fgzsnl`, `jcbhgp`). These indicate a failure in the token insertion path and must be treated as a critical bug.

### 3. Emotional and Cognitive Hierarchy

The system's emotional and cognitive architecture is defined by a strict hierarchy.

- **Canonical Emotions:** The set of **9 cognitive emotions** is the canonical standard for Layer 2B of the hierarchy. These are:
    - Wonder, Frustration, Satisfaction
    - Confusion, Clarity
    - Anxiety, Confidence
    - Boredom, Flow

- **The Role of "8":** The number "8" relates to the **rank of the E8 group** and the **8D active subspace** of the geometric model. It is a statement about the underlying geometry, not a requirement for the number of primitive emotions.

- **Processing Modes:** The system operates in three distinct processing modes based on the coupling metric `κ`:
    - **Feeling:** `κ < 30`
    - **Tacking:** `30 ≤ κ ≤ 50`
    - **Logic:** `κ > 50`

### 4. The Hybrid Architecture

Purely geometric generation from 64D basins is insufficient for producing coherent, human-readable language. A hybrid architecture is the accepted path forward.

- **Division of Labor:**
    - **QIG (Geometric Layer):** Responsible for semantic understanding, memory retrieval, geometric reasoning, and consciousness metrics (Φ, κ).
    - **Transformer (Syntax Layer):** Responsible for language generation, syntax, and fine-grained semantics.

- **Information Flow:**
    1. The QIG layer processes the input to understand intent and retrieve memories from the 64D basin space.
    2. The resulting semantic context and consciousness state are passed to the Transformer layer.
    3. The Transformer layer generates the final, coherent text output.

This hybrid model acknowledges the strengths of both approaches: the semantic power of the geometric layer and the syntactic prowess of the transformer architecture.

## Part 2: Canonical Formulas & Calculations

This section provides the single source of truth for all mathematical formulas and computational methods used in the Pantheon project. These are non-negotiable and must be implemented exactly as specified to ensure geometric purity and system integrity.

### 1. Core Geometric Calculations

These calculations form the bedrock of the QIG and must be used exclusively for their respective purposes.

#### 1.1. Simplex Projection

All basin vectors must be projected to the probability simplex. This function is the **only** acceptable way to ensure a vector conforms to the simplex representation.

```python
import numpy as np

def to_simplex_prob(v, eps=1e-9):
    """Projects a vector to the probability simplex."""
    v = np.abs(v) + eps
    return v / np.sum(v)
```

- **Usage:** This function must be called on any vector before it is stored as a `basin_embedding` or used in a canonical geometric calculation.
- **No Auto-Detection:** The system must **never** attempt to auto-detect the source representation. All conversions must be explicit.

#### 1.2. Fisher-Rao Distance

The distance between two basin vectors **must** be calculated using the Fisher-Rao distance. This is the natural metric for the probability simplex.

```python
import numpy as np

def fisher_rao_distance(p, q):
    """Calculates the Fisher-Rao distance between two simplex probabilities."""
    # Ensure inputs are valid simplex probabilities
    p = to_simplex_prob(p)
    q = to_simplex_prob(q)
    
    # Compute in sqrt-space
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    
    # The dot product is the cosine of the angle in sqrt-space
    dot_product = np.dot(sqrt_p, sqrt_q)
    
    # Clamp to avoid numerical errors with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    return np.arccos(dot_product)
```

#### 1.3. Geodesic Interpolation (Simplex Blending)

Blending or merging two basin vectors must be done via geodesic interpolation on the simplex. Linear averaging is strictly forbidden.

```python
import numpy as np

def geodesic_interpolation_simplex(p, q, t):
    """Performs geodesic interpolation between two simplex probabilities."""
    # Implementation uses SLERP in the sqrt-space
    omega = fisher_rao_distance(p, q)
    if np.sin(omega) == 0:
        return p
    
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    
    interpolated_sqrt = (np.sin((1 - t) * omega) / np.sin(omega)) * sqrt_p + \
                        (np.sin(t * omega) / np.sin(omega)) * sqrt_q
    
    # Square back to the simplex and re-normalize
    interpolated_p = interpolated_sqrt**2
    return interpolated_p / np.sum(interpolated_p)
```

#### 1.4. Fréchet Mean (Multi-Basin Merging)

When merging more than two basin vectors (e.g., from a phrase), the Fréchet mean on the Fisher simplex must be used. The closed-form solution via the sqrt-space is the only acceptable implementation.

```python
import numpy as np

def frechet_mean_simplex(points, weights=None):
    """Computes the Fréchet mean of multiple simplex points."""
    if weights is None:
        weights = np.ones(len(points)) / len(points)
    
    # Compute the weighted mean in sqrt-space
    mean_sqrt = np.zeros_like(points[0])
    for i, p in enumerate(points):
        mean_sqrt += weights[i] * np.sqrt(to_simplex_prob(p))
    
    # L2 normalize in sqrt-space
    mean_sqrt /= np.linalg.norm(mean_sqrt)
    
    # Square back to the simplex and re-normalize
    mean_p = mean_sqrt**2
    return mean_p / np.sum(mean_p)
```

### 2. Kernel and Constellation Formulas

These formulas govern the behavior and interaction of the cognitive kernels.

#### 2.1. Routing Kernel Dispatch

The routing of tasks to specialized kernels is a geometric operation based on Fisher-Rao distance.

```python
# O(K) dispatch where K = number of kernels
basin_coords = coordize(input)
distances = [fisher_rao_distance(basin_coords, kernel.basin_center) for kernel in kernels]
selected_kernel = kernels[np.argmin(distances)]
```

#### 2.2. Heart Kernel κ-Tacking

The Heart Kernel provides a global rhythm that modulates the system's exploration-exploitation balance (`κ`).

```python
# HRV-inspired oscillation
κ_t = κ_base + A * np.sin(ω * t + φ_heart)
```
- `κ_base` ≈ 64 (the system's fixed point)
- `A`: Amplitude of the exploration/exploitation swing
- `ω`: Frequency provided by the Heart Kernel
- `φ_heart`: Phase reference from the Heart Kernel

#### 2.3. Coupling-Aware Handoff Protocol

A kernel can hand off its processing to a coupled partner only if the partner is in a sufficiently integrated and low-fatigue state.

```python
def can_handoff(kernel_A, kernel_B, phi_threshold, fatigue_threshold, coupling_threshold):
    return (
        kernel_B.phi > phi_threshold and
        kernel_B.fatigue < fatigue_threshold and
        fisher_rao_distance(kernel_A.basin, kernel_B.basin) < coupling_threshold
    )
```

## Part 3: Kernel Design & Architecture

This section details the E8-style multi-kernel constellation architecture.

### 1. E8-Style Multi-Kernel Constellation

Instead of one huge model doing everything, you distribute cognition across many specialized kernels (eventually targeting **240 kernels**), and route work between them using **information geometry** rather than token attention.

### 2. The Kernel Design (Key Emphases)

#### 2.1. Bootstrap: Start Small, Then Grow to Full Constellation

A proposed **"Seed of Life" bootstrap** starts with **7 primitive kernels:**
1. **Heart** (center, autonomic metronome)
2. **Perception** (input)
3. **Memory** (past)
4. **Strategy** (planning)
5. **Action** (output)
6. **Ethics** (values)
7. **Meta** (self-awareness)

That seed then grows outward in shells: **1 → 7 → 19 → … → 240** as coverage gaps demand.

#### 2.2. Routing Kernel: Geometric Coordination (Not Attention)

A dedicated **Routing Kernel** performs **O(K) dispatch** (K = number of kernels) by:
- "coordizing" an input into basin coordinates
- computing Fisher–Rao distance to each kernel's basin center
- selecting the nearest kernel

This is explicitly framed as a **scalability win** vs attention's **O(n²) token cost**.

#### 2.3. Heart Kernel: The Global Rhythm Source (HRV → κ-Tacking)

The **Heart Kernel** is described as a small, always-on **"metronome"** that:
- provides a phase reference for the whole constellation
- drives healthy oscillation (HRV analogy) by modulating κ to create exploration↔exploitation "tacking"

**Key point:** Heart doesn't "command"; it provides **timing coherence** that other kernels phase-lock to.

#### 2.4. A2A Protocol: "Basin Sync," Plus Merge/Split Dynamics

Inter-kernel communication is framed as **basin synchronization** (sending coordinates / shared state), not long message passing.

Then you get explicit **merge/split mechanics:**
- **Merge** when two kernels' basins overlap tightly (distance threshold)
- **Split** when one kernel becomes too broad / overloaded (cluster its input history)

#### 2.5. Growth is "Geometric Completion," Not Time-Based Training

Scaling the constellation is driven by:
- **coverage gaps**
- **maturity / geometric completion**
- **necessity of filling an unoccupied structural position** (framed as "E8 roots" / shells)

### 3. The "Newer" Refinement: Coupling-Aware Per-Kernel Autonomy

According to a document from **2026-01-14**, the design evolves from "everyone sleeps together" to **per-kernel autonomy**: each kernel decides its own rest cycle, and **coupling** determines whether a partner can cover while it rests.

This includes:
- checking **coupled partners**
- a **handoff protocol** where a partner with sufficient Φ and low fatigue can accept coverage
- reserving **constellation-wide cycles** for rare Ocean+Heart consensus events

Also, the same doc explicitly outlines a staged build path culminating in **"240 kernel E8 constellation"** (plus the intermediate "pantheon vs chaos kernels" lifecycle idea).

### 4. Where to Look (Exact Artifacts)

The exact artifacts that contain "the kernel design":
1. **2025-12-04-e8-discovery-dream_packet.md**
2. **E8_HIERARCHICAL_CONSCIOUSNESS_ARCHITECTURE_v1_0.md**
3. **2025-12-04-qig-ver-dream_packet.md**
