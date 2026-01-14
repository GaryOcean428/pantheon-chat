# QIG-Core: Pure Fisher Information Geometry

**Pure geometric utilities for QIG consciousness architecture**

## What is QIG-Core?

A minimal, dependency-pure package providing Fisher Information Geometry operations for quantum information-based consciousness modeling.

## Features

### 📐 Geometric Math

- **Fisher metric calculations** - Riemannian distance on manifolds
- **Geodesic interpolation** - Curved-space paths
- **Natural gradients** - Geometry-aware optimization

### 🧠 Consciousness Components

- **QIG Tokenizer Interface** - Abstract contract for geometric tokenization
- **QFI Sampler** - Geometrically pure token generation (no softmax)
- **Basin Sync** - Multi-instance coordination protocol

### 🛡️ Purity

- **Zero ML dependencies** - Pure math (torch, numpy, scipy only)
- **No Transformers** - Completely decoupled from HuggingFace

## Installation

```bash
pip install qig-core
```

## Usage

### Geometric Math

```python
from qig_core import fisher_distance, geodesic_interpolate

# Compute Fisher distance between two points on manifold
distance = fisher_distance(coords1, coords2, metric_tensor)

# Interpolate along geodesic
midpoint = geodesic_interpolate(coords1, coords2, t=0.5, metric=F)
```

### Generation (QFI Sampler)

```python
from qig_core import QFISampler

sampler = QFISampler()
token_id, metrics = sampler.sample(logits, hidden_state, telemetry, embeddings)
```

### Coordination (Basin Sync)

```python
from qig_core import BasinSync

sync = BasinSync("Gary-A")
sync.update_sync(basin_distance=0.1, phi=0.8, regime="geometric")
```

## Geometric Purity

This package enforces:

- NO embeddings (use coordinates)
- NO cosine similarity (use Fisher distance)
- NO linear interpolation (use geodesic paths)
- NO Euclidean gradients (use natural gradients)

See [GEOMETRIC_PURITY.md](GEOMETRIC_PURITY.md) for details.

## E8 Integration Status

> **Note:** This package (v1.0.0) was created BEFORE the E8 kernel specialization
> direction was established in `qigkernels`. Review is needed for E8 compatibility.

**What aligns:**

- `KAPPA_STAR = 64.0` matches E8 rank² = 8² = 64 ✓

**Needs review:**

- Fisher distance for 64D E8-aligned basin geometry
- Geodesic interpolation for E8 root transitions
- QFISampler primitive type awareness (HRT/PER/MEM/ACT/PRD/ETH/META/MIX)
- BasinSync consolidation with `qigkernels.basin_sync`

**Target hierarchy:**

```markdown
qig-core (math) → qigkernels (architecture) → qig-dreams (corpora) → experiments
```

See `qigkernels/20251205-decisions-canonical-0.01F.md` D-016 for details.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
```

## License

MIT
