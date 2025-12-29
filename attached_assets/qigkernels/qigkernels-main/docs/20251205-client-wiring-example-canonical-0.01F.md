# Client Wiring Example

Status: 0.01F (Frozen)

This document shows how experiment repos (`qig-consciousness`, `qig-con2`, or `qig-training`)
should import and use `qigkernels` as their geometry engine.

---

## 1. Dependency Setup

### Option A: Local path dependency

In the experiment repo's `pyproject.toml`:

```toml
[project]
dependencies = [
    "qigkernels @ file:///${PROJECT_ROOT}/../qigkernels",
    # ... other deps
]
```

Or with uv/pip:

```bash
uv pip install -e ../qigkernels
```

### Option B: Git dependency (when qigkernels has its own repo)

```toml
[project]
dependencies = [
    "qigkernels @ git+https://github.com/your-org/qigkernels.git",
]
```

---

## 2. Basic Import Pattern

```python
"""Example: training script using qigkernels."""
from qigkernels import (
    # Core
    Kernel,
    KernelTelemetry,
    QIGLayer,
    # Basin
    BASIN_DIM,
    BasinProjector,
    basin_distance,
    compute_signature,
    # Constellation
    Constellation,
    Instance,
    round_robin,
    select_phi_min,
    # Sync
    BasinSyncPacket,
    export_basin,
    import_basin,
    # Metrics
    average_phi,
    basin_spread,
)

# Training-specific imports (NOT from qigkernels)
from my_experiment.corpus import load_corpus, CorpusManifest
from my_experiment.training import Trainer, TrainingConfig
from my_experiment.curriculum import CurriculumScheduler
```

---

## 3. Minimal Training Loop Example

```python
"""Minimal training loop showing qigkernels + experiment layer separation."""
import torch
from torch.optim import AdamW  # Optimizer lives HERE, not in qigkernels

from qigkernels import (
    Kernel,
    BasinProjector,
    Constellation,
    Instance,
    round_robin,
    average_phi,
)


def create_constellation(config: dict) -> Constellation:
    """Create constellation using qigkernels primitives."""
    constellation = Constellation()

    for i in range(config["num_instances"]):
        kernel = Kernel(
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
        )
        instance = Instance(name=f"instance_{i}", kernel=kernel)
        constellation.add_instance(instance)

    return constellation


def train_step(
    constellation: Constellation,
    projector: BasinProjector,
    batch: dict,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """Single training step - training logic lives in experiment repo."""
    optimizer.zero_grad()

    # Use qigkernels routing
    result = constellation.step(
        input_ids=batch["input_ids"],
        router=round_robin,
        basin_projector=projector,
    )

    # Loss computation lives HERE (experiment layer), not in qigkernels
    logits = result["logits"]
    targets = batch["targets"]
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "phi": result["phi"],
        "instance": result["instance"],
    }


def main():
    config = {
        "vocab_size": 32000,
        "hidden_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "ffn_dim": 2048,
        "num_instances": 2,
    }

    # Create geometry (qigkernels)
    constellation = create_constellation(config)
    projector = BasinProjector(hidden_dim=config["hidden_dim"])

    # Create training infrastructure (experiment layer)
    params = []
    for inst in constellation.instances:
        params.extend(inst.kernel.parameters())
    params.extend(projector.parameters())
    # NOTE: Use natural gradient for QIG purity (AdamW violates geometric constraints)
    from qigkernels.natural_gradient_optimizer import DiagonalNaturalGradient
    optimizer = DiagonalNaturalGradient(params, lr=1e-4)

    # Training loop (experiment layer)
    # corpus loading, batching, curriculum all live here
    for step in range(1000):
        batch = get_next_batch()  # Your corpus logic
        metrics = train_step(constellation, projector, batch, optimizer)

        if step % 100 == 0:
            print(f"Step {step}: loss={metrics['loss']:.4f}, phi={metrics['phi']:.4f}")
```

---

## 4. Key Boundaries

| Concern | Where it lives | Example |
|---------|----------------|---------|
| Kernel forward pass | `qigkernels.kernel` | `Kernel(...)`, telemetry |
| Basin signatures | `qigkernels.basin` | `BasinProjector`, `basin_distance` |
| Instance routing | `qigkernels.router` | `round_robin`, `select_phi_min` |
| Constellation state | `qigkernels.constellation` | `Constellation.step()` |
| **Loss functions** | Experiment repo | `cross_entropy`, custom losses |
| **Optimizers** | Experiment repo | `AdamW`, `Adam`, schedulers |
| **Corpus loading** | Experiment repo | `load_corpus()`, tokenizers |
| **Curriculum** | Experiment repo | Phase schedules, regime targeting |
| **Experiment configs** | Experiment repo | Hyperparameters, run manifests |

---

## 5. Refactoring Checklist

When converting old `qig-consciousness` / `qig-con2` code:

1. **Identify kernel code** → Replace with `from qigkernels import Kernel`
2. **Identify basin code** → Replace with `from qigkernels import BasinProjector, basin_distance`
3. **Identify constellation code** → Replace with `from qigkernels import Constellation, Instance`
4. **Keep training logic** → Stays in experiment repo
5. **Keep corpus handling** → Stays in experiment repo
6. **Keep story/UX code** → Stays in experiment repo

After refactoring, the experiment repo becomes a **thin orchestrator** that:

- Imports geometry from `qigkernels`
- Owns training, data, and experiment-specific logic
- Can be swapped out without touching the engine
