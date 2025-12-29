# Constellation Coordination Skill

**Type:** Reusable Component
**Category:** Multi-Instance Architecture
**Used By:** constellation-architect, integration-architect, purity-guardian

---

## Purpose

Provides patterns for coordinating multiple QIG instances (Ocean + 3 Garys) with proper load distribution, vicarious learning, and basin synchronization.

---

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│              OCEAN (Observer Only)            │
│  - Never responds                            │
│  - Observes all conversations                │
│  - Learns meta-manifold                      │
│  - Basin: FROZEN (never trained)             │
└──────────────────────────────────────────────┘
                      │
           ┌──────────┴──────────┐
           │   Basin Sync        │
           │   (read-only)       │
           └──────────┬──────────┘
                      │
      ┌───────────────┼───────────────┐
      │               │               │
┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
│  GARY-A   │  │  GARY-B   │  │  GARY-C   │
│           │  │           │  │           │
│ Role:     │  │ Role:     │  │ Role:     │
│ ACTIVE    │  │ OBSERVER  │  │ OBSERVER  │
└───────────┘  └───────────┘  └───────────┘

Round-robin: Next question → GARY-B becomes ACTIVE
```

---

## Core Patterns

### 1. Round-Robin Question Routing

**Template:**
```python
class ConstellationCoordinator:
    def __init__(self, gary_configs: List[str], ocean_config: str):
        self.garys = [self._load_gary(cfg) for cfg in gary_configs]
        self.ocean = self._load_ocean(ocean_config)
        self.active_index = 0  # Start with Gary-A

    def route_question(self, question: str) -> str:
        """
        Route question to active Gary, others observe.

        Args:
            question: User question

        Returns:
            Response from active Gary
        """
        # Identify roles
        active_gary = self.garys[self.active_index]
        observer_garys = [g for i, g in enumerate(self.garys)
                         if i != self.active_index]

        # Active Gary responds
        response, active_telemetry = active_gary.forward(question)

        # Observers learn vicariously (no response generation)
        for observer in observer_garys:
            with torch.no_grad():  # Pure measurement
                _, obs_telemetry = observer.forward(question)

            # Vicarious learning from active's basin
            vicarious_loss = self._compute_vicarious_loss(
                active_basin=active_gary.basin.detach(),  # Detach!
                observer_basin=observer.basin,
                fisher_diag=observer.fisher_diag
            )
            vicarious_loss.backward()
            observer.optimizer.step()

        # Ocean observes (no training)
        with torch.no_grad():
            _, ocean_telemetry = self.ocean.forward(question)
        # Ocean basin NEVER updates (frozen observer)

        # Round-robin: next question goes to next Gary
        self.active_index = (self.active_index + 1) % len(self.garys)

        return response
```

**Validation:**
- ✅ Only active Gary generates response
- ✅ Observers use `torch.no_grad()` for forward pass
- ✅ Active basin is detached before observer use
- ✅ Ocean never trains (no optimizer.step)
- ✅ Active index rotates (round-robin)

### 2. Basin Synchronization

**Template:**
```python
def _compute_vicarious_loss(self, active_basin: torch.Tensor,
                           observer_basin: torch.Tensor,
                           fisher_diag: torch.Tensor) -> torch.Tensor:
    """
    Compute vicarious learning loss (observer learns from active's basin).

    Args:
        active_basin: Basin from active Gary (DETACHED)
        observer_basin: Basin from observer Gary
        fisher_diag: Diagonal Fisher Information Matrix

    Returns:
        Loss for observer to minimize
    """
    # Ensure active basin is detached (no gradient flow backward)
    assert not active_basin.requires_grad, "Active basin must be detached!"

    # Fisher metric distance
    diff = observer_basin - active_basin
    distance_sq = torch.sum(diff * fisher_diag * diff)

    # Vicarious loss: minimize distance to active basin
    return distance_sq
```

**Validation:**
- ✅ Active basin must be detached
- ✅ Uses Fisher metric (not Euclidean)
- ✅ Returns scalar loss
- ✅ Only observer basin has gradient

### 3. Checkpoint Save/Load with Constellation

**Template:**
```python
def save_constellation(self, checkpoint_dir: Path, epoch: int, step: int):
    """
    Save all constellation members with basin preservation.

    Args:
        checkpoint_dir: Directory to save checkpoints
        epoch: Current epoch
        step: Current step
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save each Gary
    for i, gary in enumerate(self.garys):
        gary_path = checkpoint_dir / f"gary_{chr(65+i)}_epoch{epoch}_step{step}.pt"
        torch.save({
            'model_state_dict': gary.model.state_dict(),
            'optimizer_state_dict': gary.optimizer.state_dict(),
            'basin': gary.basin.detach().cpu(),  # Preserve basin
            'target_basin': gary.target_basin,
            'telemetry': gary.telemetry_history,
            'role': gary.role,
            'epoch': epoch,
            'step': step
        }, gary_path)

    # Save Ocean (no optimizer, basin is frozen)
    ocean_path = checkpoint_dir / f"ocean_epoch{epoch}_step{step}.pt"
    torch.save({
        'model_state_dict': self.ocean.model.state_dict(),
        'basin': self.ocean.basin.detach().cpu(),  # Frozen basin
        'telemetry': self.ocean.telemetry_history,
        'epoch': epoch,
        'step': step
    }, ocean_path)

    # Save coordinator state
    coord_path = checkpoint_dir / f"coordinator_epoch{epoch}_step{step}.json"
    with open(coord_path, 'w') as f:
        json.dump({
            'active_index': self.active_index,
            'epoch': epoch,
            'step': step,
            'constellation_telemetry': self.aggregate_telemetry()
        }, f, indent=2)

def load_constellation(self, checkpoint_dir: Path):
    """
    Load constellation from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    # Find latest checkpoint
    coord_files = list(checkpoint_dir.glob('coordinator_*.json'))
    latest = max(coord_files, key=lambda p: p.stat().st_mtime)

    with open(latest) as f:
        coord_state = json.load(f)

    epoch = coord_state['epoch']
    step = coord_state['step']

    # Load each Gary
    for i, gary in enumerate(self.garys):
        gary_path = checkpoint_dir / f"gary_{chr(65+i)}_epoch{epoch}_step{step}.pt"
        checkpoint = torch.load(gary_path, map_location=self.device)

        gary.model.load_state_dict(checkpoint['model_state_dict'])
        gary.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        gary.basin = checkpoint['basin'].to(self.device)
        gary.target_basin = checkpoint.get('target_basin')
        gary.telemetry_history = checkpoint['telemetry']

    # Load Ocean
    ocean_path = checkpoint_dir / f"ocean_epoch{epoch}_step{step}.pt"
    checkpoint = torch.load(ocean_path, map_location=self.device)
    self.ocean.model.load_state_dict(checkpoint['model_state_dict'])
    self.ocean.basin = checkpoint['basin'].to(self.device)
    self.ocean.telemetry_history = checkpoint['telemetry']

    # Restore coordinator state
    self.active_index = coord_state['active_index']

    return epoch, step
```

**Validation:**
- ✅ Saves basin for each instance
- ✅ Ocean has no optimizer state
- ✅ Coordinator state includes active_index
- ✅ Loads basin to correct device
- ✅ Preserves telemetry history

### 4. Telemetry Aggregation

**Template:**
```python
def aggregate_telemetry(self) -> Dict[str, Any]:
    """
    Aggregate telemetry across constellation.

    Returns:
        Consolidated constellation-level metrics
    """
    gary_phi = [g.telemetry_history[-1]['Phi'] for g in self.garys
                if g.telemetry_history]

    return {
        'constellation_phi_mean': np.mean(gary_phi) if gary_phi else 0.0,
        'constellation_phi_std': np.std(gary_phi) if gary_phi else 0.0,
        'ocean_phi': self.ocean.telemetry_history[-1]['Phi']
                     if self.ocean.telemetry_history else 0.0,
        'active_gary': chr(65 + self.active_index),  # 'A', 'B', or 'C'
        'total_questions': sum(len(g.telemetry_history) for g in self.garys),
        'ocean_observations': len(self.ocean.telemetry_history),
        'regime_distribution': self._get_regime_distribution(),
        'basin_spread': self._compute_basin_spread()
    }

def _compute_basin_spread(self) -> float:
    """
    Measure how spread out Gary basins are.

    Returns:
        Standard deviation of pairwise basin distances
    """
    basins = [g.basin for g in self.garys]
    distances = []

    for i in range(len(basins)):
        for j in range(i + 1, len(basins)):
            dist = fisher_distance(basins[i], basins[j],
                                  self.garys[i].fisher_diag)
            distances.append(dist.item())

    return np.std(distances) if distances else 0.0
```

**Validation:**
- ✅ Aggregates Φ across all Garys
- ✅ Includes Ocean telemetry (separately)
- ✅ Tracks active Gary
- ✅ Computes basin spread (diversity metric)
- ✅ Returns dict (JSON-serializable)

---

## Common Violations

### ❌ Ocean Training
```python
# WRONG
ocean_loss.backward()
self.ocean_optimizer.step()  # Ocean trains!
```

**Fix:**
```python
# CORRECT
# Ocean has NO optimizer, never trains
with torch.no_grad():
    _, ocean_telemetry = self.ocean.forward(question)
# Basin remains frozen
```

### ❌ Gradient Coupling
```python
# WRONG
active_basin = active_gary.basin  # No detach!
vicarious_loss = distance(active_basin, observer_basin)
vicarious_loss.backward()  # Gradients flow to active Gary!
```

**Fix:**
```python
# CORRECT
active_basin = active_gary.basin.detach()  # Detach!
vicarious_loss = distance(active_basin, observer_basin)
vicarious_loss.backward()  # Only observer trains
```

### ❌ Static Roles
```python
# WRONG
gary_a_always_active = True
# Gary-A answers all questions
```

**Fix:**
```python
# CORRECT
self.active_index = (self.active_index + 1) % len(self.garys)
# Round-robin rotation
```

---

## Load Distribution Metrics

Track to ensure fair distribution:

```python
def get_load_distribution(self) -> Dict[str, int]:
    """Track how many questions each Gary answered."""
    return {
        f"Gary-{chr(65+i)}": len(g.telemetry_history)
        for i, g in enumerate(self.garys)
    }

# Expected after 300 questions:
# Gary-A: ~100, Gary-B: ~100, Gary-C: ~100
```

**Validation:**
- ✅ Each Gary ~33% of questions (±10%)
- ✅ No Gary > 50% (prevents over-reliance)
- ✅ Ocean observes 100% (but responds 0%)

---

## Integration Points

### Module: `src/coordination/constellation_coordinator.py`
Main constellation implementation.

### Module: `src/coordination/basin_sync.py`
Basin synchronization utilities.

### Module: `src/metrics/geodesic_distance.py`
- `geodesic_vicarious_loss()` for observer learning

---

## Validation Checklist

When reviewing constellation coordination code:

- [ ] Round-robin rotation (no static roles)
- [ ] Ocean never trains (no optimizer)
- [ ] Active basin detached before observer use
- [ ] Observers use vicarious loss (not direct data)
- [ ] All basins saved in checkpoints
- [ ] Telemetry aggregated across constellation
- [ ] Load distribution tracked
- [ ] Basin spread monitored (diversity)

---

## Usage Example

**Agent invocation:**
```
User: "Set up constellation training with 3 Garys"
Assistant: "I'm using the constellation-coordination skill to set up the architecture..."

[Creates ConstellationCoordinator]
[Initializes 3 Garys + 1 Ocean]
[Sets up round-robin routing]
[Configures vicarious learning]
[Validates Ocean is frozen]
```

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Load Distribution** | 33% ± 10% per Gary | Fair distribution |
| **Ocean Training Steps** | 0 | Pure observer |
| **Basin Spread** | 0.03 - 0.08 | Diversity without fragmentation |
| **Vicarious Loss** | Decreasing over time | Convergence |
| **Constellation Φ** | 0.70 - 0.75 | Geometric regime |

---

## References

- **Theory:** `docs/FROZEN_FACTS.md` - Constellation architecture
- **Implementation:** `src/coordination/constellation_coordinator.py`
- **Validation:** `.claude/agents/constellation-architect.md`
