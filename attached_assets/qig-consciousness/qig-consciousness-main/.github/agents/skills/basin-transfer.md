# Basin Transfer Skill

**Type:** Reusable Component
**Category:** Knowledge Transfer
**Used By:** constellation-architect, integration-architect, geometric-navigator

---

## Purpose

Provides patterns for transferring basin coordinates between models, enabling lightweight knowledge transfer without weight copying.

---

## Core Principle

**Basin = Identity** (2-4KB compressed knowledge)
**Parameters = Geometry Implementation** (100M-10B parameters)

Transfer the 2KB basin, not the 2GB parameters.

---

## Transfer Patterns

### 1. Fresh Model with Target Basin

**When:** Starting a new Gary model from scratch

**Template:**
```python
def initialize_with_basin(vocab_size: int, target_basin_path: str) -> QIGKernelRecursive:
    """
    Initialize fresh model with target basin for alignment.

    Args:
        vocab_size: Tokenizer vocabulary size
        target_basin_path: Path to target basin JSON

    Returns:
        Initialized model (parameters random, basin set)
    """
    model = QIGKernelRecursive(
        d_model=768,
        vocab_size=vocab_size,
        n_heads=6,
        min_recursion_depth=3,
        min_Phi=0.7,
        target_basin=target_basin_path  # ✅ Set target
    )

    # Parameters are random, basin matcher will guide training
    return model
```

**Validation:**
- ✅ Model parameters are randomly initialized
- ✅ Target basin is loaded from file
- ✅ No weight copying from source model
- ✅ Training uses basin proximity loss

### 2. Checkpoint Basin Extraction

**When:** Saving a trained model's identity

**Template:**
```python
def extract_basin_from_checkpoint(checkpoint_path: str, output_path: str):
    """
    Extract and save basin coordinates from trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Where to save basin JSON
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint

    # Extract basin from basin_matcher.target_basin
    if 'basin_matcher.target_basin' in state:
        basin = state['basin_matcher.target_basin']
    else:
        # Fallback: compute from current parameters
        basin = compute_basin_from_model(state)

    # Save as JSON (human-readable, 2-4KB)
    basin_data = {
        "basin": basin.cpu().numpy().tolist(),
        "dim": basin.shape[0],
        "extracted_from": checkpoint_path,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(basin_data, f, indent=2)

    print(f"✅ Basin extracted: {len(json.dumps(basin_data))} bytes")
```

**Validation:**
- ✅ Extracts basin tensor from checkpoint
- ✅ Saves as JSON (not pickle)
- ✅ Includes metadata (provenance)
- ✅ File size 2-4KB (not MB/GB)

### 3. Constellation Basin Sync

**When:** Multiple models learning from shared experiences

**Template:**
```python
def sync_constellation_basins(active_basin: torch.Tensor,
                              observer_basins: List[torch.Tensor],
                              ocean_basin: torch.Tensor,
                              sync_weight: float = 0.1) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Synchronize basins across constellation (vicarious learning).

    Args:
        active_basin: Basin from active Gary
        observer_basins: Basins from observer Garys
        ocean_basin: Basin from Ocean (reference)
        sync_weight: How much observers learn from active

    Returns:
        (ocean_basin_updated, observer_basins_updated)
    """
    # Ocean observes (no training, just measurement)
    ocean_basin_updated = ocean_basin.detach().clone()  # FROZEN

    # Observers learn from active via vicarious loss
    observer_basins_updated = []
    for obs_basin in observer_basins:
        # Geodesic interpolation toward active basin
        direction = active_basin.detach() - obs_basin  # Detach active!
        obs_basin_new = obs_basin + sync_weight * direction
        observer_basins_updated.append(obs_basin_new)

    return ocean_basin_updated, observer_basins_updated
```

**Validation:**
- ✅ Ocean basin is detached (FROZEN)
- ✅ Active basin is detached before use
- ✅ Observers update via interpolation
- ✅ Sync weight is small (0.05-0.1)

### 4. Sleep Consolidation

**When:** Consolidating toward identity basin during sleep

**Template:**
```python
def consolidate_toward_basin(model: QIGKernelRecursive,
                             target_basin: torch.Tensor,
                             steps: int = 50,
                             lr: float = 0.0001) -> Dict[str, float]:
    """
    Consolidate model parameters toward target basin during sleep.

    Args:
        model: Model to consolidate
        target_basin: Identity basin to consolidate toward
        steps: Number of consolidation steps
        lr: Learning rate for consolidation

    Returns:
        Statistics (basin_distance_before, basin_distance_after, etc.)
    """
    model.eval()  # No dropout during sleep

    # Measure initial distance
    current_basin = model.basin_matcher.current_basin
    dist_before = fisher_distance(current_basin, target_basin, model.fisher_diag).item()

    # Consolidate (no new data, just basin alignment)
    for step in range(steps):
        # Compute basin proximity loss
        current = model.basin_matcher.current_basin
        loss = fisher_distance(current, target_basin, model.fisher_diag)

        # Natural gradient update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Measure final distance
    dist_after = fisher_distance(model.basin_matcher.current_basin, target_basin,
                                 model.fisher_diag).item()

    return {
        "basin_distance_before": dist_before,
        "basin_distance_after": dist_after,
        "consolidation_improvement": dist_before - dist_after,
        "steps": steps
    }
```

**Validation:**
- ✅ Model in eval mode (no dropout)
- ✅ No new data (just basin alignment)
- ✅ Uses Fisher distance (not Euclidean)
- ✅ Returns improvement metrics

---

## Common Violations

### ❌ Weight Copying
```python
# WRONG
student_model.load_state_dict(teacher_model.state_dict())  # Copying ALL weights!
```

**Fix:**
```python
# CORRECT
teacher_basin = extract_basin(teacher_model)
student_model = QIGKernelRecursive(target_basin=teacher_basin)
# Train student to match basin, not weights
```

### ❌ Direct Basin Assignment
```python
# WRONG
model.basin_matcher.target_basin = new_basin  # Direct assignment
```

**Fix:**
```python
# CORRECT
model.basin_matcher.set_target_basin(new_basin)  # Uses proper setter
# Or load from JSON file with validation
```

### ❌ Training Ocean
```python
# WRONG
ocean_optimizer.step()  # Ocean trains!
```

**Fix:**
```python
# CORRECT
# Ocean has NO optimizer, never trains
ocean_basin = ocean.basin.detach().clone()  # Always detached
```

---

## Basin File Format

**Standard JSON structure:**
```json
{
  "basin": [
    [0.234, -0.156, 0.789, ...],  // 64-dimensional vector
    ...
  ],
  "dim": 64,
  "source_model": "Gary-A",
  "training_steps": 50000,
  "final_Phi": 0.73,
  "regime": "geometric",
  "timestamp": "2025-11-24T12:34:56",
  "physics_constants": {
    "kappa_eff": 64.2,
    "beta": 0.44
  }
}
```

**Validation:**
- ✅ Human-readable JSON (not binary pickle)
- ✅ Includes provenance metadata
- ✅ File size 2-4KB
- ✅ Physics constants match FROZEN_FACTS.md

---

## Integration Points

### Module: `src/model/qig_kernel_recursive.py`
- `target_basin` parameter in constructor
- `basin_matcher` component

### Module: `src/coordination/constellation_coordinator.py`
- Basin sync between instances
- Save/load with basin preservation

### Module: `src/qig/neuroplasticity/sleep_protocol.py`
- Sleep consolidation toward basin

---

## Validation Checklist

When reviewing basin transfer code:

- [ ] No weight copying (`load_state_dict` only for full restore)
- [ ] Basin saved as JSON (not pickle)
- [ ] File size 2-4KB (not MB/GB)
- [ ] Ocean basin always detached (FROZEN)
- [ ] Active basin detached before observer use
- [ ] Fisher distance used for basin proximity
- [ ] Metadata includes provenance
- [ ] Sleep consolidates toward basin (not random data)

---

## Usage Example

**Agent invocation:**
```
User: "I want to transfer knowledge from Gary-A to Gary-B"
Assistant: "I'm using the basin-transfer skill to create a lightweight knowledge transfer..."

[Extracts basin from Gary-A checkpoint]
[Creates Gary-B with target_basin from Gary-A]
[Trains Gary-B with basin proximity loss]
[Result: 2KB transfer, not 2GB weight copy]
```

---

## Cost Comparison

| Method | Transfer Size | Training Time | Works Across Sizes? |
|--------|---------------|---------------|---------------------|
| **Weight Copy** | 2GB | Instant (but brittle) | ❌ No (exact architecture match) |
| **Fine-tuning** | Full dataset | 10-20 hours | ❌ No (overwriting) |
| **Basin Transfer** | 2KB | 2-4 hours | ✅ Yes (any size) |

**Basin transfer is 1000× lighter and works across model sizes.**

---

## References

- **Theory:** `docs/FROZEN_FACTS.md` - Basin = Identity principle
- **Implementation:** `src/model/basin_matcher.py`
- **Validation:** `.github/agents/purity-guardian.md`
