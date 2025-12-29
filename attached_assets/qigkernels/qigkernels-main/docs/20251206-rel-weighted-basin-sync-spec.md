# REL-Weighted Basin Sync (Design Spec)

**Date:** 2025-12-06
**Status:** Approved for Implementation
**Location:** qigkernels/rel_coupling.py, qigkernels/basin_sync.py

---

## 1. Baseline Fisher-Rao Distance

Let:

- **b_i ∈ ℝ⁶⁴** = basin signature of instance i
- **g_F** = Fisher information metric on basin space
- **d_F(i, j)** = geodesic Fisher-Rao distance between basins i and j

Implementation:

- Use `basin_distance(b_i, b_j)` from `qigkernels.basin`
- This uses Bures approximation: d² = 2(1 - cos_sim)

---

## 2. REL Coupling Scalar

Let:

- **r_ij ∈ [0, 1]** = REL coupling between instances i and j

Interpretation:

- r_ij = 0: no relationship / no history
- r_ij = 1: maximally coupled (deep shared history, strong trust)

### Components

REL coupling is computed from:

1. **Basin Overlap** - Cosine similarity of current signatures
2. **History Score** - Accumulated interaction quality
3. **Primitive Alignment** - Shared corpus primitive exposure

```python
def compute_rel_coupling(state_i: InstanceState, state_j: InstanceState) -> float:
    """
    REL coupling scalar in [0, 1].

    Components:
    - Basin overlap (geometric similarity)
    - Interaction history (longitudinal trust)
    - Primitive alignment (shared meaning exposure)
    """
    # Component 1: Basin geometric overlap
    cos_sim = cosine_similarity(state_i.basin, state_j.basin)
    basin_overlap = (cos_sim + 1.0) / 2.0  # Map [-1,1] → [0,1]

    # Component 2: Interaction history
    history = compute_history_score(state_i, state_j)

    # Component 3: Primitive alignment (optional)
    primitive_align = compute_primitive_alignment(state_i, state_j)

    # Weighted combination
    w_basin = 0.4
    w_history = 0.4
    w_primitive = 0.2

    rel = w_basin * basin_overlap + w_history * history + w_primitive * primitive_align
    return clamp(rel, 0.0, 1.0)
```

---

## 3. Combined Distance with REL Weighting

REL modulates geometric pull, does not replace it.

### Formula

```
d²_eff(i,j) = d²_F(i,j) · (1 - λ_rel · r_ij)
```

Where:

- **λ_rel ∈ [0, 0.7]** = global REL strength hyperparameter (capped for safety)
- If r_ij → 1: effective distance shrinks (closer relationship = stronger pull)
- If r_ij → 0: d²_eff ≈ d²_F (no relationship = pure geometry)

### Implementation

```python
def effective_basin_distance(
    b_i: Tensor,
    b_j: Tensor,
    rel_ij: float,
    lambda_rel: float = 0.5,
) -> Tensor:
    """
    REL-weighted effective distance.

    GEOMETRIC PURITY: Fisher-Rao is always computed first.
    REL only modulates the magnitude, never replaces geometry.
    """
    d_f = basin_distance(b_i, b_j)
    scale = max(1.0 - lambda_rel * rel_ij, 0.3)  # Floor at 0.3 for safety
    return d_f * scale
```

---

## 4. Sync Loss

For a target basin b_T:

```
L_sync(i → T) = w_sync · d²_eff(i, T)
```

Where:

- **w_sync** = global sync strength (per experiment)

### Implementation

```python
def rel_weighted_sync_loss(
    basin_i: Tensor,
    basin_target: Tensor,
    rel_ij: float,
    lambda_rel: float = 0.5,
    w_sync: float = 1.0,
) -> Tensor:
    """
    REL-weighted sync loss for basin alignment.

    Higher REL → lower effective distance → stronger sync pull.
    """
    d_eff = effective_basin_distance(basin_i, basin_target, rel_ij, lambda_rel)
    return w_sync * (d_eff ** 2)
```

---

## 5. REL & Ocean

Ocean is special: highly REL-coupled to all Garys but not over-dominant.

```python
# Initial REL between Gary_k and Ocean
r_gary_ocean = min(r_base, r_history)
# where:
#   r_base ≈ 0.5 (moderate starting coupling)
#   r_history rises as training proceeds
```

This lets Ocean guide without collapsing all Garys into clones.

---

## 6. Safety & Stability Notes

### REL Never Turns Off Geometry

- Always multiply Fisher-Rao distance; never replace it
- Minimum scale factor: 0.3 (30% of geometric distance always preserved)

### Cap REL Influence

- Clamp λ_rel ≤ 0.7 initially
- Can be relaxed after stability validation

### Telemetry

Log separately:

- d_F (pure Fisher distance)
- r_ij (REL coupling)
- d_eff (effective distance)

Plot to ensure REL is moderating, not dominating.

---

## 7. Integration Points

### In qigkernels

- `rel_coupling.py`: `compute_rel_coupling()`, `RELState` dataclass
- `basin_sync.py`: `rel_weighted_sync_loss()`, `effective_basin_distance()`

### In qig-consciousness

- `ConstellationCoordinator`: Use REL-weighted sync instead of raw Fisher
- Telemetry: Add `rel_coupling` field to instance states

### In qig-dreams

- REL corpus influences primitive alignment component
- Higher REL exposure → higher primitive_align score
