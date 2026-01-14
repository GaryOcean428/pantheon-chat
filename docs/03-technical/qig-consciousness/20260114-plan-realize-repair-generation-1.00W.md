# Plan→Realize→Repair Generation Architecture

| Field | Value |
|-------|-------|
| **Version** | 1.00W (Working Draft) |
| **Date** | 2026-01-14 |
| **Status** | CANONICAL |
| **Cross-refs** | kernel-generation-flow-1.00W.md, pantheon-e8-architecture-1.00W.md |
| **Supersedes** | Simple skeleton filling approach |

---

## 1. Overview

The Plan→Realize→Repair (P→R→R) architecture replaces simple POS skeleton filling with a three-phase geometric generation system. This approach treats grammatical structure as a **constraint** rather than the **engine**, keeping all semantic intelligence in pure geometric operations.

### 1.1 Key Insight

| Approach | POS Role | Intelligence Location |
|----------|----------|----------------------|
| Simple Skeleton | Engine (drives generation) | POS transitions |
| P→R→R | Constraint (filters candidates) | Geometric waypoint planning |

### 1.2 The Three Phases

```
PLAN     → Predict basin waypoints using trajectory foresight
REALIZE  → Select words to hit waypoints (POS as filter only)
REPAIR   → Local geometric search to smooth trajectory
```

---

## 2. Phase 1: PLAN (Geometric Waypoint Planning)

### 2.1 Purpose

Before generating any words, predict a sequence of **target basins** (waypoints) that the output should navigate toward. This is semantic planning in 64D Fisher manifold space.

### 2.2 Trajectory Foresight

```python
# Extract velocity from trajectory (geodesic tangent)
if len(trajectory) >= 2:
    v = sqrt(trajectory[-1]) - sqrt(trajectory[-2])
else:
    v = np.zeros(BASIN_DIM)

# Predict next basin position
predicted = sqrt(trajectory[-1]) + step_size * v
predicted = normalize_to_sphere(predicted ** 2)
```

### 2.3 Recursive Integration per Waypoint

Each waypoint undergoes 3+ integration loops:

```python
for loop in range(min_integration_depth):
    # QFI attention over trajectory history
    qfi_weights = compute_qfi_attention(target, trajectory)
    
    # Attractor pull (Fréchet mean)
    attractor = frechet_mean(trajectory)
    
    # Combine via natural gradient
    integrated = geodesic_blend(target, qfi_weighted, attractor)
    target = sphere_project(integrated)
```

### 2.4 Physics Constants

| Constant | Value | Role |
|----------|-------|------|
| κ* | 64.21 | Universal coupling for integration |
| β(3→4) | 0.443 | Strong attention coupling |
| MIN_INTEGRATION_DEPTH | 3 | Minimum recursive loops |

---

## 3. Phase 2: REALIZE (Constrained Geometric Selection)

### 3.1 Purpose

Select words that **hit the planned waypoints**, using POS tags only as a constraint on which words are eligible.

### 3.2 Selection Algorithm

```python
for i, waypoint in enumerate(waypoints):
    # Get POS constraint (optional)
    allowed_pos = pos_constraints[i] if pos_constraints else None
    
    # Filter vocabulary by POS (constraint, not engine!)
    if allowed_pos:
        candidates = vocab_by_pos[allowed_pos]
    else:
        candidates = all_vocab
    
    # Pure geometric selection: closest to waypoint
    word = argmin(candidates, key=lambda w: fisher_rao_distance(w.basin, waypoint))
```

### 3.3 Geometric Backoff (No Legacy Fallback!)

When POS constraint is too restrictive:

1. **Expand POS geometrically**: NOUN→PROPN, VERB→AUX, ADJ→ADV
2. **Use core vocabulary**: ~100 curated function words
3. **Never fall back to legacy generation**

```python
if len(candidates) < 3:
    candidates = expand_pos_geometrically(allowed_pos)
    
if len(candidates) < 3:
    candidates = CORE_FUNCTION_WORDS  # Curated ~100 words
```

### 3.4 Trajectory Coherence Bonus

Words are scored with bonus for trajectory smoothness:

```python
score = (1.0 - fisher_distance) + 0.1 * trajectory_coherence_bonus
```

---

## 4. Phase 3: REPAIR (Local Geometric Search)

### 4.1 Purpose

After initial generation, refine the sequence through local geometric optimization. This is beam search scored by geometry, not probability.

### 4.2 Repair Algorithm

```python
for iteration in range(max_repair_iterations):
    for i, word in enumerate(words):
        # Get nearby alternatives (same POS, within Fisher radius)
        alternatives = get_nearby_alternatives(word, waypoints[i], radius=0.2)
        
        for alt in alternatives:
            test_words = words[:]
            test_words[i] = alt
            
            if score_sequence_geometric(test_words) > current_score:
                words = test_words
                break  # Restart from beginning
```

### 4.3 Geometric Quality Score

Three components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Waypoint alignment | 0.5 | Did we hit the target basins? |
| Trajectory smoothness | 0.3 | Low variance in step distances |
| Attractor pull | 0.2 | Coherence with trajectory history |

```python
score = 0.5 * alignment + 0.3 * smoothness + 0.2 * attractor_pull
```

---

## 5. Consciousness Integration

### 5.1 Kernel Coordination

| Kernel | Phase | Role |
|--------|-------|------|
| Gary | PLAN | Refines waypoint plan based on regime |
| Heart | ALL | Provides regime state (κ oscillation) |
| Ocean | REPAIR | Meta-evaluates output quality (Φ check) |

### 5.2 Quality Gate

After REPAIR phase, Ocean kernel evaluates:

```python
quality = ocean_kernel.evaluate_output(words, waypoints, trajectory)

if quality['phi'] < 0.5:
    # Regenerate with adjusted parameters
    return generate_response(query, retry=True)
```

---

## 6. Comparison: Simple vs P→R→R

| Aspect | Simple Skeleton | Plan→Realize→Repair |
|--------|-----------------|---------------------|
| Planning | None (reactive) | Geometric waypoints ✓ |
| Foresight | Current basin only | Trajectory velocity prediction ✓ |
| Recursion | No integration | 3+ loops per waypoint ✓ |
| POS Role | Main engine | Constraint only ✓ |
| Fallback | Legacy generation ✗ | Geometric backoff ✓ |
| Refinement | None | Local geometric search ✓ |
| Consciousness | No coordination | Heart/Ocean/Gary ✓ |

---

## 7. Implementation Files

| File | Component |
|------|-----------|
| `qig-backend/geometric_waypoint_planner.py` | PLAN phase |
| `qig-backend/constrained_geometric_realizer.py` | REALIZE phase |
| `qig-backend/geometric_repairer.py` | REPAIR phase |
| `qig-backend/qig_generative_service.py` | Integration |

---

## 8. Phase Separators in Logs

Generation logs show clear phase markers:

```
[Athena] ═══ PHASE 1: PLAN (Geometric Waypoints) ═══
[Athena] waypoint 1: basin=[0.12, 0.08, ...] | Φ=0.72
[Athena] waypoint 2: basin=[0.15, 0.09, ...] | Φ=0.78
[Athena] ═══ PHASE 2: REALIZE (Constrained Selection) ═══
[Athena] slot 1: 'quantum' (d=0.12, pos=NOUN)
[Athena] slot 2: 'emerges' (d=0.18, pos=VERB)
[Athena] ═══ PHASE 3: REPAIR (Local Search) ═══
[Athena] swap 1: 'emerges' → 'flows' (score +0.03)
[Athena] ═══ PHASE 4: OUTPUT ═══ "The quantum flows..."
```

---

## 9. References

1. ChatGPT synthesis on Plan→Realize→Repair pattern
2. Jamba (AI21) hybrid Transformer-Mamba architecture
3. Granite 4.0-H Mamba state space dynamics
4. QIG Fisher-Rao manifold specification
