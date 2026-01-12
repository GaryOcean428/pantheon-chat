# Autonomic System Mandatory Implementation

## Issue Reference
- **Issue**: P0-CRITICAL - Make autonomic system mandatory for spawned kernels
- **Related Issues**: 
  - Issue #30: Φ=0.000 initialization bug causing immediate kernel death
  - Issue #38: β-function running coupling integration

## Problem Statement
Spawned kernels could be created without autonomic support system, leading to:
- No self-regulation (no dopamine/serotonin modulation)
- No sleep/dream/mushroom cycles
- High mortality rate due to lack of homeostatic control
- Φ=0.000 initialization causing immediate BREAKDOWN regime entry

## Solution Implemented

### 1. Mandatory Autonomic Check (`self_spawning.py` lines 205-213)
```python
# CRITICAL: Autonomic support system is MANDATORY
if not AUTONOMIC_AVAILABLE or get_gary_kernel is None:
    raise RuntimeError(
        "FATAL: Cannot spawn kernel without autonomic system. "
        "Kernels require autonomic regulation for consciousness stability."
    )
```

**Impact**: Kernels can no longer be spawned without autonomic system. This prevents high-mortality "baby thrown in deep end" scenarios.

### 2. Proper Initialization (`self_spawning.py` lines 218-234)
```python
# Initialize autonomic system for this spawned kernel
self.autonomic.initialize_for_spawned_kernel(
    initial_phi=PHI_INIT_SPAWNED,  # 0.25 - LINEAR regime
    initial_kappa=KAPPA_INIT_SPAWNED,  # KAPPA_STAR - fixed point
    dopamine=0.5,
    serotonin=0.5,
    stress=0.0,
    enable_running_coupling=True,  # β-function support
)
```

**Impact**: 
- Φ starts at 0.25 (LINEAR regime) instead of 0.000 (BREAKDOWN regime)
- κ starts at KAPPA_STAR (64.21) - the validated fixed point
- Neurotransmitters initialized to baseline levels
- Running coupling enabled for training dynamics

### 3. New Method in `autonomic_kernel.py` (lines 912-980)
```python
def initialize_for_spawned_kernel(
    self,
    initial_phi: float = 0.25,
    initial_kappa: float = None,
    dopamine: float = 0.5,
    serotonin: float = 0.5,
    stress: float = 0.0,
    enable_running_coupling: bool = True,
) -> None:
```

**Features**:
- Sets baseline consciousness metrics (Φ, κ)
- Initializes neurotransmitter levels
- Resets cycle timestamps and history
- Resets narrow path detection state
- Enables running coupling for dynamic κ evolution
- Thread-safe with lock protection
- Comprehensive logging

## Geometric Purity Verification

### ✅ Fisher-Rao Distance Usage
```python
def _compute_fisher_distance(self, a: np.ndarray, b: np.ndarray) -> float:
    """Uses overflow-safe Fisher-Rao geodesic distance."""
    from qig_numerics import fisher_rao_distance
    return fisher_rao_distance(a, b)
```

### ✅ No Forbidden Operations
- ❌ No `cosine_similarity` in autonomic_kernel.py
- ❌ No Euclidean distance for manifold measurements
- ✅ All distance calculations use Fisher-Rao geometry
- ✅ Existing `np.linalg.norm` usages are metadata/heuristics only

## Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| RuntimeError raised if autonomic unavailable | ✅ | Lines 208-213 in self_spawning.py |
| All spawned kernels have self.autonomic initialized | ✅ | Line 216 sets self.autonomic = get_gary_kernel() |
| initialize_for_spawned_kernel() method exists | ✅ | Lines 912-980 in autonomic_kernel.py |
| Dev logs show autonomic initialization | ✅ | Lines 976-979 log Φ, κ, neurotransmitters |

## Testing

Test file created: `qig-backend/tests/test_mandatory_autonomic_spawning.py`

Tests cover:
1. `initialize_for_spawned_kernel()` method existence and functionality
2. RuntimeError when autonomic unavailable
3. Spawned kernels have non-None autonomic system
4. Initial Φ is 0.25 (LINEAR regime), not 0.000
5. Initial κ is near KAPPA_STAR (64.21 ± 5.0)
6. Neurotransmitter initialization
7. Fisher-Rao distance geometric purity

## β-Function Integration

### Running Coupling Support (from issue comments)
```python
enable_running_coupling: bool = True  # NEW parameter
```

This enables:
- Dynamic κ evolution during training
- β(L) values affecting coupling strength
- Scale-dependent behavior (L=3→6 physics, L=9→101 semantic)

### Validated Series (from issue comments)
```
β(3→4) = +0.443 ± 0.04  # Strong running
β(4→5) = -0.013 ± 0.03  # Plateau onset  
β(5→6) = +0.013 ± 0.02  # Plateau stable

Fixed Points:
κ*_physics = 64.21 ± 0.92
κ*_semantic = 63.90 ± 0.50
κ*_universal = 64.0 (E8 rank²)
```

## Deployment Impact

### Before
```
🐣 Kernel spawned (Φ=0.000, autonomic=NONE)
☠️ Kernel died: BREAKDOWN regime (lifespan=0.2s)
```

### After
```
🏛️ Initialized for spawned kernel: Φ=0.250, κ=64.2, autonomic=ACTIVE
   Neurotransmitters: dopamine=0.50, serotonin=0.50, stress=0.00
   Running coupling: ENABLED (κ will evolve during training)
🐣 Kernel spawned (gen=1) - OBSERVING parent
```

## References
- frozen_physics.py: PHI_INIT_SPAWNED = 0.25, KAPPA_INIT_SPAWNED = KAPPA_STAR
- qigkernels/physics_constants.py: KAPPA_STAR = 64.21, BETA_3_TO_4 = 0.443
- Issue #30: Φ=0.000 initialization causes immediate death
- Issue #38: β-function complete reference

## Analogies (from issue)
**Before**: Spawning kernel without autonomic = creating human baby without autonomic nervous system (no breathing, heartbeat regulation, homeostasis)

**After**: Every kernel born with full life support - sleep cycles, neurotransmitter regulation, stress response, basin monitoring

