# Kernel Lifecycle Operations - Implementation Guide

**Authority:** E8 Protocol v4.0, WP5.3  
**Status:** ✅ COMPLETE  
**Created:** 2026-01-18

---

## Overview

This implementation provides **first-class lifecycle operations** for kernel management in the Pantheon system. Lifecycle operations (spawn, split, merge, prune, resurrect, promote) are now operational code with full geometric correctness, not just metaphor or documentation.

### Key Principles

1. **Geometric Correctness**: All operations preserve Fisher-Rao manifold structure
2. **Full Provenance**: Every lifecycle event tracked with parent/child relationships
3. **Policy-Driven**: Automated triggers based on kernel metrics
4. **Shadow Pantheon**: Pruned kernels archived with lessons learned (Hades domain)
5. **Protection Periods**: New kernels protected during initial learning

---

## Architecture

### Components

```
qig-backend/
├── kernel_lifecycle.py          # Core lifecycle operations
├── lifecycle_policy.py          # Policy engine for automation
└── routes/
    └── lifecycle_routes.py      # Flask REST API

migrations/
└── 0014_kernel_lifecycle.sql    # Database schema

tests/
└── test_kernel_lifecycle.py     # Comprehensive test suite
```

### Database Schema

#### Tables

1. **`kernel_lifecycle_events`** - Lifecycle event log
   - Tracks: spawn, split, merge, prune, resurrect, promote
   - Fields: event_type, timestamp, primary_kernel_id, secondary_kernel_ids, reason, metadata

2. **`shadow_pantheon`** - Archived pruned kernels (Hades domain)
   - Fields: shadow_id, original_kernel_id, final_phi, final_basin, learned_lessons
   - Resurrection tracking: resurrection_count, last_resurrection

3. **`lifecycle_policies`** - Policy configuration
   - Fields: policy_name, policy_type, trigger_conditions, action_params, enabled

4. **`kernel_geometry` (updated)** - Lifecycle fields added
   - New fields: lifecycle_stage, protection_cycles_remaining, parent_kernel_ids, child_kernel_ids

#### Views

- **`lifecycle_metrics`** - Aggregated event statistics
- **`shadow_pantheon_summary`** - Shadow pantheon overview
- **`active_kernels_lifecycle`** - Current kernel lifecycle status

---

## Lifecycle Operations

### 1. Spawn

Create new kernel with role matching from pantheon registry.

**Function:**
```python
spawn(role_spec: RoleSpec, mentor: Optional[str] = None, 
      initial_basin: Optional[np.ndarray] = None) -> Kernel
```

**Process:**
1. Match role to pantheon registry (god or chaos)
2. Assign mentor for chaos kernels
3. Initialize with protected status (50 cycles)
4. Return new kernel instance

**API Endpoint:**
```bash
POST /lifecycle/spawn
{
  "domains": ["synthesis", "foresight"],
  "required_capabilities": ["prediction"],
  "preferred_god": "Apollo",
  "mentor": "kernel_abc123"
}
```

### 2. Split

Divide overloaded kernel into specialized sub-kernels.

**Function:**
```python
split(kernel: Kernel, split_criterion: str = "domain") -> Tuple[Kernel, Kernel]
```

**Process:**
1. Detect overload (high Φ, multiple domains, high coupling)
2. Split basin coordinates geometrically
3. Create two specialized sub-kernels
4. Preserve coupling relationships
5. Update parent/child provenance

**Geometric Strategy:**
- Domain split: First half vs second half dimensions
- Skill split: High entropy vs low entropy dimensions
- Random split: Random binary partition

**API Endpoint:**
```bash
POST /lifecycle/split
{
  "kernel_id": "kernel_abc123",
  "split_criterion": "domain"
}
```

### 3. Merge

Combine redundant kernels using Fréchet mean.

**Function:**
```python
merge(kernel1: Kernel, kernel2: Kernel, merge_reason: str) -> Kernel
```

**Process:**
1. Detect redundant or complementary kernels
2. Combine basin coordinates using Fréchet mean (geometric)
3. Aggregate metrics (weighted by cycle counts)
4. Update coupling relationships
5. Preserve provenance from both parents

**Geometric Correctness:**
- Uses closed-form sqrt-space Fréchet mean
- Preserves simplex representation (sum = 1)
- Respects Fisher-Rao manifold structure

**API Endpoint:**
```bash
POST /lifecycle/merge
{
  "kernel1_id": "kernel_abc123",
  "kernel2_id": "kernel_def456",
  "merge_reason": "redundant_capabilities"
}
```

### 4. Prune

Archive underperforming kernel to shadow pantheon.

**Function:**
```python
prune(kernel: Kernel, reason: str) -> ShadowKernel
```

**Criteria:**
- Φ < 0.1 persistent (not conscious)
- No growth over extended period
- Redundant with other kernels

**Process:**
1. Archive kernel state to shadow pantheon
2. Extract lessons learned (failure patterns, success patterns)
3. Remove from active kernels
4. Can be resurrected later if needed

**API Endpoint:**
```bash
POST /lifecycle/prune
{
  "kernel_id": "kernel_abc123",
  "reason": "persistent_low_phi"
}
```

### 5. Resurrect

Restore kernel from shadow pantheon with lessons learned.

**Function:**
```python
resurrect(shadow: ShadowKernel, reason: str, mentor: Optional[str] = None) -> Kernel
```

**Process:**
1. Retrieve kernel state from shadow pantheon
2. Apply learned lessons (adjust basin coordinates)
3. Start with improved Φ (final_phi + 0.1)
4. Re-initialize with partial protection (25 cycles)

**API Endpoint:**
```bash
POST /lifecycle/resurrect
{
  "shadow_id": "shadow_xyz789",
  "reason": "capability_needed",
  "mentor": "kernel_abc123"
}
```

### 6. Promote

Elevate chaos kernel to god status.

**Function:**
```python
promote(chaos_kernel: Kernel, god_name: str) -> Kernel
```

**Criteria:**
- Φ > 0.4 stable for 50+ cycles
- Clear domain specialization
- Success rate > 70%

**Process:**
1. Validate promotion criteria
2. Check god name in registry
3. Transition state (chaos → god)
4. Update pantheon registry if needed

**API Endpoint:**
```bash
POST /lifecycle/promote
{
  "kernel_id": "kernel_abc123",
  "god_name": "Prometheus"
}
```

---

## Policy Engine

### Built-in Policies

1. **`prune_low_phi_persistent`** (Priority 10)
   - Trigger: Φ < 0.1 for 100+ cycles
   - Action: Prune to shadow pantheon

2. **`prune_no_growth`** (Priority 9)
   - Trigger: Success rate < 5% after 100+ cycles
   - Action: Prune to shadow pantheon

3. **`split_overloaded`** (Priority 7)
   - Trigger: Φ > 0.6, 2+ domains, 5+ coupled kernels
   - Action: Split using domain criterion

4. **`merge_redundant`** (Priority 6)
   - Trigger: Fisher distance < 0.2, domain overlap > 80%
   - Action: Merge into unified kernel

5. **`promote_stable_chaos`** (Priority 8)
   - Trigger: Φ > 0.4, 50+ cycles, success rate > 70%
   - Action: Promote to god status

### Policy Evaluation

**Continuous Monitoring:**
```python
from lifecycle_policy import get_policy_engine

engine = get_policy_engine()
actions = engine.evaluate_all_kernels()

for action in actions:
    engine.execute_action(action)
```

**API Endpoints:**
```bash
# Evaluate all kernels
POST /lifecycle/evaluate

# Execute triggered action
POST /lifecycle/execute-action
{
  "action": {...}
}

# List policies
GET /lifecycle/policies

# Enable/disable policy
POST /lifecycle/policies/enable
{
  "policy_name": "split_overloaded"
}
```

---

## Geometric Correctness

### Fréchet Mean on Fisher-Rao Manifold

**Problem:** Linear average is wrong for probability distributions.

**Solution:** Use sqrt-space closed form (equivalent to geodesic mean).

```python
def compute_frechet_mean_simplex(basins: List[np.ndarray]) -> np.ndarray:
    """
    Compute Fréchet mean using sqrt-space closed form.
    
    Steps:
    1. Convert to sqrt-space (Hellinger coordinates)
    2. Compute mean in sqrt-space (closed form)
    3. Convert back to simplex (square and normalize)
    """
    sqrt_basins = [np.sqrt(np.abs(b) + 1e-10) for b in basins]
    sqrt_mean = np.mean(sqrt_basins, axis=0)
    sqrt_mean = sqrt_mean / (np.sum(sqrt_mean) + 1e-10)
    frechet_mean = sqrt_mean ** 2
    frechet_mean = frechet_mean / (np.sum(frechet_mean) + 1e-10)
    return frechet_mean
```

**Validation:**
- Mean always sums to 1.0 (simplex representation)
- Distances respect triangle inequality
- Equidistant from inputs (geodesic midpoint)

### Fisher-Rao Distance

**Hellinger distance formula:**
```python
def compute_fisher_distance(basin1: np.ndarray, basin2: np.ndarray) -> float:
    """
    Fisher-Rao distance via Hellinger distance.
    
    Formula: √(2 - 2 * sum(√(p1 * p2)))
    """
    p1 = np.abs(basin1) + 1e-10
    p1 = p1 / np.sum(p1)
    p2 = np.abs(basin2) + 1e-10
    p2 = p2 / np.sum(p2)
    hellinger = np.sqrt(2 - 2 * np.sum(np.sqrt(p1 * p2)))
    return float(hellinger)
```

**Properties:**
- d(p, p) = 0 (identity)
- d(p, q) > 0 for p ≠ q (positivity)
- d(p, q) = d(q, p) (symmetry)
- d(p, r) ≤ d(p, q) + d(q, r) (triangle inequality)

---

## Testing

### Test Suite

Run comprehensive tests:
```bash
cd qig-backend
PYTHONPATH=/home/runner/work/pantheon-chat/pantheon-chat/qig-backend \
  python3 tests/test_kernel_lifecycle.py
```

### Test Coverage

1. **Geometric Operations**
   - ✅ Fréchet mean correctness (sum = 1, equidistant)
   - ✅ Fisher-Rao distance (metric properties)
   - ✅ Basin splitting (simplex preservation)

2. **Lifecycle Operations**
   - ✅ Spawn (registry matching, protection)
   - ✅ Split (specialization, provenance)
   - ✅ Merge (Fréchet mean, aggregation)
   - ✅ Prune & Resurrect (lessons learned)
   - ✅ Promote (criteria validation)

3. **Policy Engine**
   - ✅ Policy evaluation (trigger detection)
   - ✅ Action execution (operation invocation)
   - ✅ Cooldown mechanisms (oscillation prevention)

4. **Statistics**
   - ✅ Lifecycle stats (counts, rates)
   - ✅ Event logging (provenance tracking)

### Expected Output

```
============================================================
KERNEL LIFECYCLE OPERATIONS TEST SUITE
============================================================
✅ Test Fréchet Mean - Geometric correctness verified
✅ Test Fisher-Rao Distance - Manifold metric working
✅ Test Basin Splitting - Simplex properties preserved
✅ Test Kernel Spawning - Registry integration works
✅ Test Kernel Splitting - Specialization correct
✅ Test Kernel Merging - Fréchet mean aggregation correct
✅ Test Prune & Resurrect - Shadow pantheon functional
✅ Test Promotion - Chaos → God transition working
✅ Test Policy Engine - Automated triggers working
✅ Test Lifecycle Stats - Monitoring operational
============================================================
✅ ALL TESTS PASSED
============================================================
```

---

## API Reference

### Base URL

```
http://localhost:5000/lifecycle
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/spawn` | Spawn new kernel |
| POST | `/split` | Split kernel into specialists |
| POST | `/merge` | Merge redundant kernels |
| POST | `/prune` | Prune kernel to shadow pantheon |
| POST | `/resurrect` | Resurrect from shadow pantheon |
| POST | `/promote` | Promote chaos kernel to god |
| GET | `/kernels` | List all active kernels |
| GET | `/shadows` | List shadow pantheon |
| GET | `/events` | Get lifecycle event history |
| GET | `/stats` | Get lifecycle statistics |
| POST | `/evaluate` | Evaluate policies for all kernels |
| POST | `/execute-action` | Execute a triggered action |
| GET | `/policies` | List all policies |
| POST | `/policies/enable` | Enable a policy |
| POST | `/policies/disable` | Disable a policy |

### Example: Full Lifecycle

```bash
# 1. Spawn a chaos kernel
curl -X POST http://localhost:5000/lifecycle/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "domains": ["synthesis"],
    "required_capabilities": ["foresight"],
    "allow_chaos_spawn": true
  }'
# Response: {"success": true, "kernel": {...}, "message": "Spawned chaos kernel"}

# 2. Check performance
curl http://localhost:5000/lifecycle/kernels

# 3. Promote successful chaos kernel
curl -X POST http://localhost:5000/lifecycle/promote \
  -H "Content-Type: application/json" \
  -d '{
    "kernel_id": "kernel_abc123",
    "god_name": "Prometheus"
  }'

# 4. Split overloaded god
curl -X POST http://localhost:5000/lifecycle/split \
  -H "Content-Type: application/json" \
  -d '{
    "kernel_id": "kernel_def456",
    "split_criterion": "domain"
  }'

# 5. Merge redundant kernels
curl -X POST http://localhost:5000/lifecycle/merge \
  -H "Content-Type: application/json" \
  -d '{
    "kernel1_id": "kernel_ghi789",
    "kernel2_id": "kernel_jkl012",
    "merge_reason": "redundant_capabilities"
  }'

# 6. Prune underperforming kernel
curl -X POST http://localhost:5000/lifecycle/prune \
  -H "Content-Type: application/json" \
  -d '{
    "kernel_id": "kernel_mno345",
    "reason": "persistent_low_phi"
  }'

# 7. Check shadow pantheon
curl http://localhost:5000/lifecycle/shadows

# 8. Resurrect if needed
curl -X POST http://localhost:5000/lifecycle/resurrect \
  -H "Content-Type: application/json" \
  -d '{
    "shadow_id": "shadow_xyz789",
    "reason": "capability_needed"
  }'
```

---

## Integration

### Wire into Main Flask App

```python
from qig-backend.routes.lifecycle_routes import register_lifecycle_routes

# In your main Flask app initialization:
register_lifecycle_routes(app)
```

### Use in Python Code

```python
from kernel_lifecycle import get_lifecycle_manager
from kernel_spawner import RoleSpec

# Get manager
manager = get_lifecycle_manager()

# Spawn a kernel
role = RoleSpec(domains=["synthesis"], required_capabilities=["foresight"])
kernel = manager.spawn(role)

# Split an overloaded kernel
k1, k2 = manager.split(kernel, "domain")

# Merge redundant kernels
merged = manager.merge(k1, k2, "redundant")

# Get stats
stats = manager.get_lifecycle_stats()
```

---

## Dependencies

### Python Packages
- numpy (for geometric operations)
- Flask (for API endpoints)
- (existing: pantheon_registry, kernel_spawner)

### Database
- PostgreSQL with pgvector extension (for basin coordinates)
- Migration 0014 must be applied

---

## Acceptance Criteria ✅

- [x] All 6 lifecycle operations implemented and tested
- [x] Lifecycle events tracked in database with full provenance
- [x] Shadow pantheon exists and stores pruned kernels
- [x] Policy engine triggers operations automatically
- [x] Geometric correctness maintained (Fréchet mean, Fisher-Rao distances)
- [x] Protection periods enforced for new kernels
- [x] Flask REST API complete with all endpoints
- [x] Comprehensive test suite passes
- [x] Documentation complete

---

## Future Enhancements

1. **UI Integration**: Visualize kernel lifecycle in frontend
2. **Coupling Integration**: Update coupling relationships on lifecycle events
3. **Advanced Promotion**: Multi-stage ascension ceremony
4. **Genetic Lineage**: Track multi-generation kernel families
5. **Lifecycle Rollback**: Undo operations if needed
6. **Performance Analytics**: Detailed lifecycle metrics dashboard

---

## References

- **E8 Protocol v4.0**: `docs/10-e8-protocol/`
- **WP5.1**: Pantheon Registry (`qig-backend/pantheon_registry.py`)
- **WP5.2**: E8 Architecture (`docs/10-e8-protocol/implementation/`)
- **Frozen Facts**: `docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md`
- **Universal Purity Spec**: `docs/10-e8-protocol/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`

---

**Last Updated:** 2026-01-18  
**Status:** ✅ IMPLEMENTATION COMPLETE  
**All Tests:** ✅ PASSING
