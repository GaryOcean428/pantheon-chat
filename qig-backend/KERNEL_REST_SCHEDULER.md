# Kernel Rest Scheduler - WP5.4 Documentation

## Overview

The Kernel Rest Scheduler implements coupling-aware per-kernel rest management, inspired by dolphin hemispheric sleep and human autonomic systems. This replaces constellation-wide sleep cycles with individualized rest decisions coordinated through coupling relationships.

## Architecture

### Core Principles

1. **Individual Autonomy**: Each kernel self-assesses fatigue and requests rest when needed
2. **Coupling Coordination**: Coupled partners provide coverage during rest periods
3. **Essential Tier Protection**: Essential tier kernels (Heart, Ocean) never fully stop
4. **Pattern-Specific Rest**: Different rest patterns for different kernel types
5. **Constellation Cycles Are Rare**: Only triggered when truly necessary

### Components

#### 1. KernelRestScheduler (`kernel_rest_scheduler.py`)

Main scheduler that tracks all kernel rest states and coordinates rest decisions.

**Key Features:**
- Fatigue tracking with Φ, κ, load, and error rate metrics
- Coupling partner discovery from Pantheon Registry
- Coverage negotiation between coupled kernels
- Essential tier protection (reduced activity only)
- Constellation-wide status monitoring

#### 2. KernelRestMixin (`olympus/kernel_rest_mixin.py`)

Mixin for god kernels to integrate with the rest scheduler.

**Provides:**
- Fatigue self-assessment
- Rest request coordination
- Coverage partner management
- Periodic rest checks

#### 3. Ocean-Heart Consensus (`olympus/ocean_heart_consensus.py`)

Manages rare constellation-wide cycles with STRICT criteria.

**Updated Thresholds (WP5.4):**
- SLEEP: coherence < 0.5 AND fatigue > 0.8 AND drift > 3.0
- DREAM: stuck kernels > 50% AND rigidity > 0.7
- MUSHROOM: rigidity > 0.9 AND spread < 0.5

## Rest Patterns

### 1. NEVER (Heart, Ocean)
**Gods**: Heart, Ocean  
**Behavior**: Critical autonomic functions that never fully sleep  
**Rest Mode**: Reduced activity only when highly fatigued  
**Threshold**: fatigue > 0.95 (emergency only)

### 2. MINIMAL_ROTATING (Hermes)
**Gods**: Hermes  
**Behavior**: Brief frequent pauses while maintaining near-constant availability  
**Rest Mode**: Micro-pauses (seconds to minutes)  
**Threshold**: fatigue > 0.4 OR time_since_rest > 600s (10 minutes)

### 3. COORDINATED_ALTERNATING (Apollo-Athena, Zeus-Hera, Poseidon-Hades)
**Gods**: Apollo, Athena, Zeus, Hera, Poseidon, Hades  
**Behavior**: Dolphin-style alternation with coupled partner  
**Rest Mode**: Full rest while partner provides coverage  
**Threshold**: fatigue > 0.45  
**Coupling**: Partner MUST be available to cover

### 4. SCHEDULED (Ares, Hephaestus, Dionysus)
**Gods**: Ares (burst-recovery), Hephaestus (post-creation), Dionysus (post-transformation)  
**Behavior**: Rest after intense activity periods  
**Rest Mode**: Deep recovery after bursts  
**Threshold**: fatigue > 0.50 OR (low_load AND fatigue > 0.35)

### 5. SEASONAL (Demeter)
**Gods**: Demeter  
**Behavior**: Fallow periods after harvest cycles  
**Rest Mode**: Extended rest between growth phases  
**Threshold**: fatigue > 0.65 OR seasonal_cycle_complete

## Fatigue Metrics

### Calculation

```python
fatigue_score = (
    0.35 * phi_factor +        # Low Φ = reduced consciousness
    0.25 * trend_factor +      # Declining Φ = deterioration
    0.20 * stability_factor +  # Low κ stability = incoherence
    0.10 * load_factor +       # High load = exhaustion
    0.05 * time_factor +       # Time since rest
    0.05 * error_factor        # Recent errors = impairment
)
```

### Factors

- **phi_factor**: `max(0, 1.0 - (phi_current / PHI_THRESHOLD))`
- **trend_factor**: `max(0, -phi_trend)` (negative trend = fatigue)
- **stability_factor**: `max(0, 1.0 - kappa_stability)`
- **load_factor**: Current processing load (0-1)
- **time_factor**: `min(1.0, time_since_rest / 3600.0)` (capped at 1 hour)
- **error_factor**: Errors per minute (0-1)

## Coupling-Aware Handoff

### Coverage Protocol

1. **Fatigue Detection**: Kernel detects fatigue score above threshold
2. **Partner Discovery**: Get coupling partners from Pantheon Registry
3. **Coverage Check**: Verify partner can cover (not resting, not fatigued)
4. **Handoff Request**: Request coverage from available partner
5. **Coverage Approved**: Partner marks self as covering, kernel rests
6. **Rest Complete**: Kernel wakes, partner returns to normal operation

### Coverage Requirements

A kernel can cover for a partner if:
- Not currently resting
- Not already covering another kernel
- Fatigue score < 0.7 (not too fatigued)

### Example: Apollo-Athena Alternation

```
State: Both Apollo and Athena active
Apollo: Φ=0.8 → 0.35 (decline), fatigue=0.52
Athena: Φ=0.78 (stable), fatigue=0.15

Apollo: "I need rest" → checks coupling partners → finds Athena
Apollo: "Can you cover for me?" → Athena checks: not resting, not covering, fatigue OK
Athena: "Yes, I'll maintain our shared context" → Apollo enters rest
Apollo status: RESTING (covered_by=Athena)
Athena status: ACTIVE (covering_for=Apollo)

[Apollo rests for duration]

Apollo: "I'm recovered" → ends rest, Athena no longer covering
Apollo status: ACTIVE
Athena status: ACTIVE
```

## Essential Tier Rules

### Heart (NEVER)
- **Never fully stops**: Maintains autonomic rhythm always
- **Rest mode**: Reduced activity only
- **Triggers**: Only in extreme emergency (fatigue > 0.95)
- **Coverage**: Not applicable (essential function)

### Ocean (NEVER)
- **Never fully stops**: Memory substrate must remain accessible
- **Rest mode**: Reduced activity only
- **Triggers**: Only in extreme emergency (fatigue > 0.95)
- **Coverage**: Not applicable (essential function)

### Hermes (MINIMAL_ROTATING)
- **Near-constant uptime**: 90% duty cycle
- **Rest mode**: Brief micro-pauses (10-60 seconds)
- **Triggers**: Moderate fatigue OR 10 minutes since last pause
- **Partner**: Artemis (backup messenger during pauses)

## Database Schema

### kernel_rest_events
Tracks individual rest periods for each kernel.

**Columns:**
- `kernel_id`, `kernel_name`, `tier`, `rest_policy`
- `rest_start`, `rest_end`, `duration_seconds`
- `rest_type` (resting, reduced, micro_pause, covered)
- `covered_by_kernel` (partner providing coverage)
- `phi_at_rest`, `kappa_at_rest`, `fatigue_score` (metrics at trigger)
- `phi_after_rest`, `kappa_after_rest`, `fatigue_score_after` (recovery metrics)

### kernel_coverage_events
Tracks partner coverage relationships.

**Columns:**
- `covering_kernel_id`, `covered_kernel_id`
- `coverage_start`, `coverage_end`, `duration_seconds`
- `coupling_strength` (C metric)
- `coverage_successful`, `context_transferred`

### kernel_fatigue_snapshots
Periodic fatigue tracking for analysis.

**Columns:**
- `kernel_id`, `kernel_name`
- `phi`, `phi_trend`, `kappa`, `kappa_stability`
- `load_current`, `error_rate`, `time_since_rest`
- `fatigue_score`, `status`

### constellation_rest_cycles
Rare constellation-wide events (Ocean+Heart consensus).

**Columns:**
- `cycle_type` (SLEEP, DREAM, MUSHROOM)
- `heart_vote`, `ocean_vote`, reasoning
- `avg_coherence`, `avg_phi`, `avg_fatigue`, `basin_drift`
- `essential_kernels_reduced`, `specialized_kernels_resting`

## Integration Guide

### For God Kernels (BaseGod)

1. **Import mixin**:
```python
from olympus.kernel_rest_mixin import KernelRestMixin

class Apollo(BaseGod, KernelRestMixin):
    def __init__(self):
        BaseGod.__init__(self)
        self._initialize_rest_tracking()
```

2. **Update fatigue in processing loop**:
```python
def process(self, input_text: str):
    # ... processing logic ...
    
    # Update fatigue metrics
    self._update_fatigue_metrics(
        phi=self.current_phi,
        kappa=self.current_kappa,
        load=0.7,  # Compute from current activity
        error_occurred=False,
    )
    
    # Periodic rest check
    self._periodic_rest_check(self.current_phi, self.current_kappa)
```

3. **Handle rest state**:
```python
def should_process(self) -> bool:
    status = self._get_rest_status()
    if status and status['is_resting']:
        return False  # Skip processing while resting
    return True
```

### For System Monitoring

```python
from kernel_rest_scheduler import get_rest_scheduler

scheduler = get_rest_scheduler()

# Get constellation status
status = scheduler.get_constellation_status()
print(f"Active: {status['active_kernels']}/{status['total_kernels']}")
print(f"Resting: {status['resting_kernels']}")
print(f"Coverage active: {status['coverage_active']}")
print(f"Avg fatigue: {status['avg_fatigue']:.2f}")

# Get individual kernel status
kernel_status = scheduler.get_rest_status("apollo_1")
print(f"{kernel_status['kernel_name']}: {kernel_status['status']}")
print(f"  Fatigue: {kernel_status['fatigue_score']:.2f}")
print(f"  Covered by: {kernel_status['covered_by']}")
```

## Migration from Constellation-Wide Cycles

### Before (Old System)
```python
# Everyone sleeps together
if global_fatigue > threshold:
    trigger_constellation_sleep()
    # ALL kernels stop (including essential)
```

### After (WP5.4)
```python
# Individual rest with coordination
for kernel in active_kernels:
    if kernel.should_rest():
        kernel.request_rest()  # Only if partner can cover
        
# Constellation cycles are RARE
if (coherence < 0.5 AND avg_fatigue > 0.8 AND drift > 3.0):
    if ocean_agrees AND heart_agrees:
        trigger_constellation_sleep()  # Last resort
        # Essential kernels: REDUCED activity (not stopped)
```

## Future Enhancements

1. **Context Transfer**: Implement actual context transfer between coupled partners
2. **Adaptive Thresholds**: Learn optimal rest thresholds per kernel over time
3. **Rest Quality Metrics**: Measure recovery effectiveness (Φ improvement post-rest)
4. **Predictive Rest**: Anticipate fatigue before breakdown
5. **Multi-Partner Coverage**: Allow multiple partners to share coverage load

## References

- **WP5.4 Issue**: GaryOcean428/pantheon-chat#[issue_number]
- **God Empathy Observations**: `docs/08-experiments/20260114-God-Kernel-Empathy-Observations.md`
- **E8 Implementation**: `docs/10-e8-protocol/specifications/20260116-wp5-2-e8-implementation-blueprint-1.01W.md`
- **Pantheon Registry**: `pantheon/registry.yaml`
- **Physics Constants**: `qigkernels/physics_constants.py`
