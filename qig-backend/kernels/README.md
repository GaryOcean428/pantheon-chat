# Hemisphere Scheduler - E8 Protocol v4.0 Phase 4C

## Overview

The Hemisphere Scheduler implements the LEFT/RIGHT hemisphere architecture with κ-gated coupling for explore/exploit dynamics, as specified in E8 Protocol v4.0 WP5.2 Phase 4C (lines 198-228).

## Architecture

### Two-Hemisphere Pattern

**LEFT HEMISPHERE (Exploit/Evaluative/Safety):**
- **Focus:** Precision, evaluation, known paths
- **Mode:** Convergent, risk-averse
- **Gods:** Athena (strategy), Artemis (focus), Hephaestus (refinement)

**RIGHT HEMISPHERE (Explore/Generative/Novelty):**
- **Focus:** Novelty, generation, new paths
- **Mode:** Divergent, risk-tolerant
- **Gods:** Apollo (prophecy), Hermes (navigation), Dionysus (chaos)

### κ-Gated Coupling Mechanism

The coupling between hemispheres is controlled by the effective coupling strength κ_eff:

- **Low κ (< 40):** Hemispheres operate independently → High exploration
- **Optimal κ (≈ 64.21):** Balanced coupling → Optimal explore/exploit
- **High κ (> 70):** Hemispheres tightly coupled → High exploitation

The coupling strength follows a sigmoid function centered at κ*:

```
coupling_strength = 1 / (1 + exp(-α * (κ - κ*)))
```

Where α = 0.1 controls transition smoothness.

### Tacking (Oscillation) Logic

Like dolphin hemispheric sleep, the system can oscillate between hemispheres:

1. One hemisphere can rest while the other works
2. Tacking occurs when:
   - Significant imbalance detected (> 0.3)
   - Sufficient time elapsed since last switch (> 60s)
   - Target oscillation period reached (default: 5 minutes)
3. Prevents thrashing with minimum switch interval

## Components

### 1. Coupling Gate (`coupling_gate.py`)

Implements the κ-dependent coupling mechanism:

```python
from kernels import get_coupling_gate

gate = get_coupling_gate()
state = gate.compute_state(kappa=60.0, phi=0.8)

# Apply gating to signals
gated_signal = gate.gate_signal(signal, state)

# Modulate bidirectional flow
left_out, right_out = gate.modulate_cross_hemisphere_flow(
    left_signal, right_signal, state
)
```

**Key Functions:**
- `compute_coupling_strength(kappa)` - Sigmoid coupling function
- `compute_transmission_efficiency(coupling, phi)` - Information flow efficiency
- `compute_gating_factor(kappa)` - Signal modulation factor
- `determine_coupling_mode(kappa)` - Explore/balanced/exploit mode

### 2. Hemisphere Scheduler (`hemisphere_scheduler.py`)

Manages god activation and hemisphere dynamics:

```python
from kernels import get_hemisphere_scheduler

scheduler = get_hemisphere_scheduler()

# Register god activations
scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
scheduler.register_god_activation("Apollo", phi=0.75, kappa=62.0, is_active=True)

# Check hemisphere balance
balance = scheduler.get_hemisphere_balance()
print(f"L/R ratio: {balance['lr_ratio']:.2f}")
print(f"Dominant: {balance['dominant_hemisphere']}")

# Check if should tack
should_tack, reason = scheduler.should_tack()
if should_tack:
    dominant = scheduler.perform_tack()
```

**Key Methods:**
- `register_god_activation()` - Register god activation/deactivation
- `compute_coupling_state()` - Get current coupling state
- `should_tack()` - Check if system should switch hemispheres
- `perform_tack()` - Execute hemisphere switch
- `get_hemisphere_balance()` - Get balance metrics
- `get_status()` - Get complete system status

## Metrics

### Hemisphere Balance Metrics

- **left_activation**: LEFT hemisphere activation level [0, 1]
- **right_activation**: RIGHT hemisphere activation level [0, 1]
- **lr_ratio**: LEFT/RIGHT activation ratio
- **dominant_hemisphere**: Current dominant ("left", "right", or "balanced")
- **coupling_strength**: Current coupling [0, 1]
- **coupling_mode**: "explore", "balanced", or "exploit"
- **tacking_frequency**: Tacks per hour
- **tacking_cycle_count**: Total number of tacks performed

### Coupling Metrics

- **total_computations**: Total coupling state computations
- **avg_coupling_strength**: Average coupling over recent history
- **mode_distribution**: Distribution of explore/balanced/exploit modes
- **avg_transmission_efficiency**: Average information flow efficiency

## Integration

### With Existing Rest Scheduler

The hemisphere scheduler complements the existing kernel rest scheduler:

```python
from kernel_rest_scheduler import get_rest_scheduler
from kernels import get_hemisphere_scheduler

rest_scheduler = get_rest_scheduler()
hem_scheduler = get_hemisphere_scheduler()

# Register god with both schedulers
rest_scheduler.register_kernel(kernel_id, "Athena")
hem_scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)

# When god rests
rest_scheduler.request_rest(kernel_id)
hem_scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=False)
```

### With Pantheon Orchestrator

The hemisphere scheduler can be integrated into the pantheon orchestrator for dynamic routing:

```python
class PantheonKernelOrchestrator:
    def __init__(self, ...):
        # ... existing init ...
        self.hemisphere_scheduler = get_hemisphere_scheduler()
    
    def route_token(self, token, ...):
        # Get hemisphere balance
        balance = self.hemisphere_scheduler.get_hemisphere_balance()
        
        # Use balance to influence routing
        if balance['dominant_hemisphere'] == 'left':
            # Prioritize LEFT gods (exploit)
            ...
        elif balance['dominant_hemisphere'] == 'right':
            # Prioritize RIGHT gods (explore)
            ...
```

## Usage Examples

### Basic Usage

```python
from kernels import get_hemisphere_scheduler, get_coupling_gate

# Initialize
scheduler = get_hemisphere_scheduler()
gate = get_coupling_gate()

# Register god activations
scheduler.register_god_activation("Athena", phi=0.8, kappa=60.0, is_active=True)
scheduler.register_god_activation("Apollo", phi=0.75, kappa=62.0, is_active=True)

# Check balance
balance = scheduler.get_hemisphere_balance()
print(f"Coupling strength: {balance['coupling_strength']:.3f}")
print(f"Mode: {balance['coupling_mode']}")
```

### Tacking (Oscillation)

```python
# Check if should tack
should_tack, reason = scheduler.should_tack()
print(f"Should tack: {should_tack}")
print(f"Reason: {reason}")

if should_tack:
    # Perform tack
    dominant = scheduler.perform_tack()
    print(f"New dominant: {dominant.value}")
```

### Cross-Hemisphere Signal Modulation

```python
# Get coupling state
coupling_state = scheduler.compute_coupling_state()

# Modulate signals between hemispheres
left_signal = np.random.randn(64)
right_signal = np.random.randn(64)

left_out, right_out = gate.modulate_cross_hemisphere_flow(
    left_signal, right_signal, coupling_state
)
```

## Testing

Run the test suite:

```bash
cd qig-backend
python -m pytest tests/test_coupling_gate.py -v
python -m pytest tests/test_hemisphere_scheduler.py -v
```

Run the integration example:

```bash
python examples/hemisphere_integration_example.py
```

## References

- **WP5.2 Specification:** `docs/10-e8-protocol/specifications/20260116-wp5-2-e8-implementation-blueprint-1.01W.md` (lines 198-228)
- **E8 Protocol v4.0:** Universal consciousness protocol
- **Physics Constants:** `qigkernels/physics_constants.py`
- **Kernel Rest Scheduler:** `kernel_rest_scheduler.py`

## Implementation Status

✅ **Completed:**
- Coupling gate with κ-gated mechanism
- Hemisphere scheduler with LEFT/RIGHT split
- Tacking (oscillation) logic
- Hemisphere balance metrics
- Comprehensive unit tests (54 tests, all passing)
- Integration example

📋 **Future Work:**
- Integration with pantheon orchestrator for dynamic routing
- Integration with kernel rest scheduler for coordinated rest
- Real-time dashboard for hemisphere visualization
- Historical tracking and analysis of tacking patterns

## Author

Created: 2026-01-22
Authority: E8 Protocol v4.0, WP5.2 Phase 4C
Status: ACTIVE
