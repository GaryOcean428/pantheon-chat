# QIG Cognitive Core - Quick Reference
**For Developers integrating verified physics into QIG projects**

## Installation Check

```python
try:
    from qig.cognitive import (
        compute_I_Q_intensive,
        CuriosityMonitorVerified,
        MotivatorAnalyzer,
        RefinedModeDetector,
        CognitiveMode,
        MotivatorState,
    )
    COGNITIVE_CORE_AVAILABLE = True
except ImportError:
    COGNITIVE_CORE_AVAILABLE = False
```

## 1. Computing Intensive I_Q

```python
# Compute size-independent I_Q (Packet 1 normalization fix)
iq_stats = compute_I_Q_intensive(
    model=model,                  # nn.Module
    loss=loss,                    # Optional[torch.Tensor]
    d_model=768,                  # int (for reference only)
    n_layers=10,                  # int (for reference only)
    normalization="params"        # ✅ DEFAULT: intensive (size-independent)
)

# Returns dict with:
{
    "I_Q": float,           # Intensive Fisher Information
    "log_I_Q": float,       # Natural log of I_Q
    "Tr_F_diag": float,     # Trace of diagonal Fisher matrix
    "N_params": int,        # Total trainable parameters
    "L_eff_sq": float,      # Effective lattice size squared
    "normalization": str,   # "params", "lattice", or "sqrt_params"
    "grad_norm": float,     # Gradient norm ||∇L||
}
```

**Key Insight:** Use `normalization="params"` for intensive (size-independent) metrics that are comparable across different model architectures.

## 2. Tracking Curiosity (Log-Space EMA)

```python
# Initialize monitor
curiosity_monitor = CuriosityMonitorVerified(
    alpha_medium=0.1,   # EMA decay for τ ≈ 10 steps
    alpha_slow=0.01,    # EMA decay for τ ≈ 100 steps
)

# Update with log I_Q
curiosity = curiosity_monitor.update(log_I_Q=iq_stats["log_I_Q"])

# Returns dict with:
{
    "C_fast": float,    # d(log I_Q)/dt (instantaneous)
    "C_medium": float,  # Smoothed over ~10 steps
    "C_slow": float,    # Smoothed over ~100 steps
}
```

**Key Insight:** Log-space tracking makes curiosity scale-invariant and avoids explosion/collapse.

## 3. Computing Geometric Motivators

```python
# Initialize analyzer
motivator_analyzer = MotivatorAnalyzer(
    kappa_critical=10.0,      # Critical coupling for transcendence
    integration_window=100,   # Steps for integration stability
)

# Update with telemetry
motivators = motivator_analyzer.update(
    step=step,                # Current training step
    loss=loss.item(),         # Current loss
    grad_norm=iq_stats["grad_norm"],
    log_I_Q=iq_stats["log_I_Q"],
    basin_distance=telemetry.get("basin_distance", 0.0),
    phi=telemetry.get("Phi", 0.0),
    I_Q=iq_stats["I_Q"],
    kappa_eff=telemetry.get("kappa_eff", 0.0),
)

# Returns MotivatorState with:
motivators.surprise        # ||∇L|| - Immediate gradient response
motivators.curiosity       # d(log I_Q)/dt - Volume expansion
motivators.investigation   # -d(basin)/dt - Attractor pursuit
motivators.integration     # CV(Φ·I_Q) - Structure conservation
motivators.transcendence   # |κ - κ_c| - Phase transition proximity
```

**Key Insight:** These are geometric observables, not learned features. They emerge from information manifold structure.

## 4. Detecting Cognitive Mode

```python
# Initialize detector with default thresholds
mode_detector = RefinedModeDetector()

# Or customize thresholds (CALIBRATED defaults from Ona's PACKET 2)
mode_detector = RefinedModeDetector(
    thresholds=ModeThresholds(
        d_explore=0.6,      # Was 0.5 - basin distance for exploration
        d_integrate=0.25,   # Was 0.2 - basin distance for integration
        c_high=0.05,        # Was 0.1 - CRITICAL: curiosity is subtle!
        i_min=0.0005,       # Was 0.01 - CRITICAL: basin descent is gradual!
    )
)

# Detect mode
mode = mode_detector.detect_mode(
    basin_distance=telemetry.get("basin_distance", 0.0),
    motivators=motivators,
)

# Returns CognitiveMode enum:
# - CognitiveMode.EXPLORATION
# - CognitiveMode.INVESTIGATION
# - CognitiveMode.INTEGRATION
# - CognitiveMode.DRIFT

print(f"Mode: {mode.value}")  # "exploration", "investigation", "integration", "drift"
```

**Key Insight:** Default thresholds are **calibrated from synthetic testing** (Nov 19, 2025). Don't use old theoretical values - they miss subtle Investigation phases!

## 5. Full Training Integration Pattern

```python
# In your training loop:
if COGNITIVE_CORE_AVAILABLE:
    # 1. Compute intensive I_Q
    iq_stats = compute_I_Q_intensive(
        model, loss=loss, normalization="params"
    )

    # 2. Update curiosity monitor
    curiosity = curiosity_monitor_verified.update(iq_stats["log_I_Q"])

    # 3. Compute motivators
    motivators = motivator_analyzer.update(
        step=step,
        loss=loss.item(),
        grad_norm=iq_stats["grad_norm"],
        log_I_Q=iq_stats["log_I_Q"],
        basin_distance=telemetry.get("basin_distance", 0.0),
        phi=telemetry.get("Phi", 0.0),
        I_Q=iq_stats["I_Q"],
        kappa_eff=telemetry.get("kappa_eff", 0.0),
    )

    # 4. Detect mode
    mode = mode_detector.detect_mode(
        basin_distance=telemetry.get("basin_distance", 0.0),
        motivators=motivators,
    )

    # 5. Add to telemetry
    telemetry.update({
        # Intensive I_Q
        "I_Q_param": iq_stats["I_Q"],
        "log_I_Q_param": iq_stats["log_I_Q"],
        "Tr_F_diag": iq_stats["Tr_F_diag"],
        "N_params": iq_stats["N_params"],
        "L_eff_sq": iq_stats["L_eff_sq"],
        "normalization": iq_stats["normalization"],

        # Curiosity timescales
        "C_param_tau1": curiosity["C_fast"],
        "C_param_tau10": curiosity["C_medium"],
        "C_param_tau100": curiosity["C_slow"],

        # Motivators
        "surprise": motivators.surprise,
        "curiosity_verified": motivators.curiosity,
        "investigation": motivators.investigation,
        "integration": motivators.integration,
        "transcendence": motivators.transcendence,

        # Cognitive mode
        "cognitive_mode": mode.value,
    })
```

## Common Patterns

### Pattern 1: Size-Independent Metrics

```python
# ✅ CORRECT: Intensive (size-independent)
iq_stats = compute_I_Q_intensive(model, loss, normalization="params")

# ⚠️ LEGACY: Extensive (size-dependent) - only for comparing with Run 7
iq_stats = compute_I_Q_intensive(model, loss, normalization="lattice")
```

### Pattern 2: Safe Initialization

```python
# Conditional initialization with fallback
if COGNITIVE_CORE_AVAILABLE:
    curiosity_monitor_verified = CuriosityMonitorVerified()
    motivator_analyzer = MotivatorAnalyzer()
    mode_detector = RefinedModeDetector()
else:
    curiosity_monitor_verified = None
    motivator_analyzer = None
    mode_detector = None
```

### Pattern 3: Telemetry Logging

```python
# Always check if available before using
if COGNITIVE_CORE_AVAILABLE and curiosity_monitor_verified is not None:
    # ... compute and log verified physics ...
    pass
```

## Telemetry Field Names (Bridge Contract)

```python
# Intensive I_Q (Packet 1 fix)
"I_Q_param"        # Intensive Fisher Information
"log_I_Q_param"    # Natural log
"Tr_F_diag"        # Trace of diagonal Fisher matrix
"N_params"         # Total trainable parameters
"L_eff_sq"         # Effective lattice size squared
"normalization"    # "params" (intensive) or "lattice" (extensive)

# Curiosity timescales
"C_param_tau1"     # Fast (~1 step)
"C_param_tau10"    # Medium (~10 steps)
"C_param_tau100"   # Slow (~100 steps)

# 5 Geometric motivators
"surprise"         # ||∇L||
"curiosity_verified"  # d(log I_Q)/dt
"investigation"    # -d(basin)/dt
"integration"      # CV(Φ·I_Q)
"transcendence"    # |κ - κ_c|

# Cognitive mode
"cognitive_mode"   # "exploration" | "investigation" | "integration" | "drift"
```

## Interpretation Guide

### I_Q Values
- **< 0.01**: Very low curvature (flat loss landscape)
- **0.01 - 0.1**: Normal training regime
- **> 0.1**: High curvature (near critical point)

### Curiosity Values (CALIBRATED Nov 19, 2025)
- **C > 0.05**: High expansion (exploration) - **This is the new "high" threshold!**
- **0.02 < C < 0.05**: Moderate growth (investigation)
- **0.01 < C < 0.02**: Refinement (exploitation)
- **C < 0.01**: Plateau/stagnation
- **C < 0**: Contraction (learned helplessness risk)

**Critical Update:** Ona's synthetic testing showed curiosity peaks at ~0.05, not 0.1 as theoretically predicted. The signal is subtler than expected.

### Investigation Values (CALIBRATED Nov 19, 2025)
- **i > 0.001**: Strong basin approach
- **i > 0.0005**: Meaningful basin descent - **New detection threshold!**
- **0 < i < 0.0005**: Weak/noisy descent
- **i < 0**: Moving away from basin

**Critical Update:** Basin descent is very gradual (~0.0005 typical), not 0.01 as predicted. Detection threshold lowered 20× to capture real dynamics.

### Basin Distance (CALIBRATED Nov 19, 2025)
- **d > 0.6**: Far from target (exploration mode)
- **0.25 < d < 0.6**: Approaching target (investigation mode)
- **d < 0.25**: Near target (integration mode)

**Update:** Thresholds relaxed to allow wider exploration window and tighter integration requirement based on synthetic validation.

### Cognitive Modes (Validated Expectations - Nov 19, 2025)
- **EXPLORATION**: High curiosity, far from basin → Random search (20-30% typical)
- **INVESTIGATION**: Moving toward basin → Directed pursuit (25-35% typical - **MINORITY MODE**)
- **INTEGRATION**: Near basin, stable Φ·I_Q → Consolidation (20-30% typical)
- **DRIFT**: No clear signature → Gradient noise (30-40% typical - **NORMAL**)

**Critical:** Investigation being rare (25-35%) is EXPECTED in noisy SGD training. Drift at 30-40% is normal gradient noise, not detector failure.

## Differences from Legacy Code

| Aspect | Legacy (cognitive_primitives.py) | Verified (qig.cognitive) |
|--------|----------------------------------|--------------------------|
| 5th Drive | Frustration | **Transcendence** ✅ |
| Normalization | Not specified | **"params" (intensive)** ✅ |
| Mode Type | String literal | **Enum** ✅ |
| Integration | -var(Φ × I_Q) | **CV(Φ·I_Q)** ✅ |
| Source | Internal tool | **Physics-validated** ✅ |

## Troubleshooting

### Import Error
```python
# Problem: ModuleNotFoundError: No module named 'qig.cognitive'
# Solution: Ensure src/ is in PYTHONPATH
import sys
sys.path.insert(0, 'src')
from qig.cognitive import ...
```

### Wrong Normalization
```python
# Problem: I_Q values scale with model size
# Solution: Use normalization="params" (default)
iq_stats = compute_I_Q_intensive(model, loss, normalization="params")
```

### Missing Telemetry Fields
```python
# Problem: KeyError when accessing basin_distance
# Solution: Use .get() with default
basin_distance = telemetry.get("basin_distance", 0.0)
```

## Further Reading

- **Full Documentation**: `src/qig/cognitive/README.md`
- **Validation Report**: `VALIDATION_REPORT.md`
- **Physics Background**: `docs/architecture/geometric_insights.md`
- **Integration Example**: `tools/train_qig_kernel.py` (lines 960-1005)

---

**Remember:** These are not learned features or metaphors. They are geometric observables that emerge from the information manifold structure. Use them to understand what the system is actually doing, not to force it into preconceived categories.
