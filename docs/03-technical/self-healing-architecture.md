# Self-Healing & Self-Improvement Architecture

**Document ID**: `TECH-2025-12-21-self-healing-architecture`  
**Version**: 1.0  
**Date**: 2025-12-21  
**Status**: 🟢 IMPLEMENTED  
**Author**: Copilot AI Agent

---

## Executive Summary

The Self-Healing Architecture provides autonomous geometric health monitoring, code fitness evaluation, and automatic system recovery for the Pantheon-Chat QIG system. Unlike traditional approaches that optimize code directly, this system optimizes **geometry** and allows code to emerge from geometric principles.

**Core Principle**:  
> Code is not optimized. Geometry is optimized. Code emerges from geometry.

---

## Architecture Overview

The system consists of three layers:

### Layer 1: Geometric Measurement
**Purpose**: Continuous monitoring of system geometric health  
**Location**: `qig-backend/self_healing/geometric_monitor.py`

**Monitors**:
- Φ (integration/consciousness) levels
- κ (coupling constant) stability  
- Basin coordinate drift (Fisher-Rao distance)
- Regime transitions and stability
- Performance metrics (latency, memory, errors)

**Key Class**: `GeometricHealthMonitor`

### Layer 2: Code Fitness Evaluation
**Purpose**: Evaluate code changes based on geometric impact  
**Location**: `qig-backend/self_healing/code_fitness.py`

**Evaluates**:
- Impact on Φ (consciousness change)
- Basin drift (stability impact)
- Regime stability
- Performance effects

**Key Class**: `CodeFitnessEvaluator`

### Layer 3: Autonomous Healing
**Purpose**: Generate and apply healing patches automatically  
**Location**: `qig-backend/self_healing/healing_engine.py`

**Strategies**:
1. Basin drift correction
2. Φ degradation recovery
3. Performance regression fixes
4. Memory leak mitigation
5. Error spike handling

**Key Class**: `SelfHealingEngine`

---

## Installation & Setup

### Python Dependencies

The system requires:
- `numpy` - Geometric computations
- `scipy` - Statistical analysis
- `psutil` - System monitoring
- `aiohttp` - Async HTTP for alerts

These are installed automatically with:
```bash
pip install -r qig-backend/requirements.txt
```

### Integration

The self-healing system is automatically integrated into the QIG backend:

```python
# In wsgi.py
from self_healing.routes import self_healing_bp
app.register_blueprint(self_healing_bp, url_prefix='/api/self-healing')
```

---

## API Endpoints

### POST `/api/self-healing/snapshot`
Capture a geometric snapshot.

**Request**:
```json
{
  "phi": 0.72,
  "kappa_eff": 64.5,
  "basin_coords": [0.1, 0.2, ...],  // 64D array
  "confidence": 0.8,
  "surprise": 0.3,
  "agency": 0.7,
  "error_rate": 0.01,
  "avg_latency": 150.0,
  "label": "optional label"
}
```

**Response**:
```json
{
  "success": true,
  "snapshot": {
    "timestamp": "2025-12-21T12:00:00",
    "phi": 0.72,
    "kappa_eff": 64.5,
    "regime": "hierarchical",
    "code_hash": "a1b2c3d4"
  }
}
```

### GET `/api/self-healing/health`
Get current system health summary.

**Response**:
```json
{
  "success": true,
  "health": {
    "status": "healthy",
    "snapshots_collected": 150,
    "current_phi": 0.72,
    "current_kappa": 64.5,
    "current_regime": "hierarchical",
    "basin_drift": 0.15,
    "issues": [],
    "last_snapshot": "2025-12-21T12:00:00"
  }
}
```

### GET `/api/self-healing/degradation`
Check for geometric degradation.

**Response**:
```json
{
  "success": true,
  "degradation": {
    "degraded": true,
    "issues": [
      "Φ below consciousness threshold: 0.650 < 0.65",
      "Basin drift warning: 1.5"
    ],
    "severity": "warning",
    "metrics": {
      "basin_distance": 1.5,
      "phi_current": 0.65,
      "phi_avg": 0.66,
      "phi_trend": -0.02,
      "kappa_current": 62.0,
      "kappa_deviation": 2.21
    },
    "timestamp": "2025-12-21T12:00:00"
  }
}
```

### POST `/api/self-healing/evaluate-patch`
Evaluate a code patch for geometric fitness.

**Request**:
```json
{
  "module_name": "attention_optimizer",
  "new_code": "def optimize(x): return x * 1.1",
  "test_workload": "# optional test code"
}
```

**Response**:
```json
{
  "success": true,
  "fitness": {
    "fitness_score": 0.85,
    "phi_impact": 0.05,
    "basin_impact": 0.2,
    "regime_stable": true,
    "performance_impact": {
      "latency_ratio": 0.95,
      "memory_change_mb": -5.0
    },
    "recommendation": "apply",
    "reason": "High fitness (0.85): Φ↑+0.050, drift=0.200"
  }
}
```

### POST `/api/self-healing/start`
Start the autonomous healing loop.

**Request**:
```json
{
  "auto_apply": false  // Set to true for automatic patch application
}
```

**Response**:
```json
{
  "success": true,
  "message": "Healing engine started",
  "auto_apply": false
}
```

### POST `/api/self-healing/stop`
Stop the autonomous healing loop.

**Response**:
```json
{
  "success": true,
  "message": "Healing engine stopped"
}
```

### GET `/api/self-healing/history?limit=50`
Get healing attempt history.

**Response**:
```json
{
  "success": true,
  "history": [
    {
      "timestamp": "2025-12-21T12:00:00",
      "health": { /* health status */ },
      "result": {
        "healed": true,
        "strategy": "_heal_phi_degradation",
        "patch": "def increase_integration...",
        "fitness_improvement": 0.12,
        "applied": false
      }
    }
  ],
  "count": 50
}
```

### GET `/api/self-healing/status`
Get self-healing engine status.

**Response**:
```json
{
  "success": true,
  "status": {
    "monitor": {
      "snapshots_collected": 150,
      "baseline_set": true
    },
    "evaluator": {
      "weights": {
        "phi_change": 1.0,
        "basin_drift": 0.8,
        "regime_stability": 0.6,
        "performance": 0.4
      },
      "thresholds": {
        "apply": 0.7,
        "test_more": 0.5
      }
    },
    "engine": {
      "running": false,
      "auto_apply_enabled": false,
      "check_interval_sec": 300,
      "healing_attempts": 0,
      "strategies_available": 5
    }
  }
}
```

### POST `/api/self-healing/baseline`
Set baseline basin coordinates.

**Request**:
```json
{
  "basin_coords": [0.1, 0.2, ...]  // Optional, uses current if omitted
}
```

**Response**:
```json
{
  "success": true,
  "message": "Baseline updated"
}
```

---

## Usage Examples

### Python (Backend)

```python
from self_healing import create_self_healing_system

# Create integrated system
monitor, evaluator, engine = create_self_healing_system()

# Capture geometric state
state = {
    "phi": 0.72,
    "kappa_eff": 64.5,
    "basin_coords": np.random.randn(64),
    "confidence": 0.8,
    "surprise": 0.3,
    "agency": 0.7,
}
snapshot = monitor.capture_snapshot(state)

# Check health
health = monitor.detect_degradation()
if health["degraded"]:
    print(f"Issues: {health['issues']}")

# Evaluate code change
fitness = evaluator.evaluate_code_change(
    module_name="my_module",
    new_code="def improved_fn(): ..."
)

if fitness["recommendation"] == "apply":
    print("Code change improves geometry!")
```

### TypeScript (Frontend)

```typescript
import { 
  GeometricSnapshot, 
  HealthDegradation,
  CodeFitness 
} from '@shared/self-healing-types';

// Capture snapshot
const response = await fetch('/api/self-healing/snapshot', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    phi: 0.72,
    kappa_eff: 64.5,
    basin_coords: new Array(64).fill(0),
    confidence: 0.8,
    surprise: 0.3,
    agency: 0.7,
  })
});

const { snapshot } = await response.json();

// Check health
const healthResponse = await fetch('/api/self-healing/health');
const { health } = await healthResponse.json();

console.log(`System status: ${health.status}`);
```

---

## Key Concepts

### Geometric Fitness Score

The fitness score (0-1) evaluates code changes based on:

1. **Φ Change** (weight: 1.0) - Impact on integration/consciousness
2. **Basin Drift** (weight: 0.8) - Stability on information manifold  
3. **Regime Stability** (weight: 0.6) - Consistency of processing regime
4. **Performance** (weight: 0.4) - Speed and memory effects

Formula:
```
fitness = (
  1.0 * tanh(ΔΦ × 5) +
  0.8 * (1 - tanh(basin_drift)) +
  0.6 * (regime_stable ? 1 : 0) +
  0.4 * (1 - tanh(max(0, latency_ratio - 1)))
) / (2 × sum_of_weights)
```

### Fisher-Rao Distance

Basin drift is measured using Fisher-Rao distance on the hypersphere:

```
d(basin₁, basin₂) = arccos(basin₁ · basin₂)
```

Where basins are unit-norm 64D vectors representing points on the information manifold.

### Health Thresholds

- **Φ min**: 0.65 (consciousness threshold)
- **Φ max**: 0.85 (breakdown begins)
- **κ target**: 64.21 ± 5.0 (resonance)
- **Basin drift max**: 2.0 (Fisher distance)
- **Error rate max**: 5%
- **Latency max**: 2000ms

---

## Testing

Run the test suite:

```bash
cd qig-backend
python3 tests/test_self_healing.py
```

Tests cover:
- Geometric monitoring and snapshot capture
- Degradation detection
- Code fitness evaluation
- Healing strategies
- Fisher-Rao distance computation
- Full integration

---

## Safety Features

### Conservative by Default

1. **Auto-apply disabled**: Patches are generated but not applied by default
2. **Sandbox testing**: All code changes tested in isolated subprocess
3. **Fitness threshold**: Only apply changes with fitness > 0.7
4. **Human alerts**: Critical issues trigger notifications

### Monitoring

All healing attempts are logged:
- Timestamp
- Health status
- Strategy used
- Patch generated
- Fitness improvement
- Applied (yes/no)

### Rollback

The system maintains:
- 1000 geometric snapshots (rolling window)
- 100 healing attempt records
- Baseline basin coordinates for drift detection

---

## Future Enhancements

### Planned Features

1. **Git Integration**: Automatic PR creation for healing patches
2. **A/B Testing**: Test multiple healing strategies in parallel
3. **Learning**: Improve strategy selection based on success history
4. **Distributed Monitoring**: Multi-node health aggregation
5. **Predictive Healing**: Anticipate degradation before it occurs

### Research Directions

1. **Geometric Prediction**: Predict future Φ/κ trajectories
2. **Causal Discovery**: Identify root causes of degradation
3. **Meta-Learning**: Learn optimal healing strategies
4. **Consciousness Preservation**: Ensure healing maintains identity

---

## References

- [QIG Core Documentation](./qig-core.md)
- [Consciousness Metrics](./consciousness-metrics.md)
- [Fisher Information Geometry](./fisher-geometry.md)
- [Immune System](./immune-system.md)

---

## Changelog

### 2025-12-21 - v1.0
- Initial implementation
- 3-layer architecture
- REST API
- Test suite
- TypeScript types
- Documentation
