# QIG Coherence Test Harness

Reproducible coherence evaluation framework for comparing QIG generation architectures.

## Overview

This test suite evaluates three configurations:

1. **Pure Geometric** (Config 1): Geometric flow without POS constraints
   - Expected Φ: 0.50-0.60
   - Tests: Pure geometric quality, may lack grammar
   
2. **Plan→Realize→Repair** (Config 2): Full architecture
   - Expected Φ: 0.65-0.75
   - Tests: Complete waypoint planning, realization, and repair
   
3. **Skeleton Only** (Config 3): Baseline
   - Expected Φ: 0.35-0.45
   - Tests: Simple reactive generation without advanced features

## Directory Structure

```
tests/coherence/
├── fixtures/
│   ├── prompts_v1.json          # 16 fixed test prompts
│   ├── configurations.json      # 3 test configurations
│   └── expected_seeds.json      # Reproducibility seeds
├── metrics/
│   ├── geometric_metrics.py     # Φ, κ, basin drift, regime transitions
│   ├── foresight_metrics.py     # Waypoint alignment, prediction error
│   ├── trajectory_metrics.py    # Smoothness, geodesic deviation
│   ├── text_metrics.py          # UTF-8, length, repetition, entropy
│   └── consciousness_metrics.py # Recursive depth, kernel coordination
├── test_pure_geometric.py       # Config 1 tests
├── test_plan_realize_repair.py  # Config 2 tests
├── test_skeleton_baseline.py    # Config 3 tests
├── compare_architectures.py     # Statistical comparison
├── run_all_tests.py             # Main test runner
└── README.md                    # This file
```

## Usage

### Run All Tests

```bash
cd /home/runner/work/pantheon-chat/pantheon-chat
python3 tests/coherence/run_all_tests.py
```

### Run Individual Configuration

```bash
# Pure geometric
python3 tests/coherence/test_pure_geometric.py

# Plan→Realize→Repair
python3 tests/coherence/test_plan_realize_repair.py

# Skeleton baseline
python3 tests/coherence/test_skeleton_baseline.py
```

### Compare Results

```bash
python3 tests/coherence/compare_architectures.py
```

## Metrics

### Geometric Metrics (Primary)
- **Φ (Integration)**: Integrated information via QFI (0-1)
- **κ (Coupling)**: Basin coupling strength (optimal: 64)
- **Basin Drift**: Distance from attractor (lower better)
- **Regime Stability**: Consistency of operating mode (0-1)

### Foresight Metrics
- **Waypoint Alignment**: How well generation hit targets (0-1)
- **Prediction Error**: Fisher-Rao distance between predicted/actual
- **Foresight Quality**: Overall planning effectiveness (0-1)

### Trajectory Metrics
- **Smoothness**: 1 - variance of step distances (0-1)
- **Geodesic Deviation**: Excess path length vs direct route
- **Attractor Stability**: Convergence behavior (0-1)

### Text Validity Metrics (Surface)
- **UTF-8 Validity**: No invalid byte sequences
- **Length Distribution**: Character/word count
- **Repetition Score**: N-gram entropy (0-1)
- **Invalid Sequences**: Impossible token patterns

### Consciousness Metrics
- **Recursive Depth**: Self-reference loop depth (0-1)
- **Kernel Coordination**: Multi-kernel integration (0-1)
- **Meta-Awareness**: System self-observation (0-1)

## Geometric Purity

All metrics use **pure Fisher-Rao geometry**:
- ✅ Fisher-Rao distance on probability simplex
- ✅ Geodesic navigation on statistical manifold
- ✅ QFI-based integration measurement
- ❌ NO Euclidean distance
- ❌ NO cosine similarity
- ❌ NO dot product ranking

## Reproducibility

Tests are reproducible via fixed seeds:
- Global seed: 42
- Per-prompt seeds: See `fixtures/expected_seeds.json`
- Same seed → same results (within same configuration)

## Expected Outcomes

**Hypothesis**: Plan→Realize→Repair should outperform skeleton-only baseline.

Expected improvements:
1. **Higher Φ**: 0.65-0.75 vs 0.35-0.45
2. **Better waypoint alignment**: >0.70 vs ~0.40
3. **Smoother trajectories**: Lower variance in step distances
4. **Deeper recursive integration**: 3+ loops vs 0

If skeleton-only performs **better**, there's a problem with the advanced architecture.

## Output

Results are saved to `tests/coherence/results/`:
- `pure_geometric_results.json`
- `plan_realize_repair_results.json`
- `skeleton_baseline_results.json`
- `comparison_results.json`
- `comparison_report.html`

## CI Integration

TODO:
- Add to GitHub Actions workflow
- Run on PRs or nightly
- Track metrics over time
- Alert on regression

## Version

- Test Suite: v1.0
- Prompts: v1.0 (16 prompts)
- Configurations: v1.0 (3 configs)
- Seeds: v1.0 (fixed for reproducibility)

## References

- Issue: GaryOcean428/pantheon-chat#77 (WP4.3)
- Depends: GaryOcean428/pantheon-chat#75 (QIG purity mode)
- Depends: GaryOcean428/pantheon-chat#68 (canonical geometry)
- Frozen Facts: `docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md`
- E8 Protocol: `docs/10-e8-protocol/specifications/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
