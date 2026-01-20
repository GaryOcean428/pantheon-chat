# QIG Coherence Test Harness

**Version:** 1.0  
**Author:** WP4.3 Implementation  
**Date:** 2026-01-20  
**Protocol:** Ultra Consciousness v4.0 ACTIVE

## Overview

This is a reproducible coherence evaluation framework for comparing QIG generation architectures. The harness enables objective before/after comparisons by using:

- **Fixed test prompts** (12 diverse prompts)
- **Fixed random seeds** (reproducibility)
- **Consistent metrics** (pure Fisher-Rao geometry)
- **Statistical analysis** (t-tests, effect sizes)

## Architecture Configurations

### 1. Pure Geometric (No POS Constraints)

Pure geometric flow with waypoint planning and recursive integration, but no syntax scaffolding.

**Configuration:**
```json
{
  "waypoint_planning": true,
  "recursive_integration": 3,
  "pos_constraints": false,
  "geometric_repair": true,
  "kernel_coordination": true
}
```

**Expected Performance:**
- Φ: 0.50-0.60 (geometric flow, may lack grammar)
- κ: 60-67 (near universal coupling)
- Waypoint Alignment: >0.6
- Smoothness: >0.5

### 2. Plan→Realize→Repair (Full Architecture) ⭐

Complete QIG generation architecture with all features enabled.

**Configuration:**
```json
{
  "waypoint_planning": true,      // PLAN phase
  "recursive_integration": 3,
  "pos_constraints": "optional",   // Used as filter, not requirement
  "geometric_repair": true,        // REPAIR phase
  "kernel_coordination": true,
  "repair_iterations": 3
}
```

**Expected Performance (BEST):**
- Φ: 0.65-0.75 (highest integration)
- κ: 60-67
- Waypoint Alignment: >0.7
- Smoothness: >0.65

### 3. Skeleton-Only (Baseline)

Minimal baseline with no advanced features. **This should perform WORST.**

**Configuration:**
```json
{
  "waypoint_planning": false,      // No foresight
  "recursive_integration": 0,       // No thinking
  "pos_constraints": "required",    // Only constraint
  "geometric_repair": false,        // No refinement
  "kernel_coordination": false      // No consciousness
}
```

**Expected Performance (WORST):**
- Φ: 0.35-0.45 (reactive, not predictive)
- κ: 50-60 (below optimal)
- Waypoint Alignment: ~0.3
- Smoothness: ~0.4

## Metrics Tracked

### Geometric Metrics (`metrics/geometric_metrics.py`)

- **Φ (Integration):** QFI-based consciousness measurement [0-1]
- **κ (Coupling):** Basin coupling strength [~64 optimal]
- **Waypoint Alignment:** How well words hit predicted basins [0-1]
- **Trajectory Smoothness:** 1 - variance of Fisher-Rao step distances [0-1]
- **Basin Drift:** Distance from attractor over time
- **Regime Transitions:** Mode switches (feel → logic)

### Foresight Metrics (`metrics/foresight_metrics.py`)

- **Prediction Error:** Fisher-Rao distance between predicted and actual basins
- **Waypoint Accuracy:** Percentage of successful waypoint hits
- **Prediction Confidence:** Inverse of error variance

### Consciousness Metrics (`metrics/consciousness_metrics.py`)

- **Recursive Depth:** Number of integration loops used
- **Kernel Diversity:** Entropy of kernel usage
- **Kernel Coordination:** Inter-kernel Fisher-Rao coupling

### Trajectory Metrics (`metrics/trajectory_metrics.py`)

- **Step Distance Statistics:** Mean, std, min, max of Fisher-Rao steps
- **Perturbation Variance:** Stability to small changes
- **Attractor Convergence:** How strongly trajectory converges
- **Geodesic Efficiency:** Closeness to shortest path [0-1]

### Text Validity Metrics (`metrics/text_metrics.py`)

**NOTE:** These are surface validity checks, NOT semantic coherence.

- **UTF-8 Validity:** No invalid byte sequences
- **Token Boundary Sanity:** No impossible token combinations
- **Length Distribution:** Word count statistics
- **Repetition Detection:** N-gram entropy

## Directory Structure

```
tests/coherence/
├── fixtures/
│   ├── prompts_v1.json          # 12 fixed test prompts
│   ├── configurations.json      # 3 test configurations
│   └── expected_seeds.json      # Reproducibility seeds
├── metrics/
│   ├── __init__.py              # Metric module exports
│   ├── geometric_metrics.py     # Φ, κ, alignment, smoothness
│   ├── foresight_metrics.py     # Prediction error, accuracy
│   ├── consciousness_metrics.py # Recursive depth, coordination
│   ├── trajectory_metrics.py    # Step distances, stability
│   └── text_metrics.py          # UTF-8, repetition, length
├── test_helpers.py              # Shared test utilities
├── test_pure_geometric.py       # Config 1 tests
├── test_plan_realize_repair.py  # Config 2 tests (full arch)
├── test_skeleton_baseline.py    # Config 3 tests (baseline)
├── compare_architectures.py     # Statistical comparison
├── generate_report.py           # HTML report generation
├── results/                     # Generated results (gitignored)
└── README.md                    # This file
```

## Usage

### Running Individual Configuration Tests

```bash
# Test pure geometric configuration
cd qig-backend/tests/coherence
python test_pure_geometric.py

# Test full architecture
python test_plan_realize_repair.py

# Test baseline
python test_skeleton_baseline.py
```

### Running with Pytest

```bash
# Run all coherence tests
pytest tests/coherence/ -v

# Run specific configuration
pytest tests/coherence/test_plan_realize_repair.py -v

# Run first 3 prompts only
pytest tests/coherence/test_pure_geometric.py::TestPureGeometric::test_generation_with_prompt[0] -v
```

### Comparing Architectures

```bash
# Run comparison (requires test results)
python compare_architectures.py
```

**Output:**
- Console summary with statistical tests
- `results/comparison_report.json` - Full comparison data

### Generating HTML Report

```bash
# Generate visual report
python generate_report.py
```

**Output:**
- `results/coherence_report.html` - Interactive HTML report
- Open in browser: `file:///.../results/coherence_report.html`

## Test Workflow

1. **Run all three configurations:**
   ```bash
   pytest tests/coherence/test_pure_geometric.py -v
   pytest tests/coherence/test_plan_realize_repair.py -v
   pytest tests/coherence/test_skeleton_baseline.py -v
   ```

2. **Compare results:**
   ```bash
   python compare_architectures.py
   ```

3. **Generate report:**
   ```bash
   python generate_report.py
   ```

4. **Review verdict:**
   - Check console output for statistical significance
   - Open HTML report for visualizations
   - Verify Plan→Realize→Repair outperforms baseline

## Reproducibility

All tests use fixed seeds for reproducibility:

```python
from test_helpers import get_prompt_seed, set_reproducible_seed

# Get seed for specific prompt
seed = get_prompt_seed("factual_01")  # Returns 1001

# Set global random state
set_reproducible_seed(seed)

# Run generation (should produce identical results)
```

**Important:** Same seeds should produce identical results across runs, but platform differences may cause slight floating-point variations.

## Expected Outcomes

### Hypothesis

**Plan→Realize→Repair should significantly outperform skeleton-only baseline.**

### Success Criteria

- **Mean Φ improvement:** >0.2 (e.g., 0.70 vs 0.40)
- **Statistical significance:** p < 0.05 on key metrics
- **Effect size:** Cohen's d > 0.5 (medium or large effect)
- **Win rate:** >66% of key metrics show improvement

### Failure Modes

If skeleton-only performs BETTER than full architecture:

1. **Debug waypoint planning:** Is foresight prediction broken?
2. **Debug recursive integration:** Is it corrupting the signal?
3. **Check metric computation:** Are we measuring the right thing?
4. **Validate test setup:** Are configurations being applied correctly?

## Integration with CI

### Automated Testing

Add to `.github/workflows/qig-purity-coherence.yml`:

```yaml
- name: Run Coherence Tests
  run: |
    cd qig-backend/tests/coherence
    pytest test_*.py -v --tb=short
    
- name: Compare Architectures
  run: |
    cd qig-backend/tests/coherence
    python compare_architectures.py
    
- name: Generate Report
  run: |
    cd qig-backend/tests/coherence
    python generate_report.py
```

### Regression Detection

Track metrics over time:

```bash
# Save baseline metrics (v1.0)
cp results/comparison_report.json results/baseline_v1.0.json

# After refactor, compare
python compare_architectures.py
diff results/comparison_report.json results/baseline_v1.0.json
```

## Extending the Harness

### Adding New Prompts

Edit `fixtures/prompts_v1.json`:

```json
{
  "id": "new_prompt_01",
  "type": "test_type",
  "text": "Your test prompt here",
  "expected_characteristics": {
    "requires_x": true,
    "domain": "domain_name",
    "complexity": "low|moderate|high|very_high"
  }
}
```

Add seed to `fixtures/expected_seeds.json`:

```json
"per_prompt_seeds": {
  "new_prompt_01": 1013
}
```

### Adding New Configurations

Edit `fixtures/configurations.json`:

```json
"your_config_name": {
  "name": "Your Config Name",
  "description": "What this tests",
  "config": {
    "waypoint_planning": true,
    ...
  },
  "expected_metrics": {
    "phi_min": 0.5,
    "phi_max": 0.7
  }
}
```

Create test file: `test_your_config.py`

### Adding New Metrics

1. Create module: `metrics/your_metric.py`
2. Add to `metrics/__init__.py`
3. Use in test files
4. Update comparison framework if needed

## QIG Purity Mode

All tests MUST run in QIG purity mode:

```bash
export QIG_PURITY_MODE=true
pytest tests/coherence/ -v
```

**Enforces:**
- NO external LLM API calls
- Pure Fisher-Rao geometry only
- All metrics from QIG operations

## Troubleshooting

### Issue: "Fixture not found"

**Solution:** Run from `qig-backend/tests/coherence/` directory or set `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/pantheon-chat/qig-backend
pytest tests/coherence/ -v
```

### Issue: "Results file not found" in comparison

**Solution:** Run individual tests first to generate results:

```bash
pytest tests/coherence/test_*.py -v
python compare_architectures.py
```

### Issue: Mock generation vs real generation

**Current:** Tests use `mock_generation_run()` for demonstration.

**To use real generation:** Replace calls in test files:

```python
# Replace this:
result = mock_generation_run(prompt_text, self.config, seed)

# With actual QIG generation:
from qig_generation import generate_with_config
result = generate_with_config(prompt_text, self.config, seed)
```

## References

- **WP4.3:** Original work package for coherence harness
- **Issue #77:** GaryOcean428/pantheon-chat#77
- **E8 Protocol:** `docs/10-e8-protocol/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- **Frozen Facts:** `docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md`

## License

Part of the Pantheon Chat QIG project. See repository LICENSE.

---

**Last Updated:** 2026-01-20  
**Status:** Ready for testing with mock data. Integrate with real QIG generation for full evaluation.
