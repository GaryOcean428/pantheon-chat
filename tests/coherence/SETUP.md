# Coherence Test Harness - Setup Instructions

## Dependencies

The coherence test harness requires Python 3.8+ with the following packages:

```bash
# Core dependencies
numpy>=1.24.0
scipy>=1.11.0

# Optional (for actual generation, not mocks)
torch>=2.0.0
psycopg2-binary>=2.9.9
```

## Installation

### Option 1: Use existing qig-backend environment

```bash
cd qig-backend
pip install -r requirements.txt
```

### Option 2: Minimal install for testing only

```bash
pip install numpy scipy
```

### Option 3: Virtual environment (recommended)

```bash
cd tests/coherence
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy
```

## Running Tests

### Quick Start (Mock Generation)

The tests use mock generation by default, so you can run them without a full QIG backend:

```bash
# Install minimal dependencies
pip install numpy scipy

# Run all tests
python3 tests/coherence/run_all_tests.py

# Or run individually
python3 tests/coherence/test_skeleton_baseline.py
python3 tests/coherence/test_pure_geometric.py
python3 tests/coherence/test_plan_realize_repair.py

# Compare results
python3 tests/coherence/compare_architectures.py
```

### Full Integration (Actual Generation)

To use actual QIG generation instead of mocks:

1. Install full qig-backend dependencies
2. Modify test runners to import and use actual generation modules
3. Replace `mock_generation_result()` calls with actual generation

Example modification in test files:

```python
# Replace this:
from test_utils import mock_generation_result
result = mock_generation_result(prompt_text, config['settings'])

# With this:
from qig_generation import generate_with_config
result = generate_with_config(prompt_text, config['settings'])
```

## CI Integration

### GitHub Actions Workflow

Add to `.github/workflows/coherence-tests.yml`:

```yaml
name: Coherence Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        cd tests/coherence
        pip install numpy scipy
    
    - name: Run coherence tests
      run: |
        python3 tests/coherence/run_all_tests.py
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: coherence-results
        path: tests/coherence/results/
    
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = JSON.parse(
            fs.readFileSync('tests/coherence/results/comparison_results.json', 'utf8')
          );
          // Format and post comment with results
```

## Troubleshooting

### ModuleNotFoundError: numpy

```bash
pip install numpy scipy
```

### Tests fail with "Results file not found"

Run the individual test scripts before running comparison:

```bash
python3 tests/coherence/test_skeleton_baseline.py
python3 tests/coherence/test_pure_geometric.py
python3 tests/coherence/test_plan_realize_repair.py
python3 tests/coherence/compare_architectures.py
```

### Import errors from qig-backend

The tests mock generation by default. To use real generation:

1. Ensure qig-backend is installed
2. Modify test runners to import actual generation modules
3. Configure paths in test_utils.py

## Current Status

✅ **Infrastructure**: Complete
- Directory structure created
- Fixtures versioned (prompts v1.0, configs v1.0, seeds v1.0)
- All 5 metric modules implemented

✅ **Metrics**: Complete
- Geometric metrics (Φ, κ, drift, stability)
- Foresight metrics (alignment, prediction error)
- Trajectory metrics (smoothness, deviation)
- Text metrics (UTF-8, repetition, entropy)
- Consciousness metrics (depth, coordination)

✅ **Test Runners**: Complete
- Pure geometric configuration
- Plan→Realize→Repair configuration
- Skeleton baseline configuration
- Comparison framework
- HTML report generation

⚠️ **Integration**: Pending
- Currently uses mock generation
- Need to connect to actual QIG generation
- CI workflow pending

## Next Steps

1. **Install dependencies** and verify tests run with mocks
2. **Integrate actual generation** by replacing mock calls
3. **Add CI workflow** for automated testing
4. **Track metrics over time** for regression detection
5. **Add visualization** dashboard for trend analysis
