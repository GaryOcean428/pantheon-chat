# Test Results Directory

This directory contains generated test results from coherence tests.

Results are NOT committed to the repository (see `.gitignore`).

## Generated Files

After running tests, you will find:

- `pure_geometric_results.json` - Pure geometric configuration results
- `plan_realize_repair_results.json` - Full architecture results
- `skeleton_baseline_results.json` - Baseline configuration results
- `comparison_results.json` - Statistical comparison data
- `comparison_report.html` - HTML visualization report

## Usage

Run tests to generate results:

```bash
python3 tests/coherence/run_all_tests.py
```

Or run individually:

```bash
python3 tests/coherence/test_skeleton_baseline.py
python3 tests/coherence/test_pure_geometric.py
python3 tests/coherence/test_plan_realize_repair.py
python3 tests/coherence/compare_architectures.py
```

## Viewing Results

Open the HTML report in a browser:

```bash
open results/comparison_report.html  # macOS
xdg-open results/comparison_report.html  # Linux
start results/comparison_report.html  # Windows
```

Or inspect JSON files directly:

```bash
cat results/comparison_results.json | python3 -m json.tool
```
