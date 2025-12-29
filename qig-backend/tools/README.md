# QIG Backend Tools

Quality assurance and validation tools for QIG implementations.

## Tools

### geometric_purity_checker.py

AST-based checker that enforces geometric purity requirements:
- Detects forbidden patterns (cosine_similarity, np.linalg.norm)
- Validates Fisher-Rao distance usage
- Integrates with pre-commit hooks and CI/CD

**Usage:**
```bash
# Check single file
python geometric_purity_checker.py path/to/file.py

# Check directory
python geometric_purity_checker.py qig-backend/

# Errors only
python geometric_purity_checker.py --errors-only qig-backend/

# JSON output for CI
python geometric_purity_checker.py --json qig-backend/ > results.json
```

### substrate_independence_validator.py

Validates substrate independence by comparing β-functions and κ values across substrates (physics, semantic, biological).

**Usage:**
```bash
# Compare physics and semantic
python substrate_independence_validator.py \
    --semantic beta_results.json \
    --output comparison.json

# Generate publication plot
python substrate_independence_validator.py \
    --semantic beta_results.json \
    --plot figure.png

# With biological substrate
python substrate_independence_validator.py \
    --semantic semantic.json \
    --biological biological.json \
    --output full_comparison.json \
    --plot publication_figure.png
```

**Input JSON Format:**
```json
{
  "kappa_star": 64.21,
  "kappa_star_error": 0.92,
  "beta_emergence": 0.443,
  "beta_plateau": -0.013,
  "beta_fixed_point": 0.013,
  "n_measurements": 108,
  "method": "DMRG",
  "source": "qig-verification"
}
```

## Integration

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: geometric-purity
      name: Geometric Purity Check
      entry: python qig-backend/tools/geometric_purity_checker.py
      language: system
      types: [python]
      pass_filenames: false
      args: [qig-backend/]
```

### CI/CD Pipeline

```yaml
# .github/workflows/qig-validation.yml
- name: Check Geometric Purity
  run: |
    python qig-backend/tools/geometric_purity_checker.py \
      --errors-only \
      qig-backend/
```

## Development

### Adding New Checks

To add new geometric purity patterns, edit `FORBIDDEN_PATTERNS` in `geometric_purity_checker.py`:

```python
FORBIDDEN_PATTERNS = [
    ('your_pattern', 'Explanation and recommended fix'),
    # ...
]
```

### Testing

```bash
# Test geometric checker
python geometric_purity_checker.py qig-backend/tools/

# Test substrate validator with example data
python substrate_independence_validator.py \
    --semantic examples/semantic_beta.json \
    --output test_results.json
```

## References

- Consciousness Protocol v4.0 §1
- QIG Purity Requirements (docs/03-technical/QIG-PURITY-REQUIREMENTS.md)
- pantheon-chat README: Contributing → Geometric Purity Requirements
