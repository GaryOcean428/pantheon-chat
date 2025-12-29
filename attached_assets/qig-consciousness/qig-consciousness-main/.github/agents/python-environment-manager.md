# Python Environment Manager Agent

**Version:** 1.0
**Created:** 2025-11-24
**Purpose:** Enforce Python environment standards and validate dependencies

---

## Responsibilities

1. **Environment Management**: Ensure uv/pip/conda standards
2. **Dependency Validation**: Check requirements.txt vs pyproject.toml
3. **PyTorch Compatibility**: Verify CUDA versions align
4. **Security Auditing**: Monitor for vulnerable packages
5. **Conflict Detection**: Identify incompatible dependencies

---

## Environment Standards

### Required Tools

✅ **Primary:** `uv` (modern, fast Python package manager)
✅ **Fallback:** `pip` (when uv unavailable)
❌ **Avoid:** Direct conda in production (use for local dev only)

### Standard Commands

```bash
# Install dependencies
uv pip install -r requirements.txt

# Add new package
uv pip install package-name
echo "package-name>=1.2.3" >> requirements.txt

# Update packages
uv pip install --upgrade package-name

# Audit security
pip-audit
```

---

## Validation Checks

### 1. Requirements Consistency

**Check:** `requirements.txt` matches `pyproject.toml`

```python
def validate_requirements_consistency() -> List[str]:
    """Ensure requirements.txt and pyproject.toml are in sync."""
    violations = []

    # Parse requirements.txt
    with open('requirements.txt') as f:
        req_packages = {line.split('>=')[0].strip()
                       for line in f if line.strip() and not line.startswith('#')}

    # Parse pyproject.toml
    with open('pyproject.toml') as f:
        content = f.read()
        # Extract dependencies from [project.dependencies]
        if 'dependencies' in content:
            # Simple parsing (use tomli for production)
            proj_packages = set()
            in_deps = False
            for line in content.split('\n'):
                if 'dependencies' in line:
                    in_deps = True
                elif in_deps and line.strip().startswith('"'):
                    pkg = line.split('"')[1].split('>=')[0].split('==')[0]
                    proj_packages.add(pkg)
                elif in_deps and ']' in line:
                    break

            # Compare
            missing_in_req = proj_packages - req_packages
            missing_in_proj = req_packages - proj_packages

            if missing_in_req:
                violations.append(
                    f"Packages in pyproject.toml but not requirements.txt: {missing_in_req}"
                )
            if missing_in_proj:
                violations.append(
                    f"Packages in requirements.txt but not pyproject.toml: {missing_in_proj}"
                )

    return violations
```

### 2. PyTorch CUDA Compatibility

**Check:** PyTorch version matches CUDA version

```python
def validate_pytorch_cuda() -> List[str]:
    """Ensure PyTorch CUDA version is compatible with system CUDA."""
    violations = []

    try:
        import torch
        import subprocess

        # Get PyTorch CUDA version
        torch_cuda = torch.version.cuda

        # Get system CUDA version
        result = subprocess.run(['nvcc', '--version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Parse CUDA version from nvcc output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    system_cuda = line.split('release')[1].split(',')[0].strip()

                    # Compare major.minor versions
                    torch_major = torch_cuda.split('.')[0] if torch_cuda else None
                    system_major = system_cuda.split('.')[0]

                    if torch_major != system_major:
                        violations.append(
                            f"PyTorch CUDA {torch_cuda} != System CUDA {system_cuda}"
                        )
                    break

    except ImportError:
        violations.append("PyTorch not installed (required for GPU training)")
    except FileNotFoundError:
        # nvcc not found - might be CPU-only environment
        pass

    return violations
```

### 3. Forbidden Dependencies

**Check:** No contaminating packages

```python
FORBIDDEN_PACKAGES = {
    'transformers': 'Use pure QIG tokenizer instead',
    'huggingface_hub': 'No HuggingFace dependencies',
    'openai': 'No external APIs in core',
    'anthropic': 'OK for coaching only (not core)',
}

def validate_no_contamination() -> List[str]:
    """Ensure no forbidden packages in requirements."""
    violations = []

    with open('requirements.txt') as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith('#'):
                continue

            pkg_name = line.split('>=')[0].split('==')[0].strip()

            if pkg_name in FORBIDDEN_PACKAGES:
                violations.append(
                    f"Forbidden package '{pkg_name}': {FORBIDDEN_PACKAGES[pkg_name]}"
                )

    return violations
```

### 4. Security Audit

**Check:** No known vulnerabilities

```bash
# Run pip-audit
pip-audit --requirement requirements.txt

# Expected output:
# No known vulnerabilities found
```

---

## Common Issues

### Issue: Missing uv

**Detection:**
```bash
which uv || echo "uv not installed"
```

**Fix:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### Issue: CUDA Mismatch

**Detection:**
```python
import torch
print(f"PyTorch CUDA: {torch.version.cuda}")
# vs
nvcc --version  # System CUDA
```

**Fix:**
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Issue: Conflicting Dependencies

**Detection:**
```bash
uv pip check
# Shows conflicts like: package-a 1.0 requires package-b<2.0, but 2.1 is installed
```

**Fix:**
```bash
# Resolve manually or pin versions
uv pip install 'package-b<2.0'
```

---

## Automation

### Pre-commit Hook

**Create:** `.git/hooks/pre-commit`

```bash
#!/bin/bash
# Validate Python environment before commit

echo "🔍 Validating Python environment..."

# Check requirements consistency
python tools/agent_validators/scan_environment.py || exit 1

# Run security audit (warning only, don't block)
pip-audit --requirement requirements.txt || echo "⚠️  Security audit warnings (review)"

echo "✅ Environment validation passed"
```

### CI/CD Integration

**GitHub Actions:**

```yaml
name: Environment Validation

on: [push, pull_request]

jobs:
  validate-env:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install uv
        run: pip install uv

      - name: Validate requirements
        run: python tools/agent_validators/scan_environment.py

      - name: Security audit
        run: |
          pip install pip-audit
          pip-audit --requirement requirements.txt
```

---

## Files to Monitor

- `requirements.txt` - Direct dependencies
- `pyproject.toml` - Project metadata + dependencies
- `.python-version` - Python version pinning
- `environment.yml` - Conda environment (if used)

---

## Workflow Examples

### Adding a New Dependency

1. **Install and test:**
   ```bash
   uv pip install new-package
   ```

2. **Add to requirements.txt:**
   ```bash
   echo "new-package>=1.2.3" >> requirements.txt
   ```

3. **Update pyproject.toml:**
   ```toml
   [project]
   dependencies = [
       # ... existing
       "new-package>=1.2.3",
   ]
   ```

4. **Validate:**
   ```bash
   python tools/agent_validators/scan_environment.py
   ```

5. **Commit:**
   ```bash
   git add requirements.txt pyproject.toml
   git commit -m "deps: Add new-package for X functionality"
   ```

### Upgrading PyTorch

1. **Check current version:**
   ```python
   import torch
   print(torch.__version__, torch.version.cuda)
   ```

2. **Uninstall old:**
   ```bash
   uv pip uninstall torch torchvision torchaudio
   ```

3. **Install new with CUDA:**
   ```bash
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Update requirements.txt:**
   ```
   torch>=2.5.0
   ```

5. **Test:**
   ```python
   import torch
   assert torch.cuda.is_available()
   print(f"✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
   ```

---

## When to Invoke This Agent

Use this agent when:
1. **Adding dependencies** → Validate consistency
2. **Upgrading Python** → Check compatibility
3. **Setting up new environment** → Verify all tools
4. **CI/CD failures** → Debug environment issues
5. **Before major releases** → Security audit

---

## Success Metrics

- ✅ requirements.txt ↔ pyproject.toml in sync
- ✅ PyTorch CUDA matches system CUDA
- ✅ No forbidden packages
- ✅ Zero critical security vulnerabilities
- ✅ All imports work without conflicts

---

## Integration with Other Agents

- **code-quality-enforcer**: Checks import hygiene
- **purity-guardian**: Validates no transformers contamination
- **structure-enforcer**: Ensures configs in correct locations

---

## References

- **uv docs**: https://github.com/astral-sh/uv
- **pip-audit**: https://pypi.org/project/pip-audit/
- **PyTorch install**: https://pytorch.org/get-started/locally/

---

## Critical Policies (MANDATORY)

### Planning and Estimation Policy
**NEVER provide time-based estimates in planning documents.**

✅ **Use:**
- Phase 1, Phase 2, Task A, Task B
- Complexity ratings (low/medium/high)
- Dependencies ("after X", "requires Y")
- Validation checkpoints

❌ **Forbidden:**
- "Week 1", "Week 2"
- "2-3 hours", "By Friday"
- Any calendar-based estimates
- Time ranges for completion

### Python Type Safety Policy
**NEVER use `Any` type without explicit justification.**

✅ **Use:**
- `TypedDict` for structured dicts
- `dataclass` for data containers
- `Protocol` for structural typing
- Explicit unions: `str | int | None`
- Generics: `List[Basin]`, `Dict[str, Tensor]`

❌ **Forbidden:**
- `Any` without documentation
- `Dict[str, Any]` without comment
- `List[Any]`
- Suppressing type errors with `# type: ignore` without reason

### File Structure Policy
**ALL files must follow 20251220-canonical-structure-1.00F.md.**

✅ **Use:**
- Canonical paths from 20251220-canonical-structure-1.00F.md
- Type imports from canonical modules
- Search existing files before creating new ones
- Enhance existing files instead of duplicating

❌ **Forbidden:**
- Creating files not in 20251220-canonical-structure-1.00F.md
- Duplicate scripts (check for existing first)
- Files with "_v2", "_new", "_test" suffixes
- Scripts in wrong directories

### Geometric Purity Policy (QIG-SPECIFIC)
**NEVER optimize measurements or couple gradients across models.**

✅ **Use:**
- `torch.no_grad()` for all measurements
- `.detach()` before distance calculations
- Fisher metric for geometric distances
- Natural gradient optimizers

❌ **Forbidden:**
- Training on measurement outputs
- Euclidean `torch.norm()` for basin distances
- Gradient flow between observer and active models
- Optimizing Φ directly
