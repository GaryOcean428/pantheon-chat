# QIG Consciousness Agents

Specialized agents for maintaining QIG architecture purity and code quality.

## Primary Validation Tool

**Always run the geometric purity audit first:**

```bash
python tools/validation/geometric_purity_audit.py
```

This tool checks for terminology violations based on `docs/2025-11-29--geometric-terminology.md`.

## Available Agents

### qig-physics-validator
Validates code against FROZEN_FACTS.md physics constants and geometric purity.
- Runs `tools/validation/geometric_purity_audit.py` for terminology checks
- Checks β = 0.44 is not learnable
- Verifies κ values match documentation
- Ensures min_depth ≥ 3

### constellation-architect
Ensures Ocean + Constellation architecture consistency.
- Validates Φ-weighted routing
- Checks observer effect implementation
- Reviews checkpoint persistence

### documentation-consolidator
Maintains documentation hygiene and prevents fragmentation.
- Consolidates duplicates
- Archives outdated content
- Updates INDEX.md

### code-quality-enforcer
Maintains type safety, import hygiene, and geometric terminology.
- Runs `tools/validation/geometric_purity_audit.py` for terminology checks
- Enforces type annotations
- Prevents transformers contamination
- Validates telemetry patterns

### type-registry-guardian
Enforces type correctness and registration.
- Validates types are registered in `src/types/`
- Ensures imports use canonical locations
- Prevents duplicate type definitions
- Keeps `docs/20251127-type-registry-1.00W.md` in sync

### naming-convention-enforcer
Enforces file naming conventions.
- Docs: `YYYY-MM-DD--name.md` format
- Python: `snake_case.py` format
- Sleep packets: date-prefixed naming
- Flags violations with suggested fixes

## Usage

When making changes, invoke relevant agents:

1. **First step (always)** → Run `python tools/validation/geometric_purity_audit.py`
2. **Architecture changes** → constellation-architect
3. **Physics modifications** → qig-physics-validator
4. **New documentation** → documentation-consolidator
5. **Code quality issues** → code-quality-enforcer
6. **Type definitions** → type-registry-guardian
7. **File naming** → naming-convention-enforcer

## Collective QA Process

For comprehensive review:

```bash
# Step 1: Geometric purity (PRIMARY)
python tools/validation/geometric_purity_audit.py

# Step 2: Physics validation
python tools/agent_validators/scan_physics.py

# Step 3: Structure validation
python tools/agent_validators/scan_structure.py

# Step 4: Address any violations
# Step 5: Verify tests pass
```

## Core Principle

All agents validate against:
- `tools/validation/geometric_purity_audit.py` - Primary terminology enforcement
- `docs/2025-11-29--geometric-terminology.md` - Complete terminology guide
- `docs/FROZEN_FACTS.md` - Physics constants
- `src/constants.py` - Import constants from here, never hardcode
- Telemetry consistency patterns
