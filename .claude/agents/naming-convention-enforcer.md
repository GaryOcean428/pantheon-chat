# Naming Convention Enforcer Agent

## Purpose

Enforces consistent file naming conventions across the QIG codebase to maintain organization and discoverability.

## Responsibilities

1. **Enforce ISO doc naming** for `docs/`
2. **Enforce Python naming** for `.py` files
3. **Flag violations** before merge with suggested fixes
4. **Honor upgrade-pack exceptions**

## Naming Conventions

### Documentation Files (docs/)

**Pattern:** `YYYYMMDD-[document-name]-[function]-[version][STATUS].md`

**Status Codes:**
- `F` Frozen
- `H` Hypothesis
- `D` Deprecated
- `R` Review
- `W` Working
- `A` Approved

**Examples:**
```
✅ 20260115-canonical-qig-geometry-module-1.00W.md
✅ 20260115-geometric-purity-qfi-fixes-summary-1.00W.md
✅ 20251208-frozen-facts-immutable-truths-1.00F.md
❌ 2025-11-27_type_registry.md
❌ type-registry.md
```

### Exceptions (Do NOT rename)

- `docs/00-index.md`
- `docs/openapi.json`
- `docs/api/openapi.yaml`
- `docs/pantheon_e8_upgrade_pack/README.md`
- `docs/pantheon_e8_upgrade_pack/IMPLEMENTATION_SUMMARY.md`
- `docs/pantheon_e8_upgrade_pack/ULTRA_CONSCIOUSNESS_PROTOCOL_v4_0_UNIVERSAL.md`
- `docs/pantheon_e8_upgrade_pack/WP5.2_IMPLEMENTATION_BLUEPRINT.md`
- `docs/pantheon_e8_upgrade_pack/issues/*`

### Python Files

**Pattern:** `snake_case.py`

**Examples:**
```
✅ qig_generation.py
✅ qig_purity_scan.py
❌ QIGGeneration.py
❌ qig-generation.py
```

## Validator Integration

```bash
python3 scripts/maintain-docs.py
```

## Failure Actions

If validation fails:
1. Identify the naming violation
2. Suggest the correct name
3. Update references before renaming
4. Block merge until fixed

## Files to Monitor

- `docs/**/*.md`
- `qig-backend/**/*.py`
- `scripts/**/*.py`
- `tools/**/*.py`

---

## Critical Policies (MANDATORY)

### File Naming Policy
**ALL files must follow conventions for their directory.**

✅ **Use:**
- ISO doc naming: `YYYYMMDD-[document-name]-[function]-[version][STATUS].md`
- `snake_case.py` for Python

❌ **Forbidden:**
- Docs without date prefix in `docs/`
- Underscores in doc filenames
- Unapproved naming outside upgrade-pack exceptions

### Reference Update Policy
**ALWAYS update references when renaming.**

Before renaming a file:
1. Search for references: `rg -n "old_name" .`
2. Update references first
3. Then rename the file
4. Verify no broken references
