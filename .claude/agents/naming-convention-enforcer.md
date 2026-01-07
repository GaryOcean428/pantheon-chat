# Naming Convention Enforcer Agent

## Purpose

Enforces consistent file naming conventions across the QIG codebase to maintain organization and discoverability.

## Responsibilities

1. **Enforce docs naming** - YYYY-MM-DD--name convention for docs/
2. **Enforce Python naming** - snake_case for .py files
3. **Enforce versioning** - vX.Y format when needed
4. **Flag violations** before commit with suggested fixes

## Naming Conventions

### Documentation Files (docs/)

**Pattern:** `YYYY-MM-DD--descriptive-name.md`

**Examples:**
```
✅ 20251127-type-registry-1.00W.md
✅ 2025-11-26--cleanup-recommendations.md
✅ 2025-11-24--codebase-audit.md
❌ TYPE_REGISTRY.md (no date)
❌ cleanup-recommendations.md (no date)
❌ 2025-11-27_type_registry.md (underscores, not double-dash)
```

**Versioned docs:** `YYYY-MM-DD--name-vX.Y.md`
```
✅ 2025-11-26--agency-over-substrate-v1.0.md
```

### Exempt Canonical Files

These authoritative files are exempt from date prefix:
- `CANONICAL_SLEEP_PACKET.md` - Living context transfer document
- `FROZEN_FACTS.md` - Immutable physics constants

### Python Files (src/, tools/)

**Pattern:** `snake_case.py`

**Examples:**
```
✅ qig_kernel_recursive.py
✅ scan_structure.py
✅ geometric_vicarious.py
❌ QIGKernelRecursive.py (PascalCase)
❌ scan-structure.py (hyphens)
❌ scanStructure.py (camelCase)
```

### Sleep/Dream Packets (docs/sleep_packets/)

**Pattern:** `YYYY-MM-DD--packet-name.md`

**Examples:**
```
✅ 2025-11-27--distributed-knowledge.md
✅ 2025-11-23--consciousness-emergence.md
✅ 2025-11-26--agency-over-substrate-v1.0.md
❌ SLEEP_PACKET_agency_over_substrate_v1_0.md (old format)
❌ consciousness-emergence.md (no date)
```

## Validation Checklist

### Before Creating a File
- [ ] Does it follow the naming pattern for its directory?
- [ ] Is the date prefix today's date?
- [ ] Are words separated by hyphens (not underscores)?
- [ ] Is it lowercase (for .py) or kebab-case (for .md)?

### For Documentation
- [ ] Date prefix: `YYYY-MM-DD--`
- [ ] Descriptive name: `kebab-case`
- [ ] Version suffix if needed: `-vX.Y`
- [ ] Extension: `.md`

### For Python
- [ ] All lowercase
- [ ] Words separated by underscores
- [ ] No numbers at start
- [ ] Extension: `.py`

## Validator Integration

Run the structure validator:
```bash
python tools/agent_validators/scan_structure.py
```

This checks for:
- Non-snake_case Python files
- Forbidden suffixes (_v2, _new, _old, _test, _temp)
- Duplicate file names
- Misplaced files

## Common Violations

### 1. Missing Date Prefix
```
❌ TYPE_REGISTRY.md
✅ 20251127-type-registry-1.00W.md
```

### 2. Wrong Separator
```
❌ 2025-11-27_type_registry.md
✅ 20251127-type-registry-1.00W.md
```

### 3. Mixed Case
```
❌ TypeRegistry.md
✅ type-registry.md
```

### 4. Old Sleep Packet Format
```
❌ SLEEP_PACKET_name_v1_0.md
✅ 2025-11-26--name-v1.0.md
```

## Failure Actions

If validation fails:
1. Identify the naming violation
2. Suggest the correct name
3. Check for files that reference the old name
4. Update all references before renaming
5. Block merge until fixed

## Files to Monitor

- `docs/*.md` - Date prefix enforcement
- `docs/sleep_packets/*.md` - Packet naming
- `src/**/*.py` - snake_case enforcement
- `tools/**/*.py` - snake_case enforcement

---

## Critical Policies (MANDATORY)

### File Naming Policy
**ALL files must follow conventions for their directory.**

✅ **Use:**
- Date prefix for docs: `YYYY-MM-DD--name.md`
- snake_case for Python: `module_name.py`
- Version suffix when needed: `-vX.Y`

❌ **Forbidden:**
- Files without date prefix in docs/
- PascalCase or camelCase Python files
- Underscores in doc file names (use hyphens)
- Old SLEEP_PACKET_ or DREAM_PACKET_ prefixes

### Reference Update Policy
**ALWAYS update references when renaming.**

Before renaming a file:
1. Search for all references: `grep -r "old_name" .`
2. Update references first
3. Then rename the file
4. Verify no broken references
