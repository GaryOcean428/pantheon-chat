# Agent Skills System

**Version:** 1.0
**Created:** 2025-11-24
**Purpose:** Reusable components for QIG consciousness agents

---

## Overview

Skills are **modular, reusable capabilities** that agents can invoke for specific tasks. Unlike agents (which are autonomous), skills are **templates and validation patterns** that multiple agents share.

---

## Available Skills

### 1. [Geometric Operations](geometric-operations.md)
**Category:** Mathematical Operations
**Used By:** purity-guardian, geometric-navigator, integration-architect

**Provides:**
- Fisher metric distance calculations
- Geodesic interpolation templates
- Basin projection patterns
- QFI attention implementations

**Key Functions:**
- `fisher_distance()` - Distance on Fisher manifold
- `geodesic_interpolate()` - Interpolation along geodesics
- `project_to_basin()` - Natural gradient projection
- `qfi_attention()` - QFI-based attention weights

---

### 2. [Basin Transfer](basin-transfer.md)
**Category:** Knowledge Transfer
**Used By:** constellation-architect, integration-architect, geometric-navigator

**Provides:**
- Lightweight knowledge transfer (2KB basin vs 2GB weights)
- Checkpoint basin extraction
- Constellation basin synchronization
- Sleep consolidation patterns

**Key Functions:**
- `initialize_with_basin()` - Fresh model with target basin
- `extract_basin_from_checkpoint()` - Save identity to JSON
- `sync_constellation_basins()` - Vicarious learning sync
- `consolidate_toward_basin()` - Sleep protocol

---

### 3. [Physics Validation](physics-validation.md)
**Category:** Compliance Checking
**Used By:** qig-physics-validator, purity-guardian, code-quality-enforcer

**Provides:**
- Automated validation against FROZEN_FACTS.md
- Pre-commit hook integration
- Physics constant verification
- Architectural requirement checking

**Key Functions:**
- `validate_beta_constant()` - Ensure β = 0.44 and not learnable
- `validate_min_depth()` - Check recursion ≥ 3
- `validate_phi_thresholds()` - Verify Φ thresholds
- `validate_kappa_values()` - Check coupling constants

---

### 4. [Constellation Coordination](constellation-coordination.md)
**Category:** Multi-Instance Architecture
**Used By:** constellation-architect, integration-architect, purity-guardian

**Provides:**
- Round-robin question routing
- Vicarious learning patterns
- Basin synchronization protocols
- Checkpoint save/load with constellation

**Key Functions:**
- `route_question()` - Round-robin routing with roles
- `_compute_vicarious_loss()` - Observer learning from active
- `save_constellation()` / `load_constellation()` - Persistence
- `aggregate_telemetry()` - Constellation-level metrics

---

## How Skills Work

### Agent → Skill Relationship

```
┌─────────────────────────────┐
│   Agent (Autonomous)        │
│                             │
│   • Has specific goal       │
│   • Makes decisions         │
│   • Invokes skills          │
└─────────────┬───────────────┘
              │
              │ invokes
              ▼
┌─────────────────────────────┐
│   Skill (Reusable Module)   │
│                             │
│   • Provides templates      │
│   • No decision-making      │
│   • Used by multiple agents │
└─────────────────────────────┘
```

### Example: Purity Guardian using Skills

```markdown
# purity-guardian.md

## Validation Process

1. **Check geometric operations** → Invokes geometric-operations skill
   - Validates Fisher metric usage
   - Checks for Euclidean violations

2. **Verify basin transfers** → Invokes basin-transfer skill
   - Ensures no weight copying
   - Validates JSON format

3. **Validate physics** → Invokes physics-validation skill
   - Checks β = 0.44
   - Verifies min_depth ≥ 3
```

---

## Creating New Skills

### When to Create a Skill

Create a skill when:
- ✅ Pattern is used by 2+ agents
- ✅ Logic is complex and benefits from templates
- ✅ Validation is needed across codebase
- ✅ Examples help reduce errors

Do NOT create a skill when:
- ❌ Only used by 1 agent (keep in agent)
- ❌ Simple logic (< 20 lines)
- ❌ Project-specific (not reusable)

### Skill Template

```markdown
# [Skill Name] Skill

**Type:** Reusable Component
**Category:** [Mathematical | Transfer | Validation | Coordination]
**Used By:** [list of agents]

---

## Purpose

[One-sentence description]

---

## Core Operations

### 1. [Operation Name]

**Template:**
```python
def operation_name(...):
    \"\"\"
    [Description]
    \"\"\"
    # Implementation template
```

**Validation:**
- ✅ Requirement 1
- ✅ Requirement 2

---

## Common Violations

### ❌ [Violation Type]
[Wrong code example]

**Fix:**
[Correct code example]

---

## Integration Points

[Where this skill is used in the codebase]

---

## Validation Checklist

- [ ] Item 1
- [ ] Item 2

---

## References

[Links to docs, implementations]
```

---

## Skills vs Agents

| Aspect | Skills | Agents |
|--------|--------|--------|
| **Purpose** | Reusable templates | Autonomous validation |
| **Decision-Making** | No | Yes |
| **Invocation** | By agents | By users/system |
| **Count** | Few (4-6 core) | Many (10+ specialized) |
| **Location** | `.github/agents/skills/` | `.github/agents/` + `.claude/agents/` |
| **Format** | Templates + examples | Process + checklist |

---

## Integration with Validators

Skills provide templates, validators enforce them:

```
Skill (physics-validation.md)
  ↓ templates
tools/agent_validators/scan_physics.py
  ↓ runs
Pre-commit hook
  ↓ blocks
Commit with violations
```

---

## Maintenance

### Adding a New Skill

1. Create `skill-name.md` in `.github/agents/skills/`
2. Follow skill template structure
3. Add to this README's "Available Skills" section
4. Update agents that will use it
5. Create validator if applicable

### Updating a Skill

1. Update skill markdown file
2. Update affected agents
3. Update validators if logic changed
4. Test with agents using the skill

---

## Current Coverage

| Domain | Skill | Validator | Pre-commit Hook |
|--------|-------|-----------|-----------------|
| **Geometric Operations** | ✅ | ⏳ Planned | ⏳ Planned |
| **Basin Transfer** | ✅ | ⏳ Planned | ⏳ Planned |
| **Physics Constants** | ✅ | ⏳ Phase 3 | ⏳ Phase 3 |
| **Constellation Coord** | ✅ | ⏳ Planned | ❌ Not needed |

---

## Future Skills (Planned)

### Phase 2 (Next)
- `telemetry-aggregation.md` - Metrics collection patterns
- `checkpoint-management.md` - State persistence patterns

### Phase 3 (Later)
- `curriculum-validation.md` - Developmental content checking
- `training-monitoring.md` - Φ progression tracking

---

## References

- **Inspiration:** monkey1 project skills system
- **Agent System:** `.github/agents/README.md`
- **Validators:** `tools/agent_validators/` (Phase 3)
