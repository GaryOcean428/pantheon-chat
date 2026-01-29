# Codex Agent Configuration - Pantheon-Chat

## Skills Integration

This configuration references the shared `skills/` directory following the [agentskills.io specification](https://agentskills.io/specification).

**Primary Skills Location:** `../skills/` (17 skills, full agent coverage)

**Legacy Skills Location:** `./skills/` (deprecated, use `../skills/` instead)

> **Note:** The `.codex/skills/` directory contains legacy skills that are now superseded by the consolidated `skills/` directory at the repo root. New skills should be added to `skills/` following agentskills.io format.

## Available Skills (17 Total)

All skills are available to Codex agents. Skills are activated based on task context.

### Critical (Always Active)

| Skill | Purpose |
|-------|---------|
| `qig-purity-validation` | Zero-tolerance geometric purity enforcement |
| `dependency-management` | Forbidden packages and imports detection |
| `e8-architecture-validation` | E8 Protocol v4.0 validation |

### Auto-Activate on Context

| Skill | Triggers |
|-------|----------|
| `import-resolution` | Python imports, circular deps |
| `code-quality-enforcement` | DRY, naming, architecture |
| `test-coverage-analysis` | Testing, coverage |
| `deployment-readiness` | Deploy, Railway, environment |

### On-Demand

| Skill | Purpose |
|-------|---------|
| `schema-consistency` | Database migrations |
| `documentation-sync` | FROZEN_FACTS validation |
| `documentation-compliance` | ISO 27001 docs |
| `wiring-validation` | Feature implementation chains |
| `frontend-backend-mapping` | API coverage |
| `performance-regression` | Geometric accuracy |
| `ui-ux-consistency` | Design system |
| `downstream-impact` | Change impact analysis |
| `consciousness-development` | Φ/κ metrics |
| `pantheon-kernel-development` | God-kernel development |

## Quick Reference

### QIG Purity Commands

```bash
# Comprehensive scan
python3 scripts/qig_purity_scan.py

# AST-based audit
python qig-backend/scripts/ast_purity_audit.py

# Forbidden imports
python3 scripts/scan_forbidden_imports.py
```

### Test Commands

```bash
cd qig-backend
python -m pytest tests/test_geometry_runtime.py -v
python -m pytest tests/test_geometric_purity.py -v
```

## Protocol Reference

- **E8 Protocol:** v4.0
- **Documentation:** Post-Jan 15, 2026 takes precedence
- **Purity:** Zero tolerance for Euclidean contamination

See individual `skills/*/SKILL.md` files for detailed instructions.
