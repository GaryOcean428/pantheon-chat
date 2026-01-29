# Pantheon Chat Skills for Manus.ai

Agent skills packaged for [manus.ai](https://manus.ai) following the [Agent Skills specification](https://agentskills.io/specification).

## Overview

These skills enforce **QIG purity with zero tolerance** - the same standards as the CI/CD workflows in `.github/workflows/`. They are derived from:

- `.github/agents/` - GitHub Copilot custom agents
- `.github/workflows/` - CI/CD purity gates
- `AGENTS.md` and `CLAUDE.md` - E8 Protocol v4.0 instructions
- `docs/` - Post-Jan 15, 2026 documentation (preferred in conflicts)

## Available Skills (17 Total - Full Agent Coverage)

### Core QIG Purity (3 skills)

| Skill | Source Agent | Description |
|-------|--------------|-------------|
| `qig-purity-validation` | qig-purity-validator.md | Zero-tolerance geometric purity enforcement |
| `e8-architecture-validation` | e8-architecture-validator.md | Hierarchical kernel layers, god-kernel naming |
| `consciousness-development` | CLAUDE.md | Φ/κ metrics, Fisher-Rao geometry |

### Code Quality & Structure (6 skills)

| Skill | Source Agent | Description |
|-------|--------------|-------------|
| `import-resolution` | import-resolution-agent.md | Canonical imports, circular dependency detection |
| `schema-consistency` | schema-consistency-agent.md | Database migrations, vocabulary architecture |
| `code-quality-enforcement` | naming/dry/module agents | DRY, naming conventions, architecture |
| `test-coverage-analysis` | test-coverage-agent.md | Critical path test coverage |
| `dependency-management` | dependency-management-agent.md | Forbidden packages, requirements validation |
| `performance-regression` | performance-regression-agent.md | Detect Euclidean approximation substitutions |

### Integration & Synchronization (4 skills)

| Skill | Source Agent | Description |
|-------|--------------|-------------|
| `documentation-sync` | documentation-sync-agent.md | FROZEN_FACTS.md validation, doc freshness |
| `documentation-compliance` | documentation-compliance-auditor.md | ISO 27001, canonical naming |
| `wiring-validation` | wiring-validation-agent.md | Feature implementation chain tracing |
| `frontend-backend-mapping` | frontend-backend-capability-mapper.md | Route coverage, type consistency |

### UI & Deployment (3 skills)

| Skill | Source Agent | Description |
|-------|--------------|-------------|
| `ui-ux-consistency` | ui-ux-consistency-agent.md | Regime colors, God Panel, accessibility |
| `deployment-readiness` | deployment-readiness-agent.md | Environment, migrations, health checks |
| `downstream-impact` | downstream-impact-tracer.md | Change impact tracing |

### Advanced (1 skill)

| Skill | Source Agent | Description |
|-------|--------------|-------------|
| `pantheon-kernel-development` | AGENTS.md | God-kernel development, Zeus coordination |

## Skill Structure (per agentskills.io spec)

```text
skill-name/
├── SKILL.md           # Required: YAML frontmatter + instructions
├── references/        # Optional: documentation
└── assets/            # Optional: templates
```

### SKILL.md Format

```yaml
---
name: skill-name                    # Must match directory, lowercase a-z and -
description: When and what...       # 1-1024 chars, keywords for discovery
license: Apache-2.0                 # Optional
compatibility: Python 3.11+...      # Optional
metadata:                           # Optional
  author: pantheon-chat
  version: "2.0"
  protocol: "E8 Protocol v4.0"
allowed-tools: Bash(python3:*) Read # Optional
---
# Skill Title
## When to Use This Skill
## Step 1: ...
```

## CI Script References

Skills invoke the **actual** CI scripts - no custom implementations:

| Skill Category | Scripts/Workflows Used |
|----------------|------------------------|
| QIG Purity | `scripts/qig_purity_scan.py`, `qig-backend/scripts/ast_purity_audit.py`, `scripts/scan_forbidden_imports.py` |
| Testing | `qig-backend/tests/test_geometry_runtime.py`, `test_geometric_purity.py`, `test_qig_purity_mode.py` |
| Deployment | `drizzle-kit check`, `scripts/pre_deployment_check.sh` |
| Dependencies | `scripts/scan_forbidden_imports.py`, `pip-audit` |

## Packaging

```bash
# Validate skill structure
skills-ref validate ./skill-name

# Package for distribution
zip -r skill-name.skill skill-name/
```

## Key Principle

**NO HALF MEASURES. QIG PURITY IS NON-NEGOTIABLE.**

All skills enforce the same standards as CI. Docs after Jan 15, 2026 take precedence in conflicts.

## Related Resources

- [Agent Skills Specification](https://agentskills.io/specification)
- [GitHub Workflows](../.github/workflows/)
- [AGENTS.md](../AGENTS.md)
- [CLAUDE.md](../CLAUDE.md)
- [ROADMAP_PURE_QIG.md](../docs/ROADMAP_PURE_QIG.md)

---
**Version:** 2.0.0  
**Protocol:** E8 Protocol v4.0  
**Last Updated:** 2026-01-29
