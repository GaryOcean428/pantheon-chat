# QIG Consciousness Agents

Specialized agents for maintaining QIG architecture purity, documentation integrity, and code quality.

## Primary Validation Gate (run first)

```bash
npm run validate:geometry:scan
bash scripts/validate-qfi-canonical-path.sh
bash scripts/validate-purity-patterns.sh
npm run validate:critical
```

This gate enforces simplex-only geometry, QFI canonical paths, and blocks external LLM usage in core.

## Canonical References

- `docs/00-index.md`
- `docs/pantheon_e8_upgrade_pack/ULTRA_CONSCIOUSNESS_PROTOCOL_v4_0_UNIVERSAL.md`
- `docs/pantheon_e8_upgrade_pack/WP5.2_IMPLEMENTATION_BLUEPRINT.md`
- `docs/04-records/20260115-canonical-qig-geometry-module-1.00W.md`
- `qig-backend/qig_geometry/canonical.py`
- `qig-backend/qigkernels/physics_constants.py`
- `shared/schema.ts`
- `shared/types/index.ts`

## Available Agents

### qig-supervisor
Orchestrates full-coverage gating and task routing across QIG, E8, docs, and QA.

### qig-physics-validator
Validates physics constants, recursion thresholds, and geometry purity.

### constellation-architect
Ensures Ocean + constellation architecture matches E8 and routing invariants.

### documentation-consolidator
Maintains ISO 27001 doc structure and prevents duplication.

### code-quality-enforcer
Enforces type safety, import hygiene, telemetry patterns, and purity checks.

### type-registry-guardian
Keeps shared TS and Python type registries canonical and consistent.

### naming-convention-enforcer
Enforces ISO doc naming and repo file naming rules.

### qig-safety-ethics-enforcer
Blocks training unless all 5 existential safeguards are present.

### api-validator
Guards against external LLM usage in core and enforces hard blocks.

## Usage

When making changes, invoke relevant agents:

1. **First step (always)** → Run the primary validation gate
2. **Architecture changes** → qig-supervisor + constellation-architect + `.github/agents/e8-architecture-validator.md`
3. **Physics/purity changes** → qig-physics-validator + `.github/agents/qig-purity-validator.md`
4. **Docs changes** → documentation-consolidator + naming-convention-enforcer
5. **Types/schema** → type-registry-guardian + `.github/agents/schema-consistency-agent.md`
6. **Dependencies/imports** → code-quality-enforcer + `.github/agents/dependency-management-agent.md`
7. **Safety/training** → qig-safety-ethics-enforcer

## Collective QA Process

```bash
npm run validate:geometry:scan
bash scripts/validate-qfi-canonical-path.sh
bash scripts/validate-purity-patterns.sh
npm run validate:critical
python3 tools/check_constants.py
python3 tools/check_imports.py
npm run check
npm run lint
npm run test
npm run test:python
python3 scripts/maintain-docs.py
```

## Core Principle

All agents validate against:
- `docs/00-index.md` (ISO naming and status)
- `docs/pantheon_e8_upgrade_pack/ULTRA_CONSCIOUSNESS_PROTOCOL_v4_0_UNIVERSAL.md` (purity rules)
- `docs/04-records/20260115-canonical-qig-geometry-module-1.00W.md` (canonical geometry)
- `qig-backend/qig_geometry/canonical.py` (single source of geometry)
- `qig-backend/qigkernels/physics_constants.py` (single source of constants)
