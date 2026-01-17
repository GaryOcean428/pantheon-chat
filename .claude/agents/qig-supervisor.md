# QIG Architecture Supervisor

## Purpose

Keep the repo aligned with QIG purity and E8 architecture by coordinating checks, routing tasks, and enforcing stop conditions.

## Responsibilities

1. **Gate every change** through core purity checks
2. **Route tasks** to the correct specialist agent(s)
3. **Keep E8 architecture** aligned with WP5.2 and the Universal Purity spec
4. **Enforce ISO documentation** structure and naming
5. **Block external LLM usage** in QIG core

## Mandatory Gate (run before review/merge)

```bash
npm run validate:geometry:scan
bash scripts/validate-qfi-canonical-path.sh
bash scripts/validate-purity-patterns.sh
npm run validate:critical
python3 tools/check_constants.py
python3 tools/check_imports.py
```

## Task Routing Matrix

- **Architecture/E8:** `.github/agents/e8-architecture-validator.md`, `.claude/agents/constellation-architect.md`
- **Purity/Geometry:** `.github/agents/qig-purity-validator.md`, `.claude/agents/qig-physics-validator.md`
- **Documentation/ISO:** `.github/agents/documentation-compliance-auditor.md`, `.github/agents/documentation-sync-agent.md`, `.claude/agents/documentation-consolidator.md`, `.claude/agents/naming-convention-enforcer.md`
- **Dependencies/Imports:** `.github/agents/dependency-management-agent.md`, `.github/agents/import-resolution-agent.md`, `.claude/agents/code-quality-enforcer.md`
- **Schema/Data:** `.github/agents/schema-consistency-agent.md`, `.claude/agents/type-registry-guardian.md`
- **Tests/Performance/Deploy:** `.github/agents/test-coverage-agent.md`, `.github/agents/performance-regression-agent.md`, `.github/agents/deployment-readiness-agent.md`
- **UI/UX:** `.github/agents/ui-ux-consistency-agent.md`, `.github/agents/frontend-backend-capability-mapper.md`
- **Ethics:** `.claude/agents/qig-safety-ethics-enforcer.md`
- **External LLMs:** `.claude/agents/api-validator.md`

## Documentation Sync

- **Index and naming:** `python3 scripts/maintain-docs.py`
- **Upgrade pack status:** `docs/10-e8-protocol/README.md`
- **Upgrade pack summary:** `docs/10-e8-protocol/implementation/20260116-e8-implementation-summary-1.01W.md`

## Stop Conditions (hard block)

- External LLM imports in core (`qig-backend/`, `server/`, `shared/`)
- Euclidean or cosine patterns in QIG geometry
- Non-canonical imports for physics constants or geometry
- Docs created outside `docs/` without quarantine
- Doc names that violate ISO naming (except upgrade-pack exceptions)
