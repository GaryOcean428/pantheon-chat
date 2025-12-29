# Agent Coordination Map

**Version:** 1.0
**Created:** 2025-11-24
**Purpose:** Visual guide to agent relationships, dependencies, and invocation patterns

---

## Agent Hierarchy

```
Meta-Agent Creator (Orchestrator)
├── Purity Guardian
│   ├── calls: physics-validation skill
│   ├── calls: geometric-operations skill
│   └── validates: Fisher metric usage, gradient detachment
│
├── Structure Enforcer
│   ├── validates: 20251220-canonical-structure-1.00F.md compliance
│   ├── detects: Forbidden patterns (_v2, _new, etc.)
│   └── suggests: Migrations and archival
│
├── Type Registry Guardian
│   ├── detects: Duplicate type definitions
│   ├── validates: Canonical imports
│   └── generates: TYPE_REGISTRY.md
│
├── Python Environment Manager
│   ├── validates: Dependency consistency
│   ├── checks: PyTorch CUDA compatibility
│   └── audits: Security (pip-audit)
│
├── Integration Architect
│   ├── calls: constellation-coordination skill
│   ├── calls: basin-transfer skill
│   └── orchestrates: Multi-instance coordination
│
├── Geometric Navigator
│   ├── calls: geometric-operations skill
│   ├── implements: Fisher metric calculations
│   └── validates: Natural gradient usage
│
├── Documentation Weaver
│   ├── tracks: FROZEN_FACTS.md milestones
│   ├── cross-references: Commit ↔ doc updates
│   └── generates: Status reports
│
└── Test Synthesizer
    ├── generates: Property-based tests
    ├── validates: Telemetry completeness
    └── creates: Integration test suites
```

---

## Agent-to-Skill Mapping

### Skills System

Located in `.github/agents/skills/`:

1. **geometric-operations.md**
   - Used by: Purity Guardian, Geometric Navigator, Integration Architect
   - Provides: Fisher metric templates, geodesic interpolation, QFI attention patterns

2. **basin-transfer.md**
   - Used by: Integration Architect, Documentation Consolidator, Constellation Architect
   - Provides: Lightweight 2KB basin transfer, checkpoint extraction, sleep consolidation

3. **physics-validation.md**
   - Used by: Purity Guardian, Code Quality Enforcer, QIG Physics Validator
   - Provides: Automated β, κ, Φ validation, FROZEN_FACTS.md compliance checks

4. **constellation-coordination.md**
   - Used by: Integration Architect, Constellation Architect, Test Synthesizer
   - Provides: Round-robin routing, vicarious learning, basin sync patterns

### Agent Invocation Matrix

| Agent | Calls Skills | Calls Agents | Called By |
|-------|--------------|--------------|-----------|
| Meta-Agent Creator | - | All agents | User |
| Purity Guardian | geometric-operations, physics-validation | - | Meta-Agent, Code Quality Enforcer |
| Structure Enforcer | - | - | Meta-Agent, Pre-commit |
| Type Registry Guardian | - | - | Meta-Agent, Pre-commit |
| Python Environment Manager | - | - | Meta-Agent, CI/CD |
| Integration Architect | constellation-coordination, basin-transfer | - | Meta-Agent, Constellation Architect |
| Geometric Navigator | geometric-operations | - | Purity Guardian, Integration Architect |
| Documentation Weaver | All skills (for docs) | - | Meta-Agent, Release |
| Test Synthesizer | constellation-coordination | - | Meta-Agent, CI/CD |

---

## Workflow Triggers

### On Commit (Pre-commit Hook)

```
User: git commit
│
├─▶ Structure Enforcer
│   ├── Validates: File locations, naming, duplicates
│   └── Blocks if: Non-canonical files, bad suffixes
│
├─▶ Purity Guardian (via scan_physics.py)
│   ├── Validates: β, κ, Φ, min_depth
│   ├── Checks: Fisher vs Euclidean
│   └── Blocks if: Physics violations
│
└─▶ Type Registry Guardian (via scan_types.py)
    ├── Validates: No duplicate types
    ├── Checks: Canonical imports
    └── Blocks if: Import violations

If all pass → Commit succeeds
If any fail → Commit blocked, show violations
```

### On Pull Request

```
PR Opened
│
├─▶ CI/CD (GitHub Actions)
│   ├── scan_physics.py
│   ├── scan_structure.py
│   └── scan_types.py
│
├─▶ Integration Architect
│   ├── Reviews: Multi-instance coordination
│   ├── Validates: Basin sync patterns
│   └── Comments: Integration suggestions
│
└─▶ Test Synthesizer
    ├── Checks: Test coverage
    ├── Validates: Telemetry completeness
    └── Suggests: Missing test cases
```

### On Release

```
Release Tag Created
│
├─▶ Documentation Weaver
│   ├── Updates: CHANGELOG.md
│   ├── Generates: Release notes
│   └── Archives: Version snapshot
│
├─▶ Constellation Architect
│   ├── Validates: Checkpoint compatibility
│   ├── Exports: Basin snapshots
│   └── Documents: Model states
│
└─▶ QIG Physics Validator
    ├── Confirms: All physics constants frozen
    ├── Validates: No experimental values
    └── Certifies: Physics compliance
```

---

## Decision Trees

### When to Use Which Agent?

#### Scenario: Adding New Module

```
Is it a new Python file?
├─ YES → Call Structure Enforcer
│         └─ Validate location is canonical
│            ├─ In src/model/ → OK
│            ├─ In chat_interfaces/ → Check if 5th entry point (ERROR)
│            └─ In root/ → ERROR (wrong location)
│
└─ NO → Is it a new type?
          └─ YES → Call Type Registry Guardian
                   └─ Check no duplicates
                      └─ Register in CANONICAL_TYPES
```

#### Scenario: Refactoring Code

```
What kind of refactor?
│
├─ Geometric operations → Call Geometric Navigator
│  └─ Ensures Fisher metric, natural gradient
│
├─ Multi-instance coordination → Call Integration Architect
│  └─ Uses constellation-coordination skill
│
├─ Moving files → Call Structure Enforcer
│  └─ Validates new location is canonical
│
└─ Changing types → Call Type Registry Guardian
   └─ Updates imports across codebase
```

#### Scenario: Code Review

```
Review Checklist:
│
├─ Physics constants used?
│  └─ YES → Invoke Purity Guardian
│           └─ Validates against FROZEN_FACTS.md
│
├─ New files added?
│  └─ YES → Invoke Structure Enforcer
│           └─ Checks 20251220-canonical-structure-1.00F.md compliance
│
├─ Imports changed?
│  └─ YES → Invoke Type Registry Guardian
│           └─ Validates canonical imports
│
└─ Coordination logic?
   └─ YES → Invoke Integration Architect
            └─ Reviews basin sync patterns
```

---

## Communication Patterns

### Agent-to-Agent Messages

Agents communicate via structured messages:

```python
{
    "from": "purity-guardian",
    "to": "geometric-navigator",
    "action": "validate_distance_calculation",
    "context": {
        "file": "src/training/loss.py",
        "line": 42,
        "code": "torch.norm(basin_a - basin_b) ** 2"
    },
    "response": {
        "valid": false,
        "violation": "euclidean_distance",
        "suggestion": "Use geodesic_distance() from geometric-operations skill"
    }
}
```

### Agent-to-User Messages

Agents provide actionable feedback:

```
❌ VIOLATION DETECTED

Agent: Purity Guardian
File: src/training/loss.py:42
Issue: Euclidean distance instead of Fisher metric

Found:
    loss = torch.norm(basin_a - basin_b) ** 2

Should be:
    from src.metrics.geodesic_distance import geodesic_vicarious_loss
    loss = geodesic_vicarious_loss(basin_a, basin_b, fisher_diag)

Reason: Geometric purity requires Fisher metric for basin distances.

Fix: See geometric-operations skill for templates.
```

---

## Automation Scripts

### Pre-commit Hook

```bash
.git/hooks/pre-commit
├── scan_physics.py      # Purity Guardian validator
├── scan_structure.py    # Structure Enforcer validator
└── scan_types.py        # Type Registry Guardian validator
```

**Runs:** Before every commit
**Blocks:** If any validator fails
**Bypass:** `git commit --no-verify` (not recommended)

### CI/CD Pipeline

```yaml
.github/workflows/validate.yml
├── Physics validation (scan_physics.py)
├── Structure validation (scan_structure.py)
└── Type validation (scan_types.py)
```

**Runs:** On push to main, on PR
**Status:** Reports to GitHub checks
**Blocks:** PR merge if fails

### Manual Validation

```bash
# Run all validators
tools/agent_validators/scan_physics.py
tools/agent_validators/scan_structure.py
tools/agent_validators/scan_types.py

# Or comprehensive check
make validate  # (if Makefile configured)
```

---

## Agent Responsibilities

### By Domain

#### Physics & Purity
- **Purity Guardian**: 100% geometric purity enforcement
- **QIG Physics Validator**: FROZEN_FACTS.md compliance
- **Geometric Navigator**: Fisher metric operations

#### Structure & Organization
- **Structure Enforcer**: 20251220-canonical-structure-1.00F.md compliance
- **Type Registry Guardian**: Type definition management
- **Documentation Consolidator**: Doc organization

#### Coordination & Integration
- **Integration Architect**: Multi-instance coordination
- **Constellation Architect**: Ocean + Gary orchestration
- **Meta-Agent Creator**: Agent system orchestration

#### Quality & Testing
- **Code Quality Enforcer**: Type safety, import hygiene
- **Test Synthesizer**: Comprehensive test generation
- **Python Environment Manager**: Dependency validation

#### Documentation
- **Documentation Weaver**: Cross-referencing, status tracking
- **Documentation Consolidator**: Doc hygiene

---

## Success Metrics

### Per-Agent KPIs

| Agent | Metric | Target | Current |
|-------|--------|--------|---------|
| Purity Guardian | Physics violations | 0 | 18 (pre-Phase 4) |
| Structure Enforcer | Non-canonical files | 0 | TBD |
| Type Registry Guardian | Duplicate types | 0 | TBD |
| Python Environment Manager | Dependency conflicts | 0 | TBD |
| Documentation Weaver | Outdated docs | <5% | TBD |
| Test Synthesizer | Test coverage | >90% | ~70% |

### System-Wide KPIs

- **Commit Block Rate**: % of commits blocked by validators (target: <10% after stabilization)
- **False Positive Rate**: % of validator errors that are false positives (target: <2%)
- **Fix Time**: Average time to fix validator violations (target: <30 mins)
- **Agent Invocation Rate**: % of tasks that use agents (target: >80%)

---

## Troubleshooting

### Validator Failed - What Now?

1. **Read the violation report** - Shows file, line, issue, fix
2. **Check the relevant skill** - Templates and examples in `.github/agents/skills/`
3. **Consult the agent docs** - Full agent spec in `.github/agents/` or `.claude/agents/`
4. **Fix the violation** - Apply suggested fix
5. **Re-run validator** - Confirm fix before committing

### False Positive?

1. **Verify it's truly false** - Check FROZEN_FACTS.md, 20251220-canonical-structure-1.00F.md
2. **If legitimate exception** - Add comment explaining why
3. **Update validator** - If systematic false positive, fix validator logic
4. **Report to team** - Document for future reference

### Need New Agent?

1. **Check existing agents** - Can current agents be enhanced?
2. **Check skills** - Can skills be reused?
3. **Consult Meta-Agent Creator** - For agent creation guidance
4. **Follow template** - See `.github/agents/meta-agent-creator.md`

---

## Future Enhancements

### Planned

1. **Agent Metrics Dashboard**
   - Real-time KPI tracking
   - Violation trends
   - Agent usage statistics

2. **AI-Powered Agent Suggestions**
   - Context-aware agent invocation
   - Predictive violation detection
   - Auto-fix suggestions

3. **Cross-Agent Learning**
   - Agents learn from violations
   - Adaptive thresholds
   - Pattern recognition

4. **Agent Composition**
   - Chain multiple agents
   - Conditional workflows
   - Parallel execution

---

## References

- **Agent Specifications:** `.github/agents/*.md`, `.claude/agents/*.md`
- **Skills Library:** `.github/agents/skills/*.md`
- **Validators:** `tools/agent_validators/`
- **Structure:** `20251220-canonical-structure-1.00F.md`
- **Physics:** `FROZEN_FACTS.md`
- **Rules:** `20251220-canonical-rules-1.00F.md`, `.github/copilot-instructions.md`

---

## Updates

- **2025-11-24:** Initial creation (Phase 5 of agent system improvement)
- **Future:** Add metrics dashboard, AI suggestions, cross-agent learning
