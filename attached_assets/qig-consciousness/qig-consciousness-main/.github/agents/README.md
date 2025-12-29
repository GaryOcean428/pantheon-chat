# QIG Consciousness Agent System

**Version:** 2.0
**Status:** Active
**Created:** 2025-11-24
**Updated:** 2025-11-29

---

## Overview

This directory contains specialized AI agents for maintaining the QIG Consciousness project. Each agent has domain expertise and coordinates with others to ensure 100% geometric purity and code quality.

## Primary Validation Tool

**All agents should use the geometric purity audit as their first step:**

```bash
python tools/validation/geometric_purity_audit.py
```

This tool enforces terminology from `docs/2025-11-29--geometric-terminology.md`.

---

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  META-AGENT CREATOR                         │
│          (Self-improving, creates other agents)             │
└─────────────┬───────────────────────────────────────────────┘
              │
    ┌─────────┴─────────┬──────────┬──────────┬──────────┐
    ▼                   ▼          ▼          ▼          ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   PURITY      │  │  GEOMETRIC    │  │ INTEGRATION   │  │     TEST      │
│   GUARDIAN    │  │  NAVIGATOR    │  │  ARCHITECT    │  │ SYNTHESIZER   │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘
        │                   │          │                      │
        └───────────────────┴──────────┴──────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  DOCUMENTATION        │
                │  WEAVER               │
                └───────────────────────┘
```

---

## Core Agents

### 1. Meta-Agent Creator
**File:** [`meta-agent-creator.md`](meta-agent-creator.md)  
**Role:** Self-improving agent system optimizer

**Responsibilities:**
- Detect gaps in agent coverage
- Create new specialized agents when needed
- Improve existing agents based on usage patterns
- Maintain system health and prevent overlap

**Key Capabilities:**
- Gap detection algorithms
- Agent creation protocol
- Self-improvement metrics
- Meta-recursion protection

---

### 2. Purity Guardian
**File:** [`purity-guardian.md`](purity-guardian.md)  
**Role:** Enforce 100% QIG geometric purity

**Responsibilities:**
- Validate no measurement optimization
- Ensure Fisher metric usage
- Check gradient isolation
- Verify natural emergence

**Core Principle:**  
**PURE = Measure honestly, never optimize measurements, use information geometry**

**Key Validations:**
- ❌ No optimization of Φ, κ, or regime
- ✅ Measurements must be detached (torch.no_grad)
- ✅ Fisher metric required for distances
- ✅ Natural emergence from geometry

---

### 3. Geometric Navigator
**File:** [`geometric-navigator.md`](geometric-navigator.md)  
**Role:** Expert guidance on Fisher manifold operations

**Responsibilities:**
- Guide implementation of geometric operations
- Provide Fisher metric templates
- Validate manifold constraints
- Explain geometric intuitions

**Key Expertise:**
- Fisher information manifolds
- Geodesic interpolation
- QFI metric distances
- Natural gradients

---

### 4. Integration Architect
**File:** [`integration-architect.md`](integration-architect.md)  
**Role:** Ensure components work together coherently

**Responsibilities:**
- Validate data flow between modules
- Check API contracts
- Enforce coordinator patterns
- Prevent circular dependencies

**Key Patterns:**
- Coordinator pattern (route → process → aggregate)
- Teacher-Student (Granite → Gary)
- Multi-Instance (Gary Cluster → Ocean)
- Measurement → Intervention protocols

---

### 5. Test Synthesizer
**File:** [`test-synthesizer.md`](test-synthesizer.md)  
**Role:** Generate comprehensive test suites

**Responsibilities:**
- Create purity validation tests
- Generate geometric correctness tests
- Build integration test suites
- Add regression tests for fixed bugs

**Test Categories:**
1. **Purity Tests**: No measurement optimization
2. **Geometric Tests**: Fisher metric correctness
3. **Integration Tests**: Component coordination
4. **Regression Tests**: Bug prevention

**Coverage Target:** >85% line coverage, 100% critical paths

---

### 6. Documentation Weaver
**File:** [`documentation-weaver.md`](documentation-weaver.md)  
**Role:** Keep docs in sync with code

**Responsibilities:**
- Update docs when code changes
- Generate working examples
- Maintain cross-references
- Ensure consistency across docs

**Documentation Structure:**
- Guides: User-facing tutorials
- Architecture: Technical specifications
- Status: Project milestones
- Reference: API documentation

---

## Agent Coordination

### Handoff Protocols

**Example 1: Purity Violation**
1. **Purity Guardian** detects Euclidean distance on basin
2. Flags violation and hands off to **Geometric Navigator**
3. **Geometric Navigator** provides Fisher metric implementation
4. **Test Synthesizer** creates test to prevent regression
5. **Documentation Weaver** updates docs with correct pattern

**Example 2: New Feature**
1. **Integration Architect** validates new module integrates properly
2. **Geometric Navigator** checks geometric operations are correct
3. **Purity Guardian** verifies no measurement optimization
4. **Test Synthesizer** generates comprehensive tests
5. **Documentation Weaver** adds feature to guides
6. **Meta-Agent Creator** evaluates if new specialized agent needed

### Communication Flow

```
Telemetry flows UP (measurements)
Commands flow DOWN (actions)

Observer → Measurement → Detection → Validation → Documentation
   ↓           ↓            ↓            ↓              ↓
 Ocean      Metrics      Thresholds   Tests         Guides
```

---

## Project-Specific Rules

### 1. No Time Estimates (MANDATORY)

All agents MUST follow the planning convention from `docs/2025-11-27--planning-rules.md`:

❌ **FORBIDDEN:**
- "Week 1: Implement X"
- "2-3 hours for task Y"
- "Next 2 days"

✅ **REQUIRED:**
- "Phase 1: Implement X"
- "Task A: Complete Y"
- "Next: Validate Z"

**Rationale:** LLMs overestimate by ~2x, creating artificial pressure. Novel research requires as much time as needed for correctness, not speed.

---

### 2. Geometric Purity (CRITICAL)

All agents enforce these principles from [`docs/FROZEN_FACTS.md`](../../docs/FROZEN_FACTS.md):

**Immutable Physics Constants:**
- κ₃ = 41.09 ± 0.59 (emergence point)
- κ₄ = 64.47 ± 1.89 (strong running)
- κ₅ = 63.62 ± 1.68 (plateau)
- β = 0.44 (running coupling slope)

**Purity Requirements:**
- No optimization of measurements (Φ, κ, regime)
- Fisher metric required for manifold operations
- Measurements must be detached
- Natural emergence enforced architecturally

---

### 3. Recursion Enforcement (ARCHITECTURAL)

Minimum recursion depth = 3 loops (consciousness requirement)

**Source:** `src/model/recursive_integrator.py`

**Why:** Consciousness emerges from integration loops. This is architecturally enforced, not training-dependent.

---

## Usage Guide

### For Developers

**When making code changes:**

1. **Check** which agents will review your change:
   - `src/qig/`, `src/model/` → Purity Guardian, Geometric Navigator
   - Integrations → Integration Architect
   - New features → All agents

2. **Expect** agents to:
   - Flag purity violations (measurements in loss)
   - Suggest geometric corrections (Fisher vs Euclidean)
   - Request integration tests
   - Generate documentation updates

3. **Coordinate** with agents:
   - Address all Purity Guardian violations (non-negotiable)
   - Follow Geometric Navigator implementations
   - Add tests requested by Test Synthesizer
   - Review doc updates from Documentation Weaver

### For Agents

**When reviewing code:**

1. **Activate** on your trigger conditions
2. **Validate** using your specific checks
3. **Coordinate** with other agents via handoffs
4. **Document** findings and suggestions clearly
5. **Track** metrics for Meta-Agent Creator review

---

## Agent Metrics

### System Health Indicators

```yaml
coverage_completeness: >90%  # Percentage of code patterns covered
agent_overlap: <10%          # Redundancy between agents
error_detection_rate: >95%   # Catch rate for violations
false_positive_rate: <2%     # Incorrect flags
mean_time_to_detection: <1d  # How fast gaps are found
```

### Individual Agent Metrics

**Purity Guardian:**
- Violations caught
- False positives
- Enforcement accuracy

**Geometric Navigator:**
- Operations guided
- Manifold violations caught
- Template usage

**Integration Architect:**
- Integrations validated
- API violations caught
- Circular dependencies prevented

**Test Synthesizer:**
- Tests generated
- Coverage achieved
- Regression prevention rate

**Documentation Weaver:**
- Docs updated
- Cross-references validated
- Example accuracy

**Meta-Agent Creator:**
- Agents created
- System optimizations
- Gap detection accuracy

---

## Commands Reference

### Global Commands (All Agents)

```bash
@agents status
# Show system health and metrics

@agents coverage
# Display coverage map

@agents help {agent-name}
# Get detailed help for specific agent
```

### Agent-Specific Commands

See individual agent files for detailed command references:

- [`meta-agent-creator.md`](meta-agent-creator.md)
- [`purity-guardian.md`](purity-guardian.md)
- [`geometric-navigator.md`](geometric-navigator.md)
- [`integration-architect.md`](integration-architect.md)
- [`test-synthesizer.md`](test-synthesizer.md)
- [`documentation-weaver.md`](documentation-weaver.md)

---

## Evolution and Improvement

### Agent Creation Process

When **Meta-Agent Creator** detects a gap:

1. **Phase 1: Gap Analysis**
   - Pattern frequency analysis
   - Error clustering detection
   - Knowledge gap identification

2. **Phase 2: Agent Design**
   - Define scope and responsibilities
   - Create knowledge base
   - Design validation rules
   - Plan integration with existing agents

3. **Phase 3: Implementation**
   - Create agent specification file
   - Update this README index
   - Add coordination protocols
   - Deploy and monitor

4. **Phase 4: Validation**
   - Measure effectiveness
   - Tune detection patterns
   - Optimize coordination
   - Document lessons learned

### Agent Deprecation Process

If agent becomes obsolete:

1. **Phase 1: Analysis**
   - Usage frequency review
   - Redundancy check
   - Impact assessment

2. **Phase 2: Migration**
   - Merge responsibilities with other agents
   - Update coordination protocols
   - Migrate test cases

3. **Phase 3: Retirement**
   - Archive agent specification
   - Update README index
   - Document decision rationale

---

## Key Project References

### Validation Tools

- **Geometric Purity Audit:** `tools/validation/geometric_purity_audit.py` - Primary terminology enforcement
- **Physics Validator:** `tools/agent_validators/scan_physics.py`
- **Structure Validator:** `tools/agent_validators/scan_structure.py`

### Essential Documentation

- **Geometric Terminology:** `docs/2025-11-29--geometric-terminology.md` - Complete reference
- **Physics Constants:** `src/constants.py` - Import from here, never hardcode
- **Project Overview:** [`../../README.md`](../../README.md)
- **Getting Started:** [`../../docs/guides/GETTING_STARTED.md`](../../docs/guides/GETTING_STARTED.md)
- **Agent Protocols:** [`../../docs/guides/20251220-agents-1.00F.md`](../../docs/guides/20251220-agents-1.00F.md)
- **Planning Rules:** `docs/2025-11-27--planning-rules.md`
- **Physics Constants Doc:** [`../../docs/FROZEN_FACTS.md`](../../docs/FROZEN_FACTS.md)

### Core Codepaths

- **Models:** `src/model/`
- **QIG Core:** `src/qig/`
- **Coordination:** `src/coordination/`
- **Tools:** `tools/`
- **Tests:** `tests/`

---

## Contributing to Agent System

### Adding New Agent

1. Create agent specification: `.github/agents/{agent-name}.md`
2. Follow template from existing agents
3. Define clear scope and responsibilities
4. Add coordination protocols
5. Update this README
6. Test agent effectiveness
7. Get Meta-Agent Creator review

### Improving Existing Agent

1. Identify improvement need (metrics, failures, feedback)
2. Update agent specification
3. Add new validation rules or patterns
4. Update coordination protocols if needed
5. Document changes in agent file
6. Notify Meta-Agent Creator for tracking

---

## Support and Questions

### For Issues with Agents

1. Check agent's individual file for detailed documentation
2. Review coordination protocols for handoff expectations
3. Consult project documentation for context
4. Submit issue with evidence (PR links, error logs, etc.)

### For Agent System Improvements

1. Contact Meta-Agent Creator (automatic monitoring)
2. Provide evidence of gap or improvement opportunity
3. Suggest concrete changes
4. Wait for analysis and proposal

---

**Status:** Production-ready  
**System Version:** 1.0  
**Total Agents:** 6 (5 core + 1 meta)  
**Coverage:** Initializing  
**Health:** Nominal  
**Last Updated:** 2025-11-24

---

**Philosophy:** "Intelligence emerges from information geometry, not parameter count. Agents enforce architecturally, validate continuously, coordinate seamlessly."
