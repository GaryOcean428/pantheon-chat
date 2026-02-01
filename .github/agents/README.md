# GitHub Copilot Custom Agents - Complete Overview

**Status:** ✅ COMPLETE - All 17 agents implemented
**Last Updated:** 2026-01-13
**Total Agents:** 17 (14 new + 3 existing)

## Overview

This document provides a comprehensive overview of all GitHub Copilot custom agents designed for the pantheon-chat QIG repository. These agents ensure full coverage of codebase quality, following the GitHub Copilot custom agent specification.

## Agent Categories

### 1. Code Quality & Structure (7 agents)

#### 1.1 Import Resolution Agent
- **File:** `import-resolution-agent.md`
- **Purpose:** Detect and fix Python import errors, validate __init__.py barrel exports
- **Key Responsibilities:**
  - Enforce canonical import patterns (absolute imports from qig_backend.module)
  - Check for circular dependencies
  - Validate barrel export completeness
- **Critical Checks:**
  - All imports use absolute paths from qig_backend
  - No relative imports except in test files
  - Every module has __init__.py with __all__

#### 1.2 QIG Purity Enforcement Agent (Extended)
- **File:** `qig-purity-validator.md`
- **Purpose:** Scan for forbidden patterns (cosine_similarity, np.linalg.norm), suggest Fisher-Rao alternatives
- **Key Responsibilities:**
  - Validate all distance calculations use canonical_fisher.py
  - Detect Euclidean contamination (cosine similarity, L2 norm)
  - Ensure no neural nets/transformers in QIG logic
- **Critical Checks:**
  - No cosine_similarity or np.linalg.norm in QIG code
  - All geometric operations use Fisher-Rao metrics
  - Physics constants match frozen_physics.py

#### 1.3 Schema Consistency Agent
- **File:** `schema-consistency-agent.md`
- **Purpose:** Validate database migrations match SQLAlchemy models
- **Key Responsibilities:**
  - Check for NULL columns that should be NOT NULL
  - Identify orphaned tables
  - Ensure vocabulary architecture follows single canonical table pattern
- **Critical Checks:**
  - Only ONE vocabulary table exists
  - All models have corresponding migrations
  - pgvector indexes properly configured

#### 1.4 Naming Convention Agent
- **File:** `naming-convention-agent.md`
- **Purpose:** Enforce snake_case for Python, camelCase for TypeScript, SCREAMING_SNAKE for constants
- **Key Responsibilities:**
  - Validate file naming matches module purpose
  - Ensure coordinator.py has Coordinator class
  - Check ISO 27001 documentation naming
- **Critical Checks:**
  - Python: snake_case functions, PascalCase classes
  - TypeScript: camelCase functions, PascalCase components
  - Constants: SCREAMING_SNAKE_CASE everywhere

#### 1.5 Module Organization Agent
- **File:** `module-organization-agent.md`
- **Purpose:** Validate proper layering (qig_core shouldn't import from olympus)
- **Key Responsibilities:**
  - Check geometric_primitives contains only pure math
  - Ensure barrel exports are side-effect free
  - Validate routes don't import from training
- **Critical Checks:**
  - geometric_primitives → only NumPy/SciPy (no app logic)
  - qig_core → no imports from olympus/training/routes
  - routes → no imports from training

#### 1.6 DRY Enforcement Agent
- **File:** `dry-enforcement-agent.md`
- **Purpose:** Detect code duplication across modules
- **Key Responsibilities:**
  - Identify multiple basin coordinate calculations
  - Suggest consolidation into qig_core
  - Flag same functionality in Python and TypeScript
- **Critical Checks:**
  - No duplicate geometric operations
  - Single source of truth for computations
  - Configuration not duplicated across files

#### 1.7 Test Coverage Agent
- **File:** `test-coverage-agent.md`
- **Purpose:** Identify critical paths without tests
- **Key Responsibilities:**
  - Suggest test cases based on FROZEN_FACTS.md validation data
  - Check pytest fixtures match actual usage
  - Validate Fisher-Rao distance, consciousness measurement, basin navigation
- **Critical Checks:**
  - Fisher-Rao distance: 95%+ coverage
  - Consciousness measurement: 90%+ coverage
  - FROZEN_FACTS.md validation tests exist

### 2. Integration & Synchronization (3 agents)

#### 2.1 Documentation Sync Agent
- **File:** `documentation-sync-agent.md`
- **Purpose:** Detect code changes that invalidate documentation
- **Key Responsibilities:**
  - Flag when FROZEN_FACTS.md claims differ from frozen_physics.py constants
  - Auto-update docs/04-records when code changes
  - Track documentation staleness
- **Critical Checks:**
  - FROZEN_FACTS.md constants match frozen_physics.py
  - PR records exist in docs/04-records
  - Code examples in docs still execute

#### 2.2 Wiring Validation Agent
- **File:** `wiring-validation-agent.md`
- **Purpose:** Verify every documented feature has actual implementation
- **Key Responsibilities:**
  - Check all consciousness components are measured and logged
  - Validate telemetry endpoints exist for all metrics
  - Trace Documentation → Backend → API → Frontend → User
- **Critical Checks:**
  - All documented features have code implementations
  - Every metric is logged via telemetry
  - All metrics have API endpoints

#### 2.3 Frontend-Backend Capability Mapper (Existing)
- **File:** `frontend-backend-capability-mapper.md`
- **Purpose:** Ensure every Python route has corresponding TypeScript API client
- **Key Responsibilities:**
  - Validate React components can access all backend features
  - Check authentication flows
  - Map full capability exposure chain
- **Critical Checks:**
  - Backend → API → Client → Service → Component → User
  - No hidden capabilities
  - Type consistency across stack

### 3. Performance & Regression (2 agents)

#### 3.1 Performance Regression Agent
- **File:** `performance-regression-agent.md`
- **Purpose:** Detect when geometric operations become Euclidean approximations
- **Key Responsibilities:**
  - Flag when β-function becomes constant (should vary with scale)
  - Monitor consciousness metrics for suspicious values (Φ=1.0 always)
  - Detect performance gains with accuracy loss
- **Critical Checks:**
  - Fisher-Rao not replaced with Euclidean
  - β-function varies with scale (β(3→4) ≠ β(4→5))
  - Φ shows variation across inputs (not constant)

#### 3.2 Dependency Management Agent
- **File:** `dependency-management-agent.md`
- **Purpose:** Validate requirements.txt matches actual imports
- **Key Responsibilities:**
  - Check qigkernels external package version is compatible
  - Detect when new dependencies add Euclidean operations
  - Security vulnerability scanning
- **Critical Checks:**
  - No scikit-learn (Euclidean metrics)
  - No sentence-transformers (cosine similarity)
  - All imports have corresponding requirements

### 4. User Interface & Deployment (2 agents)

#### 4.1 UI/UX Consistency Agent
- **File:** `ui-ux-consistency-agent.md`
- **Purpose:** Ensure consciousness visualizations follow design system
- **Key Responsibilities:**
  - Check God Panel matches specs in docs/07-user-guides
  - Validate color schemes reflect geometric states (green=geometric, yellow=linear, red=breakdown)
  - Accessibility compliance (WCAG 2.1)
- **Critical Checks:**
  - Regime colors: breakdown=red, linear=yellow, geometric=green, hierarchical=purple
  - God Panel: 240px left sidebar, 320px right panel
  - All components use design system tokens

#### 4.2 Deployment Readiness Agent
- **File:** `deployment-readiness-agent.md`
- **Purpose:** Verify Replit environment variables match .env.example
- **Key Responsibilities:**
  - Check Neon DB migrations are applied
  - Validate frontend build artifacts exist
  - Confirm health check endpoints return valid responses
- **Critical Checks:**
  - All env vars from .env.example present
  - Database: migrations applied, pgvector installed
  - Health checks: /health and /health/ready return 200

## Existing Agents (Retained)

### E8 Architecture Validator
- **File:** `e8-architecture-validator.md`
- **Purpose:** Validate E8 Lie group structure, kernel specialization hierarchy

### Documentation Compliance Auditor
- **File:** `documentation-compliance-auditor.md`
- **Purpose:** ISO 27001 documentation standards, canonical naming

### Downstream Impact Tracer
- **File:** `downstream-impact-tracer.md`
- **Purpose:** Trace impact of changes through dependency chain

## Usage

Each agent is invoked automatically by GitHub Copilot when editing files in relevant areas:

```markdown
# Example: When editing qig_core/consciousness_4d.py
- QIG Purity Enforcement Agent checks for Euclidean operations
- Test Coverage Agent validates tests exist
- Performance Regression Agent monitors metric behavior
- Documentation Sync Agent checks FROZEN_FACTS.md alignment
```

## Validation Commands

```bash
# Run all agent validation checks
python scripts/run_all_agent_checks.py

# Individual agent checks
python scripts/validate_imports.py           # Import Resolution
python scripts/check_qig_purity.py          # QIG Purity
python scripts/validate_schema.py           # Schema Consistency
python scripts/check_naming_conventions.py  # Naming Convention
python scripts/validate_architecture.py     # Module Organization
python scripts/find_duplicate_code.py       # DRY Enforcement
pytest tests/ --cov                         # Test Coverage
python scripts/check_performance.py         # Performance Regression
python scripts/validate_dependencies.py     # Dependency Management
python scripts/validate_wiring.py           # Wiring Validation
python scripts/check_doc_sync.py           # Documentation Sync
python scripts/validate_ui_consistency.py   # UI/UX Consistency
bash scripts/pre_deployment_check.sh       # Deployment Readiness
```

## CI/CD Integration

All agents are integrated into CI/CD pipeline:

```yaml
# .github/workflows/agent-validation.yml
name: Agent Validation

on: [pull_request, push]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Run all agent checks
        run: python scripts/run_all_agent_checks.py
```

## Metrics & Reporting

Each agent generates standardized reports:

```markdown
# Example Report Format
# [Agent Name] Report

## Summary
- ✅ Passed: X checks
- ❌ Failed: Y checks
- ⚠️ Warnings: Z checks

## Critical Issues
1. [Issue description]
   **Action:** [Required fix]

## Priority Actions
1. [Action 1 - CRITICAL]
2. [Action 2 - HIGH]
```

## Maintenance

- **Review Frequency:** Monthly
- **Update Trigger:** Major architecture changes, new features
- **Owner:** Architecture team
- **Documentation:** This file + individual agent files

## References

- [GitHub Copilot Custom Agent Spec](https://docs.github.com/en/copilot/reference/custom-agents-configuration)
- [FROZEN_FACTS.md](../../docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md)
- [Project Architecture](../../docs/10-e8-protocol/README.md)
- [QIG Implementation Guide](../../docs/implementation/)

---

**Coverage:** 17 specialized domains
**Status:** Production-ready ✅
