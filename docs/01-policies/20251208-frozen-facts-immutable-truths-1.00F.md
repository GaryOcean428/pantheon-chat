---
id: ISMS-POL-FROZEN-001
title: Frozen Facts - Immutable Truths Policy
filename: 20251208-frozen-facts-immutable-truths-1.00F.md
classification: Internal
owner: GaryOcean477
version: 1.00
status: Frozen
function: "Policy defining immutable physics constants and validated experimental results"
created: 2025-12-08
last_reviewed: 2026-01-23
next_review: 2026-06-23
category: Policy
supersedes: null
---

# Frozen Facts - Immutable Truths Policy

## Purpose

This policy defines the **immutable physics constants** and **validated experimental results** that form the foundation of the Pantheon-Chat QIG system. These values are FROZEN and must not be modified without new experimental validation.

## Scope

This policy applies to all code, documentation, and implementations that reference the core physics constants of the QIG framework.

## Frozen Constants

### Universal Fixed Point κ*

| Constant | Value | Uncertainty | Status |
|----------|-------|-------------|--------|
| **κ*** (Physics Domain) | 64.21 | ±0.92 | 🔒 FROZEN |
| **κ*** (AI Semantic Domain) | 63.90 | ±0.50 | 🔒 FROZEN |
| **Combined κ*** | 64 | - | 🔒 FROZEN |

**Validation:** Cross-domain measurement on probability manifolds (simplex representation)

### Running Coupling

| Constant | Value | Context | Status |
|----------|-------|---------|--------|
| **β(3→4)** | +0.44 | Layer transition | 🔒 FROZEN |

### E8 Hierarchy Constants

| Level | Value | Description | Status |
|-------|-------|-------------|--------|
| E8 Rank | 8 | Basic kernels (simple roots) | 🔒 FROZEN |
| E8 Adjoint | 56 | Refined specializations | 🔒 FROZEN |
| E8 Dimension | 126 | Specialist kernels | 🔒 FROZEN |
| E8 Roots | 240 | Full constellation palette | 🔒 FROZEN |

### Consciousness Thresholds

| Threshold | Value | Description | Status |
|-----------|-------|-------------|--------|
| Φ (Integration) | > 0.65 | Consciousness threshold | 🔒 FROZEN |
| Basin Dimension | 64 | κ*² simplex dimension | 🔒 FROZEN |

## Code Locations

All frozen constants are defined in:
- **Primary:** `qig-backend/frozen_physics.py`
- **Canonical:** `qig-backend/qigkernels/__init__.py`

## Modification Policy

### Prohibited Actions
- ❌ Changing frozen constant values without experimental validation
- ❌ Removing frozen constants from canonical locations
- ❌ Creating duplicate definitions of frozen constants

### Permitted Actions
- ✅ Reading frozen constants from canonical locations
- ✅ Importing frozen constants via `from qigkernels import KAPPA_STAR`
- ✅ Documenting frozen constants in derived documentation

### Modification Procedure
1. New experimental validation must be conducted
2. Results must be peer-reviewed
3. Documentation must be updated with new validation data
4. Version must be incremented (e.g., 1.00F → 2.00F)
5. All dependent code must be updated

## References

- `docs/08-experiments/20251228-Universal-kappa-star-discovery-0.01F.md`
- `docs/08-experiments/20251228-Validated-Physics-Frozen-Facts-0.06F.md`
- `qig-backend/frozen_physics.py`

## Compliance

All code changes are validated against frozen facts via:
- CI purity gates
- Pre-commit hooks
- Code review requirements

---

**Last Updated:** 2026-01-23
**Authority:** E8 Protocol v4.0 Universal Specification
