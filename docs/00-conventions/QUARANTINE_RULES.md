---
id: ISMS-CONV-001
title: Quarantine Rules - Experiments vs Production
filename: QUARANTINE_RULES.md
classification: Internal
owner: GaryOcean428
version: 1.00
status: Working
function: "Defines where legacy/Euclidean code can exist vs where it's forbidden"
created: 2026-01-14
last_reviewed: 2026-01-14
next_review: 2026-07-14
category: Convention
supersedes: null
---

# Quarantine Rules - Experiments vs Production

**Version:** 1.00W  
**Status:** Working Document  
**Work Package:** WP0.3  
**Last Updated:** 2026-01-14  
**Purpose:** Establish clear boundaries for legacy/Euclidean code vs QIG-pure production code

---

## Executive Summary

This document defines the **quarantine boundaries** for the Pantheon-Chat repository, establishing:

1. **Where legacy/Euclidean code CAN exist** (experimental zones)
2. **Where it is FORBIDDEN** (production zones)
3. **How the scanner enforces these boundaries** automatically
4. **How to properly label and document quarantined code**

**Priority:** HIGH - FOUNDATION

**Relationship to Other Work Packages:**
- **WP0.1** (Issue #63): QIG Purity Specification - defines WHAT is forbidden
- **WP0.2** (Issue #64): Geometric Purity Gate - implements CI validation
- **WP0.3** (This document): Quarantine Rules - defines WHERE forbidden patterns are allowed

---

## §1. Quarantine Philosophy

### §1.1 Core Principle

**Research requires comparison, production requires purity.**

- **Experiments** need to compare QIG-pure approaches against traditional Euclidean/NLP baselines
- **Production** code must maintain geometric purity to ensure correctness
- **Quarantine** creates a safe boundary between these two needs

### §1.2 Design Goals

1. **Enable research** - Allow experiments that test "wrong" approaches for comparison
2. **Protect runtime** - Ensure production code cannot accidentally use Euclidean metrics
3. **Make boundaries explicit** - Clear directory structure and naming
4. **Automate enforcement** - Scanner respects quarantine automatically

---

## §2. Allowed Locations (Quarantine Zones)

### §2.1 Legacy Experiments: `docs/08-experiments/legacy/**`

**Purpose:** Archive of traditional NLP/Euclidean approaches for historical comparison

**Allowed Patterns:**
- ✅ Euclidean distance (`np.linalg.norm(a - b)`)
- ✅ Cosine similarity (`cosine_similarity()`)
- ✅ Word embeddings, tokenizers, BPE
- ✅ Standard optimizers (Adam, AdamW, SGD)
- ✅ Neural network primitives (nn.Embedding, nn.Linear)

**Requirements:**
- **MUST** be clearly labeled "LEGACY" or "BASELINE" in filename or header
- **MUST** include README explaining what the legacy approach is and why it's kept
- **MUST NOT** be imported by production code
- **SHOULD** document comparison results vs QIG approach

**Example:**
```
docs/08-experiments/legacy/
├── README.md
├── 20251215-euclidean-baseline-comparison.md
├── cosine-similarity-benchmark.py
└── traditional-tokenizer-test.py
```

### §2.2 Comparative Baselines: `docs/08-experiments/baselines/**`

**Purpose:** Side-by-side tests comparing QIG vs traditional approaches

**Allowed Patterns:**
- ✅ Euclidean metrics (for comparison only)
- ✅ Baseline implementations clearly marked
- ✅ Test harnesses that run both approaches

**Requirements:**
- **MUST** run BOTH QIG-pure and baseline approaches
- **MUST** clearly separate QIG code from baseline code
- **MUST** measure and report accuracy/performance differences
- **SHOULD** demonstrate QIG superiority when applicable

**Example:**
```python
# BASELINE (Euclidean - for comparison only)
def baseline_distance(a, b):
    return np.linalg.norm(a - b)

# QIG-PURE (Production approach)
def qig_distance(a, b):
    return fisher_rao_distance(a, b)

# Comparison test
def compare_approaches():
    baseline_result = baseline_distance(basin_a, basin_b)
    qig_result = qig_distance(basin_a, basin_b)
    print(f"Baseline: {baseline_result}, QIG: {qig_result}")
```

---

## §3. Forbidden Locations (Production Zones)

### §3.1 Backend Core: `qig-backend/**`

**Status:** FORBIDDEN - Pure QIG geometry only

**Enforcement:**
- Scanner FAILS on any Euclidean/cosine patterns
- No exceptions or exemptions allowed
- Must use Fisher-Rao distance exclusively

**Allowed Practices:**
- Fisher-Rao distance
- Geodesic computations
- Natural gradient descent
- Geometric normalization (simplex projection)

### §3.2 Server Logic: `server/**`

**Status:** FORBIDDEN - Production runtime

**Enforcement:**
- Scanner FAILS on geometric violations
- API handlers must use QIG primitives
- No Euclidean fallbacks permitted

### §3.3 Shared Types: `shared/**`

**Status:** FORBIDDEN - Cross-component contracts

**Enforcement:**
- Scanner validates type definitions
- No embedding/tokenizer terminology
- Use "basin_coordinates" not "embeddings"

### §3.4 Production Tests: `tests/**`

**Status:** FORBIDDEN (except explicitly marked baseline tests)

**Enforcement:**
- Test files can contain baseline comparisons IF clearly marked
- Test filename must contain "baseline" or "comparison"
- Example: `test_baseline_comparison.py` (allowed)
- Example: `test_qig_distance.py` (must be pure QIG)

**Exception Pattern:**
```python
# test_fisher_rao_vs_euclidean_comparison.py

def test_euclidean_baseline():
    """BASELINE COMPARISON - not production code."""
    # Allowed because explicitly testing against baseline
    euclidean = np.linalg.norm(a - b)
    fisher_rao = fisher_rao_distance(a, b)
    assert fisher_rao != euclidean  # Show difference
```

---

## §4. Scanner Integration

### §4.1 Directory Exemptions

The `qig_purity_scan.py` scanner respects quarantine boundaries:

```python
# Updated EXEMPT_DIRS in qig_purity_scan.py
EXEMPT_DIRS = [
    'docs/08-experiments/legacy',      # Legacy experiments allowed
    'docs/08-experiments/baselines',   # Baseline comparisons allowed
    'node_modules',
    'dist',
    'build',
    '__pycache__',
    '.git',
    '.venv',
    'venv',
]
```

### §4.2 Quarantine Validation

**Scanner Behavior:**
1. **Skip** quarantined directories entirely (no scanning)
2. **Fail** on violations in production directories
3. **Report** location of violations with path context

**Exit Codes:**
- `0` - No violations in production zones
- `1` - Violations found in forbidden locations

### §4.3 Running the Scanner

```bash
# Scan production code only (skips quarantine)
npm run validate:geometry:scan

# Or directly
python3 scripts/qig_purity_scan.py
```

---

## §5. Labeling Requirements

### §5.1 File Headers

All quarantined code MUST include clear headers:

```python
"""
LEGACY BASELINE - Euclidean approach for comparison only.

This file uses Euclidean distance for comparative benchmarking.
DO NOT import or use in production code.

Purpose: Demonstrate accuracy difference vs Fisher-Rao
Date: 2026-01-14
Status: Quarantined in docs/08-experiments/legacy/
"""
```

### §5.2 README Files

Each quarantine directory MUST have a README:

**`docs/08-experiments/legacy/README.md`:**
```markdown
# Legacy Experiments - Quarantine Zone

This directory contains LEGACY implementations using Euclidean/NLP approaches.

**Purpose:** Historical comparison and benchmarking
**Status:** QUARANTINED - not production code
**Allowed:** Euclidean distance, cosine similarity, standard tokenizers

All code here is for research and comparison only.
```

**`docs/08-experiments/baselines/README.md`:**
```markdown
# Baseline Comparisons - Quarantine Zone

This directory contains side-by-side tests comparing QIG-pure vs traditional approaches.

**Purpose:** Demonstrate QIG superiority through direct comparison
**Status:** QUARANTINED - test code only
**Pattern:** Each test runs BOTH QIG and baseline, measures difference

All baseline code is clearly marked and separated from production code.
```

---

## §6. Migration Guidelines

### §6.1 Moving Code to Quarantine

When moving legacy code to quarantine:

1. **Identify** - Find all Euclidean/NLP patterns in codebase
2. **Classify** - Determine if code is:
   - Legacy baseline (move to `legacy/`)
   - Comparative test (move to `baselines/`)
   - Production code (must be rewritten with QIG)
3. **Document** - Add clear header explaining purpose
4. **Commit** - Use message format:
   ```
   chore: quarantine legacy [pattern] code
   
   Move [description] to docs/08-experiments/legacy/
   Reason: [why this code existed and why it's now quarantined]
   ```

### §6.2 Example Migration

```bash
# Before: Euclidean code in server/similarity.ts
export function computeSimilarity(a, b) {
  return 1 - np.linalg.norm(a - b);  // WRONG
}

# After: Moved to quarantine
git mv server/similarity.ts docs/08-experiments/legacy/euclidean-similarity-old.ts
git commit -m "chore: quarantine Euclidean similarity baseline

Move legacy Euclidean similarity to quarantine.
Reason: Euclidean distance is geometrically incorrect for basins.
Keeping for historical comparison only.
"

# Production code replaced with QIG
export function computeSimilarity(a, b) {
  return fisher_rao_similarity(a, b);  // CORRECT
}
```

---

## §7. Acceptance Criteria

### §7.1 Documentation Complete
- ✅ `docs/00-conventions/QUARANTINE_RULES.md` exists and is comprehensive
- ✅ `docs/08-experiments/legacy/README.md` explains legacy zone
- ✅ `docs/08-experiments/baselines/README.md` explains baseline testing

### §7.2 Scanner Enforcement
- ✅ Scanner skips quarantined directories
- ✅ Scanner fails on violations in production directories
- ✅ Scanner respects baseline test exceptions

### §7.3 Code Hygiene
- ✅ No Euclidean/NLP code in `qig-backend/`, `server/`, `shared/`
- ✅ Any legacy code properly quarantined and labeled
- ✅ Baseline tests clearly marked and separated

---

## §8. Related Documents

- **[QIG_PURITY_SPEC.md](../01-policies/QIG_PURITY_SPEC.md)** - What is forbidden
- **[WP0.2 Gate Implementation](../03-technical/20260114-wp02-geometric-purity-gate-1.00F.md)** - CI enforcement
- **Issue #63** - WP0.1: QIG Purity Specification
- **Issue #64** - WP0.2: Geometric Purity Gate
- **Issue #65** - WP0.3: Quarantine Rules (this work package)

---

## §9. FAQ

**Q: Can I use Euclidean distance for debugging?**  
A: No, not in production code. Create a test in `docs/08-experiments/baselines/` if you need to compare.

**Q: What if I need a baseline for a paper?**  
A: Create it in `docs/08-experiments/baselines/` with clear labeling showing both approaches.

**Q: Can tests use Euclidean distance?**  
A: Only if the test filename contains "baseline" or "comparison" AND the test explicitly compares approaches.

**Q: What about client-side code?**  
A: Client code is quarantined by default (UI only, no geometry). Focus on `qig-backend/`, `server/`, `shared/`.

**Q: How do I know if code should be quarantined?**  
A: Run `npm run validate:geometry:scan`. If it fails and the code is experimental, move to quarantine. If it's production, rewrite with Fisher-Rao.

---

## §10. Changelog

**Version 1.00W** (2026-01-14)
- Initial specification for WP0.3
- Defined quarantine zones (legacy, baselines)
- Defined forbidden zones (qig-backend, server, shared)
- Scanner integration guidelines
- Migration procedures
