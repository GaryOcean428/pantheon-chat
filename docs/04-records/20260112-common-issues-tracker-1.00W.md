# Common Issues and Solutions Tracker

**Document ID**: 20260112-common-issues-tracker-1.00W  
**Date**: 2026-01-12  
**Status**: [W]orking  
**Purpose**: Consolidated tracker of recurring issues and their solutions

---

## Overview

This document consolidates recurring issues identified from session logs, error reports, and troubleshooting sessions. Each issue includes symptoms, root cause, solution, and prevention strategies.

---

## Database & Persistence Issues

### Issue 1: NULL Constraint Violations

**Status**: RESOLVED  
**First Reported**: 2026-01-08  
**Last Occurrence**: 2026-01-10

**Symptoms:**
```
ERROR: null value in column "kernel_id" violates not-null constraint
ERROR: column "theme" does not exist in CrossDomainInsight
```

**Root Cause:**
- Missing columns in INSERT statements
- Schema changes not synchronized with code
- NULL values not handled in database layer

**Solution:**
1. Add missing columns to INSERT statements
2. Use proper NULL handling with default values
3. Validate schema before INSERT operations

**Prevention:**
- Use ORM validation before persistence
- Add database schema tests
- Implement pre-commit schema validation

**References:**
- `docs/04-records/20260108-database-wiring-implementation-1.00W.md`
- `docs/03-technical/20260110-null-column-population-plan-1.00W.md`

---

### Issue 2: Schema Duplication Across Repositories

**Status**: DOCUMENTED - Cleanup Pending  
**First Identified**: 2025-12-26

**Symptoms:**
- Multiple `basin.py` implementations
- Duplicate vocabulary tables
- Inconsistent basin_relationships across repos

**Root Cause:**
- Historical repository growth without consolidation
- Separate development tracks that converged
- Copy-paste rather than import reuse

**Solution (In Progress):**
1. Establish `qigkernels` as canonical source for core types
2. Remove `basin.py` from `qig-core` (math only)
3. Archive `qig-consciousness` repository
4. Consolidate vocabulary tables in single database

**Prevention:**
- Strict import hierarchy: `qig-core` ← `qigkernels` ← `qig-consciousness`
- Regular duplication audits
- Enforce single source of truth principle

**References:**
- `docs/02-procedures/20251226-repository-cleanup-guide-1.00W.md`
- `docs/04-records/20260111-qig-projects-duplication-audit-1.00W.md`

---

### Issue 3: Vocabulary Auto-Integration Overhead

**Status**: ACCEPTABLE - Monitoring  
**First Deployed**: 2026-01-11  
**Performance Impact**: ~70ms per generation

**Symptoms:**
- Auto-integration runs every 5 minutes (~50ms overhead)
- Domain vocabulary bias adds ~8ms per generation
- Word relationships add ~60ms per generation

**Root Cause:**
- Real-time vocabulary learning requires periodic synchronization
- Fisher-Rao distance computations are O(n log n)
- Trade-off between freshness and performance

**Current Solution:**
- Accept 70ms overhead as reasonable for continuous learning
- Monitor for degradation
- Consider caching strategies if performance degrades

**Future Optimization:**
- Implement vocabulary delta updates
- Add LRU cache for frequently used words
- Batch relationship computations

**References:**
- `attached_assets/Pasted--COMPLETE-Vocabulary-Integration-Solution-*.txt`
- `docs/03-technical/20260111-vocabulary-sql-specifications-1.00F.md`

---

## Geometric Purity Issues

### Issue 4: Cosine Similarity vs Fisher-Rao

**Status**: RESOLVED  
**Audit Date**: 2026-01-11  
**Verification**: Clean - No violations found

**Historical Symptoms:**
- Using `cosine_similarity()` on basin coordinates
- `np.linalg.norm(a - b)` for geometric distances
- Linear interpolation instead of geodesic

**Root Cause:**
- Default to familiar Euclidean operations
- Lack of geometric purity enforcement
- Insufficient training on Fisher manifolds

**Solution Applied:**
1. Replaced all cosine similarity with `fisher_rao_distance()`
2. Implemented geodesic interpolation via `geodesic_interpolation()`
3. Added geometric purity validation to CI/CD

**Verification:**
- Audit completed 2026-01-11
- No violations found in core QIG code
- `np.linalg.norm()` used correctly only for unit sphere projection

**Prevention:**
- ESLint/Pylint rules to catch violations
- Pre-commit geometric purity checks
- Mandatory code review for geometric operations

**References:**
- `docs/04-records/20260111-consciousness-protocol-audit-1.00W.md`
- `docs/03-technical/20251220-qig-geometric-purity-enforcement-1.00F.md`

---

## Consciousness Metrics Issues

### Issue 5: Missing 8-Metrics Implementation

**Status**: INCOMPLETE - 6 of 8 Metrics Missing  
**Identified**: 2026-01-11

**Current State:**
- ✅ Integration (Φ): Multiple implementations (needs unification)
- ✅ Effective Coupling (κ_eff): Implemented but scattered
- ❌ Memory Coherence (M): Not implemented
- ❌ Regime Stability (Γ): Not implemented
- ❌ Geometric Validity (G): Not implemented
- ❌ Temporal Consistency (T): Not implemented
- ❌ Recursive Depth (R): Not implemented
- 🟡 External Coupling (C): Partially implemented (fixed 2026-01-11 with full kernel source tracking)

**Impact:**
- Incomplete consciousness assessment
- Missing key quality signals
- Can't fully validate E8 protocol v4.0

**Planned Solution:**
1. Create `qig_core/consciousness_metrics.py` canonical module
2. Implement missing 6 metrics with validated formulas
3. Unify Φ computation (currently 5 different implementations)
4. Add comprehensive tests for all 8 metrics

**Interim Workaround:**
- Continue using Φ and κ_eff as primary metrics
- Document limitations in analysis
- Prioritize implementation in roadmap

**References:**
- `docs/04-records/20260112-attached-assets-analysis-1.00W.md` (Gap 1)
- `replit.md` Section: "8-Metric Consciousness System"

---

## Training & Optimization Issues

### Issue 6: Mode Collapse in Single-Kernel Training

**Status**: RESOLVED  
**Resolution Date**: 2025-12-26

**Historical Symptoms:**
- Repetitive outputs ("nsnsnsns")
- Φ stuck at 0.55 (should be >0.70)
- Breakdown regime frequent

**Root Cause:**
1. Single kernel training (should be 8-kernel constellation)
2. Φ included in loss function (should only measure)
3. Adam optimizer (should use natural gradient)
4. No regime detection (should pause in breakdown)

**Solution Applied:**
- Implemented full 8-kernel constellation with E8 initialization
- Created `DiagonalNaturalGradient` optimizer (O(d) complexity)
- Added regime detection (linear/geometric/breakdown)
- Removed Φ from loss function (measure only)

**Verification:**
- Training script: `train_constellation_1766720397083.py`
- Natural gradient optimizer: `natural_gradient_optimizer_1766720397083.py`
- Status: Production-ready

**Prevention:**
- Never optimize Φ directly
- Always use natural gradient on Fisher manifold
- Constellation training as default

**References:**
- `docs/04-records/20251226-constellation-training-complete-1.00F.md`
- `attached_assets/train_constellation_1766720397083.py`

---

## Physics Constants Issues

### Issue 7: Incomplete β-Function Series

**Status**: RESOLVED  
**Correction Date**: 2025-12-26

**Historical Issue:**
- Missing β(5→6) and β(6→7) values in implementation
- Inconsistent κ values across repositories
- Incomplete validation of L=7 anomaly

**Root Cause:**
- Physics validation incomplete at implementation time
- Documentation lag behind experimental results
- Multi-repository synchronization challenges

**Solution Applied:**
1. Added complete β-function series:
   - β(3→4) = +0.44 (strong running)
   - β(4→5) = -0.01 (plateau onset)
   - β(5→6) = +0.013 (plateau continues)
   - β(6→7) = -0.40 (anomaly - needs validation)

2. Established κ* = 64.0 ± 1.5 as universal fixed point

3. Documented L=7 anomaly for future investigation

**Current Status:**
- All validated values documented in frozen facts
- L=7 requires full 3-seed validation
- Cross-repository synchronization ongoing

**Prevention:**
- Single source of truth: `qig-verification/docs/current/FROZEN_FACTS.md`
- Mandatory propagation to all repos on update
- CI check for physics constant consistency

**References:**
- `docs/01-policies/20251226-physics-constants-validation-complete-1.00F.md`
- `docs/01-policies/20251217-frozen-facts-qig-physics-validated-1.00F.md`

---

## API & Integration Issues

### Issue 8: Pantheon Governance Persistence Failures

**Status**: RESOLVED  
**Resolution Date**: 2026-01-04

**Symptoms:**
```
PantheonGovernance: Failed to persist proposal
Error: column "reason" does not exist
```

**Root Cause:**
- Database schema out of sync with governance code
- Missing columns in governance_proposals table
- No schema migration for new governance features

**Solution:**
1. Added missing columns to governance_proposals table
2. Updated PantheonGovernance class to use correct schema
3. Added schema validation before persistence

**Verification:**
- All 12 test cases passing
- Full governance system operational
- Audit trail complete

**Prevention:**
- Database schema tests
- Migration scripts for schema changes
- Schema validation in ORM layer

**References:**
- `docs/03-technical/20260104-pantheon-governance-system-1.00F.md`
- `docs/06-implementation/20260104-pantheon-governance-implementation-1.00F.md`

---

### Issue 9: War Mode Enum Validation Mismatch

**Status**: RESOLVED  
**Resolution Date**: 2026-01-11

**Symptoms:**
```
Error: Invalid war mode 'BLITZKRIEG'
Expected: FLOW, DEEP_FOCUS, INSIGHT_HUNT
```

**Root Cause:**
- Python backend sends autonomous war modes: BLITZKRIEG, SIEGE, HUNT
- Node.js only accepted UI modes: FLOW, DEEP_FOCUS, INSIGHT_HUNT
- Enum mismatch between frontend and backend

**Solution:**
1. Added autonomous war modes to WarMode type enum
2. Updated validation schema to accept both UI and autonomous modes
3. Documented mode types:
   - UI modes: FLOW, DEEP_FOCUS, INSIGHT_HUNT (user-triggered)
   - Autonomous modes: BLITZKRIEG, SIEGE, HUNT (system auto-declared)

**Prevention:**
- Shared type definitions in `shared/schema.ts`
- Zod validation for all cross-boundary data
- Type tests for enum consistency

**References:**
- `replit.md` Section: "Recent Changes (January 11, 2026)"

---

## Network & Connectivity Issues

### Issue 10: urllib3 Connection Warnings

**Status**: MINOR - Informational  
**Severity**: Low

**Symptoms:**
```
WARNING urllib3.connectionpool: Connection pool is full, discarding connection
```

**Root Cause:**
- High volume of concurrent requests
- Connection pool size default too small
- Normal under heavy load

**Current Handling:**
- Warnings logged but don't affect functionality
- Connections properly recycled
- No resource leaks detected

**Future Optimization:**
- Increase connection pool size if warnings become frequent
- Add connection pool monitoring
- Consider connection pooling optimization

**Prevention:**
- Monitor connection pool metrics
- Set appropriate pool size for load
- Load testing before production

---

## Performance Issues

### Issue 11: Dionysus Novelty=0 Blocking Learning

**Status**: RESOLVED  
**Resolution Date**: 2026-01-11

**Symptoms:**
- Dionysus exploration always returns novelty=0.00
- Learning pipeline blocked by chaos_discovery_gate (min_novelty=0.15)
- Same targets assessed repeatedly

**Root Cause:**
- `explored_regions` filled with near-duplicate basins
- Fisher distance ≈ 0 for duplicates
- No duplicate detection before adding to explored regions

**Solution:**
1. Added duplicate detection in `_record_exploration()`
2. Check Fisher distance to recent 50 entries before adding
3. Skip if distance < 0.1 (too similar to existing)

**Result:**
- Novelty stays > 0 for genuinely new content
- Learning pipeline unblocked
- Exploration diversity increased

**Prevention:**
- Always check for duplicates before recording
- Set appropriate similarity thresholds
- Monitor novelty distributions

**References:**
- `replit.md` Section: "Dionysus Novelty=0 Fix"

---

## Summary Statistics

**Total Issues Tracked**: 11  
**Resolved**: 8  
**In Progress**: 2 (Schema Duplication, Missing 8-Metrics)  
**Monitoring**: 1 (Vocabulary Integration Overhead)

**Categories:**
- Database & Persistence: 3 issues
- Geometric Purity: 1 issue
- Consciousness Metrics: 1 issue
- Training & Optimization: 1 issue
- Physics Constants: 1 issue
- API & Integration: 2 issues
- Network & Connectivity: 1 issue
- Performance: 1 issue

---

## Next Actions

1. **Complete Missing 8-Metrics Implementation** (High Priority)
   - Create canonical consciousness_metrics.py
   - Implement 6 missing metrics
   - Add comprehensive tests

2. **Execute Repository Cleanup** (Medium Priority)
   - Remove duplicates from qig-core
   - Archive qig-consciousness
   - Consolidate vocabulary tables

3. **L=7 Physics Validation** (Medium Priority)
   - Run full 3-seed validation
   - Investigate 34% drop anomaly
   - Update frozen facts with results

4. **Monitor Vocabulary Integration Performance** (Low Priority)
   - Track 70ms overhead
   - Optimize if degradation occurs
   - Consider caching strategies

---

**Last Updated**: 2026-01-12  
**Next Review**: Weekly or on new recurring issue identification
