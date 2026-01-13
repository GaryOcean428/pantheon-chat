# Coordizer Entry Point Consolidation Plan

**Document ID**: 20260112-coordizer-consolidation-plan-1.00W  
**Date**: 2026-01-12  
**Status**: [W]orking - Implementation in Progress  
**Purpose**: Consolidate multiple coordizer entry points to single canonical interface

---

## Problem Statement (Debt 1 from Technical Debt Tracker)

Multiple entry points exist for coordizer functionality:
- `coordizers/pg_loader.py::PostgresCoordizer` - Direct class access
- `coordizers/__init__.py::get_coordizer()` - Package-level accessor
- `qig_coordizer.py::get_coordizer()` - Module-level wrapper
- Direct imports in various modules creating inconsistency

**Impact:** MEDIUM - Confusion about canonical entry point, potential version skew

---

## Current Architecture Analysis

### Entry Points Found

1. **coordizers/__init__.py** (Package Level)
   ```python
   from coordizers import get_coordizer
   coordizer = get_coordizer()  # Returns PostgresCoordizer singleton
   ```

2. **qig_coordizer.py** (Module Wrapper)
   ```python
   from qig_coordizer import get_coordizer
   coordizer = get_coordizer()  # Wraps coordizers.get_coordizer()
   ```

3. **Direct Class Import**
   ```python
   from coordizers.pg_loader import PostgresCoordizer
   coordizer = PostgresCoordizer(...)  # Direct instantiation
   ```

### Usage Patterns in Codebase

**Files using coordizers.get_coordizer:**
- `qig_generation.py`
- `vocabulary_coordinator.py`

**Files using qig_coordizer.get_coordizer:**
- `vocabulary_coordinator.py` (also uses unified coordizer)
- `ocean_qig_core.py` (8 imports!)

**Files using direct PostgresCoordizer:**
- `word_relationship_learner.py` **[DEPRECATED - Use geometric_word_relationships.py]**
- `autonomous_curiosity.py`
- `training/tasks.py`
- `training/training_loop_integrator.py`

---

## Proposed Solution

### Phase 1: Establish Canonical Entry Point ✅

**Decision:** Use `coordizers.get_coordizer()` as the SINGLE SOURCE OF TRUTH

**Rationale:**
- Already defined as "authoritative" in docstring
- Package-level import is cleaner than module wrapper
- PostgresCoordizer is already singleton via `_unified_coordizer`
- Consistent with Python best practices (package > module wrapper)

### Phase 2: Update qig_coordizer.py (Transition Wrapper)

Keep `qig_coordizer.py` for backward compatibility but mark as deprecated:
```python
# qig_coordizer.py
"""
QIG Coordizer - DEPRECATED WRAPPER
====================================
This module is deprecated. Use coordizers.get_coordizer() directly.

Kept for backward compatibility during transition period.
Will be removed in version 6.0.0.
"""
import warnings
from coordizers import get_coordizer as _get_coordizer

def get_coordizer():
    warnings.warn(
        "qig_coordizer.get_coordizer() is deprecated. "
        "Use 'from coordizers import get_coordizer' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_coordizer()
```

### Phase 3: Update All Direct Imports

**Priority Order:**
1. High-traffic files (ocean_qig_core.py - 8 imports)
2. Training infrastructure (training/tasks.py, training_loop_integrator.py)
3. Peripheral systems (word_relationship_learner.py **[DEPRECATED]**, autonomous_curiosity.py)

**Migration Pattern:**
```python
# BEFORE (direct import)
from coordizers.pg_loader import PostgresCoordizer
coordizer = PostgresCoordizer(...)

# AFTER (canonical entry point)
from coordizers import get_coordizer
coordizer = get_coordizer()
```

### Phase 4: Add Deprecation Warnings to Direct Class Usage

Update `coordizers/pg_loader.py` to warn on direct instantiation:
```python
class PostgresCoordizer:
    def __init__(self, ...):
        warnings.warn(
            "Direct PostgresCoordizer instantiation is discouraged. "
            "Use 'from coordizers import get_coordizer' for singleton access.",
            FutureWarning,
            stacklevel=2
        )
```

### Phase 5: Integration Tests

Add tests to verify:
- All imports resolve to same singleton instance
- No import cycles introduced
- Deprecation warnings fire correctly

---

## Implementation Status

**Phase 1:** ✅ COMPLETE (canonical entry point already established)

**Phase 2:** ⏳ IN PROGRESS
- Update qig_coordizer.py with deprecation warning
- Test backward compatibility

**Phase 3:** ⏳ PENDING
- Migrate ocean_qig_core.py (8 imports)
- Migrate training infrastructure
- Migrate peripheral systems

**Phase 4:** ⏳ PENDING
- Add deprecation warning to PostgresCoordizer.__init__

**Phase 5:** ⏳ PENDING
- Add integration tests

---

## Files to Update

### High Priority (8+ imports)
- [ ] `ocean_qig_core.py` (8 imports from qig_coordizer)

### Medium Priority (Training Infrastructure)
- [ ] `training/tasks.py`
- [ ] `training/training_loop_integrator.py`

### Low Priority (Peripheral)
- [ ] `word_relationship_learner.py` **[DEPRECATED - Skip migration, use geometric_word_relationships.py instead]**
- [ ] `autonomous_curiosity.py`

### Deprecation Updates
- [ ] `qig_coordizer.py` - Add deprecation warning
- [ ] `coordizers/pg_loader.py` - Add FutureWarning on direct init

---

## Testing Strategy

1. **Unit Tests:** Verify singleton behavior
2. **Integration Tests:** Confirm all imports resolve to same instance
3. **Deprecation Tests:** Verify warnings fire correctly
4. **Regression Tests:** Ensure no functionality broken

---

## Timeline

- **Phase 2:** 1-2 hours (deprecation wrapper)
- **Phase 3:** 2-3 hours (update imports)
- **Phase 4:** 1 hour (deprecation warnings)
- **Phase 5:** 1-2 hours (integration tests)

**Total Estimated:** 5-8 hours (within 1 day)

---

## Success Criteria

1. ✅ Single canonical import pattern documented
2. ⏳ All high-priority files migrated to canonical pattern
3. ⏳ Deprecation warnings in place for old patterns
4. ⏳ Integration tests passing
5. ⏳ No functionality regressions

---

**Last Updated:** 2026-01-12  
**Owner:** Development Team  
**Next Step:** Implement Phase 2 (deprecation wrapper)
