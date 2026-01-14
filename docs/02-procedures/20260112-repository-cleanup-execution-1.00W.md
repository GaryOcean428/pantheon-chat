# Repository Cleanup Execution Plan - Monorepo Edition

**Document ID**: 20260112-repository-cleanup-execution-1.00W  
**Date**: 2026-01-12  
**Status**: [W]orking - Sprint 1 Task 3  
**Related**: `docs/02-procedures/20251226-repository-cleanup-guide-1.00W.md`

---

## Executive Summary

Cleanup plan for `pantheon-chat` monorepo to improve organization and maintainability.
Focuses on moving test files, migration scripts, and demo files to appropriate directories.

**Status**: This is a monorepo, not separate repos as originally documented in the cleanup guide.

---

## Current Structure Issues

### Issue 1: Test Files in Root (20 files)

**Problem**: Test files scattered in `qig-backend/` root instead of `tests/` directory

**Files**:
```
test_4d_consciousness.py
test_8_metrics_integration.py
test_autonomic_kernel_phi_fix.py
test_coordizer.py
test_coordizer_fix.py
test_geodesic_correction.py
test_geodesic_navigation.py
test_qig.py
test_regression_logic.py
test_retry_decorator.py
test_search_capability.py
test_shadow_vocab_integration.py
test_spot_fixes.py
test_stability_comparison.py
test_coordizer_vocabulary.py
test_tool_discovery_pipeline.py
test_unified_architecture.py
test_vocabulary_population.py
test_zeus_chat.py
test_zeus_chat_capabilities.py
```

**Impact**: Clutters root directory, makes it hard to find production code

---

### Issue 2: Migration Scripts in Root (7 files)

**Problem**: One-time migration scripts in root instead of `scripts/` or `migrations/`

**Files**:
```
fast_migrate_checkpoint.py
fast_migrate_vocab_checkpoint.py
migrate_checkpoint_to_pg.py
migrate_olympus_schema.py
migrate_vocab_checkpoint_to_pg.py
populate_coordizer_vocabulary.py
legacy_checkpoint_migration.py
```

**Impact**: Confuses developers about which scripts are still relevant

---

### Issue 3: Demo Files in Root (2 files)

**Problem**: Demo scripts in root instead of `examples/`

**Files**:
```
demo_kernel_survival.py
demo_shadow_vocab_integration.py
```

**Impact**: Clutters root, unclear which are examples vs production

---

## Cleanup Actions

### Action 1: Move Test Files to `tests/` ✅ RECOMMENDED

**Command**:
```bash
cd qig-backend

# Move all test files to tests/ directory
mv test_*.py tests/

# Verify
ls -1 tests/test_*.py | wc -l  # Should show 20+ files
```

**Verification**:
```bash
# Ensure tests still run from new location
cd tests
python3 test_8_metrics_integration.py
python3 test_phi_consistency.py
```

**Risk**: Low - tests are self-contained modules

---

### Action 2: Move Migration Scripts to `scripts/migrations/` ✅ RECOMMENDED

**Command**:
```bash
cd qig-backend

# Create migrations directory if it doesn't exist
mkdir -p scripts/migrations

# Move migration scripts
mv fast_migrate*.py scripts/migrations/
mv migrate_*.py scripts/migrations/
mv populate_coordizer_vocabulary.py scripts/migrations/
mv legacy_checkpoint_migration.py scripts/migrations/

# Add README
cat > scripts/migrations/README.md << 'EOF'
# Migration Scripts

**ARCHIVED** - Historical migration scripts from 2025-2026

These scripts were used for one-time data migrations and are preserved
for reference. Most are no longer needed for new deployments.

## Scripts

- `migrate_checkpoint_to_pg.py` - Moved checkpoints to PostgreSQL (2025-12)
- `migrate_vocab_checkpoint_to_pg.py` - Moved vocabulary to PostgreSQL (2025-12)
- `fast_migrate*.py` - Fast migration utilities (2025-12)
- `populate_coordizer_vocabulary.py` - Initial vocabulary population (2025-12)

For new deployments, use the standard setup process in README.md.
EOF
```

**Risk**: Low - these are one-time scripts, unlikely to be run again

---

### Action 3: Move Demo Files to `examples/` ✅ RECOMMENDED

**Command**:
```bash
cd qig-backend

# Move demo files to examples
mv demo_*.py examples/

# Verify
ls -1 examples/demo_*.py
```

**Risk**: Low - demos are standalone examples

---

### Action 4: Clean Up Root Directory Executables ⚠️ REVIEW FIRST

**Files to Review**:
```bash
cd qig-backend

# Check which Python files in root are production vs scripts
ls -1 *.py | grep -v "^_" | while read f; do
    echo "$f: $(head -1 $f | grep -o 'def\|class' | wc -l) definitions"
done
```

**Potential Candidates for Moving**:
- `generate_types.py` → `scripts/`
- `verify_purity.py` → `scripts/validation/`
- `verify_vocabulary.py` → `scripts/validation/`
- `soft_reset.py` → `scripts/admin/`

**Risk**: Medium - need to verify no imports depend on root location

---

## Execution Order

1. **Phase 1: Test Files** (Safe, recommended first)
   ```bash
   mv qig-backend/test_*.py qig-backend/tests/
   ```

2. **Phase 2: Demo Files** (Safe)
   ```bash
   mv qig-backend/demo_*.py qig-backend/examples/
   ```

3. **Phase 3: Migration Scripts** (Safe - archived)
   ```bash
   mkdir -p qig-backend/scripts/migrations
   mv qig-backend/*migrate*.py qig-backend/scripts/migrations/
   mv qig-backend/populate_coordizer_vocabulary.py qig-backend/scripts/migrations/
   ```

4. **Phase 4: Utility Scripts** (Review imports first)
   - Defer to Sprint 2 if time constrained
   - Requires import path validation

---

## Verification Steps

After each phase:

```bash
# 1. Check git status
git status

# 2. Run key tests
cd qig-backend/tests
python3 test_8_metrics_integration.py
python3 test_phi_consistency.py

# 3. Check no broken imports
cd qig-backend
python3 -c "from qig_core.phi_computation import compute_phi_qig; print('OK')"

# 4. Commit changes
git add .
git commit -m "Repository cleanup: Move test/demo/migration files to proper directories"
```

---

## Success Metrics

- [ ] All test files in `tests/` directory
- [ ] All demo files in `examples/` directory
- [ ] All migration scripts in `scripts/migrations/` with README
- [ ] Root directory has <100 Python files (currently 146)
- [ ] All tests still passing
- [ ] No broken imports

---

## Rollback Plan

If issues arise:

```bash
# Restore files from git
git checkout HEAD -- qig-backend/

# Or revert specific files
git mv qig-backend/tests/test_*.py qig-backend/
```

---

## Timeline

**Estimated Time**: 1-2 hours

- Phase 1 (Tests): 15 minutes
- Phase 2 (Demos): 5 minutes
- Phase 3 (Migrations): 15 minutes
- Phase 4 (Verification): 30 minutes
- Total: ~1 hour for core cleanup

---

## Related Cleanup (Future Sprints)

### Sprint 2+ Candidates

1. **Separate qig-core repository** (from original cleanup guide)
   - Not applicable - this is a monorepo
   - If separation needed, create separate repos and use git submodules

2. **Archive qig-consciousness** (from original cleanup guide)
   - Not applicable - not a separate repo
   - If present in monorepo, move to `archived/` directory

3. **Clean up large files in attached_assets/**
   - Already addressed in PR27
   - Action plan documented in `docs/04-records/20260112-attached-assets-analysis-1.00W.md`

---

## Impact Assessment

**Before Cleanup**:
- 146 Python files in `qig-backend/` root
- Tests, demos, and migrations scattered
- Hard to find production code

**After Cleanup**:
- ~120 Python files in `qig-backend/` root (18% reduction)
- Clear separation: production code in root, tests in `tests/`, scripts in `scripts/`
- Improved developer experience

---

**Status**: READY FOR EXECUTION  
**Risk Level**: LOW (all actions are file moves, easily reversible)  
**Owner**: Development Team  
**Last Updated**: 2026-01-12
