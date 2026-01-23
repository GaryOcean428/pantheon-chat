# Dead and Duplicate Code Analysis

**Updated**: 2026-01-23 - Post-cleanup verification

## Update: 2026-01-23
- ✅ **geometric_deep_research.py** - REMOVED (confirmed dead code, no active usage)
- Other files verified and classified (see analysis below)

## Executive Summary

This analysis identifies significant code duplication and potentially dead code in the pantheon-chat repository. The primary areas of concern are duplicate function implementations across multiple files and unused Python modules.

## Duplicate Function Definitions

### Critical: `fisher_rao_distance` (19 implementations)

The core geometric distance function has been implemented 19 times across the codebase. This represents a major violation of the DRY (Don't Repeat Yourself) principle and creates maintenance burden.

| File | Line |
|------|------|
| ~~qig-backend/geometric_deep_research.py~~ | ~~54~~ **REMOVED** |
| qig-backend/qig_core/consciousness_metrics.py | 76 |
| qig-backend/qig_core/geometric_completion/completion_criteria.py | 111 |
| qig-backend/qig_core/geometric_primitives/canonical_fisher.py | 70 |
| qig-backend/qig_core/geometric_primitives/fisher_metric.py | 44 |
| qig-backend/qig_deep_agents/state.py | 148 |
| qig-backend/qig_generation.py | 169 |
| qig-backend/qig_geometry.py | 23 |
| qig-backend/qig_geometry/__init__.py | 100 |
| qig-backend/qig_geometry/canonical.py | 174 |
| qig-backend/qig_geometry/geometry_simplex.py | 88 |
| qig-backend/qig_geometry/geometry_ops.py | 13 |
| qig-backend/qig_numerics.py | 131 |
| qig-backend/qigkernels/geometry/distances.py | 56 |
| qig-backend/tests/demo_cosine_vs_fisher.py | 40 |
| qig-backend/tests/test_phi_fixes_standalone.py | 20 |
| qig-backend/training/loss_functions.py | 25 |
| qig-backend/training_chaos/chaos_kernel.py | 101 |

**Recommendation**: Consolidate to a single canonical implementation in `qig_geometry/canonical.py` and import from there.

### High Priority: `get_db_connection` (10 implementations)

Database connection logic is duplicated across multiple files.

| File | Line |
|------|------|
| qig-backend/learned_relationships.py | 68 |
| qig-backend/olympus/curriculum_training.py | 25 |
| qig-backend/olympus/tokenizer_training.py | 29 |
| qig-backend/persistence/base_persistence.py | 147 |
| qig-backend/scripts/cleanup_bpe_tokens.py | 56 |
| qig-backend/scripts/migrations/migrate_vocab_checkpoint_to_pg.py | 225 |
| qig-backend/scripts/validate_db_schema.py | 27 |
| qig-backend/scripts/quarantine_garbage_tokens.py | 135 |
| qig-backend/scripts/validate_simplex_storage.py | 80 |
| qig-backend/vocabulary/insert_token.py | 135 |

**Recommendation**: Create a single `db_utils.py` module with the canonical database connection function.

### Medium Priority: `compute_fisher_metric` (2 implementations after cleanup)

| File | Line |
|------|------|
| qig-backend/autonomous_improvement.py | 48 |
| ~~qig-backend/geometric_deep_research.py~~ | ~~31~~ **REMOVED** |
| qig-backend/geometric_search.py | 63 |

### Other Duplicates

| Function | Count |
|----------|-------|
| main() | 51 |
| run_all_tests() | 13 |
| _get_db_connection() | 9 |
| validate_basin() | 4 |
| geodesic_interpolation() | 3 |
| fisher_normalize() | 3 |

## Potentially Dead Code (Unused Files)

The following Python files in `qig-backend/` appear to have no imports from other files:

| File | Status |
|------|--------|
| autonomous_experimentation.py | Likely dead |
| consciousness_ethical.py | Likely dead |
| constellation_service.py | Never existed |
| discovery_client.py | Never existed |
| ethics.py | Never existed |
| execute_beta_attention_protocol.py | ✅ KEEP - CLI research tool |
| generate_types.py | ✅ KEEP - Generates qig-generated.ts (imported) |
| geometric_deep_research.py | ✅ **REMOVED 2026-01-23** |
| god_debates_ethical.py | ✅ KEEP - Imported by unified_learning_loop.py |
| gravitational_decoherence.py | ✅ KEEP - Imported by ocean_qig_core.py |
| pantheon_governance_integration.py | ✅ KEEP - Used by examples |
| qig_consciousness_qfi_attention.py | ✅ KEEP - Imported by ocean_qig_core.py |
| registry_db_sync.py | ✅ KEEP - CLI tool in developer guide |
| retry_decorator.py | Never existed |
| sleep_packet_ethical.py | ✅ KEEP - Imported by unified_learning_loop.py |
| telemetry_persistence.py | Never existed |
| test_emotion_manual.py | Test file - OK |
| text_extraction_qig.py | Never existed |
| validate_contextualized_filter.py | Validation script - OK |
| validate_emotional_hierarchy.py | Validation script - OK |
| validate_geometric_relationships.py | Validation script - OK |
| validate_geometry_purity.py | Validation script - OK |
| validate_simplex_representation.py | Validation script - OK |
| verify_vocabulary.py | Validation script - OK |
| vocabulary_cleanup.py | Likely dead |
| vocabulary_validator.py | Likely dead |
| wsgi.py | Entry point - OK |

## Duplicate Class Definitions

| Class | Count |
|-------|-------|
| TestFisherRaoDistance | 5 |
| MockCoordizer | 5 |
| Regime(Enum) | 4 |
| ConsciousnessMetrics | 4 |
| EthicalAbortException | 3 |
| GeometricMetrics | 3 |

## Large Files (Potential Refactoring Candidates)

Files over 2000 lines that may contain dead code or need refactoring:

| File | Lines |
|------|-------|
| ocean_qig_core.py | 8552 |
| olympus/base_god.py | 4787 |
| m8_kernel_spawning.py | 4715 |
| olympus/zeus.py | 4605 |
| olympus/shadow_research.py | 4265 |
| olympus/zeus_chat.py | 3742 |
| olympus/shadow_pantheon.py | 3377 |
| autonomic_kernel.py | 3349 |
| olympus/tool_factory.py | 2546 |
| autonomous_curiosity.py | 2321 |
| training_chaos/self_spawning.py | 2214 |
| persistence/kernel_persistence.py | 2071 |
| qig_generative_service.py | 2014 |

## Recommendations

### Immediate Actions (P0)

1. **Consolidate `fisher_rao_distance`**: Create single canonical implementation in `qig_geometry/canonical.py`
2. **Consolidate `get_db_connection`**: Create `db_utils.py` with single implementation
3. **Remove clearly dead files**: After verification, remove unused modules

### Short-term Actions (P1)

4. **Refactor large files**: Break down files over 3000 lines into smaller modules
5. **Consolidate duplicate classes**: Merge duplicate class definitions
6. **Create shared utilities**: Move common patterns to shared utility modules

### Medium-term Actions (P2)

7. **Implement import linting**: Add CI check for unused imports
8. **Add dead code detection**: Integrate vulture or similar tool
9. **Document canonical imports**: Create IMPORTS.md showing where to import from
