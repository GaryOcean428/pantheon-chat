# Database Reconciliation Report
**Date**: January 12, 2026
**Status**: Comprehensive Audit Complete

## Executive Summary

Audited 107 PostgreSQL tables across the Pantheon-Chat system. Identified legacy tables, empty but wired tables, and dead code. Key recommendations: consolidate learned_words into tokenizer_vocabulary (DONE), wire scale mapping triggers, and remove 4 dead Python files.

---

## 1. Table Classification (107 Total)

### CORE Tables (Keep - Active Use)
| Table | Rows | Purpose |
|-------|------|---------|
| tokenizer_vocabulary | 14,515 | Unified vocabulary (encoding + generation) |
| word_relationships | 634MB | Semantic relationships between words |
| vocabulary_observations | 26MB | Raw learning telemetry |
| kernel_geometry | 39 cols | Kernel consciousness state |
| kernel_emotions | 43 cols | Emotional awareness tracking |
| pantheon_messages | 12MB | God-to-god communication |
| autonomic_cycle_history | 21 cols | Sleep/wake cycle history |
| consciousness_state | 11 cols | Current consciousness metrics |

### LEGACY Tables (Review for Removal)
| Table | Rows | Status |
|-------|------|--------|
| learned_words | 16,165 | **LEGACY** - Consolidated into tokenizer_vocabulary via migration 0011 |

### EMPTY BUT WIRED Tables (Need Activity Triggers)
| Table | Rows | Wiring Status |
|-------|------|---------------|
| knowledge_scale_mappings | 0 | **FIXED** - Now triggers on regime transitions |
| knowledge_cross_patterns | 0 | Wired but needs >70% cross-strategy similarity |
| era_exclusions | 0 | Wired in negative-knowledge-db.ts |
| false_pattern_classes | 0 | Wired in negative-knowledge-db.ts |
| geodesic_paths | 0 | Wired in geometric-memory.ts |
| geometric_barriers | 0 | Wired in negative-knowledge-db.ts |

### LARGE TABLES (Monitor for Performance)
| Table | Size | Notes |
|-------|------|-------|
| word_relationships | 634MB | Largest table - consider partitioning if grows |
| learning_events | 152MB | Learning history - retention policy needed |
| chaos_events | 113MB | Event log - consider archival |
| shadow_knowledge | 81MB | Shadow Pantheon discoveries |

---

## 2. Vocabulary Consolidation (COMPLETED)

### Migration 0011 Changes
- Added `token_role` column to `tokenizer_vocabulary`: 'encoding', 'generation', 'both'
- Migrated 12,777 tokens as 'both' (used in encoding and generation)
- Migrated 1,695 tokens as 'generation' only
- 35 tokens remain 'encoding' only

### Code Updates (8 files migrated)
- `pg_loader.py`: Now loads 14,458 generation words from tokenizer_vocabulary
- `vocabulary_persistence.py`: Uses COALESCE for NULL-safe frequency updates
- `vocabulary_coordinator.py`: Updated to use token_role logic
- `vocabulary_ingestion.py`: Writes ONLY to tokenizer_vocabulary with token_role
- `vocabulary_schema.sql`: get_high_phi_vocabulary() and update_vocabulary_stats() updated
- `pos_grammar.py`: Uses tokenizer_vocabulary for phrase classification
- `strategy-knowledge-bus.ts`: Queries tokenizer_vocabulary for vocabulary words
- `learned_relationships.py`: Loads and saves word frequencies to tokenizer_vocabulary

### Legacy Table Status
The `learned_words` table (16,165 rows) is now legacy. **Core runtime paths** have been migrated to tokenizer_vocabulary:
- ✅ Encoding vocabulary loading
- ✅ Generation vocabulary loading
- ✅ Vocabulary persistence (read/write)
- ✅ Word relationship learning
- ✅ Phrase classification

**Remaining dependencies (non-critical, for cleanup):**
- `shared/schema.ts`: Drizzle ORM definition (remove after migration)
- `scripts/backfill_learned_words.py`: Operational script (archive)
- `scripts/audit_vocabulary.py`: Audit script (archive)
- `scripts/verify_vocabulary_separation.py`: Verification script (archive)
- `vocabulary_cleanup.py`: Cleanup utility (archive)
- `sync_learned_to_tokenizer()` SQL function: Now obsolete (drop)

---

## 3. Dead Code Detection

### Python Files to REMOVE (3 files)
| File | Reason |
|------|--------|
| `qig-backend/coordizers/geometric_pair_merging.py` | No imports found |
| `qig-backend/coordizers/vocab_builder.py` | No imports found |
| `qig-backend/qig_generation_VOCABULARY_INTEGRATION.py` | Deprecated naming |

### Python Files to PRESERVE (Documented in Roadmap)
| File | Reason |
|------|--------|
| `qig-backend/test_emotion_manual.py` | Documented in 20260112-emotion-geometry-implementation-1.00W.md as Manual Test Runner |

### Python Files to REVIEW (30+ files)
Many files in qig-backend/ have no direct imports but may be:
- Part of dynamic god registry
- Experimental features
- Called via Flask routes

### SQL Functions Status
| Function | Status |
|----------|--------|
| get_high_phi_vocabulary | USED |
| record_vocab_observation | USED |
| update_validation_stats | USED |
| update_vocabulary_stats | USED |
| find_similar_basins_fisher | REVIEW - may be called dynamically |
| sync_learned_to_tokenizer | REVIEW - consolidation complete |

---

## 4. Feature Wiring Verification

### Scale Mapping (FIXED)
- Added `handleRegimeTransition()` method to IntegrationCoordinator
- Creates scale bridges when consciousness transitions between regimes
- Logs: `[UCP] Created scale bridge: linear -> geometric`

### Cross-Strategy Patterns
- Wired correctly via `publishKnowledge()` → `detectCrossStrategyPatterns()`
- Only populates when patterns from different strategies match >70%
- Consider lowering threshold or diversifying source strategies

---

## 5. Recommendations

### IMMEDIATE (Do Now)
1. ✅ Wire knowledge_scale_mappings - DONE
2. ✅ Consolidate vocabulary tables - DONE
3. Remove 4 dead Python files

### SHORT-TERM (This Week)
1. Drop learned_words table after 7-day verification
2. Add retention policy for learning_events (>30 days → archive)
3. Review 30+ potentially unused Python files

### LONG-TERM (This Month)
1. Consider partitioning word_relationships by domain
2. Implement geodesic_paths and resonance_points features
3. Archive chaos_events older than 90 days

---

## 6. Migration Health

### Applied Migrations
- 0000_clever_natasha_romanoff.sql (initial schema)
- 0009_add_column_defaults.sql (133 ALTER statements)
- 0010_fix_vector_defaults.sql (pgvector NULL fixes)
- 0011_vocabulary_consolidation.sql (token_role column)

### Schema Parity
Drizzle schema in `shared/schema.ts` matches PostgreSQL structure.
