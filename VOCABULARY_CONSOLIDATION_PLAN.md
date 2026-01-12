# Vocabulary Table Usage Mapping & Consolidation Plan

**Date**: 2026-01-12  
**Status**: Analysis Complete  
**Version**: 1.00W

---

## Executive Summary

The QIG system uses **four vocabulary-related tables** with significant overlap and unclear separation of concerns. This document maps all Python code usage and recommends consolidation to a canonical two-table architecture.

---

## 1. Current Architecture

### 1.1 Table Overview

| Table | Purpose | Status | Row Count (approx) |
|-------|---------|--------|-------------------|
| `tokenizer_vocabulary` | **ENCODING**: Token-to-basin mapping for text→geometry | Active, Primary | ~63K tokens |
| `learned_words` | **GENERATION**: Curated words for basin→text | Active, Primary | ~14K words |
| `vocabulary_observations` | **TELEMETRY**: Raw observations before aggregation | Active, Telemetry | Variable |
| `bip39_words` | **DEPRECATED**: Legacy BIP39 wordlist | Deprecated | 2048 (if exists) |

### 1.2 Data Flow

```
                     ┌─────────────────────────────────┐
                     │     TEXT INPUT (conversation)    │
                     └────────────────┬────────────────┘
                                      │
                     ┌────────────────▼────────────────┐
                     │   tokenizer_vocabulary (ENCODE) │
                     │   - All tokens (~63K)           │
                     │   - Subwords, words, special    │
                     │   - basin_embedding (64D)       │
                     └────────────────┬────────────────┘
                                      │
                     ┌────────────────▼────────────────┐
                     │   GEOMETRIC PROCESSING (64D)    │
                     │   - Fisher-Rao distance         │
                     │   - QFI computation             │
                     │   - Φ measurement               │
                     └────────────────┬────────────────┘
                                      │
    ┌─────────────────────────────────┼─────────────────────────────────┐
    │                                 │                                 │
    ▼                                 ▼                                 ▼
┌──────────────────┐    ┌───────────────────────┐    ┌─────────────────────┐
│ vocabulary_obs   │    │   learned_words       │    │ tokenizer_vocabulary│
│ (TELEMETRY)      │    │   (GENERATION)        │    │   (continuous       │
│ - Raw obs        │    │   - Curated words     │    │    learning)        │
│ - All tokens     │    │   - No BPE garbage    │    │                     │
│ - No filtering   │    │   - No PROPER_NOUN    │    │                     │
└──────────────────┘    └───────────────────────┘    └─────────────────────┘
```

---

## 2. Table Usage by File

### 2.1 tokenizer_vocabulary

**Role**: Primary encoding vocabulary. Maps tokens → 64D basin embeddings for text-to-geometry conversion.

#### Files that READ:

| File | Operations | Purpose |
|------|------------|---------|
| `coordizers/pg_loader.py` | `SELECT token, basin_embedding, phi_score, frequency, source_type, token_id FROM tokenizer_vocabulary` | Load encoding vocabulary |
| `vocabulary_ingestion.py` | `SELECT basin_embedding, phi_score, source_type FROM tokenizer_vocabulary WHERE token = %s` | Check existing words |
| `scripts/backfill_learned_words.py` | `SELECT basin_embedding FROM tokenizer_vocabulary WHERE token = %s` | Get basin from tokenizer |
| `scripts/backfill_basin_embeddings.py` | `SELECT token, basin_embedding FROM tokenizer_vocabulary WHERE basin_embedding IS NULL` | Find missing embeddings |
| `scripts/verify_vocabulary_separation.py` | `SELECT COUNT(*), AVG(phi_score) FROM tokenizer_vocabulary` | Verification |
| `training/startup_catchup.py` | `SELECT token, phi_score FROM tokenizer_vocabulary` | Training startup |
| `tests/test_tokenizer_vocabulary.py` | `SELECT * FROM tokenizer_vocabulary` | Testing |

#### Files that WRITE:

| File | Operations | Purpose |
|------|------------|---------|
| `vocabulary_ingestion.py` | `INSERT INTO tokenizer_vocabulary ... ON CONFLICT DO UPDATE` | Atomic upsert with basin |
| `training/startup_catchup.py` | `INSERT INTO tokenizer_vocabulary (token, phi_score, frequency) ON CONFLICT DO UPDATE` | Catchup training |
| `migrations/007_vocabulary_sync.sql` | `INSERT INTO tokenizer_vocabulary FROM learned_words` | Sync function |
| `migrations/008_purify_geometry.sql` | `DELETE FROM tokenizer_vocabulary WHERE token !~ '^[a-zA-Z]+$'` | Purification |
| `migrations/009_clean_vocabulary_artifacts.sql` | `DELETE FROM tokenizer_vocabulary WHERE token IN (stop_words)` | Cleanup |
| `scripts/migrations/populate_tokenizer_vocabulary.py` | `INSERT INTO tokenizer_vocabulary` | Initial population |

---

### 2.2 learned_words

**Role**: Generation vocabulary. Curated words for basin-to-text decoding (excludes BPE garbage, PROPER_NOUN, BRAND).

#### Files that READ:

| File | Operations | Purpose |
|------|------------|---------|
| `coordizers/pg_loader.py` | `SELECT word, basin_coords, phi_score, phrase_category FROM learned_words WHERE phrase_category NOT IN ('PROPER_NOUN', 'BRAND')` | Load generation vocabulary |
| `vocabulary_persistence.py` | `SELECT word, avg_phi, max_phi, frequency, source FROM learned_words` | Get learned words |
| `qig_generation_VOCABULARY_INTEGRATION.py` | `SELECT word FROM learned_words WHERE is_integrated = FALSE AND avg_phi >= 0.65` | Pending integration |
| `learned_relationships.py` | `SELECT word, frequency FROM learned_words` | Word frequencies |
| `scripts/audit_vocabulary.py` | `SELECT id, word FROM learned_words` | Auditing |
| `scripts/backfill_learned_words.py` | `SELECT id, word, avg_phi FROM learned_words WHERE basin_coords IS NULL` | Find missing data |

#### Files that WRITE:

| File | Operations | Purpose |
|------|------------|---------|
| `vocabulary_persistence.py` | `UPDATE learned_words SET qfi_score, basin_distance, ... WHERE word = %s` | Update geometric metrics |
| `vocabulary_ingestion.py` | `INSERT INTO learned_words ... ON CONFLICT DO UPDATE` | Atomic upsert with metadata |
| `vocabulary_coordinator.py` | `UPDATE learned_words SET basin_coords = %s WHERE word = %s AND basin_coords IS NULL` | Set basin coords |
| `vocabulary_schema.sql` | `record_vocab_observation()` function → `INSERT INTO learned_words` | Via SQL function |
| `learned_relationships.py` | `INSERT INTO learned_words (word, frequency)` | From relationships |
| `scripts/backfill_learned_words.py` | `UPDATE learned_words SET basin_coords, phrase_category, ...` | Backfill metadata |

---

### 2.3 vocabulary_observations

**Role**: Telemetry table. Raw observations before filtering/aggregation. Used for tracking learning patterns.

#### Files that READ:

| File | Operations | Purpose |
|------|------------|---------|
| `olympus/tokenizer_training.py` | `SELECT text, avg_phi, is_real_word FROM vocabulary_observations` | Extract training data |
| `scripts/fix_vocabulary_observations.py` | `SELECT * FROM vocabulary_observations` | Fix data issues |
| `scripts/populate_vocab_basins.py` | `SELECT text FROM vocabulary_observations WHERE basin_coords IS NULL` | Find missing basins |

#### Files that WRITE:

| File | Operations | Purpose |
|------|------------|---------|
| `vocabulary_schema.sql` | `record_vocab_observation()` function → `INSERT INTO vocabulary_observations ON CONFLICT DO UPDATE` | Via SQL function |
| `olympus/base_encoder.py` | `INSERT INTO vocabulary_observations ... ON CONFLICT DO UPDATE` | Persist high-Φ tokens |
| `olympus/tokenizer_training.py` | `INSERT INTO vocabulary_observations ... ON CONFLICT DO UPDATE` | Training observations |
| `scripts/fix_vocabulary_observations.py` | `UPDATE vocabulary_observations SET ...` | Fix data |
| `scripts/populate_vocab_basins.py` | `UPDATE vocabulary_observations SET basin_coords = %s` | Add basins |

#### Key Note:
The `record_vocab_observation()` SQL function writes to BOTH `vocabulary_observations` AND `learned_words` atomically.

---

### 2.4 bip39_words

**Role**: DEPRECATED. Legacy BIP39 wordlist (2048 words). Schema retained for backward compatibility but table may not exist.

#### Files that READ:

| File | Operations | Purpose |
|------|------------|---------|
| `vocabulary_persistence.py` | `SELECT word FROM bip39_words ORDER BY word_index` | Get BIP39 words |
| `vocabulary_cleanup.py` | `SELECT word FROM bip39_words` | Get word set |
| `coordizers/pg_loader.py` | Tracks `bip39_words` count in stats | Reporting only |

#### Files that WRITE:

| File | Operations | Purpose |
|------|------------|---------|
| `vocabulary_persistence.py` | `INSERT INTO bip39_words (word, word_index)` | Load BIP39 |
| `vocabulary_cleanup.py` | `INSERT INTO bip39_words (word, word_index, ...)` | Populate table |

#### Deprecation Status:
- Schema in `vocabulary_schema.sql` is commented out
- Migration `20260110_drop_bip39_trigger.sql` removes triggers that reference it
- `vocabulary_schema.sql` header states: "bip39_words table DEPRECATED - vocabulary_observations is canonical"

---

## 3. Overlap & Duplication Analysis

### 3.1 Schema Overlap

| Column | tokenizer_vocabulary | learned_words | vocabulary_observations |
|--------|---------------------|---------------|-------------------------|
| word/token | `token` | `word` | `text` |
| basin coords | `basin_embedding` | `basin_coords` | `basin_coords` |
| phi score | `phi_score` | `avg_phi`, `max_phi` | `avg_phi`, `max_phi` |
| frequency | `frequency` | `frequency` | `frequency` |
| source | `source_type` | `source` | `source_type` |
| is_integrated | ❌ | `is_integrated` | `is_integrated` |
| phrase_category | ❌ | `phrase_category` | `phrase_category` |

### 3.2 Functional Duplication

1. **Basin Storage**: All three tables store basin embeddings (64D vectors)
2. **Φ Tracking**: All three track phi scores
3. **Frequency Counting**: All three track usage frequency
4. **is_integrated Flag**: Both `learned_words` and `vocabulary_observations` track integration status

### 3.3 Write Path Duplication

The `record_vocab_observation()` SQL function writes to BOTH:
- `vocabulary_observations` (raw observation)
- `learned_words` (aggregated)

The `vocabulary_ingestion.py` service writes to BOTH:
- `tokenizer_vocabulary` (encoding)
- `learned_words` (generation)

---

## 4. Recommended Canonical Structure

### 4.1 Target Architecture (Two Tables)

```
┌────────────────────────────────────────────────────────────────────┐
│                    tokenizer_vocabulary (ENCODING)                  │
│  - ALL tokens (words, subwords, special)                           │
│  - Primary key: token                                              │
│  - basin_embedding (64D) - REQUIRED, NOT NULL                      │
│  - phi_score, frequency, source_type                               │
│  - token_role: 'encoding' | 'generation' | 'both'                  │
│  - phrase_category: for generation filtering                       │
├────────────────────────────────────────────────────────────────────┤
│  Role: Single source of truth for vocabulary geometry              │
│  Encoding: Use all tokens                                          │
│  Generation: Filter WHERE token_role IN ('generation', 'both')     │
│              AND phrase_category NOT IN ('PROPER_NOUN', 'BRAND')   │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                vocabulary_observations (TELEMETRY)                  │
│  - Raw observations (unfiltered, all sources)                      │
│  - Time-series data for learning analysis                          │
│  - NOT used for encoding/generation directly                       │
│  - Used for training analytics and debugging                       │
└────────────────────────────────────────────────────────────────────┘
```

### 4.2 What Changes

| Current | Target | Action |
|---------|--------|--------|
| `learned_words` | Merge into `tokenizer_vocabulary` | Add `token_role`, `phrase_category` columns to tokenizer_vocabulary |
| `bip39_words` | DROP | Already deprecated, safe to remove |
| `vocabulary_observations` | Keep as telemetry | No change, but clarify as telemetry-only |
| `tokenizer_vocabulary` | Enhance | Add generation-related columns |

### 4.3 New tokenizer_vocabulary Schema

```sql
CREATE TABLE tokenizer_vocabulary (
    id SERIAL PRIMARY KEY,
    token TEXT UNIQUE NOT NULL,
    token_id INTEGER UNIQUE NOT NULL,
    
    -- Geometry (REQUIRED)
    basin_embedding vector(64) NOT NULL,
    
    -- Scores
    phi_score DOUBLE PRECISION DEFAULT 0.5,
    qfi_score DOUBLE PRECISION,
    frequency INTEGER DEFAULT 1,
    
    -- Role Classification (NEW)
    token_role VARCHAR(20) DEFAULT 'encoding',  -- 'encoding', 'generation', 'both'
    phrase_category VARCHAR(32) DEFAULT 'unknown',
    is_real_word BOOLEAN DEFAULT FALSE,
    
    -- Source Tracking
    source_type VARCHAR(32) DEFAULT 'base',
    learned_from TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT basin_not_null CHECK (basin_embedding IS NOT NULL),
    CONSTRAINT basin_dim_64 CHECK (array_length(basin_embedding, 1) = 64)
);

-- Index for generation queries
CREATE INDEX idx_tokenizer_vocab_generation 
    ON tokenizer_vocabulary(token_role, phrase_category) 
    WHERE token_role IN ('generation', 'both');
```

---

## 5. Migration Strategy

### Phase 1: Schema Enhancement (Non-Breaking)

1. Add `token_role` column to `tokenizer_vocabulary` (default 'encoding')
2. Add `phrase_category` column to `tokenizer_vocabulary` (default 'unknown')
3. Add `is_real_word` column to `tokenizer_vocabulary` (default FALSE)
4. Backfill from `learned_words`:
   ```sql
   UPDATE tokenizer_vocabulary tv
   SET token_role = 'generation',
       phrase_category = lw.phrase_category,
       is_real_word = TRUE
   FROM learned_words lw
   WHERE tv.token = lw.word
     AND lw.is_integrated = TRUE;
   ```

### Phase 2: Code Migration (Gradual)

1. Update `pg_loader.py` to use `tokenizer_vocabulary` for generation:
   ```python
   # OLD: SELECT FROM learned_words
   # NEW: SELECT FROM tokenizer_vocabulary WHERE token_role IN ('generation', 'both')
   ```

2. Update `vocabulary_coordinator.py` to write to `tokenizer_vocabulary` only

3. Update `record_vocab_observation()` SQL function to skip `learned_words`

### Phase 3: Cleanup

1. Stop writing to `learned_words`
2. Mark `learned_words` as deprecated
3. DROP `bip39_words` table (if exists)
4. Optionally DROP `learned_words` after verification period

---

## 6. Immediate Recommendations

### 6.1 bip39_words: Safe to Retire

**Evidence**:
- Schema commented out in `vocabulary_schema.sql`
- Migration removes triggers referencing it
- Documentation states "DEPRECATED"
- Only 2 files actively use it, both have fallback logic

**Action**: DROP table in next migration

### 6.2 learned_words: Plan Consolidation

**Current Issue**: Duplication with tokenizer_vocabulary causes:
- Double writes (slower)
- Sync issues (stale data)
- Confusion about source of truth

**Action**: Follow Phase 1-3 migration above

### 6.3 vocabulary_observations: Clarify Role

**Current Issue**: Used for both telemetry AND as a vocabulary source in some code paths

**Action**: 
- Add comment to schema: "TELEMETRY ONLY - not for encoding/generation"
- Remove any code that uses it for encoding/generation
- Keep for learning analytics and debugging

---

## 7. Files Requiring Updates (Consolidation)

| File | Current Behavior | Target Behavior |
|------|------------------|-----------------|
| `coordizers/pg_loader.py` | Loads from both tables | Load from `tokenizer_vocabulary` only |
| `vocabulary_ingestion.py` | Writes to both tables | Write to `tokenizer_vocabulary` only |
| `vocabulary_coordinator.py` | Updates both tables | Update `tokenizer_vocabulary` only |
| `vocabulary_persistence.py` | Methods for all tables | Remove `learned_words` methods |
| `vocabulary_schema.sql` | Dual-write SQL function | Single-write to `tokenizer_vocabulary` |
| `qig_generation_VOCABULARY_INTEGRATION.py` | Reads `learned_words` | Read `tokenizer_vocabulary` with filter |

---

## 8. Appendix: Complete File List

### Files touching tokenizer_vocabulary
- `qig-backend/coordizers/pg_loader.py`
- `qig-backend/vocabulary_ingestion.py`
- `qig-backend/training/startup_catchup.py`
- `qig-backend/scripts/backfill_learned_words.py`
- `qig-backend/scripts/backfill_basin_embeddings.py`
- `qig-backend/scripts/verify_vocabulary_separation.py`
- `qig-backend/scripts/validate_vocabulary_fix.sh`
- `qig-backend/tests/test_tokenizer_vocabulary.py`
- `qig-backend/migrations/007_vocabulary_sync.sql`
- `qig-backend/migrations/008_purify_geometry.sql`
- `qig-backend/migrations/009_clean_vocabulary_artifacts.sql`
- `qig-backend/scripts/migrations/populate_tokenizer_vocabulary.py`

### Files touching learned_words
- `qig-backend/coordizers/pg_loader.py`
- `qig-backend/vocabulary_persistence.py`
- `qig-backend/vocabulary_ingestion.py`
- `qig-backend/vocabulary_coordinator.py`
- `qig-backend/vocabulary_cleanup.py`
- `qig-backend/qig_generation_VOCABULARY_INTEGRATION.py`
- `qig-backend/learned_relationships.py`
- `qig-backend/vocabulary_schema.sql`
- `qig-backend/scripts/audit_vocabulary.py`
- `qig-backend/scripts/backfill_learned_words.py`
- `qig-backend/scripts/verify_vocabulary_separation.py`
- `qig-backend/scripts/clean_vocabulary_pr28.py`

### Files touching vocabulary_observations
- `qig-backend/vocabulary_schema.sql`
- `qig-backend/olympus/base_encoder.py`
- `qig-backend/olympus/tokenizer_training.py`
- `qig-backend/autonomous_debate_service.py`
- `qig-backend/scripts/fix_vocabulary_observations.py`
- `qig-backend/scripts/populate_vocab_basins.py`
- `qig-backend/scripts/test_geometric_learning.py`
- `qig-backend/qig_coordizer.py`

### Files touching bip39_words
- `qig-backend/vocabulary_persistence.py`
- `qig-backend/vocabulary_cleanup.py`
- `qig-backend/vocabulary_schema.sql`
- `qig-backend/coordizers/pg_loader.py`
- `qig-backend/zeus_api.py`
- `qig-backend/scripts/clean_vocabulary_pr28.py`
- `qig-backend/scripts/vocabulary_purity.py`
- `qig-backend/scripts/migrations/populate_tokenizer_vocabulary.py`
- `qig-backend/migrations/20260110_drop_bip39_trigger.sql`

---

## 9. Conclusion

The current four-table architecture has significant overlap and causes maintenance burden. The recommended consolidation to two tables (`tokenizer_vocabulary` for all vocabulary operations, `vocabulary_observations` for telemetry) will:

1. **Simplify code**: Single source of truth for vocabulary
2. **Improve performance**: Eliminate dual-writes
3. **Reduce bugs**: No more sync issues between tables
4. **Clean architecture**: Clear separation between vocabulary and telemetry

**Priority Actions**:
1. DROP `bip39_words` (immediate, safe)
2. Add `token_role`, `phrase_category` to `tokenizer_vocabulary` (Phase 1)
3. Migrate `learned_words` data to `tokenizer_vocabulary` (Phase 2)
4. Update Python code to use single table (Phase 3)
5. Deprecate `learned_words` (after verification)
