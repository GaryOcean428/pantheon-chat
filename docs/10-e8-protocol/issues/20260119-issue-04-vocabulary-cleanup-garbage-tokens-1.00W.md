# ISSUE 04: Vocabulary Cleanup - Garbage Tokens & learned_words Deprecation

**Priority:** HIGH  
**Phase:** 3 (Data Quality)  
**Status:** TODO  
**Blocks:** Generation quality, vocabulary integrity
**Created:** 2026-01-19

---

## PROBLEM STATEMENT

### Current State
The coordizer_vocabulary table contains BPE tokenizer artifacts and garbage tokens that contaminate the generation vocabulary. Additionally, the learned_words table (17,324 entries) still exists but should be deprecated.

**Evidence:**
- Token "wjvq" found in coordizer_vocabulary with `token_role='both'` or `'generation'`
- This is BPE garbage from GPT-2/HuggingFace tokenizer migration
- No validation prevented non-dictionary words from entering generation vocabulary
- learned_words table never properly deprecated/dropped after consolidation

**Impact:**
- Generated text contains garbage tokens
- Consciousness metrics computed on invalid data
- Vocabulary pollution reduces generation quality
- Dual storage creates maintenance burden

---

## ROOT CAUSES

### 1. BPE Tokenizer Artifacts Not Cleaned
**Culprit:** Initial vocabulary migration from BPE tokenizers

```sql
-- Example garbage tokens currently in database:
SELECT token, token_role, source_type 
FROM coordizer_vocabulary 
WHERE token ~ '^[^a-z]|[0-9]|^.{1,2}$'
LIMIT 10;
```

**Problems:**
- BPE subword markers (Ġ, ċ, ##, ▁) not filtered
- Short fragments (2 chars or less) from tokenization
- Numeric-only tokens
- Special characters that aren't words

### 2. No Validation Gate on Generation Vocabulary
**Culprit:** pg_loader.py loads tokens without strict validation

```python
# qig-backend/coordizers/pg_loader.py
def _load_generation_vocabulary(self):
    # Loads from coordizer_vocabulary with token_role filter
    # BUT: No validation that tokens are valid English words!
    cur.execute("""
        SELECT token, basin_coords, qfi_score
        FROM coordizer_vocabulary
        WHERE token_role IN ('generation', 'both')
    """)
```

**What's Missing:**
```python
# SHOULD BE:
from word_validation import is_valid_english_word, is_bpe_garbage

def _load_generation_vocabulary(self):
    cur.execute("""
        SELECT token, basin_coords, qfi_score
        FROM coordizer_vocabulary
        WHERE token_role IN ('generation', 'both')
    """)
    
    for token, coords, qfi in cur.fetchall():
        # Validate before adding to generation vocab
        if not is_valid_english_word(token, strict=True):
            logger.warning(f"Rejecting garbage token: {token}")
            continue
        
        if is_bpe_garbage(token):
            logger.warning(f"Rejecting BPE artifact: {token}")
            continue
        
        self.generation_vocab[token] = {'coords': coords, 'qfi': qfi}
```

### 3. learned_words Table Never Properly Deprecated

**Current State:**
```sql
-- Still exists with 17,324 entries:
SELECT COUNT(*) FROM learned_words;  -- 17324

-- Overlap with coordizer_vocabulary:
SELECT COUNT(DISTINCT lw.word)
FROM learned_words lw
JOIN coordizer_vocabulary cv ON lw.word = cv.token;
```

**Problems:**
- Duplicate storage of same words
- No clear migration path
- May contain valuable data not in coordizer_vocabulary
- Creates confusion about which table is canonical

---

## PROPOSED SOLUTION

### Phase 1: Audit & Document

**Script:** `qig-backend/scripts/audit_vocabulary.py` (already exists)

```bash
# Run audit to identify garbage tokens
cd qig-backend
python scripts/audit_vocabulary.py --report

# Expected output:
# Total tokens: 50,000
# Valid English words: 48,500 (97%)
# BPE artifacts: 1,200 (2.4%)
# Technical garbage: 300 (0.6%)
# 
# Top garbage tokens:
# - wjvq (freq=10, role=both)
# - ĠTheĠ (freq=50, role=generation)
# - ##ing (freq=30, role=both)
```

**Action Items:**
1. Run audit_vocabulary.py to generate full report
2. Export list of garbage tokens to `garbage_tokens.txt`
3. Verify no false positives (legitimate abbreviations, etc.)
4. Document overlap between learned_words and coordizer_vocabulary

### Phase 2: Clean coordizer_vocabulary

**Migration:** `qig-backend/migrations/016_clean_vocabulary_garbage.sql`

```sql
-- ============================================================================
-- Migration 016: Clean Vocabulary Garbage
-- ============================================================================
-- Date: 2026-01-19
-- Purpose: Remove BPE artifacts and technical garbage from generation vocabulary
--
-- CRITICAL: Only removes from token_role='generation' and 'both'
-- Encoding vocabulary (token_role='encoding') untouched for backward compatibility
-- ============================================================================

-- Step 1: Identify garbage tokens
CREATE TEMP TABLE garbage_tokens AS
SELECT token, token_role, source_type, frequency
FROM coordizer_vocabulary
WHERE token_role IN ('generation', 'both')
  AND (
    -- BPE markers
    token ~ '^[ĠġĊċ]' OR
    token ~ '^##' OR
    token ~ '^▁' OR
    -- Numeric-only
    token ~ '^\d+$' OR
    -- Too short (unless whitelisted)
    (LENGTH(token) < 3 AND token NOT IN ('a', 'i', 'to', 'of', 'in', 'on', 'at', 'by', 'or', 'an', 'is', 'be', 'do', 'go', 'it', 'me', 'we', 'he', 'up', 'no')) OR
    -- Technical garbage patterns
    token ~ 'obj$' OR
    token ~ '^api' OR
    token ~ 'callback' OR
    token ~ 'handler' OR
    -- Known garbage from audit
    token = ANY(ARRAY['wjvq', 'xyzw', 'qwerty'])  -- Add from audit
  );

-- Step 2: Log what we're removing
DO $$
DECLARE
    garbage_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO garbage_count FROM garbage_tokens;
    RAISE NOTICE '[016] Identified % garbage tokens for removal', garbage_count;
    
    -- Sample of what's being removed
    RAISE NOTICE '[016] Sample garbage tokens:';
    FOR rec IN (SELECT token, frequency FROM garbage_tokens LIMIT 10)
    LOOP
        RAISE NOTICE '  - % (freq=%)', rec.token, rec.frequency;
    END LOOP;
END $$;

-- Step 3: Remove from generation vocabulary
-- Option A: Delete entirely (if not used for encoding)
DELETE FROM coordizer_vocabulary
WHERE token IN (SELECT token FROM garbage_tokens)
  AND token_role = 'generation';  -- Only generation-only tokens

-- Option B: Downgrade to encoding-only (preserve for backward compat)
UPDATE coordizer_vocabulary
SET token_role = 'encoding'
WHERE token IN (SELECT token FROM garbage_tokens)
  AND token_role = 'both';

-- Step 4: Verification
DO $$
DECLARE
    remaining_garbage INTEGER;
BEGIN
    SELECT COUNT(*) INTO remaining_garbage
    FROM coordizer_vocabulary
    WHERE token_role IN ('generation', 'both')
      AND token IN (SELECT token FROM garbage_tokens);
    
    IF remaining_garbage = 0 THEN
        RAISE NOTICE '[016] ✓ All garbage tokens removed from generation vocabulary';
    ELSE
        RAISE WARNING '[016] ⚠ Still have % garbage tokens in generation', remaining_garbage;
    END IF;
END $$;

DROP TABLE garbage_tokens;
```

### Phase 3: Migrate & Deprecate learned_words

**Migration:** `qig-backend/migrations/017_deprecate_learned_words.sql`

```sql
-- ============================================================================
-- Migration 017: Migrate learned_words to coordizer_vocabulary & Deprecate
-- ============================================================================

-- Step 1: Migrate valid words not yet in coordizer_vocabulary
INSERT INTO coordizer_vocabulary (
    token, basin_coords, qfi_score, frequency, 
    token_role, source_type, created_at, updated_at
)
SELECT 
    lw.word,
    lw.basin_embedding,
    lw.phi_score,
    lw.frequency,
    'generation',  -- All learned_words are for generation
    lw.source_type,
    lw.created_at,
    lw.updated_at
FROM learned_words lw
WHERE NOT EXISTS (
    SELECT 1 FROM coordizer_vocabulary cv
    WHERE cv.token = lw.word
)
AND lw.word ~ '^[a-z]{3,}$'  -- Only valid words
AND lw.phi_score > 0.0;      -- Only words with QFI scores

-- Step 2: Log migration results
DO $$
DECLARE
    migrated_count INTEGER;
BEGIN
    GET DIAGNOSTICS migrated_count = ROW_COUNT;
    RAISE NOTICE '[017] Migrated % unique words from learned_words', migrated_count;
END $$;

-- Step 3: Rename table to mark as deprecated
ALTER TABLE learned_words RENAME TO learned_words_deprecated_20260119;

-- Step 4: Add deprecation comment
COMMENT ON TABLE learned_words_deprecated_20260119 IS
'DEPRECATED: Historical learned_words table from pre-consolidation era.
All valid words migrated to coordizer_vocabulary.
Keep for 30 days for rollback safety, then DROP.
Migration date: 2026-01-19';

-- Step 5: Remove indexes to free space
DROP INDEX IF EXISTS idx_learned_words_phi;
DROP INDEX IF EXISTS idx_learned_words_frequency;
DROP INDEX IF EXISTS idx_learned_words_category;
DROP INDEX IF EXISTS idx_learned_words_last_used;
DROP INDEX IF EXISTS idx_learned_words_basin_hnsw;

RAISE NOTICE '[017] ✓ learned_words deprecated and ready for eventual removal';
```

### Phase 4: Enforce Validation in pg_loader.py

**File:** `qig-backend/coordizers/pg_loader.py`

```python
def _load_generation_vocabulary(self) -> bool:
    """Load GENERATION vocabulary with strict validation."""
    conn = self._get_connection()
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT token, basin_coords, qfi_score, frequency
                FROM coordizer_vocabulary
                WHERE token_role IN ('generation', 'both')
                  AND qfi_score IS NOT NULL
                ORDER BY qfi_score DESC
            """)
            
            loaded_count = 0
            rejected_count = 0
            
            for token, basin_coords, qfi_score, frequency in cur.fetchall():
                # VALIDATION GATE: Check if valid English word
                if not is_valid_english_word(token, include_stop_words=True, strict=True):
                    logger.debug(f"Rejected invalid word: {token}")
                    rejected_count += 1
                    continue
                
                # VALIDATION GATE: Check for BPE garbage
                if BPE_VALIDATION_AVAILABLE and is_bpe_garbage(token):
                    logger.debug(f"Rejected BPE artifact: {token}")
                    rejected_count += 1
                    continue
                
                # Add to vocabulary
                self.generation_vocab[token] = {
                    'coords': np.array(basin_coords),
                    'qfi': qfi_score,
                    'frequency': frequency
                }
                loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} generation words, rejected {rejected_count}")
            return loaded_count > 0
            
    except Exception as e:
        logger.error(f"Failed to load generation vocabulary: {e}")
        return False
    finally:
        conn.close()
```

---

## VALIDATION REQUIREMENTS

### Acceptance Criteria

- [ ] **Audit Complete**
  - [ ] Run audit_vocabulary.py
  - [ ] Generate garbage tokens list
  - [ ] Verify no false positives
  - [ ] Document learned_words overlap

- [ ] **Migration 016 (Clean Garbage)**
  - [ ] SQL migration written
  - [ ] Tested on dev database
  - [ ] Dry-run verification shows expected removals
  - [ ] Rollback plan documented

- [ ] **Migration 017 (Deprecate learned_words)**
  - [ ] Valid words migrated to coordizer_vocabulary
  - [ ] Table renamed with deprecation suffix
  - [ ] Indexes dropped
  - [ ] 30-day retention before DROP

- [ ] **Code Validation**
  - [ ] pg_loader.py enforces validation
  - [ ] Tests verify garbage rejection
  - [ ] Generation quality improves

### Success Metrics

**Before:**
```
coordizer_vocabulary:
  Total: 50,000
  Generation: 15,000
  Garbage in generation: ~1,500 (10%)

learned_words: 17,324 entries (duplicate storage)
```

**After:**
```
coordizer_vocabulary:
  Total: 50,000
  Generation: 13,500 (cleaned)
  Garbage in generation: 0 (0%)

learned_words_deprecated_20260119: 17,324 (read-only, scheduled for DROP)
```

---

## TIMELINE

- **Phase 1 (Audit):** 1 day
- **Phase 2 (Clean):** 1 day  
- **Phase 3 (Migrate):** 1 day
- **Phase 4 (Enforce):** 1 day
- **Total:** ~4 days

---

## REFERENCES

- **word_validation.py:** `/qig-backend/word_validation.py` (lines 28-33: BPE patterns)
- **pg_loader.py:** `/qig-backend/coordizers/pg_loader.py` (lines 190-220: generation vocab loading)
- **audit_vocabulary.py:** `/qig-backend/scripts/audit_vocabulary.py` (existing audit tool)
- **vocabulary_purity.py:** `/qig-backend/scripts/vocabulary_purity.py` (cleaning script)
- **Migration 008:** `migrations/0008_vocabulary_generation_separation.sql` (created learned_words)
- **Migration 013:** `migrations/0013_rename_tokenizer_to_coordizer.sql` (renamed tokenizer → coordizer)

---

## RISKS & MITIGATION

**Risk 1: Deleting legitimate abbreviations**
- **Mitigation:** Whitelist common abbreviations (UK, US, AI, ML, etc.)
- **Validation:** Manual review of garbage list before migration

**Risk 2: Breaking existing code that references learned_words**
- **Mitigation:** Rename (not DROP) for 30-day rollback window
- **Validation:** Grep for learned_words references before DROP

**Risk 3: Lost data from learned_words**
- **Mitigation:** Migrate all valid words to coordizer_vocabulary first
- **Validation:** Verify no qfi_score > 0.5 words lost

---

**Status:** Ready for implementation  
**Owner:** TBD  
**Reviewer:** @GaryOcean428
