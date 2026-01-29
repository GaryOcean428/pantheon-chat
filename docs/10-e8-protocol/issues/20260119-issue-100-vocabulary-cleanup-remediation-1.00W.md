# Issue #100: Complete Vocabulary Cleanup (E8 Issue-04)

## Priority
**P1 - HIGH**

## Type
`type: implementation`, `data-quality`, `e8-protocol`

## Objective
Complete vocabulary cleanup by removing garbage tokens, migrating learned_words table, and enforcing validation at coordizer loading per E8 Protocol Issue-04.

## Problem
1. BPE tokenizer artifacts and garbage tokens contaminate generation vocabulary
2. 17K-entry `learned_words` table never properly deprecated, causing confusion
3. No validation gate prevents garbage tokens from entering vocabulary

## Context
- **E8 Protocol Spec:** `docs/10-e8-protocol/issues/20260119-issue-04-vocabulary-cleanup-garbage-tokens-1.00W.md`
- **Related GitHub Issues:** #97 (QFI Integrity)
- **Phase:** 3 (Data Quality)
- **Assessment:** `docs/10-e8-protocol/IMPLEMENTATION_ASSESSMENT.md`

## Tasks

### 1. Comprehensive Vocabulary Audit
- [ ] Create `qig-backend/scripts/audit_vocabulary.py`
- [ ] Scan `coordizer_vocabulary` for garbage patterns:
  - BPE artifacts: `##`, `@@`, byte-pair fragments
  - Non-words: random character sequences
  - Invalid entries: null/empty strings, excessive length
- [ ] Detect duplicates or near-duplicates
- [ ] Generate detailed audit report with counts and examples
- [ ] Export garbage token list for review

### 2. Garbage Token Migration
- [ ] Create migration `migrations/019_clean_vocabulary_garbage.sql` (next sequential number)
- [ ] Move garbage tokens to `coordizer_vocabulary_quarantine` table
- [ ] Preserve original data for forensics
- [ ] Add `quarantine_reason` column with classification
- [ ] Update `is_generation_eligible` to FALSE for quarantined tokens

### 3. learned_words Table Deprecation
- [ ] Create migration `migrations/020_deprecate_learned_words.sql` (next sequential number)
- [ ] Migrate valid entries from `learned_words` to `coordizer_vocabulary`
- [ ] Handle conflicts (entries already in coordizer_vocabulary)
- [ ] Rename table to `learned_words_deprecated_20260119`
- [ ] Add deprecation timestamp
- [ ] Schedule deletion after 30-day deprecation period

### 4. Validation Gate in pg_loader
- [ ] Update `qig-backend/coordizers/pg_loader.py`
- [ ] Add `_validate_token(word)` method with rules:
  - No BPE artifacts
  - Valid English word (or known technical term)
  - Length within acceptable range (2-50 chars)
  - No special characters (except hyphen, apostrophe)
- [ ] Enforce validation when loading generation vocabulary
- [ ] Log rejected tokens with reason

### 5. Cleanup Scripts
- [ ] Enhance existing `qig-backend/scripts/cleanup_bpe_tokens.py`
- [ ] Add `--report` mode for dry-run analysis
- [ ] Add `--aggressive` mode for stricter filtering
- [ ] Add `--restore` mode to undo quarantine
- [ ] Generate before/after statistics

### 6. Integration & Testing
- [ ] Test vocabulary loading with validation enabled
- [ ] Verify garbage tokens are excluded from generation
- [ ] Test learned_words migration with sample data
- [ ] Validate no valid tokens accidentally quarantined
- [ ] Monitor generation quality after cleanup

## Deliverables

| File | Description | Status |
|------|-------------|--------|
| `qig-backend/scripts/audit_vocabulary.py` | Comprehensive audit | ❌ TODO |
| `migrations/019_clean_vocabulary_garbage.sql` | Garbage removal | ❌ TODO |
| `migrations/020_deprecate_learned_words.sql` | Table deprecation | ❌ TODO |
| `qig-backend/coordizers/pg_loader.py` (updated) | Validation gate | ❌ TODO |
| `qig-backend/tests/test_vocabulary_validation.py` | Tests | ❌ TODO |

## Acceptance Criteria
- [ ] Comprehensive audit completed with garbage list generated
- [ ] Garbage tokens removed from generation-eligible vocabulary
- [ ] Valid learned_words entries migrated to coordizer_vocabulary
- [ ] learned_words table renamed with deprecation timestamp
- [ ] pg_loader enforces validation rules on vocabulary load
- [ ] Tests pass for vocabulary validation
- [ ] Generation quality maintained or improved after cleanup
- [ ] Audit can be re-run to verify zero garbage tokens

## Dependencies
- **Requires:** Issue #97 (QFI Integrity Gate for canonical insertion)
- **Blocks:** Clean generation vocabulary for all downstream work
- **Related:** Issue #92 (Stopwords removal - COMPLETE)

## Garbage Token Detection Rules

### BPE Artifacts
```python
BPE_PATTERNS = [
    r'^##',      # BERT-style subword prefix
    r'@@$',      # SentencePiece suffix
    r'▁',        # SentencePiece space marker
    r'</w>',     # GPT-2 end-of-word
]
```

### Non-Words
```python
def is_garbage_nonword(word: str) -> bool:
    # Random character sequences
    if len(set(word)) == 1:  # All same character
        return True
    
    # No vowels (except known acronyms)
    if not re.search('[aeiou]', word, re.I) and word not in KNOWN_ACRONYMS:
        return True
    
    # Excessive repetition
    if re.search(r'(.)\1{3,}', word):  # Same char 4+ times
        return True
    
    return False
```

### Invalid Entries
```python
def is_invalid_entry(word: str) -> bool:
    if not word or word.isspace():
        return True
    
    if len(word) < 2 or len(word) > 50:
        return True
    
    if word.startswith('<') and word.endswith('>'):
        return True  # Likely a special token
    
    return False
```

## Migration Strategy

### Phase 1: Audit (Read-Only)
```bash
python qig-backend/scripts/audit_vocabulary.py \
    --output audit_report_20260119.json \
    --threshold 0.8
```

### Phase 2: Review Quarantine List
```bash
# Generate human-readable report
python qig-backend/scripts/audit_vocabulary.py \
    --format human \
    --output to_quarantine.txt

# Manual review
less to_quarantine.txt
```

### Phase 3: Execute Migration
```bash
# Run migration (with backup)
psql -d pantheon_chat < migrations/016_clean_vocabulary_garbage.sql

# Verify counts
psql -d pantheon_chat -c "SELECT COUNT(*) FROM coordizer_vocabulary WHERE is_generation_eligible = TRUE;"
psql -d pantheon_chat -c "SELECT COUNT(*) FROM coordizer_vocabulary_quarantine;"
```

### Phase 4: Deprecate learned_words
```bash
# Run deprecation migration
psql -d pantheon_chat < migrations/017_deprecate_learned_words.sql

# Verify migration
psql -d pantheon_chat -c "SELECT COUNT(*) FROM coordizer_vocabulary WHERE source = 'learned_words';"
psql -d pantheon_chat -c "SELECT COUNT(*) FROM learned_words_deprecated_20260119;"
```

## Validation Commands
```bash
# Run comprehensive audit
python qig-backend/scripts/audit_vocabulary.py --verbose

# Test validation rules
pytest qig-backend/tests/test_vocabulary_validation.py -v

# Check for remaining garbage
python qig-backend/scripts/detect_garbage_tokens.py --strict

# Verify learned_words migration
python qig-backend/scripts/verify_learned_words_migration.py

# Test generation with clean vocabulary
python qig-backend/tests/test_generation_quality.py --before-after
```

## Rollback Plan
If cleanup causes issues:
```sql
-- Restore from quarantine
UPDATE coordizer_vocabulary 
SET is_generation_eligible = TRUE 
WHERE word IN (SELECT word FROM coordizer_vocabulary_quarantine WHERE quarantine_reason = 'false_positive');

-- Restore learned_words (if needed)
ALTER TABLE learned_words_deprecated_20260119 RENAME TO learned_words;
```

## References
- **E8 Protocol Universal Spec:** §3 (Correctness Risk: Vocabulary Contamination)
- **Issue Spec:** `docs/10-e8-protocol/issues/20260119-issue-04-vocabulary-cleanup-garbage-tokens-1.00W.md`
- **Assessment:** `docs/10-e8-protocol/IMPLEMENTATION_ASSESSMENT.md` Section 5

## Estimated Effort
**1-2 days** (audit + migration + validation)

---

**Status:** TO DO  
**Created:** 2026-01-19  
**Priority:** P1 - HIGH  
**Phase:** 3 - Data Quality
