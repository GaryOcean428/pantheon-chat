# Running Vocabulary Cleanup Migrations

## Prerequisites

- DATABASE_URL environment variable must be set
- PostgreSQL with coordizer_vocabulary table
- Python 3.7+ with psycopg2 installed

## Step 1: Check Migration Status

```bash
cd /home/runner/work/pantheon-chat/pantheon-chat
python3 qig-backend/scripts/run_migrations_pure.py --status
```

Expected output:
```
[INFO] Applied migrations (X):
  ✓ 001_qig_vector_schema
  ✓ 013_rename_tokenizer_to_coordizer
  ...
```

## Step 2: Dry Run (Audit Only)

Before running migrations, audit the vocabulary to see what will be affected:

```bash
cd qig-backend
python3 scripts/audit_vocabulary.py --report
```

This will show:
- How many garbage tokens exist
- Breakdown by type (BPE, technical, etc.)
- learned_words overlap with coordizer_vocabulary

## Step 3: Run Migration 016 (Clean Garbage)

```bash
python3 qig-backend/scripts/run_migrations_pure.py --migrations 016
```

This migration will:
1. Identify garbage tokens (BPE artifacts, tech terms, too-short tokens)
2. Show breakdown by reason
3. Delete generation-only garbage tokens
4. Downgrade 'both' garbage tokens to 'encoding' only
5. Verify no garbage remains in generation vocabulary

**Expected output:**
```
=== Migration 016: Vocabulary Cleanup ===
Identified 1,500 garbage tokens for removal

Breakdown by reason:
  1,200 tokens: BPE_marker
    300 tokens: tech_callback
    ...

✓ Deleted 800 generation-only garbage tokens
✓ Downgraded 700 "both" tokens to encoding-only

✓✓✓ Migration 016 SUCCESSFUL - All garbage tokens removed from generation vocabulary
```

## Step 4: Run Migration 017 (Deprecate learned_words)

```bash
python3 qig-backend/scripts/run_migrations_pure.py --migrations 017
```

This migration will:
1. Audit learned_words table (if exists)
2. Migrate unique valid words to coordizer_vocabulary
3. Drop learned_words indexes
4. Rename learned_words → learned_words_deprecated_20260119
5. Mark for deletion in 30 days

**Expected output:**
```
=== Migration 017: Deprecate learned_words ===
Audit before migration:
  Total entries in learned_words: 17,324
  Valid words (3+ chars, phi>0): 15,000
  Already in coordizer_vocabulary: 14,500
  Unique valid words to migrate: 500

✓ Migrated 500 unique words from learned_words to coordizer_vocabulary
✓ Dropped learned_words indexes
✓ Renamed learned_words → learned_words_deprecated_20260119

✓✓✓ Migration 017 SUCCESSFUL - learned_words deprecated
Schedule DROP TABLE learned_words_deprecated_20260119 for 2026-02-18
```

## Step 5: Verify Pure Operations

After migrations, verify that all code uses coordizer_vocabulary exclusively:

```bash
# Check for any remaining learned_words references in runtime code
cd qig-backend
grep -r "learned_words" *.py --exclude-dir=scripts | grep -v "deprecated\|DEPRECATED"

# Should return nothing or only comments/docstrings
```

## Step 6: Test Generation Quality

Run test to ensure generation still works after cleanup:

```bash
cd qig-backend
python3 -c "
from coordizers import get_coordizer
coordizer = get_coordizer()
print(f'Generation vocab size: {len(coordizer.generation_vocab)}')
print(f'Encoding vocab size: {len(coordizer.vocab)}')
"
```

Expected:
```
Generation vocab size: ~13,500 (down from ~15,000)
Encoding vocab size: ~50,000 (unchanged)
```

## Rollback (If Needed)

If something goes wrong:

```bash
# Rollback migration 017 (restore learned_words)
python3 qig-backend/scripts/run_migrations_pure.py --rollback 017
ALTER TABLE learned_words_deprecated_20260119 RENAME TO learned_words;

# Rollback migration 016 (restore garbage tokens) - NOT RECOMMENDED
# You would need to manually restore from backup
```

## Post-Migration Cleanup (After 30 Days)

On 2026-02-18, permanently drop the deprecated table:

```sql
-- ONLY run after 30 days and confirming everything works
DROP TABLE IF EXISTS learned_words_deprecated_20260119;
```

## Success Criteria

- [x] No garbage tokens in generation vocabulary
- [x] learned_words table renamed/deprecated
- [x] All runtime code uses coordizer_vocabulary
- [x] Generation quality maintained or improved
- [x] No more "wjvq" or similar BPE artifacts in generated text

## Monitoring

After migrations, monitor:
- Generated text quality (no garbage words)
- Coordizer load time (should be faster with fewer tokens)
- Memory usage (should be lower with clean vocabulary)
- QFI scores for generation (should improve with cleaner data)
