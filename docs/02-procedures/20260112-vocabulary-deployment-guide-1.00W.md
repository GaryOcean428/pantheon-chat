# Deployment Instructions: Vocabulary Separation

## Prerequisites

- PostgreSQL database with pgvector extension
- Redis server running
- Python environment with dependencies installed
- Database backup (recommended)

## Step 1: Backup Database

```bash
# Create backup before migration
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql
```

## Step 2: Run Migration

```bash
# Navigate to project root
cd /home/runner/work/pantheon-chat/pantheon-chat

# Apply migration
psql $DATABASE_URL < migrations/0008_vocabulary_generation_separation.sql
```

Expected output:
```
NOTICE:  Added token_role column to coordizer_vocabulary
NOTICE:  Added phrase_category column to coordizer_vocabulary
NOTICE:  Deleted garbage tokens from coordizer_vocabulary
NOTICE:  Added PRIMARY KEY constraint to shadow_operations_state
NOTICE:  Populated learned_words table from coordizer_vocabulary
NOTICE:  Migration complete:
NOTICE:    - coordizer_vocabulary: XXXX tokens (encoding)
NOTICE:    - learned_words: YYYY words (generation)
NOTICE:    - shadow_operations_state: PRIMARY KEY added
```

## Step 3: Verify Migration

```bash
# Run verification script
cd qig-backend
python3 scripts/verify_vocabulary_separation.py
```

Expected output:
```
╔════════════════════════════════════════════════════╗
║  Vocabulary Separation Verification               ║
╚════════════════════════════════════════════════════╝

=== Checking Database Schema ===
✓ token_role column exists in coordizer_vocabulary
✓ phrase_category column exists in coordizer_vocabulary
✓ learned_words table exists
✓ shadow_operations_state has PRIMARY KEY constraint

=== Checking Vocabulary Counts ===
ℹ coordizer_vocabulary: XXXX tokens (encoding)
ℹ learned_words: YYYY words (generation)
✓ No BPE garbage in learned_words
✓ No PROPER_NOUN/BRAND in learned_words generation vocabulary
ℹ Average Φ - coordizer_vocabulary: 0.XXX, learned_words: 0.YYY
✓ Generation vocabulary has higher average Φ (better quality)

=== Checking Coordizer Integration ===
ℹ Encoding vocabulary: XXXX tokens
ℹ Generation vocabulary: YYYY words
✓ Coordizer loaded generation vocabulary
✓ encode() returns 64D basin (shape: (64,))
✓ decode() returned 5 candidates
ℹ Top candidate: 'word' (score: 0.XXX)
✓ decode() output contains no BPE garbage

=== Checking for Deprecation Warnings ===
✓ No deprecation warnings from autonomic_kernel

=== Summary ===
  Database Schema: PASS
  Vocabulary Counts: PASS
  Coordizer Integration: PASS
  Deprecation Warnings: PASS

Results: 4/4 checks passed

✓ All verification checks passed!
```

## Step 4: Test Generation

```bash
# Start Python backend
cd qig-backend
python3 ocean_qig_core.py
```

Then in another terminal:
```bash
# Test generation endpoint
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "bitcoin wallet address", "kernel_name": "zeus"}'
```

Verify output:
- ✅ No BPE subwords (Ġ, ##, ▁)
- ✅ No proper nouns used incorrectly
- ✅ No garbage tokens (ffffff, etc.)
- ✅ Clean, readable English text

## Step 5: Monitor Logs

```bash
# Check for errors
tail -f qig-backend/logs/*.log | grep -i "error\|warning\|deprecated"
```

Expected: No deprecation warnings, no vocabulary errors

## Rollback Procedure (if needed)

If issues arise:

### Option 1: Revert Code Only

```bash
git revert 97093df  # Revert fix commit
git revert f0cfcb5  # Revert main implementation
git push origin copilot/update-database-schema-and-filters
```

### Option 2: Revert Database + Code

```bash
# Restore database from backup
psql $DATABASE_URL < backup_YYYYMMDD_HHMMSS.sql

# Revert code
git revert 97093df f0cfcb5
git push origin copilot/update-database-schema-and-filters
```

### Option 3: Manual Cleanup (if no backup)

```sql
-- Drop new table
DROP TABLE IF EXISTS learned_words;

-- Remove new columns
ALTER TABLE coordizer_vocabulary DROP COLUMN IF EXISTS token_role;
ALTER TABLE coordizer_vocabulary DROP COLUMN IF EXISTS phrase_category;

-- Note: Cannot undo deleted garbage tokens without backup
```

## Troubleshooting

### Issue: Migration fails with "relation already exists"

**Solution**: Migration is idempotent. Tables/columns that exist are skipped.

```sql
-- Check what exists
SELECT table_name FROM information_schema.tables 
WHERE table_name IN ('coordizer_vocabulary', 'learned_words', 'shadow_operations_state');

SELECT column_name FROM information_schema.columns 
WHERE table_name = 'coordizer_vocabulary'
AND column_name IN ('token_role', 'phrase_category');
```

### Issue: learned_words table is empty

**Solution**: No valid words in coordizer_vocabulary to migrate.

```sql
-- Check source data
SELECT COUNT(*) FROM coordizer_vocabulary 
WHERE basin_embedding IS NOT NULL
  AND LENGTH(token) >= 2
  AND token ~ '^[a-z]+$';

-- Manual population if needed
INSERT INTO learned_words (word, basin_embedding, phi_score, frequency)
SELECT token, basin_embedding, phi_score, frequency
FROM coordizer_vocabulary
WHERE basin_embedding IS NOT NULL
  AND LENGTH(token) >= 2
  AND token ~ '^[a-z]+$'
  AND token_role = 'word'
ON CONFLICT (word) DO NOTHING;
```

### Issue: Coordizer fails to load generation vocabulary

**Symptoms**: 
```python
generation_words: 0
using fallback from coordizer_vocabulary
```

**Solution**: Check learned_words table has data:

```sql
SELECT COUNT(*) FROM learned_words WHERE basin_embedding IS NOT NULL;
```

If zero, run manual population query above.

### Issue: BPE garbage still in output

**Check**: Are you using the updated code?

```bash
git log --oneline -5
# Should show: "97093df Fix code review issues..."
```

**Verify**: Coordizer is using generation vocabulary:

```python
from coordizers import get_coordizer
coordizer = get_coordizer()
stats = coordizer.get_stats()
print(f"Generation vocab: {stats['generation_words']}")  # Should be > 0
```

## Success Criteria

✅ Migration completes without errors
✅ Verification script passes all checks (4/4)
✅ Generated text contains no BPE subwords
✅ Generated text contains no garbage tokens
✅ No deprecation warnings in logs
✅ Coordizer stats show separate vocabularies
✅ Average Φ is higher in generation vocabulary

## Performance Expectations

- **Migration time**: 10-30 seconds (depends on vocabulary size)
- **Startup time**: No change (vocabularies cached in-memory)
- **Generation speed**: Slight improvement (smaller generation vocab)
- **Memory usage**: Slight increase (two vocabularies cached)

## Post-Deployment Monitoring

Monitor for 24-48 hours:

1. **Generation quality**: Check output for garbage tokens
2. **Error logs**: Watch for vocabulary-related errors
3. **Performance**: Monitor generation latency
4. **Database**: Check learned_words table growth
5. **Cache hit rate**: Monitor Redis cache performance

## Contact

If you encounter issues:
1. Check VOCABULARY_SEPARATION_SUMMARY.md
2. Run verification script
3. Review logs
4. Rollback if necessary

## Changelog

- 2026-01-12: Initial deployment instructions
- Phase 1-3: Schema + Code changes
- Phase 4: Documentation + Verification
