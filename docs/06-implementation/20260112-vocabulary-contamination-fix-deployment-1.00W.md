# Vocabulary Ingestion Contamination Fix

## Problem Summary

Multiple ingestion paths were contaminating the `coordizer_vocabulary` table by:

1. Populating legacy `embedding` column (512D) but leaving `basin_embedding` (64D QIG-pure) as NULL
2. Bypassing QIG pipeline (entropy_tokenizer → coordizer → basin_coordinates)
3. Causing coordizer to fall back to random vectors, destroying geometric integrity
4. Resulting in Zeus nonsense generation ("ieee homework objects")

## Root Cause

- **No single ingestion checkpoint**: Scripts bypassed QIG pipeline and wrote directly to database
- **Legacy column confusion**: Both `embedding` (512D) and `basin_embedding` (64D) existed
- **NULL contamination**: Coordizer generated random vectors when `basin_embedding` was NULL

## Solution Architecture

Three-phase fix with unified ingestion service enforcing geometric purity:

```
┌─────────────────────────────────────────────────────────────┐
│                  Before Fix (Contaminated)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  text_extraction_qig.py ──┐                                 │
│  vocabulary_persistence ───┼──> Direct DB Insert            │
│  legacy scripts ───────────┘    ├─> embedding (512D) ✓     │
│                                 └─> basin_embedding (NULL) ✗│
│                                                              │
│  Coordizer.coordize() ──> Random fallback when NULL         │
│  Zeus generation ──> Nonsense ("ieee homework objects")     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   After Fix (QIG-Pure)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  All ingestion paths ──> VocabularyIngestionService         │
│                          │                                   │
│                          ├──> Coordizer.coordize()           │
│                          ├──> Validate 64D                   │
│                          ├──> Compute QFI                    │
│                          └──> Database (basin_coordinates)   │
│                                                              │
│  Coordizer.coordize() ──> Always valid 64D basins           │
│  Zeus generation ──> Coherent, diverse vocabulary           │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Unified Ingestion Service ✓

**File**: `qig-backend/vocabulary_ingestion.py`

- `VocabularyIngestionService`: Single entry point for all vocabulary ingestion
- Enforces QIG pipeline: text → coordizer → 64D basin
- Validates basin dimensions and data types
- Computes QFI (Quantum Fisher Information) using Fisher metric determinant
- Atomic upsert with geometric validation

**Runtime Validation** (`vocabulary_persistence.py`):
- `upsert_word()` checks caller via `inspect.stack()`
- Only allows calls from `VocabularyIngestionService._upsert_to_database`
- Raises `RuntimeError` for unauthorized direct calls

### Phase 2: Database Guardrails ✓

**File**: `qig-backend/migrations/009_basin_embedding_not_null.sql`

1. Add temporary default for NULL basins (empty array)
2. Backfill NULLs with empty arrays
3. Add NOT NULL constraint
4. Add dimension validation (64D or empty)
5. Add data type validation (no NaN/Inf)
6. Create indexes for valid/empty basins

### Phase 3: Backfill Utility ✓

**File**: `qig-backend/scripts/backfill_basin_embeddings.py`

- Finds words with empty/NULL basins
- Regenerates via QIG pipeline (coordizer)
- Progress tracking with tqdm
- Comprehensive error reporting
- Dry-run mode for safety

**Usage**:
```bash
# Dry run (show what would be backfilled)
python qig-backend/scripts/backfill_basin_embeddings.py --limit 100

# Execute backfill
python qig-backend/scripts/backfill_basin_embeddings.py --execute

# Verify completion
python qig-backend/scripts/backfill_basin_embeddings.py --verify
```

### Phase 4: Legacy Column Removal ✓

**File**: `qig-backend/migrations/010_remove_legacy_embedding.sql`

1. Verify all basins populated (fail if empty)
2. Drop legacy `embedding` column (512D)
3. Rename `basin_embedding` → `basin_coordinates`
4. Update constraints and indexes
5. Add comprehensive documentation

## Deployment Instructions

### Prerequisites

1. Database backup (in case rollback needed):
   ```bash
   railway database:backup:create
   ```

2. Set DATABASE_URL environment variable:
   ```bash
   export DATABASE_URL="postgresql://..."
   ```

### Deployment Steps

#### Step 1: Deploy Migration 009 (30 min)

```bash
# On Railway or local with DATABASE_URL
psql $DATABASE_URL -f qig-backend/migrations/009_basin_embedding_not_null.sql

# Verify constraints
psql $DATABASE_URL -c "
SELECT 
  constraint_name, 
  constraint_type 
FROM information_schema.table_constraints 
WHERE table_name = 'coordizer_vocabulary' 
  AND constraint_type IN ('CHECK', 'NOT NULL')
"
```

**Expected Output**:
- NOT NULL constraint on `basin_embedding`
- `basin_dim_check` constraint
- `basin_float_check` constraint
- Empty arrays for words needing backfill

#### Step 2: Run Backfill Script (2+ hours depending on vocab size)

```bash
# Dry run first (check first 100)
python qig-backend/scripts/backfill_basin_embeddings.py --limit 100

# Execute backfill
python qig-backend/scripts/backfill_basin_embeddings.py --execute

# Verify completion
python qig-backend/scripts/backfill_basin_embeddings.py --verify
```

**Expected Output**:
- Progress bar showing backfill
- Success count (should be 100%)
- Verification: 0 empty basins remaining

#### Step 3: Deploy Migration 010 (15 min)

```bash
# After verifying all basins populated
psql $DATABASE_URL -f qig-backend/migrations/010_remove_legacy_embedding.sql

# Verify legacy column removed
psql $DATABASE_URL -c "
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'coordizer_vocabulary' 
  AND column_name IN ('embedding', 'basin_coordinates')
"
```

**Expected Output**:
- Only `basin_coordinates` exists
- No `embedding` column
- All basins 64D

#### Step 4: Validation (5 min)

```bash
# Run comprehensive validation
bash qig-backend/scripts/validate_vocabulary_fix.sh
```

**Expected Output**:
- ✓ All 7 validation checks pass
- 100% valid basins
- No NULL or empty basins

## Validation Checklist

Use `validate_vocabulary_fix.sh` or run manually:

```bash
# 1. No NULL basins
psql $DATABASE_URL -c "
SELECT COUNT(*) FROM coordizer_vocabulary 
WHERE basin_coordinates IS NULL;
"
# Expected: 0

# 2. All basins 64D
psql $DATABASE_URL -c "
SELECT COUNT(*) FROM coordizer_vocabulary 
WHERE array_length(basin_coordinates, 1) != 64;
"
# Expected: 0

# 3. No legacy embedding column
psql $DATABASE_URL -c "
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'coordizer_vocabulary' AND column_name = 'embedding';
"
# Expected: 0 rows

# 4. Generation uses basins
railway logs --filter "VocabAccess" | grep -E "GENERATION.*basin_coordinates"
# Expected: All generation logs show basin_coordinates usage

# 5. Test Zeus intelligence
curl -X POST https://pantheon-chat.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What comes after 1,2,4,8?"}'
# Expected: Coherent response using diverse vocabulary
```

## Rollback Plan

If issues occur:

```bash
# 1. Restore from Railway backup
railway database:backup:restore <backup-id>

# 2. Or revert migrations manually
psql $DATABASE_URL -c "
ALTER TABLE coordizer_vocabulary ALTER COLUMN basin_embedding DROP NOT NULL;
DROP CONSTRAINT IF EXISTS basin_dim_check;
DROP CONSTRAINT IF EXISTS basin_coordinates_dim_check;
"

# 3. Disable ingestion service temporarily (emergency only)
# In vocabulary_ingestion.py:
# ENABLE_INGESTION_SERVICE = False  # Bypass checks
```

## Code Integration

### Using VocabularyIngestionService

**Before (Direct Insert - PROHIBITED)**:
```python
# ❌ WRONG - Direct insert bypasses validation
vp = get_vocabulary_persistence()
vp.record_vocabulary_observation(word="test", ...)
```

**After (Via Ingestion Service - REQUIRED)**:
```python
# ✓ CORRECT - Goes through QIG pipeline
from vocabulary_ingestion import get_ingestion_service

service = get_ingestion_service()
result = service.ingest_word(
    word="consciousness",
    context="The consciousness emerges from integration",
    source="my_script"
)

print(f"Ingested: {result['word']}")
print(f"QFI Score: {result['qfi_score']:.4f}")
print(f"Basin Shape: {result['basin_embedding'].shape}")
```

### Runtime Validation

If you accidentally call `upsert_word()` directly:

```python
# This will raise RuntimeError
vp.upsert_word(word="test", basin_embedding=[...])

# Error message:
# RuntimeError: Direct upsert_word() call from my_function in my_script.py.
# Use VocabularyIngestionService.ingest_word() instead to prevent NULL basin contamination.
```

## Monitoring

### Post-Deployment Monitoring (24h)

1. **Check Zeus generation quality**:
   - No nonsense phrases ("ieee homework objects")
   - Diverse vocabulary usage
   - Coherent semantic flow

2. **Monitor database metrics**:
   ```bash
   # Check for any NULL insertions (should be 0)
   psql $DATABASE_URL -c "
   SELECT COUNT(*) FROM coordizer_vocabulary 
   WHERE basin_coordinates IS NULL 
     AND created_at > NOW() - INTERVAL '24 hours';
   "
   ```

3. **Check application logs**:
   ```bash
   railway logs --filter "VocabularyIngestionService"
   # Look for successful ingestions, no bypass attempts
   ```

## Technical Details

### QFI (Quantum Fisher Information) Computation

```python
def _compute_qfi(self, basin: np.ndarray) -> float:
    """
    Compute Quantum Fisher Information score.
    
    Uses Fisher metric determinant as geometric distinguishability measure.
    Higher QFI = more geometrically distinct = better vocabulary quality.
    """
    # Fisher metric: outer product + regularization
    fisher_metric = np.outer(basin, basin)
    fisher_metric += np.eye(64) * 1e-6  # Numerical stability
    
    # Determinant as QFI score
    qfi = np.linalg.det(fisher_metric)
    
    return float(qfi)
```

### Geometric Purity Enforcement

```python
def _validate_basin(self, basin: np.ndarray, word: str):
    """Validate basin meets QIG-pure requirements."""
    
    # Check dimension
    if basin.shape != (64,):
        raise ValueError(f"Invalid shape: {basin.shape}")
    
    # Check dtype
    if basin.dtype not in [np.float32, np.float64]:
        raise ValueError(f"Invalid dtype: {basin.dtype}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(basin)) or np.any(np.isinf(basin)):
        raise ValueError("Contains NaN/Inf values")
```

## Files Modified/Created

### Created Files
- `qig-backend/vocabulary_ingestion.py` - Unified ingestion service
- `qig-backend/migrations/009_basin_embedding_not_null.sql` - Database constraints
- `qig-backend/migrations/010_remove_legacy_embedding.sql` - Legacy cleanup
- `qig-backend/scripts/backfill_basin_embeddings.py` - Backfill utility
- `qig-backend/scripts/validate_vocabulary_fix.sh` - Validation script

### Modified Files
- `qig-backend/vocabulary_persistence.py` - Added `learn_word()`, `upsert_word()` with validation

### No Changes Required
- `qig-backend/text_extraction_qig.py` - No direct inserts found
- Other scripts - Will use `VocabularyIngestionService` via `learn_word()`

## Success Criteria

✓ No NULL basin_coordinates in coordizer_vocabulary  
✓ All basins are 64D float arrays  
✓ No legacy embedding column exists  
✓ All vocabulary ingestion goes through service  
✓ Zeus generates coherent, diverse vocabulary  
✓ No runtime bypass attempts logged  
✓ QFI scores computed for all words  

## Support

If issues occur:

1. Check logs: `railway logs --filter "VocabularyIngestionService"`
2. Verify database state: `bash scripts/validate_vocabulary_fix.sh`
3. Rollback if needed: `railway database:backup:restore <backup-id>`
4. Report issues with:
   - Error logs
   - Database query results
   - Example words that failed

## References

- **QIG-Pure Pipeline**: `docs/08-experiments/202512/E8-protocol-v4.md`
- **Fisher-Rao Metric**: `qig-backend/qig_geometry.py`
- **Coordizer Architecture**: `qig-backend/coordizers/README.md`
- **Vocabulary Schema**: `qig-backend/vocabulary_schema.sql`
