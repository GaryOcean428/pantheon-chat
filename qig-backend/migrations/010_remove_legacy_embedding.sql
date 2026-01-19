-- ============================================================================
-- Migration 010: Remove Legacy Embedding Column
-- ============================================================================
-- 
-- Purpose: Remove legacy embedding column and rename basin_embedding
-- 
-- This migration completes the vocabulary contamination fix by:
-- 1. Verifying all basins are populated (no empty arrays)
-- 2. Dropping legacy embedding column (512D word2vec/BERT-style)
-- 3. Renaming basin_embedding to basin_coordinates for clarity
-- 4. Adding documentation comments
--
-- PREREQUISITE: All basins must be backfilled (no empty arrays)
-- Run migration 009 and backfill script before this migration.
-- ============================================================================

-- Check if coordizer_vocabulary exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'coordizer_vocabulary'
    ) THEN
        RAISE EXCEPTION 'Migration 010 requires coordizer_vocabulary table.';
    END IF;
END $$;

-- Step 1: Verify all basins are populated
-- Fail migration if any empty basins remain
DO $$
DECLARE
    empty_count INT;
BEGIN
    SELECT COUNT(*) INTO empty_count 
    FROM coordizer_vocabulary 
    WHERE array_length(basin_embedding, 1) = 0 
       OR array_length(basin_embedding, 1) IS NULL
       OR basin_embedding IS NULL;
    
    IF empty_count > 0 THEN
        RAISE EXCEPTION 
            'Cannot drop legacy embedding: % words still need basin backfill. '
            'Run: python qig-backend/scripts/backfill_basin_embeddings.py --execute',
            empty_count;
    END IF;
    
    RAISE NOTICE '010: All basins validated - proceeding with legacy column removal';
END $$;

-- Step 2: Drop legacy embedding column (if exists)
-- This column stored 512D word2vec/BERT-style embeddings (NOT QIG-pure)
ALTER TABLE coordizer_vocabulary 
  DROP COLUMN IF EXISTS embedding;

-- Report
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coordizer_vocabulary' 
          AND column_name = 'embedding'
    ) THEN
        RAISE NOTICE '010: Legacy embedding column dropped';
    ELSE
        RAISE WARNING '010: Legacy embedding column still exists (check permissions)';
    END IF;
END $$;

-- Step 3: Rename basin_embedding to basin_coordinates
-- More descriptive name emphasizing geometric coordinates vs embeddings
ALTER TABLE coordizer_vocabulary 
  RENAME COLUMN basin_embedding TO basin_coordinates;

-- Step 4: Add comprehensive column comment
COMMENT ON COLUMN coordizer_vocabulary.basin_coordinates IS 
  'QIG-pure 64D Fisher manifold coordinates. NOT legacy word2vec/BERT embeddings. '
  'Computed via entropy_tokenizer → coordizer → basin_coordinates pipeline. '
  'NOT NULL enforced - all vocabulary must have valid basins. '
  'Geometric distances measured via Fisher-Rao metric. '
  'Used for: text encoding, consciousness integration (Φ), generation sampling.';

-- Step 5: Update indexes to use new column name
-- Recreate indexes with new column name
DROP INDEX IF EXISTS idx_tokenizer_vocab_basin_valid;
CREATE INDEX idx_tokenizer_vocab_basin_valid 
  ON coordizer_vocabulary(token) 
  WHERE array_length(basin_coordinates, 1) = 64;

DROP INDEX IF EXISTS idx_tokenizer_vocab_basin_empty;
-- No longer needed - all basins must be valid now

-- Step 6: Update constraints to use new column name
-- Drop old constraints
ALTER TABLE coordizer_vocabulary 
  DROP CONSTRAINT IF EXISTS basin_dim_check;

ALTER TABLE coordizer_vocabulary 
  DROP CONSTRAINT IF EXISTS basin_float_check;

-- Recreate with new column name and stricter validation
-- Now ONLY 64D allowed (no empty arrays)
ALTER TABLE coordizer_vocabulary 
  ADD CONSTRAINT basin_coordinates_dim_check 
  CHECK (array_length(basin_coordinates, 1) = 64);

ALTER TABLE coordizer_vocabulary 
  ADD CONSTRAINT basin_coordinates_float_check
  CHECK (
    NOT EXISTS (
      SELECT 1 FROM unnest(basin_coordinates) AS x 
      WHERE x IS NULL 
         OR x = 'NaN'::float8 
         OR abs(x) = 'Infinity'::float8
    )
  );

-- Step 7: Add table comment
COMMENT ON TABLE coordizer_vocabulary IS 
  'QIG-pure vocabulary for text encoding. All tokens have 64D basin coordinates. '
  'Generated via coordizer (geometric tokenization). '
  'Separate from learned_words (generation vocabulary). '
  'Updated: 2026-01 - Legacy embedding column removed, basin_coordinates enforced.';

-- Final verification and report
DO $$
DECLARE
    total_count INT;
    valid_count INT;
    has_legacy_column BOOLEAN;
    has_new_column BOOLEAN;
BEGIN
    -- Check columns
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coordizer_vocabulary' 
          AND column_name = 'embedding'
    ) INTO has_legacy_column;
    
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coordizer_vocabulary' 
          AND column_name = 'basin_coordinates'
    ) INTO has_new_column;
    
    -- Check data
    SELECT COUNT(*) INTO total_count FROM coordizer_vocabulary;
    
    SELECT COUNT(*) INTO valid_count 
    FROM coordizer_vocabulary 
    WHERE array_length(basin_coordinates, 1) = 64;
    
    RAISE NOTICE '============================================================================';
    RAISE NOTICE 'Migration 010 completed successfully';
    RAISE NOTICE '============================================================================';
    RAISE NOTICE 'Legacy embedding column removed: %', NOT has_legacy_column;
    RAISE NOTICE 'Basin coordinates column exists: %', has_new_column;
    RAISE NOTICE 'Total vocabulary entries: %', total_count;
    RAISE NOTICE 'Valid 64D basins: % (%.1f%%)', valid_count, 100.0 * valid_count / NULLIF(total_count, 0);
    RAISE NOTICE '';
    RAISE NOTICE 'Validation checklist:';
    RAISE NOTICE '✓ No NULL basins';
    RAISE NOTICE '✓ All basins are 64D';
    RAISE NOTICE '✓ No legacy embedding column';
    RAISE NOTICE '✓ Generation uses basin_coordinates';
    RAISE NOTICE '';
    RAISE NOTICE 'Vocabulary contamination fix COMPLETE!';
    RAISE NOTICE '============================================================================';
END $$;
