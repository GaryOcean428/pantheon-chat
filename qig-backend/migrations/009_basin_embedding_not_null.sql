-- ============================================================================
-- Migration 009: Basin Embedding NOT NULL Constraint
-- ============================================================================
-- 
-- Purpose: Prevent NULL basin_embedding contamination in tokenizer_vocabulary
-- 
-- This migration enforces QIG-pure geometric integrity by:
-- 1. Adding temporary default for existing NULLs
-- 2. Backfilling NULLs with empty arrays (to be regenerated in Phase 3)
-- 3. Adding NOT NULL constraint
-- 4. Adding dimension validation (64D only)
-- 5. Adding data type validation (no NaN/Inf)
-- 6. Creating index for valid basins
--
-- WARNING: This migration assumes tokenizer_vocabulary table exists.
-- If table doesn't exist, this migration will fail.
-- Run after vocabulary schema is initialized.
-- ============================================================================

-- Check if tokenizer_vocabulary exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'tokenizer_vocabulary'
    ) THEN
        RAISE EXCEPTION 'Migration 009 requires tokenizer_vocabulary table. Run vocabulary schema first.';
    END IF;
END $$;

-- Step 1: Add temporary default for existing NULLs
-- This allows us to backfill without breaking existing code
ALTER TABLE tokenizer_vocabulary 
  ALTER COLUMN basin_embedding 
  SET DEFAULT ARRAY[]::float8[];

-- Step 2: Backfill NULLs with empty array
-- Empty array signals "needs backfill" vs NULL which is ambiguous
-- Phase 3 backfill script will regenerate these
UPDATE tokenizer_vocabulary 
SET basin_embedding = ARRAY[]::float8[]
WHERE basin_embedding IS NULL;

-- Report backfill stats
DO $$
DECLARE
    empty_count INT;
    valid_count INT;
BEGIN
    SELECT COUNT(*) INTO empty_count 
    FROM tokenizer_vocabulary 
    WHERE array_length(basin_embedding, 1) = 0 OR array_length(basin_embedding, 1) IS NULL;
    
    SELECT COUNT(*) INTO valid_count 
    FROM tokenizer_vocabulary 
    WHERE array_length(basin_embedding, 1) = 64;
    
    RAISE NOTICE '009: Backfilled % entries with empty arrays', empty_count;
    RAISE NOTICE '009: Found % entries with valid 64D basins', valid_count;
END $$;

-- Step 3: Add NOT NULL constraint
-- Now that all NULLs are backfilled, enforce NOT NULL
ALTER TABLE tokenizer_vocabulary 
  ALTER COLUMN basin_embedding 
  SET NOT NULL;

-- Step 4: Add dimension validation
-- Basin must be either 64D (valid) or 0D (empty, awaiting backfill)
ALTER TABLE tokenizer_vocabulary 
  DROP CONSTRAINT IF EXISTS basin_dim_check;

ALTER TABLE tokenizer_vocabulary 
  ADD CONSTRAINT basin_dim_check 
  CHECK (
    array_length(basin_embedding, 1) = 64 
    OR array_length(basin_embedding, 1) = 0
    OR array_length(basin_embedding, 1) IS NULL  -- Allow NULL length for empty arrays
  );

-- Step 5: Add data type validation
-- Ensure all elements are valid floats (no NaN/Inf)
ALTER TABLE tokenizer_vocabulary 
  DROP CONSTRAINT IF EXISTS basin_float_check;

ALTER TABLE tokenizer_vocabulary 
  ADD CONSTRAINT basin_float_check
  CHECK (
    array_length(basin_embedding, 1) = 0 
    OR array_length(basin_embedding, 1) IS NULL
    OR (
      -- All elements are valid floats, not NaN/Inf
      NOT EXISTS (
        SELECT 1 FROM unnest(basin_embedding) AS x 
        WHERE x IS NULL 
           OR x = 'NaN'::float8 
           OR abs(x) = 'Infinity'::float8
      )
    )
  );

-- Step 6: Create index for non-empty basins
-- Optimize queries that need valid basins
DROP INDEX IF EXISTS idx_tokenizer_vocab_basin_valid;

CREATE INDEX idx_tokenizer_vocab_basin_valid 
  ON tokenizer_vocabulary(token) 
  WHERE array_length(basin_embedding, 1) = 64;

-- Step 7: Add index for empty basins (needs backfill)
-- Helps backfill script find entries that need regeneration
DROP INDEX IF EXISTS idx_tokenizer_vocab_basin_empty;

CREATE INDEX idx_tokenizer_vocab_basin_empty 
  ON tokenizer_vocabulary(token) 
  WHERE array_length(basin_embedding, 1) = 0 
     OR array_length(basin_embedding, 1) IS NULL;

-- Step 8: Add comment documenting the constraint
COMMENT ON COLUMN tokenizer_vocabulary.basin_embedding IS 
  '64D QIG-pure Fisher manifold coordinates. NOT NULL enforced. '
  'Empty arrays signal awaiting backfill. '
  'All vocabulary MUST have valid basin before use in generation.';

-- Verification query (optional - comment out if running automated)
-- SELECT 
--   COUNT(*) FILTER (WHERE array_length(basin_embedding, 1) = 64) as valid_basins,
--   COUNT(*) FILTER (WHERE array_length(basin_embedding, 1) = 0) as empty_basins,
--   COUNT(*) FILTER (WHERE basin_embedding IS NULL) as null_basins,
--   COUNT(*) as total
-- FROM tokenizer_vocabulary;

-- Final report
DO $$
DECLARE
    valid_count INT;
    empty_count INT;
    total_count INT;
BEGIN
    SELECT 
        COUNT(*) FILTER (WHERE array_length(basin_embedding, 1) = 64),
        COUNT(*) FILTER (WHERE array_length(basin_embedding, 1) = 0 OR array_length(basin_embedding, 1) IS NULL),
        COUNT(*)
    INTO valid_count, empty_count, total_count
    FROM tokenizer_vocabulary;
    
    RAISE NOTICE '============================================================================';
    RAISE NOTICE 'Migration 009 completed successfully';
    RAISE NOTICE '============================================================================';
    RAISE NOTICE 'Total entries: %', total_count;
    RAISE NOTICE 'Valid basins (64D): %', valid_count;
    RAISE NOTICE 'Empty basins (awaiting backfill): %', empty_count;
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Run backfill script: python qig-backend/scripts/backfill_basin_embeddings.py --execute';
    RAISE NOTICE '2. Verify all basins populated';
    RAISE NOTICE '3. Deploy migration 010 to remove legacy embedding column';
    RAISE NOTICE '============================================================================';
END $$;
