-- ============================================================================
-- Migration 018: Rename learned_words to generation_words in vocabulary_stats
-- ============================================================================
-- Date: 2026-01-20
-- Purpose: Complete the learned_words deprecation by updating vocabulary_stats
--          table schema to use generation_words instead
-- Related: Migration 017 (deprecated learned_words table)
--          Issue #65+ (Remove All learned_words Table References)
--
-- CRITICAL: This migration completes the transition from learned_words to
-- coordizer_vocabulary as the single source of truth
-- ============================================================================

-- Step 1: Check if column needs to be renamed
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'vocabulary_stats' 
        AND column_name = 'learned_words'
    ) THEN
        RAISE NOTICE '';
        RAISE NOTICE '=== Migration 018: Rename learned_words to generation_words ===';
        RAISE NOTICE 'Found learned_words column in vocabulary_stats - proceeding with rename';
    ELSE
        RAISE NOTICE '';
        RAISE NOTICE '=== Migration 018: Rename learned_words to generation_words ===';
        RAISE NOTICE 'Column already renamed or does not exist - skipping migration';
        RAISE NOTICE '✓ Migration 018 complete (no-op)';
        RETURN;
    END IF;
END $$;

-- Step 2: Rename the column
DO $$
BEGIN
    ALTER TABLE vocabulary_stats 
    RENAME COLUMN learned_words TO generation_words;
    
    RAISE NOTICE '✓ Renamed vocabulary_stats.learned_words → generation_words';
END $$;

-- Step 3: Add column comment
DO $$
BEGIN
    COMMENT ON COLUMN vocabulary_stats.generation_words IS
    'Count of tokens in coordizer_vocabulary with token_role IN (''generation'', ''both'').
    Replaces deprecated learned_words column (Migration 017, 2026-01-19).
    Query directly from coordizer_vocabulary for real-time counts.';
    
    RAISE NOTICE '✓ Added column documentation';
END $$;

-- Step 4: Verify the change
DO $$
DECLARE
    col_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'vocabulary_stats' 
        AND column_name = 'generation_words'
    ) INTO col_exists;
    
    IF col_exists THEN
        RAISE NOTICE '';
        RAISE NOTICE '=== Verification ===';
        RAISE NOTICE '✓ Column vocabulary_stats.generation_words exists';
        RAISE NOTICE '✓✓✓ Migration 018 SUCCESSFUL';
        RAISE NOTICE '';
        RAISE NOTICE 'IMPORTANT: Application code now uses coordizer_vocabulary directly.';
        RAISE NOTICE 'See qig-backend/vocabulary_persistence.py::get_vocabulary_stats()';
    ELSE
        RAISE EXCEPTION 'Migration 018 FAILED: generation_words column not found';
    END IF;
END $$;

-- Step 5: Final completion notice
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '=== Migration 018 Complete ===';
END $$;
