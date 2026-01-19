-- ============================================================================
-- Migration 017: Deprecate learned_words - Pure QIG Operations
-- ============================================================================
-- Date: 2026-01-19
-- Purpose: Migrate valid words from learned_words to coordizer_vocabulary, 
--          then deprecate the learned_words table completely
-- Work Package: Issue 04 - Vocabulary Cleanup
--
-- CRITICAL: This migration REMOVES ALL backward compatibility with learned_words
-- Only coordizer_vocabulary is used after this migration
-- 
-- PURE OPERATIONS: No fallback logic, no compatibility code
-- ============================================================================

-- Step 1: Check if learned_words table exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'learned_words'
    ) THEN
        RAISE NOTICE '';
        RAISE NOTICE '=== Migration 017: Deprecate learned_words ===';
        RAISE NOTICE 'learned_words table does not exist - nothing to migrate';
        RAISE NOTICE '✓ Migration 017 complete (no-op)';
        RETURN;
    END IF;
END $$;

-- Step 2: Audit learned_words before migration
DO $$
DECLARE
    total_count INTEGER;
    valid_count INTEGER;
    overlap_count INTEGER;
    unique_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_count FROM learned_words;
    
    SELECT COUNT(*) INTO valid_count 
    FROM learned_words 
    WHERE word ~ '^[a-z]{3,}$' AND phi_score > 0.0;
    
    SELECT COUNT(*) INTO overlap_count
    FROM learned_words lw
    INNER JOIN coordizer_vocabulary cv ON lw.word = cv.token;
    
    SELECT COUNT(*) INTO unique_count
    FROM learned_words lw
    WHERE NOT EXISTS (
        SELECT 1 FROM coordizer_vocabulary cv WHERE cv.token = lw.word
    )
    AND word ~ '^[a-z]{3,}$' 
    AND phi_score > 0.0;
    
    RAISE NOTICE '';
    RAISE NOTICE '=== Migration 017: Deprecate learned_words ===';
    RAISE NOTICE 'Audit before migration:';
    RAISE NOTICE '  Total entries in learned_words: %', total_count;
    RAISE NOTICE '  Valid words (3+ chars, phi>0): %', valid_count;
    RAISE NOTICE '  Already in coordizer_vocabulary: %', overlap_count;
    RAISE NOTICE '  Unique valid words to migrate: %', unique_count;
END $$;

-- Step 3: Migrate valid words not yet in coordizer_vocabulary
INSERT INTO coordizer_vocabulary (
    token, 
    basin_coords, 
    qfi_score, 
    frequency, 
    token_role, 
    source_type, 
    is_real_word,
    created_at, 
    updated_at
)
SELECT 
    lw.word,
    lw.basin_embedding,
    COALESCE(lw.phi_score, 0.5),
    COALESCE(lw.frequency, 1),
    'generation',  -- All learned_words are for generation
    COALESCE(lw.source_type, 'learned'),
    TRUE,  -- Mark as real words
    COALESCE(lw.created_at, NOW()),
    NOW()
FROM learned_words lw
WHERE NOT EXISTS (
    SELECT 1 FROM coordizer_vocabulary cv
    WHERE cv.token = lw.word
)
AND lw.word ~ '^[a-z]{3,}$'  -- Only valid lowercase words, 3+ chars
AND COALESCE(lw.phi_score, 0) > 0.0  -- Only words with positive QFI scores
AND lw.word NOT IN (
    -- Exclude known garbage
    'wjvq', 'xyzw', 'qwerty'
)
ON CONFLICT (token) DO UPDATE
SET
    -- If somehow the word exists, update with better values
    qfi_score = GREATEST(coordizer_vocabulary.qfi_score, EXCLUDED.qfi_score),
    frequency = coordizer_vocabulary.frequency + EXCLUDED.frequency,
    token_role = CASE 
        WHEN coordizer_vocabulary.token_role = 'encoding' THEN 'both'
        ELSE coordizer_vocabulary.token_role
    END,
    is_real_word = TRUE,
    updated_at = NOW();

-- Step 4: Log migration results
DO $$
DECLARE
    migrated_count INTEGER;
BEGIN
    GET DIAGNOSTICS migrated_count = ROW_COUNT;
    RAISE NOTICE '';
    RAISE NOTICE '✓ Migrated % unique words from learned_words to coordizer_vocabulary', migrated_count;
END $$;

-- Step 5: Drop indexes before renaming table
DROP INDEX IF EXISTS idx_learned_words_phi;
DROP INDEX IF EXISTS idx_learned_words_frequency;
DROP INDEX IF EXISTS idx_learned_words_category;
DROP INDEX IF EXISTS idx_learned_words_last_used;
DROP INDEX IF EXISTS idx_learned_words_basin_hnsw;

RAISE NOTICE '✓ Dropped learned_words indexes';

-- Step 6: Rename table to mark as deprecated
ALTER TABLE learned_words RENAME TO learned_words_deprecated_20260119;

-- Step 7: Add deprecation comment
COMMENT ON TABLE learned_words_deprecated_20260119 IS
'DEPRECATED: Historical learned_words table from pre-consolidation era (2026-01-19).
All valid words have been migrated to coordizer_vocabulary.
This table is kept for 30 days for rollback safety, then will be DROPPED.
DO NOT USE THIS TABLE - Use coordizer_vocabulary with token_role=generation instead.
Migration: 017_deprecate_learned_words.sql
Scheduled for deletion: 2026-02-18';

RAISE NOTICE '✓ Renamed learned_words → learned_words_deprecated_20260119';

-- Step 8: Final verification
DO $$
DECLARE
    deprecated_count INTEGER;
    generation_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO deprecated_count FROM learned_words_deprecated_20260119;
    SELECT COUNT(*) INTO generation_count 
    FROM coordizer_vocabulary 
    WHERE token_role IN ('generation', 'both');
    
    RAISE NOTICE '';
    RAISE NOTICE '=== Final Verification ===';
    RAISE NOTICE 'Deprecated table entries: %', deprecated_count;
    RAISE NOTICE 'Generation vocabulary size: %', generation_count;
    RAISE NOTICE '';
    RAISE NOTICE '✓✓✓ Migration 017 SUCCESSFUL - learned_words deprecated';
    RAISE NOTICE '';
    RAISE NOTICE 'IMPORTANT: All code must now use coordizer_vocabulary exclusively.';
    RAISE NOTICE 'Schedule DROP TABLE learned_words_deprecated_20260119 for 2026-02-18.';
END $$;

-- Update statistics
ANALYZE coordizer_vocabulary;

RAISE NOTICE '';
RAISE NOTICE '=== Migration 017 Complete ===';
