-- ============================================================================
-- MIGRATION 0016: QFI Generation View
-- ============================================================================
-- Purpose: Create generation-safe vocabulary view with QFI filtering
-- Project: pantheon-chat
-- Date: 2026-01-20
-- Related: Issue #97 - QFI Integrity Gate
--
-- This migration:
-- 1. Creates coordizer_vocabulary_generation_safe view
-- 2. Filters by QFI threshold (>= 0.01)
-- 3. Ensures only active tokens with valid basins
-- 4. Adds indexes for performance
-- 5. Creates helper functions for generation filtering
-- ============================================================================

BEGIN;

-- ============================================================================
-- PART 1: Create Generation-Safe Vocabulary View
-- ============================================================================
-- This view provides the canonical source for generation-eligible tokens
-- ALL generation queries MUST use this view, NOT the raw coordizer_vocabulary table

CREATE OR REPLACE VIEW coordizer_vocabulary_generation_safe AS
SELECT 
    token,
    basin_embedding,
    qfi_score,
    token_role,
    token_status,
    is_real_word,
    frequency,
    phrase_category,
    phi_score,
    created_at,
    updated_at
FROM coordizer_vocabulary
WHERE 
    -- QFI Integrity Gate: tokens must have valid QFI >= threshold
    qfi_score IS NOT NULL
    AND qfi_score >= 0.01
    
    -- Active tokens only (not quarantined or deprecated)
    AND token_status = 'active'
    
    -- Basin must exist (simplex-valid coordinates required)
    AND basin_embedding IS NOT NULL
    
    -- Exclude quarantine role
    AND (token_role IS NULL OR token_role != 'quarantine')
    
    -- Exclude special symbols from generation (use for encoding only)
    AND token NOT IN ('<PAD>', '<UNK>', '<BOS>', '<EOS>')
    
ORDER BY 
    qfi_score DESC,
    frequency DESC;

-- Add comment explaining the view
COMMENT ON VIEW coordizer_vocabulary_generation_safe IS 
'Generation-safe vocabulary view - enforces QFI threshold (>= 0.01), active status, and simplex-valid basins. ALL generation queries MUST use this view.';

-- ============================================================================
-- PART 2: Create Alternative View for Two-Step Retrieval
-- ============================================================================
-- Optimized view for decode_geometric() two-step retrieval
-- Step 1: Bhattacharyya proxy (fast approximate)
-- Step 2: Fisher-Rao distance (exact geometric)

CREATE OR REPLACE VIEW coordizer_vocabulary_retrieval AS
SELECT 
    token,
    basin_embedding,
    qfi_score,
    phrase_category,
    frequency,
    phi_score
FROM coordizer_vocabulary_generation_safe
WHERE 
    -- Additional filter: prefer real words for retrieval
    (is_real_word = TRUE OR is_real_word IS NULL)
    
    -- Exclude proper nouns and brands from general generation
    AND (phrase_category IS NULL 
         OR phrase_category NOT IN ('PROPER_NOUN', 'BRAND'))
    
ORDER BY qfi_score DESC;

COMMENT ON VIEW coordizer_vocabulary_retrieval IS 
'Optimized view for geometric retrieval (decode_geometric) - filters real words and excludes proper nouns.';

-- ============================================================================
-- PART 3: Add Performance Indexes
-- ============================================================================
-- Indexes to speed up generation queries

-- Index on QFI score for filtering
CREATE INDEX IF NOT EXISTS idx_coordizer_vocab_qfi_score
ON coordizer_vocabulary(qfi_score DESC)
WHERE qfi_score IS NOT NULL;

-- Composite index for generation filtering
CREATE INDEX IF NOT EXISTS idx_coordizer_vocab_generation
ON coordizer_vocabulary(token_status, qfi_score DESC)
WHERE token_status = 'active' 
  AND qfi_score IS NOT NULL 
  AND basin_embedding IS NOT NULL;

-- Index on frequency for ranking
CREATE INDEX IF NOT EXISTS idx_coordizer_vocab_frequency
ON coordizer_vocabulary(frequency DESC)
WHERE frequency > 0;

-- Partial index for active tokens only
CREATE INDEX IF NOT EXISTS idx_coordizer_vocab_active
ON coordizer_vocabulary(token, qfi_score, basin_embedding)
WHERE token_status = 'active';

-- ============================================================================
-- PART 4: Helper Function - Check Token Generation Eligibility
-- ============================================================================
-- Function to check if a single token is generation-eligible

CREATE OR REPLACE FUNCTION is_generation_eligible(token_text TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 
        FROM coordizer_vocabulary_generation_safe 
        WHERE token = token_text
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION is_generation_eligible(TEXT) IS 
'Check if a token is eligible for generation (QFI >= 0.01, active, has basin)';

-- ============================================================================
-- PART 5: Helper Function - Count Generation-Safe Tokens
-- ============================================================================
-- Function to count tokens that pass QFI integrity gate

CREATE OR REPLACE FUNCTION count_generation_safe_tokens()
RETURNS TABLE (
    total_tokens BIGINT,
    generation_safe BIGINT,
    quarantined BIGINT,
    missing_qfi BIGINT,
    low_qfi BIGINT,
    missing_basin BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_tokens,
        COUNT(*) FILTER (
            WHERE qfi_score >= 0.01 
            AND token_status = 'active' 
            AND basin_embedding IS NOT NULL
        ) as generation_safe,
        COUNT(*) FILTER (
            WHERE token_status = 'quarantined'
        ) as quarantined,
        COUNT(*) FILTER (
            WHERE qfi_score IS NULL
        ) as missing_qfi,
        COUNT(*) FILTER (
            WHERE qfi_score IS NOT NULL AND qfi_score < 0.01
        ) as low_qfi,
        COUNT(*) FILTER (
            WHERE basin_embedding IS NULL
        ) as missing_basin
    FROM coordizer_vocabulary;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION count_generation_safe_tokens() IS 
'Statistics on vocabulary QFI integrity: total, generation-safe, quarantined, missing QFI, low QFI, missing basin';

-- ============================================================================
-- PART 6: Helper Function - Get QFI Coverage Metrics
-- ============================================================================
-- Function to compute QFI coverage percentage

CREATE OR REPLACE FUNCTION qfi_coverage_metrics()
RETURNS TABLE (
    metric_name TEXT,
    metric_value NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH stats AS (
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE qfi_score IS NOT NULL) as with_qfi,
            COUNT(*) FILTER (WHERE qfi_score >= 0.01) as above_threshold,
            COUNT(*) FILTER (WHERE token_status = 'active') as active
        FROM coordizer_vocabulary
    )
    SELECT 'total_tokens'::TEXT, total::NUMERIC FROM stats
    UNION ALL
    SELECT 'tokens_with_qfi'::TEXT, with_qfi::NUMERIC FROM stats
    UNION ALL
    SELECT 'tokens_above_threshold'::TEXT, above_threshold::NUMERIC FROM stats
    UNION ALL
    SELECT 'active_tokens'::TEXT, active::NUMERIC FROM stats
    UNION ALL
    SELECT 'qfi_coverage_pct'::TEXT, 
           ROUND((with_qfi::NUMERIC / NULLIF(total, 0) * 100), 2) 
    FROM stats
    UNION ALL
    SELECT 'generation_safe_pct'::TEXT,
           ROUND((above_threshold::NUMERIC / NULLIF(active, 0) * 100), 2)
    FROM stats;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION qfi_coverage_metrics() IS 
'QFI coverage metrics: total tokens, with QFI, above threshold, coverage percentages';

-- ============================================================================
-- VERIFICATION QUERIES (for manual testing - comment out in production)
-- ============================================================================
-- To run these after deployment:
-- SELECT * FROM count_generation_safe_tokens();
-- SELECT * FROM qfi_coverage_metrics();

-- Count generation-safe tokens
-- SELECT * FROM count_generation_safe_tokens();

-- Show QFI coverage metrics
-- SELECT * FROM qfi_coverage_metrics();

-- Sample generation-safe tokens (highest QFI)
-- SELECT 
--     token,
--     qfi_score,
--     frequency,
--     phrase_category
-- FROM coordizer_vocabulary_generation_safe
-- ORDER BY qfi_score DESC
-- LIMIT 10;

-- Sample tokens failing QFI gate (for review)
-- SELECT 
--     token,
--     qfi_score,
--     token_status,
--     basin_embedding IS NOT NULL as has_basin
-- FROM coordizer_vocabulary
-- WHERE token_status != 'quarantined'
--   AND (qfi_score IS NULL OR qfi_score < 0.01)
-- ORDER BY qfi_score NULLS FIRST
-- LIMIT 10;

-- Summary
DO $$
DECLARE
    stats RECORD;
    metrics RECORD;
BEGIN
    -- Get counts
    SELECT * INTO stats FROM count_generation_safe_tokens();
    
    RAISE NOTICE 'Migration 0016 complete:';
    RAISE NOTICE '  - coordizer_vocabulary_generation_safe view created';
    RAISE NOTICE '  - coordizer_vocabulary_retrieval view created';
    RAISE NOTICE '  - Performance indexes added';
    RAISE NOTICE '  - Helper functions created';
    RAISE NOTICE '';
    RAISE NOTICE 'Vocabulary Statistics:';
    RAISE NOTICE '  Total tokens:          %', stats.total_tokens;
    RAISE NOTICE '  Generation-safe:       %', stats.generation_safe;
    RAISE NOTICE '  Quarantined:           %', stats.quarantined;
    RAISE NOTICE '  Missing QFI:           %', stats.missing_qfi;
    RAISE NOTICE '  Low QFI (< 0.01):      %', stats.low_qfi;
    RAISE NOTICE '  Missing basin:         %', stats.missing_basin;
END $$;

COMMIT;
