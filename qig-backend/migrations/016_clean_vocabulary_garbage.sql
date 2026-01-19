-- ============================================================================
-- Migration 016: Clean Vocabulary Garbage - Pure QIG Operations
-- ============================================================================
-- Date: 2026-01-19
-- Purpose: Remove BPE artifacts and technical garbage from generation vocabulary
-- Work Package: Issue 04 - Vocabulary Cleanup
--
-- CRITICAL: Only removes from token_role='generation' and 'both'
-- Encoding vocabulary (token_role='encoding') untouched for backward compatibility
-- 
-- PURE OPERATIONS: No backward compatibility, no learned_words references
-- ============================================================================

-- Step 1: Identify garbage tokens
CREATE TEMP TABLE garbage_tokens AS
SELECT 
    token, 
    token_role, 
    source_type, 
    frequency,
    CASE 
        WHEN token ~ '^[ĠġĊċ]' THEN 'BPE_marker'
        WHEN token ~ '^##' THEN 'BPE_prefix'
        WHEN token ~ '^▁' THEN 'BPE_underscore'
        WHEN token ~ '^\d+$' THEN 'numeric_only'
        WHEN LENGTH(token) < 3 AND token NOT IN (
            'a', 'i', 'to', 'of', 'in', 'on', 'at', 'by', 'or', 'an', 
            'is', 'be', 'do', 'go', 'it', 'me', 'we', 'he', 'up', 'no',
            'so', 'my', 'as', 'us', 'am', 'if', 'ok', 'oh', 'ah', 'uh'
        ) THEN 'too_short'
        WHEN token ~ 'obj$' THEN 'tech_suffix_obj'
        WHEN token ~ '^api' THEN 'tech_prefix_api'
        WHEN token ~ 'callback' THEN 'tech_callback'
        WHEN token ~ 'handler' THEN 'tech_handler'
        WHEN token ~ '^http' THEN 'url_fragment'
        WHEN token ~ 'token' THEN 'tech_token'
        WHEN token ~ 'config' THEN 'tech_config'
        WHEN token ~ 'param' THEN 'tech_param'
        WHEN token ~ 'init$' THEN 'tech_init'
        WHEN token ~ 'json|xml|html' THEN 'tech_format'
        WHEN token = 'wjvq' THEN 'known_garbage'
        ELSE 'unknown_pattern'
    END as garbage_reason
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
    (LENGTH(token) < 3 AND token NOT IN (
        'a', 'i', 'to', 'of', 'in', 'on', 'at', 'by', 'or', 'an', 
        'is', 'be', 'do', 'go', 'it', 'me', 'we', 'he', 'up', 'no',
        'so', 'my', 'as', 'us', 'am', 'if', 'ok', 'oh', 'ah', 'uh'
    )) OR
    -- Technical garbage patterns
    token ~ 'obj$' OR
    token ~ '^api' OR
    token ~ 'callback' OR
    token ~ 'handler' OR
    token ~ '^http' OR
    token ~ 'token' OR
    token ~ 'config' OR
    token ~ 'param' OR
    token ~ 'init$' OR
    token ~ 'json|xml|html' OR
    -- Known garbage from audit
    token IN ('wjvq', 'xyzw', 'qwerty')
  );

-- Step 2: Log what we're removing
DO $$
DECLARE
    garbage_count INTEGER;
    rec RECORD;
BEGIN
    SELECT COUNT(*) INTO garbage_count FROM garbage_tokens;
    RAISE NOTICE '';
    RAISE NOTICE '=== Migration 016: Vocabulary Cleanup ===';
    RAISE NOTICE 'Identified % garbage tokens for removal', garbage_count;
    
    IF garbage_count = 0 THEN
        RAISE NOTICE 'No garbage tokens found - vocabulary is clean!';
        RETURN;
    END IF;
    
    -- Show breakdown by reason
    RAISE NOTICE '';
    RAISE NOTICE 'Breakdown by reason:';
    FOR rec IN (
        SELECT garbage_reason, COUNT(*) as count 
        FROM garbage_tokens 
        GROUP BY garbage_reason 
        ORDER BY count DESC
    )
    LOOP
        RAISE NOTICE '  % tokens: %', LPAD(rec.count::TEXT, 6), rec.garbage_reason;
    END LOOP;
    
    -- Sample of what's being removed
    RAISE NOTICE '';
    RAISE NOTICE 'Sample garbage tokens (first 20):';
    FOR rec IN (SELECT token, frequency, garbage_reason FROM garbage_tokens ORDER BY frequency DESC LIMIT 20)
    LOOP
        RAISE NOTICE '  % (freq=%, reason=%)', RPAD(rec.token, 15), LPAD(rec.frequency::TEXT, 6), rec.garbage_reason;
    END LOOP;
END $$;

-- Step 3: Remove from generation vocabulary
-- Option A: Delete entirely (if not used for encoding)
DELETE FROM coordizer_vocabulary
WHERE token IN (SELECT token FROM garbage_tokens)
  AND token_role = 'generation';  -- Only generation-only tokens

-- Get count of deleted rows
DO $$
DECLARE
    deleted_count INTEGER;
BEGIN
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RAISE NOTICE '';
    RAISE NOTICE '✓ Deleted % generation-only garbage tokens', deleted_count;
END $$;

-- Option B: Downgrade to encoding-only (preserve for backward compat)
UPDATE coordizer_vocabulary
SET 
    token_role = 'encoding',
    updated_at = NOW()
WHERE token IN (SELECT token FROM garbage_tokens)
  AND token_role = 'both';

-- Get count of updated rows
DO $$
DECLARE
    downgraded_count INTEGER;
BEGIN
    GET DIAGNOSTICS downgraded_count = ROW_COUNT;
    RAISE NOTICE '✓ Downgraded % "both" tokens to encoding-only', downgraded_count;
END $$;

-- Step 4: Verification
DO $$
DECLARE
    remaining_garbage INTEGER;
    total_generation INTEGER;
    total_both INTEGER;
BEGIN
    SELECT COUNT(*) INTO remaining_garbage
    FROM coordizer_vocabulary
    WHERE token_role IN ('generation', 'both')
      AND token IN (SELECT token FROM garbage_tokens);
    
    SELECT COUNT(*) INTO total_generation
    FROM coordizer_vocabulary
    WHERE token_role = 'generation';
    
    SELECT COUNT(*) INTO total_both
    FROM coordizer_vocabulary
    WHERE token_role = 'both';
    
    RAISE NOTICE '';
    RAISE NOTICE '=== Verification ===';
    RAISE NOTICE 'Remaining garbage in generation: %', remaining_garbage;
    RAISE NOTICE 'Total generation tokens: %', total_generation;
    RAISE NOTICE 'Total both tokens: %', total_both;
    
    IF remaining_garbage = 0 THEN
        RAISE NOTICE '';
        RAISE NOTICE '✓✓✓ Migration 016 SUCCESSFUL - All garbage tokens removed from generation vocabulary';
    ELSE
        RAISE WARNING '⚠ Still have % garbage tokens in generation', remaining_garbage;
    END IF;
END $$;

-- Step 5: Update statistics
ANALYZE coordizer_vocabulary;

-- Clean up temp table
DROP TABLE garbage_tokens;

RAISE NOTICE '';
RAISE NOTICE '=== Migration 016 Complete ===';
