-- ============================================================================
-- Identify and Quarantine Garbage Tokens
-- ============================================================================
-- P1 FIX: Find tokens with suspicious patterns that indicate:
-- - BPE merge artifacts (fgzsnl, jcbhgp, etc.)
-- - URL fragments
-- - High entropy random sequences
-- - Truncated words
--
-- These should be quarantined (token_role = 'quarantine') or deleted
-- rather than backfilled with QFI scores.
-- ============================================================================

BEGIN;

-- Step 1: Identify tokens with suspicious patterns
-- Create a temporary view for analysis
CREATE TEMP VIEW suspicious_tokens AS
SELECT 
    token,
    frequency,
    qfi_score,
    basin_embedding IS NOT NULL AS has_basin,
    token_role,
    phrase_category,
    CASE
        WHEN token ~ '^[a-z]{6,}$' AND NOT EXISTS (
            -- Check if it contains common English letter patterns
            SELECT 1 WHERE token ~ '(th|he|in|er|an|re|on|at|en|ed|te|ti|or|st|ar|nd|to|nt|is|of|it|al|as|ha|ng|co|se|me|de)'
        ) THEN 'high_entropy_sequence'
        WHEN token ~ '^[a-zA-Z0-9]{3,5}[_-]' THEN 'url_fragment'
        WHEN token ~ '^[bcdfghjklmnpqrstvwxyz]{5,}$' THEN 'consonant_cluster'  -- No vowels
        WHEN LENGTH(token) < 3 THEN 'too_short'
        WHEN token ~ '[0-9]{3,}' THEN 'numeric_heavy'
        ELSE 'ok'
    END AS suspicion_reason
FROM coordizer_vocabulary
WHERE LENGTH(token) > 0;

-- Step 2: Summary statistics
SELECT 
    suspicion_reason,
    COUNT(*) AS count,
    COUNT(*) FILTER (WHERE has_basin) AS with_basin,
    COUNT(*) FILTER (WHERE qfi_score IS NULL) AS missing_qfi,
    ROUND(AVG(frequency)::numeric, 2) AS avg_frequency,
    string_agg(DISTINCT token, ', ' ORDER BY token) FILTER (WHERE suspicion_reason != 'ok') AS examples
FROM suspicious_tokens
GROUP BY suspicion_reason
ORDER BY count DESC;

-- Step 3: Find specific garbage patterns (from issue)
SELECT 
    'Specific Garbage Patterns' AS category,
    token,
    frequency,
    qfi_score,
    token_role
FROM coordizer_vocabulary
WHERE token IN ('fgzsnl', 'jcbhgp', 'tlkzfn', 'mxqwvb', 'hdryzp', 'kjwpqm')
   OR token ~ '^[bcdfghjklmnpqrstvwxyz]{6,}$'  -- All consonants, 6+ chars
ORDER BY frequency DESC
LIMIT 50;

-- Step 4: Recommended action - UPDATE to quarantine
-- Uncomment to execute:
/*
UPDATE coordizer_vocabulary
SET 
    token_role = 'quarantine',
    updated_at = NOW()
WHERE token IN (
    SELECT token 
    FROM suspicious_tokens 
    WHERE suspicion_reason IN ('high_entropy_sequence', 'consonant_cluster', 'url_fragment')
      AND frequency < 10  -- Only quarantine low-frequency suspicious tokens
);
*/

-- Step 5: Recommended action - DELETE (use with caution)
-- Uncomment to execute:
/*
DELETE FROM coordizer_vocabulary
WHERE token IN (
    SELECT token 
    FROM suspicious_tokens 
    WHERE suspicion_reason IN ('high_entropy_sequence', 'consonant_cluster')
      AND frequency = 0  -- Only delete never-used tokens
      AND qfi_score IS NULL  -- Only delete if not yet validated
);
*/

ROLLBACK;  -- Change to COMMIT when ready to apply changes
