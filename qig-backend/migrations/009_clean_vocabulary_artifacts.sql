-- ============================================================================
-- VOCABULARY CLEANUP MIGRATION
-- Date: 2026-01-09
-- Purpose: Remove curriculum artifacts and stop words from coordizer_vocabulary
--
-- Fixes:
-- - Removes training document artifacts (docusaurus, mihalcea, howsearchworks, etc.)
-- - Marks stop words (pronouns, common words) for exclusion
-- - Cleans up camelCase and other non-vocabulary tokens
-- ============================================================================

-- 1. Remove known training artifacts that shouldn't be in vocabulary
DELETE FROM coordizer_vocabulary
WHERE token IN (
    'docusaurus', 'mihalcea', 'howsearchworks', 'arxiv', 'github',
    'stackoverflow', 'wikipedia', 'readme', 'changelog', 'dockerfile'
)
   OR token LIKE '%arxiv%'
   OR token LIKE '%http%'
   OR token LIKE '%www%'
   OR token LIKE '%github%'
   OR token ~ '^[A-Z][a-z]+[A-Z]'  -- camelCase artifacts (e.g., "docuSaurus")
   OR source_type = 'training'
   OR source_type = 'curriculum';

-- 2. Mark stop words for exclusion from generation (don't delete, just mark)
-- These are high-frequency but low-semantic-value tokens
UPDATE coordizer_vocabulary
SET source_type = 'stop_word'
WHERE token IN (
    'the', 'and', 'for', 'that', 'this', 'with', 'was', 'are', 'but', 'not',
    'you', 'all', 'can', 'had', 'her', 'his', 'him', 'one', 'our', 'out',
    'they', 'what', 'when', 'who', 'will', 'from', 'have', 'been', 'has',
    'more', 'she', 'there', 'than', 'into', 'other', 'which', 'its', 'about',
    'just', 'over', 'such', 'through', 'most', 'your', 'because', 'would',
    'also', 'some', 'these', 'then', 'how', 'any', 'each', 'only', 'could',
    'very', 'them', 'being', 'may', 'should', 'between', 'where', 'before',
    'own', 'both', 'those', 'same', 'during', 'after', 'much', 'does', 'did'
)
AND source_type NOT IN ('bip39', 'special');

-- 3. Lower phi scores for pronouns that remain (reduce their generation probability)
UPDATE coordizer_vocabulary
SET phi_score = LEAST(phi_score, 0.3)
WHERE token IN ('her', 'his', 'our', 'my', 'she', 'they', 'them', 'him', 'i', 'we')
AND source_type = 'stop_word';

-- 4. Report cleanup results
DO $$
DECLARE
    stop_word_count INTEGER;
    deleted_count INTEGER;
    total_remaining INTEGER;
BEGIN
    SELECT COUNT(*) INTO stop_word_count
    FROM coordizer_vocabulary WHERE source_type = 'stop_word';

    SELECT COUNT(*) INTO total_remaining
    FROM coordizer_vocabulary WHERE source_type NOT IN ('stop_word', 'special');

    RAISE NOTICE 'Vocabulary cleanup complete:';
    RAISE NOTICE '  - Stop words marked: %', stop_word_count;
    RAISE NOTICE '  - Active vocabulary tokens: %', total_remaining;
END $$;

-- Usage: Run with psql or pg_query
-- psql $DATABASE_URL -f migrations/009_clean_vocabulary_artifacts.sql
