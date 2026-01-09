-- ============================================================================
-- VOCABULARY CLEANUP MIGRATION
-- Removes code artifacts, fixes null values, validates real words
-- Run this once after fixing the vocabulary pipeline
-- ============================================================================

-- 1. Remove obvious code artifacts
DELETE FROM vocabulary_observations
WHERE text IN (
    'type', 'word', 'frequency', 'avgPhi', 'maxPhi', 'avgphi', 'maxphi',
    'observation_type', 'basin_coords', 'contexts', 'source_type',
    'def', 'return', 'import', 'class', 'self', 'none', 'true', 'false',
    'if', 'else', 'elif', 'try', 'except', 'finally', 'raise', 'assert',
    'lambda', 'yield', 'async', 'await', 'pass', 'break', 'continue',
    'print', 'len', 'str', 'int', 'float', 'dict', 'list', 'tuple',
    'isinstance', 'hasattr', 'getattr', 'setattr'
);

-- 2. Remove camelCase tokens (code identifiers)
DELETE FROM vocabulary_observations
WHERE text ~ '[a-z][A-Z]';

-- 3. Remove snake_case tokens (Python identifiers)
DELETE FROM vocabulary_observations
WHERE text ~ '_';

-- 4. Remove tokens starting with numbers
DELETE FROM vocabulary_observations
WHERE text ~ '^\d';

-- 5. Remove tokens containing dots (file paths, URLs)
DELETE FROM vocabulary_observations
WHERE text ~ '\.';

-- 6. Remove single characters
DELETE FROM vocabulary_observations
WHERE length(text) < 2;

-- 7. Remove very long tokens (likely URLs or paths)
DELETE FROM vocabulary_observations
WHERE length(text) > 30;

-- 8. Fix max_phi = 0 (should never be less than avg_phi)
UPDATE vocabulary_observations
SET max_phi = avg_phi
WHERE max_phi < avg_phi OR max_phi = 0;

-- 9. Mark entries with is_real_word = false as NULL (needs re-validation)
UPDATE vocabulary_observations
SET is_real_word = NULL
WHERE is_real_word = FALSE;

-- 10. Set reasonable defaults for NULL phi values
UPDATE vocabulary_observations
SET avg_phi = 0.5, max_phi = 0.5
WHERE avg_phi IS NULL OR avg_phi = 0;

-- 11. Set phrase_category based on type
UPDATE vocabulary_observations
SET phrase_category =
    CASE
        WHEN type = 'phrase' THEN 'phrase'
        WHEN type = 'sequence' THEN 'sequence'
        WHEN type = 'pattern' THEN 'pattern'
        ELSE 'word'
    END
WHERE phrase_category = 'unknown' OR phrase_category IS NULL;

-- 12. Report cleanup results
DO $$
DECLARE
    total_count INT;
    null_is_real INT;
    has_basin INT;
    avg_freq NUMERIC;
BEGIN
    SELECT COUNT(*) INTO total_count FROM vocabulary_observations;
    SELECT COUNT(*) INTO null_is_real FROM vocabulary_observations WHERE is_real_word IS NULL;
    SELECT COUNT(*) INTO has_basin FROM vocabulary_observations WHERE basin_coords IS NOT NULL;
    SELECT AVG(frequency) INTO avg_freq FROM vocabulary_observations;

    RAISE NOTICE 'Vocabulary cleanup complete:';
    RAISE NOTICE '  Total entries: %', total_count;
    RAISE NOTICE '  Needs validation: %', null_is_real;
    RAISE NOTICE '  Has basin coords: %', has_basin;
    RAISE NOTICE '  Avg frequency: %', ROUND(avg_freq, 2);
END $$;
