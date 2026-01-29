-- Migration 0021: Deprecate learned_words table
-- Authority: E8 Protocol v4.0 §04 - Vocabulary Cleanup
-- Date: 2026-01-23
-- Status: ACTIVE
-- 
-- Purpose: Deprecate the learned_words table in favor of coordizer_vocabulary
-- The generation vocabulary is now consolidated into coordizer_vocabulary with
-- the 'relationships' JSONB column for learned relationships.

-- Step 1: Add deprecation notice to learned_words table
COMMENT ON TABLE learned_words IS 
'DEPRECATED: Use coordizer_vocabulary instead. This table is kept for backward compatibility only.
Migration path: All generation vocabulary should use coordizer_vocabulary.relationships JSONB column.
See E8 Protocol v4.0 §04 for details.';

-- Step 2: Create a view that redirects learned_words queries to coordizer_vocabulary
CREATE OR REPLACE VIEW learned_words_compat AS
SELECT 
    cv.id,
    cv.token AS word,
    cv.basin AS basin_embedding,
    cv.phi AS phi_score,
    COALESCE((cv.relationships->>'frequency')::integer, 0) AS frequency,
    COALESCE(cv.relationships->>'phrase_category', 'general') AS phrase_category,
    cv.created_at,
    cv.updated_at AS last_used_at,
    cv.relationships AS metadata
FROM coordizer_vocabulary cv
WHERE cv.is_active = true;

COMMENT ON VIEW learned_words_compat IS 
'Compatibility view for legacy code that queries learned_words. 
Maps to coordizer_vocabulary with relationships JSONB.';

-- Step 3: Create a function to log deprecation warnings
CREATE OR REPLACE FUNCTION log_learned_words_deprecation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE WARNING 'learned_words table is deprecated. Use coordizer_vocabulary instead. See E8 Protocol v4.0 §04.';
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Step 4: Add deprecation trigger to learned_words table
DROP TRIGGER IF EXISTS learned_words_deprecation_warning ON learned_words;
CREATE TRIGGER learned_words_deprecation_warning
    BEFORE INSERT OR UPDATE ON learned_words
    FOR EACH STATEMENT
    EXECUTE FUNCTION log_learned_words_deprecation();

-- Step 5: Migrate any remaining data from learned_words to coordizer_vocabulary
-- Only migrate words that don't already exist in coordizer_vocabulary
INSERT INTO coordizer_vocabulary (token, basin, phi, relationships, is_active, created_at, updated_at)
SELECT 
    lw.word,
    lw.basin_embedding,
    lw.phi_score,
    jsonb_build_object(
        'frequency', lw.frequency,
        'phrase_category', lw.phrase_category,
        'migrated_from', 'learned_words',
        'migration_date', NOW()
    ),
    true,
    lw.created_at,
    NOW()
FROM learned_words lw
WHERE NOT EXISTS (
    SELECT 1 FROM coordizer_vocabulary cv WHERE cv.token = lw.word
)
ON CONFLICT (token) DO UPDATE SET
    relationships = coordizer_vocabulary.relationships || EXCLUDED.relationships,
    updated_at = NOW();

-- Step 6: Record migration in schema_migrations
INSERT INTO schema_migrations (version, description, applied_at)
VALUES ('0021', 'Deprecate learned_words table in favor of coordizer_vocabulary', NOW())
ON CONFLICT (version) DO NOTHING;

-- Note: We do NOT drop the learned_words table to maintain backward compatibility
-- It can be dropped in a future migration after all code has been updated
