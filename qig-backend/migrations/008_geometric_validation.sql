-- ============================================================================
-- GEOMETRIC VOCABULARY VALIDATION MIGRATION
-- Date: 2026-01-12
-- Purpose: Add QIG-pure geometric metrics to vocabulary tables
-- ============================================================================

-- Add geometric validation columns to learned_words
ALTER TABLE learned_words
ADD COLUMN IF NOT EXISTS qfi_score FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS basin_distance FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS curvature_std FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS entropy_score FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS is_geometrically_valid BOOLEAN DEFAULT NULL,
ADD COLUMN IF NOT EXISTS validation_reason TEXT DEFAULT NULL;

-- Add indices for querying
CREATE INDEX IF NOT EXISTS idx_learned_words_geom_valid 
ON learned_words(is_geometrically_valid) 
WHERE is_geometrically_valid IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_learned_words_qfi 
ON learned_words(qfi_score) 
WHERE qfi_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_learned_words_basin_dist 
ON learned_words(basin_distance) 
WHERE basin_distance IS NOT NULL;

-- Update vocabulary_stats to include validation metrics
ALTER TABLE vocabulary_stats
ADD COLUMN IF NOT EXISTS validated_words INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS invalid_words INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS truncated_words INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS garbled_words INTEGER DEFAULT 0;

-- Function to update validation stats
CREATE OR REPLACE FUNCTION update_validation_stats()
RETURNS void AS $$
BEGIN
    UPDATE vocabulary_stats SET
        validated_words = (SELECT COUNT(*) FROM learned_words WHERE is_geometrically_valid = TRUE),
        invalid_words = (SELECT COUNT(*) FROM learned_words WHERE is_geometrically_valid = FALSE),
        truncated_words = (SELECT COUNT(*) FROM learned_words WHERE validation_reason LIKE '%TRUNCATED%'),
        garbled_words = (SELECT COUNT(*) FROM learned_words WHERE validation_reason LIKE '%GARBLED%'),
        last_updated = NOW();
END;
$$ LANGUAGE plpgsql;

-- Comment on columns
COMMENT ON COLUMN learned_words.qfi_score IS 'Quantum Fisher Information: semantic structure measure (>1.0 = valid)';
COMMENT ON COLUMN learned_words.basin_distance IS 'Fisher-Rao distance to nearest stable basin (<0.5 = valid)';
COMMENT ON COLUMN learned_words.curvature_std IS 'Ricci curvature variance along word trajectory (<0.5 = valid)';
COMMENT ON COLUMN learned_words.entropy_score IS 'Entropy at token boundaries (>1.5 = natural structure)';
COMMENT ON COLUMN learned_words.is_geometrically_valid IS 'TRUE if passes all 4 geometric validation metrics';
COMMENT ON COLUMN learned_words.validation_reason IS 'Rejection reason if invalid (e.g. GARBLED, TRUNCATED, TECHNICAL)';
