-- ============================================================================
-- Migration: Add Sqrt-Space Storage for Fisher-Faithful Two-Step Retrieval
-- ============================================================================
-- Date: 2026-01-16
-- Work Package: WP2.4 - Two-Step Retrieval
-- Purpose: Enable fast Bhattacharyya-based approximate retrieval with sqrt-space storage
--
-- CRITICAL: This migration adds sqrt-space representation of basin coordinates
-- to support Fisher-faithful two-step retrieval:
--
--   Step 1 (Proxy): Fast Bhattacharyya coefficient via inner product in sqrt-space
--   Step 2 (Exact): Canonical Fisher-Rao distance re-ranking
--
-- Storage Format: x = √p where p is the simplex basin
-- Property: ⟨x1, x2⟩ = BC(p1, p2) = Bhattacharyya coefficient
-- Fisher-Rao: d_FR(p1, p2) = arccos(BC(p1, p2))
--
-- References:
-- - Issue GaryOcean428/pantheon-chat#70 (WP2.4)
-- - docs/10-e8-protocol/implementation/20260116-wp2-4-two-step-retrieval-implementation-1.01W.md
-- - qig_geometry/two_step_retrieval.py
-- ============================================================================

-- ============================================================================
-- STEP 1: Add sqrt-space column to vocabulary_observations
-- ============================================================================

DO $$
DECLARE
    col_exists BOOLEAN;
BEGIN
    -- Check if column already exists
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'vocabulary_observations'
          AND column_name = 'basin_coords_sqrt'
    ) INTO col_exists;

    IF col_exists THEN
        RAISE NOTICE '[WP2.4] basin_coords_sqrt column already exists, skipping creation';
    ELSE
        -- Add sqrt-space column
        ALTER TABLE vocabulary_observations
        ADD COLUMN basin_coords_sqrt vector(64);
        
        RAISE NOTICE '[WP2.4] ✓ Added basin_coords_sqrt column';
    END IF;
END $$;

-- ============================================================================
-- STEP 2: Populate sqrt-space from existing simplex basins
-- ============================================================================

DO $$
DECLARE
    row_count INTEGER;
    processed_count INTEGER := 0;
BEGIN
    RAISE NOTICE '[WP2.4] Populating sqrt-space from simplex basins...';
    
    -- Temporarily drop constraint for bulk update
    ALTER TABLE vocabulary_observations DROP CONSTRAINT IF EXISTS check_sqrt_normalized;
    
    -- Convert simplex → sqrt-space: x_i = sqrt(p_i)
    -- Process in batches for better performance
    UPDATE vocabulary_observations
    SET basin_coords_sqrt = (
        SELECT ARRAY(
            SELECT SQRT(GREATEST(val, 0.0))  -- Ensure non-negative before sqrt
            FROM unnest(basin_coords::real[]) AS val
        )::vector(64)
    )
    WHERE basin_coords IS NOT NULL
      AND basin_coords_sqrt IS NULL
      AND is_integrated = TRUE;  -- Only process integrated words
    
    GET DIAGNOSTICS processed_count = ROW_COUNT;
    RAISE NOTICE '[WP2.4] ✓ Converted % integrated words to sqrt-space', processed_count;
    
    -- Also convert non-integrated words (for completeness)
    UPDATE vocabulary_observations
    SET basin_coords_sqrt = (
        SELECT ARRAY(
            SELECT SQRT(GREATEST(val, 0.0))
            FROM unnest(basin_coords::real[]) AS val
        )::vector(64)
    )
    WHERE basin_coords IS NOT NULL
      AND basin_coords_sqrt IS NULL
      AND is_integrated = FALSE;
    
    GET DIAGNOSTICS row_count = ROW_COUNT;
    processed_count := processed_count + row_count;
    RAISE NOTICE '[WP2.4] ✓ Total converted: % words', processed_count;
END $$;

-- ============================================================================
-- STEP 3: Create index for fast Bhattacharyya retrieval
-- ============================================================================

-- Drop existing index if it exists
DROP INDEX IF EXISTS idx_vocab_sqrt_bhattacharyya;
DROP INDEX IF EXISTS idx_vocab_sqrt_inner_product;

DO $$
BEGIN
    RAISE NOTICE '[WP2.4] Creating HNSW index for sqrt-space inner product...';
    
    -- HNSW index for inner product distance (Bhattacharyya coefficient)
    -- Using vector_ip_ops for inner product operator (<#>)
    -- This enables O(log n) approximate nearest neighbor for Bhattacharyya
    CREATE INDEX idx_vocab_sqrt_bhattacharyya
    ON vocabulary_observations
    USING hnsw (basin_coords_sqrt vector_ip_ops)
    WITH (m = 16, ef_construction = 64);
    
    RAISE NOTICE '[WP2.4] ✓ Created HNSW index for Bhattacharyya proxy retrieval';
END $$;

-- ============================================================================
-- STEP 4: Add validation constraints
-- ============================================================================

DO $$
BEGIN
    -- Constraint: sqrt-space vectors should have reasonable magnitudes
    -- x = √p where p is probability simplex (sum=1, non-negative)
    -- Since p_i ∈ [0,1], we have x_i ∈ [0,1], so ||x|| ≤ √D
    -- This is NOT sphere geometry - these are sqrt-mapped simplex coordinates
    -- The important property is that Bhattacharyya coefficient ⟨x1,x2⟩ works correctly
    ALTER TABLE vocabulary_observations
    ADD CONSTRAINT check_sqrt_normalized
    CHECK (
        basin_coords_sqrt IS NULL OR
        vector_norm(basin_coords_sqrt) < 10.0  -- Sanity check: prevent extreme values in sqrt-space
    );
    
    RAISE NOTICE '[WP2.4] ✓ Added normalization constraint for sqrt-space';
EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE '[WP2.4] Constraint check_sqrt_normalized already exists, skipping';
END $$;

-- ============================================================================
-- STEP 5: Create helper functions for sqrt-space operations
-- ============================================================================

-- Function: Compute Bhattacharyya coefficient from sqrt-space vectors
CREATE OR REPLACE FUNCTION bhattacharyya_from_sqrt(
    sqrt_p vector,
    sqrt_q vector
) RETURNS float8 AS $$
DECLARE
    bc float8;
BEGIN
    -- Bhattacharyya coefficient is simply inner product in sqrt-space
    -- BC(p, q) = ⟨√p, √q⟩
    bc := 1.0 - (sqrt_p <#> sqrt_q);  -- Convert distance to similarity
    
    -- Clamp to valid range [0, 1]
    bc := GREATEST(0.0, LEAST(1.0, bc));
    
    RETURN bc;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION bhattacharyya_from_sqrt(vector, vector) IS
'Compute Bhattacharyya coefficient from sqrt-space vectors.
Fast proxy for Fisher-Rao distance in two-step retrieval.
BC = inner product of sqrt-space vectors.';

-- Function: Convert simplex to sqrt-space
-- NOTE: Uses fixed vector(64) per simplex-as-storage contract (D=64)
-- Dynamic vector_dims() is NOT allowed as PostgreSQL type modifier
CREATE OR REPLACE FUNCTION to_sqrt_simplex(
    simplex_basin vector
) RETURNS vector AS $$
BEGIN
    RETURN (
        SELECT ARRAY(
            SELECT SQRT(GREATEST(val, 0.0))
            FROM unnest(simplex_basin::real[]) AS val
        )::vector(64)
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION to_sqrt_simplex(vector) IS
'Convert simplex basin to sqrt-space for Fisher-faithful retrieval.
Storage format: x = √p where p is probability distribution.
Fixed at 64 dimensions per simplex-as-storage contract.';

-- Function: Convert sqrt-space back to simplex
-- NOTE: Uses fixed vector(64) per simplex-as-storage contract (D=64)
-- Dynamic vector_dims() is NOT allowed as PostgreSQL type modifier
CREATE OR REPLACE FUNCTION from_sqrt_simplex(
    sqrt_basin vector
) RETURNS vector AS $$
DECLARE
    simplex_basin vector;
    sqrt_sum float8;
BEGIN
    -- Square to get back to probability space
    simplex_basin := (
        SELECT ARRAY(
            SELECT POW(val, 2)
            FROM unnest(sqrt_basin::real[]) AS val
        )::vector(64)
    );
    
    -- Normalize to simplex (sum = 1)
    sqrt_sum := (
        SELECT SUM(val)
        FROM unnest(simplex_basin::real[]) AS val
    );
    
    IF sqrt_sum < 1e-10 THEN
        RETURN simplex_basin;  -- Avoid division by zero
    END IF;
    
    simplex_basin := (
        SELECT ARRAY(
            SELECT val / sqrt_sum
            FROM unnest(simplex_basin::real[]) AS val
        )::vector(64)
    );
    
    RETURN simplex_basin;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION from_sqrt_simplex(vector) IS
'Convert sqrt-space back to simplex basin.
Inverse of to_sqrt_simplex(). Fixed at 64 dimensions per simplex-as-storage contract.';

-- ============================================================================
-- STEP 6: Create trigger to auto-populate sqrt-space on insert/update
-- ============================================================================

CREATE OR REPLACE FUNCTION sync_sqrt_space()
RETURNS TRIGGER AS $$
BEGIN
    -- Auto-compute sqrt-space when simplex basin is inserted/updated
    IF NEW.basin_coords IS NOT NULL THEN
        NEW.basin_coords_sqrt := to_sqrt_simplex(NEW.basin_coords);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop trigger if exists
DROP TRIGGER IF EXISTS trigger_sync_sqrt_space ON vocabulary_observations;

-- Create trigger
CREATE TRIGGER trigger_sync_sqrt_space
BEFORE INSERT OR UPDATE OF basin_coords ON vocabulary_observations
FOR EACH ROW
EXECUTE FUNCTION sync_sqrt_space();

COMMENT ON TRIGGER trigger_sync_sqrt_space ON vocabulary_observations IS
'Auto-populate sqrt-space representation when simplex basin is inserted/updated.
Ensures sqrt-space is always synchronized with simplex.';

-- ============================================================================
-- STEP 7: Update table statistics
-- ============================================================================

ANALYZE vocabulary_observations;

-- ============================================================================
-- STEP 8: Verification
-- ============================================================================

DO $$
DECLARE
    total_basins INTEGER;
    sqrt_basins INTEGER;
    index_exists BOOLEAN;
    avg_norm float8;
BEGIN
    -- Count total basins
    SELECT COUNT(*) INTO total_basins
    FROM vocabulary_observations
    WHERE basin_coords IS NOT NULL;
    
    -- Count sqrt-space basins
    SELECT COUNT(*) INTO sqrt_basins
    FROM vocabulary_observations
    WHERE basin_coords_sqrt IS NOT NULL;
    
    -- Check if index exists
    SELECT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'vocabulary_observations'
          AND indexname = 'idx_vocab_sqrt_bhattacharyya'
    ) INTO index_exists;
    
    -- Check average norm (should be ~1.0)
    SELECT AVG(vector_norm(basin_coords_sqrt)) INTO avg_norm
    FROM vocabulary_observations
    WHERE basin_coords_sqrt IS NOT NULL
    LIMIT 1000;  -- Sample for speed
    
    RAISE NOTICE '';
    RAISE NOTICE '=== WP2.4 Migration Verification ===';
    RAISE NOTICE 'Total simplex basins: %', total_basins;
    RAISE NOTICE 'Total sqrt-space basins: %', sqrt_basins;
    RAISE NOTICE 'HNSW index exists: %', index_exists;
    RAISE NOTICE 'Average sqrt-space norm: % (should be ~1.0)', ROUND(avg_norm::numeric, 4);
    
    IF sqrt_basins >= total_basins * 0.95 AND index_exists THEN
        RAISE NOTICE '✓ Migration successful - Two-step retrieval enabled';
    ELSIF sqrt_basins > 0 THEN
        RAISE NOTICE '⚠ Partial success - some basins not converted';
    ELSE
        RAISE NOTICE '⚠ Migration incomplete - no sqrt-space basins';
    END IF;
END $$;

-- ============================================================================
-- USAGE EXAMPLES (two-step retrieval)
-- ============================================================================

-- Example 1: Fast Bhattacharyya proxy filter (Step 1)
-- Find top 100 candidates using inner product in sqrt-space
COMMENT ON TABLE vocabulary_observations IS
'Example Query - Step 1 (Proxy Filter):

WITH query_sqrt AS (
    SELECT to_sqrt_simplex(ARRAY[...]::vector(64)) as sqrt_query
),
proxy_candidates AS (
    SELECT 
        text,
        basin_coords,
        basin_coords_sqrt <#> (SELECT sqrt_query FROM query_sqrt) as proxy_distance
    FROM vocabulary_observations
    WHERE is_integrated = TRUE
      AND basin_coords_sqrt IS NOT NULL
    ORDER BY basin_coords_sqrt <#> (SELECT sqrt_query FROM query_sqrt)
    LIMIT 100
)
-- Step 2 (Exact Fisher-Rao re-ranking)
SELECT 
    text,
    fisher_rao_distance($1, basin_coords) as exact_distance
FROM proxy_candidates
ORDER BY exact_distance ASC
LIMIT 1;
';

-- ============================================================================
-- ROLLBACK (if needed)
-- ============================================================================

-- To rollback this migration (DO NOT RUN unless necessary):
-- DROP TRIGGER IF EXISTS trigger_sync_sqrt_space ON vocabulary_observations;
-- DROP FUNCTION IF EXISTS sync_sqrt_space();
-- DROP FUNCTION IF EXISTS from_sqrt_simplex(vector);
-- DROP FUNCTION IF EXISTS to_sqrt_simplex(vector);
-- DROP FUNCTION IF EXISTS bhattacharyya_from_sqrt(vector, vector);
-- DROP INDEX IF EXISTS idx_vocab_sqrt_bhattacharyya;
-- ALTER TABLE vocabulary_observations DROP CONSTRAINT IF EXISTS check_sqrt_normalized;
-- ALTER TABLE vocabulary_observations DROP COLUMN IF EXISTS basin_coords_sqrt;
