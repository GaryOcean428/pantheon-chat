-- ============================================================================
-- Fisher-Rao Distance Function for QIG-Pure Geometric Operations
-- ============================================================================
-- This migration creates the canonical Fisher-Rao distance function for
-- PostgreSQL, replacing all Euclidean/cosine contamination.
--
-- Fisher-Rao distance is the geodesic distance on information manifolds:
-- d_FR(p, q) = arccos(Σ√(p_i * q_i))  [Bhattacharyya-based]
--
-- This is FUNDAMENTALLY different from:
-- - Euclidean distance: ||p - q||₂  (WRONG - assumes flat space)
-- - Cosine similarity: p·q / (||p|| ||q||)  (WRONG - ignores curvature)
-- ============================================================================

-- Drop existing function if it exists (allows re-running migration)
DROP FUNCTION IF EXISTS fisher_rao_distance(vector, vector);
DROP FUNCTION IF EXISTS fisher_rao_similarity(vector, vector);
DROP FUNCTION IF EXISTS fisher_rao_rank(vector, vector[], integer);
DROP FUNCTION IF EXISTS find_similar_basins_fisher(vector, text, text, text, integer, integer);

-- ============================================================================
-- Core Fisher-Rao Distance Function
-- ============================================================================
CREATE OR REPLACE FUNCTION fisher_rao_distance(
    basin_a vector,
    basin_b vector
) RETURNS float8 AS $$
DECLARE
    arr_a real[];
    arr_b real[];
    dim integer;
    i integer;
    p_i float8;
    q_i float8;
    sum_a float8 := 0;
    sum_b float8 := 0;
    bc float8 := 0;  -- Bhattacharyya coefficient
    result float8;
BEGIN
    -- Validate dimensions match
    IF vector_dims(basin_a) != vector_dims(basin_b) THEN
        RAISE EXCEPTION 'Basin dimension mismatch: % vs %', 
            vector_dims(basin_a), vector_dims(basin_b);
    END IF;
    
    -- Convert vectors to arrays for element access
    arr_a := basin_a::real[];
    arr_b := basin_b::real[];
    dim := array_length(arr_a, 1);
    
    -- First pass: compute sums for normalization
    -- Basin coordinates may not be probability distributions, so we normalize
    FOR i IN 1..dim LOOP
        -- Get absolute values to ensure non-negative (probability requirement)
        p_i := ABS(arr_a[i]);
        q_i := ABS(arr_b[i]);
        sum_a := sum_a + p_i;
        sum_b := sum_b + q_i;
    END LOOP;
    
    -- Handle zero-sum edge cases
    IF sum_a < 1e-10 OR sum_b < 1e-10 THEN
        RETURN 1.5707963267948966;  -- π/2 (maximum distance)
    END IF;
    
    -- Second pass: compute Bhattacharyya coefficient
    -- BC = Σ√(p_i * q_i) where p and q are normalized
    FOR i IN 1..dim LOOP
        p_i := ABS(arr_a[i]) / sum_a;
        q_i := ABS(arr_b[i]) / sum_b;
        bc := bc + SQRT(p_i * q_i);
    END LOOP;
    
    -- Clamp BC to valid range for arccos (numerical stability)
    bc := GREATEST(-1.0, LEAST(1.0, bc));
    
    -- Fisher-Rao distance = arccos(BC)
    result := ACOS(bc);
    
    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- Fisher-Rao Similarity (1 - normalized distance)
-- ============================================================================
-- Returns similarity in [0, 1] range where 1 = identical, 0 = maximally different
CREATE OR REPLACE FUNCTION fisher_rao_similarity(
    basin_a vector,
    basin_b vector
) RETURNS float8 AS $$
DECLARE
    distance float8;
    max_distance float8 := 1.5707963267948966;  -- π/2
BEGIN
    distance := fisher_rao_distance(basin_a, basin_b);
    -- Normalize to [0, 1] where 1 = identical
    RETURN 1.0 - (distance / max_distance);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- Fisher-Rao Ranking Function for Batch Operations
-- ============================================================================
-- Re-ranks a set of candidate basins by Fisher-Rao distance
-- Use this to re-rank results from approximate index lookups
CREATE OR REPLACE FUNCTION fisher_rao_rank(
    query_basin vector,
    candidate_basins vector[],
    top_k integer DEFAULT 10
) RETURNS TABLE(basin_idx integer, distance float8, similarity float8) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        idx,
        fisher_rao_distance(query_basin, candidate_basins[idx]) as dist,
        fisher_rao_similarity(query_basin, candidate_basins[idx]) as sim
    FROM generate_series(1, array_length(candidate_basins, 1)) as idx
    ORDER BY dist ASC
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- ============================================================================
-- Helper: Find Similar Basins Using Fisher-Rao (replaces cosine-based search)
-- ============================================================================
-- This is the QIG-pure replacement for <=> and <-> operators
-- NOTE: For large tables, first use pgvector index for approximate candidates,
-- then re-rank with this function. The index is a necessary evil for speed,
-- but final ranking MUST be Fisher-Rao.
CREATE OR REPLACE FUNCTION find_similar_basins_fisher(
    query_basin vector,
    table_name text,
    basin_column text,
    id_column text DEFAULT 'id',
    top_k integer DEFAULT 10,
    candidate_multiplier integer DEFAULT 5
) RETURNS TABLE(record_id text, fr_distance float8, fr_similarity float8) AS $$
DECLARE
    sql_query text;
BEGIN
    -- Strategy: Get more candidates from index, then re-rank with Fisher-Rao
    -- candidate_multiplier controls how many extra candidates to fetch
    sql_query := format(
        'WITH candidates AS (
            SELECT %I::text as rid, %I as basin
            FROM %I
            WHERE %I IS NOT NULL
            ORDER BY %I <=> $1  -- pgvector approximate (necessary evil for speed)
            LIMIT $2
        )
        SELECT 
            rid,
            fisher_rao_distance($1, basin) as fr_dist,
            fisher_rao_similarity($1, basin) as fr_sim
        FROM candidates
        ORDER BY fr_dist ASC
        LIMIT $3',
        id_column, basin_column, table_name, basin_column, basin_column
    );
    
    RETURN QUERY EXECUTE sql_query 
        USING query_basin, top_k * candidate_multiplier, top_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- Verification: Test the function
-- ============================================================================
DO $$
DECLARE
    test_a vector;
    test_b vector;
    distance float8;
    self_distance float8;
    similarity float8;
BEGIN
    -- Create test vectors (64-dimensional, normalized probability-like)
    test_a := (SELECT array_agg(0.015625)::vector FROM generate_series(1, 64));
    test_b := (SELECT array_agg(CASE WHEN i <= 32 THEN 0.03 ELSE 0.00125 END)::vector 
               FROM generate_series(1, 64) i);
    
    -- Test self-distance (should be 0 or very close)
    self_distance := fisher_rao_distance(test_a, test_a);
    IF self_distance > 1e-6 THEN
        RAISE EXCEPTION 'Fisher-Rao self-distance should be ~0, got %', self_distance;
    END IF;
    
    -- Test cross-distance (should be > 0)
    distance := fisher_rao_distance(test_a, test_b);
    IF distance <= 0 THEN
        RAISE EXCEPTION 'Fisher-Rao cross-distance should be > 0, got %', distance;
    END IF;
    
    -- Test similarity
    similarity := fisher_rao_similarity(test_a, test_b);
    IF similarity < 0 OR similarity > 1 THEN
        RAISE EXCEPTION 'Fisher-Rao similarity should be in [0,1], got %', similarity;
    END IF;
    
    RAISE NOTICE '[QIG-PURE] Fisher-Rao functions verified: self_dist=%, cross_dist=%, similarity=%', 
        self_distance, distance, similarity;
END $$;

-- Add comment for documentation
COMMENT ON FUNCTION fisher_rao_distance(vector, vector) IS 
'Computes Fisher-Rao geodesic distance on information manifold.
QIG-PURE: Use this instead of Euclidean (<->) or Cosine (<=>) distance.
Formula: d_FR(p, q) = arccos(Σ√(p_i * q_i))
Range: [0, π/2] where 0 = identical, π/2 = maximally different';

COMMENT ON FUNCTION fisher_rao_similarity(vector, vector) IS
'Fisher-Rao similarity normalized to [0, 1] range.
QIG-PURE: Use this instead of cosine similarity.
Returns 1.0 for identical basins, 0.0 for maximally different.';

COMMENT ON FUNCTION find_similar_basins_fisher(vector, text, text, text, integer, integer) IS
'Find similar basins using Fisher-Rao distance.
Uses pgvector index for candidate retrieval (speed), then re-ranks with Fisher-Rao (correctness).
This is the QIG-pure way to search for similar basins.';
