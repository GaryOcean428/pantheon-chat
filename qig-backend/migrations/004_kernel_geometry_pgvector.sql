-- Migration: Convert kernel_geometry basin_coordinates to pgvector
-- Date: 2026-01-07
-- Purpose: Enable HNSW O(log n) approximate nearest neighbor for kernel proximity queries
--
-- Problem: basin_coordinates was created as real[] (array) instead of vector(64)
-- This prevents pgvector operators (<->) and HNSW indexing
--
-- Solution: Convert column type and create HNSW index

-- ============================================================================
-- STEP 1: Enable pgvector Extension
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- STEP 2: Check current column type and convert if needed
-- ============================================================================

DO $$
DECLARE
    current_type TEXT;
    row_count INTEGER;
BEGIN
    -- Get current column type
    SELECT udt_name INTO current_type
    FROM information_schema.columns
    WHERE table_name = 'kernel_geometry' AND column_name = 'basin_coordinates';

    IF current_type IS NULL THEN
        RAISE NOTICE 'basin_coordinates column does not exist, nothing to migrate';
        RETURN;
    END IF;

    RAISE NOTICE 'Current basin_coordinates type: %', current_type;

    -- If already vector type, skip conversion
    IF current_type = 'vector' THEN
        RAISE NOTICE 'basin_coordinates is already vector type, skipping conversion';
        RETURN;
    END IF;

    -- If it's an array type (_float4 = real[], _float8 = double precision[])
    IF current_type IN ('_float4', '_float8', '_numeric') THEN
        RAISE NOTICE 'Converting basin_coordinates from array to vector(64)...';

        -- Add temporary vector column
        ALTER TABLE kernel_geometry ADD COLUMN IF NOT EXISTS basin_coordinates_vec vector(64);

        -- Convert array to vector (pad/truncate to 64 dimensions)
        UPDATE kernel_geometry
        SET basin_coordinates_vec = (
            SELECT (
                CASE
                    WHEN array_length(basin_coordinates, 1) IS NULL THEN NULL
                    WHEN array_length(basin_coordinates, 1) >= 64 THEN
                        (basin_coordinates[1:64])::vector(64)
                    ELSE
                        (basin_coordinates || array_fill(0::real, ARRAY[64 - array_length(basin_coordinates, 1)]))::vector(64)
                END
            )
        )
        WHERE basin_coordinates IS NOT NULL
          AND basin_coordinates_vec IS NULL;

        GET DIAGNOSTICS row_count = ROW_COUNT;
        RAISE NOTICE 'Converted % rows to vector format', row_count;

        -- Drop old column and rename new one
        ALTER TABLE kernel_geometry DROP COLUMN basin_coordinates;
        ALTER TABLE kernel_geometry RENAME COLUMN basin_coordinates_vec TO basin_coordinates;

        RAISE NOTICE '✓ Successfully converted basin_coordinates to vector(64)';
    ELSE
        RAISE NOTICE 'Unknown column type: %, attempting direct cast', current_type;

        -- Try to alter column type directly (works for some types)
        BEGIN
            ALTER TABLE kernel_geometry
            ALTER COLUMN basin_coordinates TYPE vector(64)
            USING basin_coordinates::vector(64);
            RAISE NOTICE '✓ Direct conversion successful';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Direct conversion failed: %, manual migration needed', SQLERRM;
        END;
    END IF;
END $$;

-- ============================================================================
-- STEP 3: Create HNSW Index for Approximate Nearest Neighbor
-- ============================================================================

-- Drop existing indexes that might conflict
DROP INDEX IF EXISTS idx_kernel_geometry_basin_ann;
DROP INDEX IF EXISTS idx_kernel_geometry_basin_hnsw;

-- Create HNSW index (only if column is vector type)
DO $$
DECLARE
    current_type TEXT;
BEGIN
    SELECT udt_name INTO current_type
    FROM information_schema.columns
    WHERE table_name = 'kernel_geometry' AND column_name = 'basin_coordinates';

    IF current_type = 'vector' THEN
        -- HNSW parameters:
        -- m = 16: connections per layer (higher = better recall, more memory)
        -- ef_construction = 64: build-time candidate list size (higher = better quality)
        CREATE INDEX idx_kernel_geometry_basin_ann
        ON kernel_geometry
        USING hnsw (basin_coordinates vector_l2_ops)
        WITH (m = 16, ef_construction = 64);

        RAISE NOTICE '✓ Created HNSW index idx_kernel_geometry_basin_ann';
    ELSE
        RAISE NOTICE 'Skipping HNSW index - basin_coordinates is not vector type (%)' , current_type;
    END IF;
END $$;

-- ============================================================================
-- STEP 4: Create additional indexes for kernel queries
-- ============================================================================

-- Index for status filtering (E8 cap enforcement)
CREATE INDEX IF NOT EXISTS idx_kernel_geometry_status ON kernel_geometry(status);

-- Index for phi-based queries (evolution selection)
CREATE INDEX IF NOT EXISTS idx_kernel_geometry_phi ON kernel_geometry(phi);

-- Composite index for live kernel queries
CREATE INDEX IF NOT EXISTS idx_kernel_geometry_live
ON kernel_geometry(status, phi DESC, spawned_at DESC)
WHERE status IN ('active', 'observing', 'shadow');

-- ============================================================================
-- STEP 5: Update table statistics
-- ============================================================================

ANALYZE kernel_geometry;

-- ============================================================================
-- STEP 6: Verification
-- ============================================================================

DO $$
DECLARE
    col_type TEXT;
    index_exists BOOLEAN;
    kernel_count INTEGER;
    vector_count INTEGER;
BEGIN
    -- Check column type
    SELECT udt_name INTO col_type
    FROM information_schema.columns
    WHERE table_name = 'kernel_geometry' AND column_name = 'basin_coordinates';

    -- Check if HNSW index exists
    SELECT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE tablename = 'kernel_geometry'
        AND indexname = 'idx_kernel_geometry_basin_ann'
    ) INTO index_exists;

    -- Count kernels and valid vectors
    SELECT COUNT(*) INTO kernel_count FROM kernel_geometry;
    SELECT COUNT(*) INTO vector_count
    FROM kernel_geometry
    WHERE basin_coordinates IS NOT NULL;

    RAISE NOTICE '';
    RAISE NOTICE '=== Migration Verification ===';
    RAISE NOTICE 'Column type: %', col_type;
    RAISE NOTICE 'HNSW index exists: %', index_exists;
    RAISE NOTICE 'Total kernels: %', kernel_count;
    RAISE NOTICE 'Kernels with vectors: %', vector_count;

    IF col_type = 'vector' AND index_exists THEN
        RAISE NOTICE '✓ Migration successful - O(log n) ANN queries enabled';
    ELSIF col_type = 'vector' THEN
        RAISE NOTICE '⚠ Partial success - vector type OK but HNSW index missing';
    ELSE
        RAISE NOTICE '⚠ Migration incomplete - still using array type (O(n) fallback)';
    END IF;
END $$;

-- ============================================================================
-- USAGE EXAMPLES (after migration)
-- ============================================================================

-- Find 10 nearest kernels to a target basin:
-- SELECT kernel_id, god_name, phi,
--        basin_coordinates <-> '[0.1,0.2,...]'::vector(64) as distance
-- FROM kernel_geometry
-- WHERE status IN ('active', 'observing')
-- ORDER BY basin_coordinates <-> '[0.1,0.2,...]'::vector(64)
-- LIMIT 10;

-- Find strongest kernel within radius:
-- SELECT kernel_id, god_name, phi
-- FROM kernel_geometry
-- WHERE basin_coordinates <-> '[0.1,0.2,...]'::vector(64) < 2.0
-- ORDER BY phi DESC
-- LIMIT 1;
