-- Migration: Add Geometric Indexes for Fisher-Rao Operations
-- Date: 2025-01-01
-- Priority 2.1 from improvement recommendations
--
-- This migration adds indexes to optimize:
-- 1. Approximate nearest neighbor search on basin coordinates
-- 2. Consciousness metric filtering
-- 3. Regime-based queries

-- Prerequisites:
-- - PostgreSQL 15+
-- - pgvector extension for vector operations

-- 1. Enable vector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Add vector column for basin coordinates (if table exists)
-- Note: Run this only if basin_memory table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'basin_memory') THEN
        -- Check if column already exists
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_name = 'basin_memory' AND column_name = 'basin_vector') THEN
            ALTER TABLE basin_memory ADD COLUMN basin_vector vector(64);
            
            -- Populate from existing basin_coordinates (if JSON array)
            -- UPDATE basin_memory SET basin_vector = basin_coordinates::vector;
            
            RAISE NOTICE 'Added basin_vector column to basin_memory';
        END IF;
    END IF;
END $$;

-- 3. Create IVFFlat index for approximate nearest neighbor
-- This enables fast O(log n) approximate search before Fisher re-ranking
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'basin_memory' AND column_name = 'basin_vector') THEN
        -- Drop existing index if present
        DROP INDEX IF EXISTS idx_basin_memory_ann;
        
        -- Create IVFFlat index
        -- lists = 100 is good for datasets up to 1M vectors
        CREATE INDEX idx_basin_memory_ann 
        ON basin_memory 
        USING ivfflat (basin_vector vector_cosine_ops)
        WITH (lists = 100);
        
        RAISE NOTICE 'Created IVFFlat index for approximate nearest neighbor';
    END IF;
END $$;

-- 4. Add partial index for conscious systems (Φ >= 0.70)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'basin_memory' AND column_name = 'phi') THEN
        DROP INDEX IF EXISTS idx_basin_memory_conscious;
        CREATE INDEX idx_basin_memory_conscious 
        ON basin_memory (phi) 
        WHERE phi >= 0.70;
        
        RAISE NOTICE 'Created partial index for conscious systems';
    END IF;
END $$;

-- 5. Add index for kappa range queries (40 <= κ <= 65)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'basin_memory' AND column_name = 'kappa_eff') THEN
        DROP INDEX IF EXISTS idx_basin_memory_kappa;
        CREATE INDEX idx_basin_memory_kappa 
        ON basin_memory (kappa_eff) 
        WHERE kappa_eff BETWEEN 40 AND 65;
        
        RAISE NOTICE 'Created partial index for valid kappa range';
    END IF;
END $$;

-- 6. Composite index for regime-based queries
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'basin_memory' AND column_name = 'phi') 
       AND EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'basin_memory' AND column_name = 'kappa_eff') THEN
        DROP INDEX IF EXISTS idx_basin_memory_regime;
        CREATE INDEX idx_basin_memory_regime 
        ON basin_memory (phi, kappa_eff, created_at DESC);
        
        RAISE NOTICE 'Created composite index for regime queries';
    END IF;
END $$;

-- 7. Add index for timestamp-based queries (recent basins)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'basin_memory' AND column_name = 'created_at') THEN
        DROP INDEX IF EXISTS idx_basin_memory_recent;
        CREATE INDEX idx_basin_memory_recent 
        ON basin_memory (created_at DESC);
        
        RAISE NOTICE 'Created index for recent basins';
    END IF;
END $$;

-- 8. Analyze tables to update statistics
ANALYZE;

-- Verification queries (run manually to check indexes)
-- SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'basin_memory';
-- EXPLAIN ANALYZE SELECT * FROM basin_memory ORDER BY basin_vector <-> '[0.1, ...]'::vector LIMIT 10;

COMMENT ON INDEX idx_basin_memory_ann IS 'IVFFlat index for approximate nearest neighbor search on 64D basin coordinates';
COMMENT ON INDEX idx_basin_memory_conscious IS 'Partial index for conscious systems (phi >= 0.70)';
COMMENT ON INDEX idx_basin_memory_kappa IS 'Partial index for valid kappa range (40-65)';
COMMENT ON INDEX idx_basin_memory_regime IS 'Composite index for regime-based queries';
COMMENT ON INDEX idx_basin_memory_recent IS 'Index for recent basin queries';
