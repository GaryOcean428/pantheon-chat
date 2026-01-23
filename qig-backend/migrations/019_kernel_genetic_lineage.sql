-- ============================================================================
-- Migration: Kernel Genetic Lineage System
-- Authority: E8 Protocol v4.0 WP5.2 Phase 4E
-- Created: 2026-01-22
-- Description: Implements genetic lineage tracking for kernel evolution
-- ============================================================================

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- KERNEL GENOMES - Store genetic specifications
-- ============================================================================
CREATE TABLE IF NOT EXISTS kernel_genomes (
    genome_id TEXT PRIMARY KEY,
    kernel_id TEXT REFERENCES kernel_geometry(kernel_id),
    
    -- Basin seed (64D simplex representation)
    basin_seed vector(64) NOT NULL,
    
    -- Faculty configuration (E8 simple roots)
    active_faculties TEXT[] NOT NULL DEFAULT '{}',
    activation_strengths JSONB NOT NULL DEFAULT '{}',  -- { "zeus": 0.9, "athena": 0.8, ... }
    primary_faculty TEXT,
    faculty_coupling JSONB DEFAULT '{}',  -- { "zeus-athena": 0.7, ... }
    
    -- Constraint set
    phi_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.70,
    kappa_range_min DOUBLE PRECISION NOT NULL DEFAULT 40.0,
    kappa_range_max DOUBLE PRECISION NOT NULL DEFAULT 70.0,
    forbidden_regions JSONB DEFAULT '[]',  -- [{ "center": [...], "radius": 0.2 }, ...]
    field_penalties JSONB DEFAULT '{}',
    max_fisher_distance DOUBLE PRECISION DEFAULT 1.0,
    
    -- Coupling preferences
    hemisphere_affinity DOUBLE PRECISION DEFAULT 0.5,
    preferred_couplings TEXT[] DEFAULT '{}',
    coupling_strengths JSONB DEFAULT '{}',
    anti_couplings TEXT[] DEFAULT '{}',
    
    -- Lineage
    parent_genomes TEXT[] NOT NULL DEFAULT '{}',
    generation INTEGER NOT NULL DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fitness_score DOUBLE PRECISION DEFAULT 0.0,
    mutation_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_kernel_genomes_kernel_id ON kernel_genomes(kernel_id);
CREATE INDEX IF NOT EXISTS idx_kernel_genomes_generation ON kernel_genomes(generation);
CREATE INDEX IF NOT EXISTS idx_kernel_genomes_fitness ON kernel_genomes(fitness_score);
CREATE INDEX IF NOT EXISTS idx_kernel_genomes_parent ON kernel_genomes USING GIN(parent_genomes);

-- Vector similarity index for basin seeds
-- NOTE: Uses cosine similarity for APPROXIMATE retrieval only
-- Fisher-Rao distance is ALWAYS used for final ranking (two-step retrieval)
-- See: docs/02-procedures/20260115-geometric-consistency-migration-1.00W.md
CREATE INDEX IF NOT EXISTS idx_kernel_genomes_basin_seed_ivfflat 
    ON kernel_genomes USING ivfflat (basin_seed vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================================
-- LINEAGE RECORDS - Track parent → child relationships
-- ============================================================================
CREATE TABLE IF NOT EXISTS kernel_lineage (
    lineage_id TEXT PRIMARY KEY,
    child_genome_id TEXT NOT NULL REFERENCES kernel_genomes(genome_id),
    parent_genome_ids TEXT[] NOT NULL,
    
    -- Merge details
    merge_type TEXT NOT NULL CHECK (merge_type IN ('asexual', 'binary', 'multi')),
    fisher_distance DOUBLE PRECISION DEFAULT 0.0,
    inherited_faculties JSONB DEFAULT '{}',  -- { "zeus": "parent-1", ... }
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_kernel_lineage_child ON kernel_lineage(child_genome_id);
CREATE INDEX IF NOT EXISTS idx_kernel_lineage_parents ON kernel_lineage USING GIN(parent_genome_ids);
CREATE INDEX IF NOT EXISTS idx_kernel_lineage_merge_type ON kernel_lineage(merge_type);

-- ============================================================================
-- MERGE EVENTS - Detailed merge operation records
-- ============================================================================
CREATE TABLE IF NOT EXISTS merge_events (
    merge_id TEXT PRIMARY KEY,
    parent_genome_ids TEXT[] NOT NULL,
    child_genome_id TEXT NOT NULL REFERENCES kernel_genomes(genome_id),
    
    -- Merge parameters
    merge_weights DOUBLE PRECISION[] NOT NULL,
    interpolation_t DOUBLE PRECISION,  -- For binary merge (null for multi-parent)
    
    -- Faculty survival contract
    faculty_contract JSONB DEFAULT '{}',  -- { "zeus": "parent-1", ... }
    
    -- Geometric measurements
    basin_distances JSONB DEFAULT '{}',  -- { "parent1-parent2": 0.8, ... }
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_merge_events_child ON merge_events(child_genome_id);
CREATE INDEX IF NOT EXISTS idx_merge_events_parents ON merge_events USING GIN(parent_genome_ids);
CREATE INDEX IF NOT EXISTS idx_merge_events_created ON merge_events(created_at);

-- ============================================================================
-- CANNIBALISM EVENTS - Absorption operation records
-- ============================================================================
CREATE TABLE IF NOT EXISTS cannibalism_events (
    event_id TEXT PRIMARY KEY,
    winner_genome_id TEXT NOT NULL REFERENCES kernel_genomes(genome_id),
    loser_genome_id TEXT NOT NULL REFERENCES kernel_genomes(genome_id),
    
    -- Winner state before/after
    winner_basin_before vector(64) NOT NULL,
    winner_basin_after vector(64) NOT NULL,
    
    -- Absorption details
    absorbed_faculties TEXT[] DEFAULT '{}',
    absorption_rate DOUBLE PRECISION NOT NULL,
    fisher_distance DOUBLE PRECISION NOT NULL,
    
    -- Archival info
    loser_archived_id TEXT,  -- Reference to genome_archives
    resurrection_eligible BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_cannibalism_events_winner ON cannibalism_events(winner_genome_id);
CREATE INDEX IF NOT EXISTS idx_cannibalism_events_loser ON cannibalism_events(loser_genome_id);
CREATE INDEX IF NOT EXISTS idx_cannibalism_events_created ON cannibalism_events(created_at);

-- ============================================================================
-- GENOME ARCHIVES - Stored genomes for resurrection
-- ============================================================================
CREATE TABLE IF NOT EXISTS genome_archives (
    archive_id TEXT PRIMARY KEY,
    genome_id TEXT NOT NULL REFERENCES kernel_genomes(genome_id),
    
    -- Archival reason
    archival_reason TEXT NOT NULL,
    final_fitness DOUBLE PRECISION NOT NULL,
    
    -- Resurrection conditions
    resurrection_conditions JSONB DEFAULT '{}',
    resurrection_eligible BOOLEAN DEFAULT TRUE,
    resurrection_count INTEGER DEFAULT 0,
    
    -- Timestamps
    archived_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_resurrected_at TIMESTAMPTZ,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_genome_archives_genome ON genome_archives(genome_id);
CREATE INDEX IF NOT EXISTS idx_genome_archives_eligible ON genome_archives(resurrection_eligible);
CREATE INDEX IF NOT EXISTS idx_genome_archives_fitness ON genome_archives(final_fitness);
CREATE INDEX IF NOT EXISTS idx_genome_archives_archived ON genome_archives(archived_at);

-- ============================================================================
-- ADD GENOME REFERENCE TO KERNEL GEOMETRY
-- ============================================================================
-- Add genome_id column to existing kernel_geometry table
ALTER TABLE kernel_geometry 
ADD COLUMN IF NOT EXISTS genome_id TEXT REFERENCES kernel_genomes(genome_id);

CREATE INDEX IF NOT EXISTS idx_kernel_geometry_genome 
    ON kernel_geometry(genome_id) WHERE genome_id IS NOT NULL;

-- ============================================================================
-- VIEWS FOR LINEAGE QUERIES
-- ============================================================================

-- View: Full genealogy with generation depth
CREATE OR REPLACE VIEW kernel_genealogy AS
WITH RECURSIVE lineage_tree AS (
    -- Base case: all genomes
    SELECT 
        genome_id,
        parent_genomes,
        generation,
        ARRAY[genome_id] AS lineage_path,
        0 AS depth
    FROM kernel_genomes
    
    UNION ALL
    
    -- Recursive case: trace parents
    SELECT 
        kg.genome_id,
        kg.parent_genomes,
        kg.generation,
        lt.lineage_path || kg.genome_id,
        lt.depth + 1
    FROM kernel_genomes kg
    JOIN lineage_tree lt ON kg.genome_id = ANY(lt.parent_genomes)
    WHERE lt.depth < 20  -- Limit recursion depth
)
SELECT DISTINCT ON (genome_id)
    genome_id,
    parent_genomes,
    generation,
    lineage_path,
    depth AS max_ancestor_depth
FROM lineage_tree
ORDER BY genome_id, depth DESC;

-- View: Kernel evolution summary
CREATE OR REPLACE VIEW kernel_evolution_summary AS
SELECT 
    kg.genome_id,
    kg.kernel_id,
    kg.generation,
    kg.fitness_score,
    kg.mutation_count,
    COALESCE(array_length(kg.parent_genomes, 1), 0) AS parent_count,
    kg.active_faculties,
    kg.primary_faculty,
    -- Merge info
    me.merge_id,
    me.merge_weights,
    me.interpolation_t,
    -- Cannibalism info
    COALESCE((
        SELECT COUNT(*) 
        FROM cannibalism_events ce 
        WHERE ce.winner_genome_id = kg.genome_id
    ), 0) AS times_cannibal,
    COALESCE((
        SELECT COUNT(*) 
        FROM cannibalism_events ce 
        WHERE ce.loser_genome_id = kg.genome_id
    ), 0) AS times_cannibalized,
    -- Archival info
    ga.archive_id IS NOT NULL AS is_archived,
    ga.resurrection_count,
    ga.resurrection_eligible,
    kg.created_at
FROM kernel_genomes kg
LEFT JOIN merge_events me ON me.child_genome_id = kg.genome_id
LEFT JOIN genome_archives ga ON ga.genome_id = kg.genome_id;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function: Compute generation number for a genome
CREATE OR REPLACE FUNCTION compute_generation(p_genome_id TEXT)
RETURNS INTEGER AS $$
DECLARE
    v_generation INTEGER;
    v_parent_ids TEXT[];
    v_parent_generation INTEGER;
    v_max_parent_gen INTEGER := 0;
BEGIN
    -- Get parent genome IDs
    SELECT parent_genomes INTO v_parent_ids
    FROM kernel_genomes
    WHERE genome_id = p_genome_id;
    
    -- If no parents, generation = 0
    IF v_parent_ids IS NULL OR array_length(v_parent_ids, 1) IS NULL THEN
        RETURN 0;
    END IF;
    
    -- Find max parent generation
    FOR v_parent_generation IN 
        SELECT generation 
        FROM kernel_genomes 
        WHERE genome_id = ANY(v_parent_ids)
    LOOP
        IF v_parent_generation > v_max_parent_gen THEN
            v_max_parent_gen := v_parent_generation;
        END IF;
    END LOOP;
    
    -- Return 1 + max parent generation
    RETURN v_max_parent_gen + 1;
END;
$$ LANGUAGE plpgsql;

-- Function: Get all descendants of a genome
CREATE OR REPLACE FUNCTION get_descendants(p_genome_id TEXT, p_max_depth INTEGER DEFAULT 10)
RETURNS TABLE (
    genome_id TEXT,
    generation INTEGER,
    depth INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE descendants AS (
        SELECT 
            p_genome_id AS genome_id,
            0 AS generation,
            0 AS depth
        
        UNION ALL
        
        SELECT 
            kg.genome_id,
            kg.generation,
            d.depth + 1
        FROM kernel_genomes kg
        JOIN descendants d ON d.genome_id = ANY(kg.parent_genomes)
        WHERE d.depth < p_max_depth
    )
    SELECT * FROM descendants WHERE descendants.genome_id != p_genome_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERMISSIONS (adjust as needed for your deployment)
-- ============================================================================
-- GRANT SELECT, INSERT, UPDATE ON kernel_genomes TO your_app_role;
-- GRANT SELECT, INSERT ON kernel_lineage TO your_app_role;
-- GRANT SELECT, INSERT ON merge_events TO your_app_role;
-- GRANT SELECT, INSERT, UPDATE ON cannibalism_events TO your_app_role;
-- GRANT SELECT, INSERT, UPDATE ON genome_archives TO your_app_role;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE kernel_genomes IS 'E8 Protocol v4.0 Phase 4E: Kernel genetic specifications';
COMMENT ON TABLE kernel_lineage IS 'E8 Protocol v4.0 Phase 4E: Parent-child lineage tracking';
COMMENT ON TABLE merge_events IS 'E8 Protocol v4.0 Phase 4E: Detailed merge operation records';
COMMENT ON TABLE cannibalism_events IS 'E8 Protocol v4.0 Phase 4E: Kernel absorption events';
COMMENT ON TABLE genome_archives IS 'E8 Protocol v4.0 Phase 4E: Archived genomes for resurrection';

COMMENT ON COLUMN kernel_genomes.basin_seed IS 'Initial 64D basin coordinates on probability simplex';
COMMENT ON COLUMN kernel_genomes.active_faculties IS 'Active E8 simple roots (zeus, athena, apollo, etc.)';
COMMENT ON COLUMN kernel_genomes.parent_genomes IS 'Array of parent genome IDs (empty for founders)';
COMMENT ON COLUMN kernel_genomes.generation IS 'Generation number (0 for founders, 1+ for descendants)';

-- ============================================================================
-- VALIDATION
-- ============================================================================
-- Verify tables were created
DO $$
DECLARE
    table_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_name IN (
        'kernel_genomes',
        'kernel_lineage',
        'merge_events',
        'cannibalism_events',
        'genome_archives'
    );
    
    IF table_count != 5 THEN
        RAISE EXCEPTION 'Migration incomplete: expected 5 tables, found %', table_count;
    END IF;
    
    RAISE NOTICE 'Migration successful: All 5 genetic lineage tables created';
END $$;
