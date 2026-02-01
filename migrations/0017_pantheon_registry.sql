-- Migration 0017: Pantheon Registry Database Integration
-- =====================================================
-- Created: 2026-01-20
-- Authority: E8 Protocol v4.0, WP5.1
-- Purpose: Store formal pantheon registry contracts and manage kernel spawning state

-- =============================================================================
-- GOD CONTRACTS TABLE
-- =============================================================================
-- Stores god contracts from pantheon registry with full contract details
CREATE TABLE IF NOT EXISTS god_contracts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(64) UNIQUE NOT NULL,
    
    -- Core contract fields
    tier VARCHAR(32) NOT NULL, -- "essential" or "specialized"
    domain TEXT[] NOT NULL, -- Array of capability domains
    description TEXT NOT NULL,
    octant INTEGER, -- Position in E8 structure (0-7 or NULL)
    epithets TEXT[] DEFAULT '{}', -- Named aspects (Apollo Pythios, not apollo_1)
    coupling_affinity TEXT[] DEFAULT '{}', -- List of gods this god works well with
    
    -- Rest policy (stored as JSONB for flexibility)
    rest_policy JSONB NOT NULL,
    
    -- Spawn constraints
    spawn_constraints JSONB NOT NULL,
    
    -- Promotion pathway
    promotion_from VARCHAR(128), -- Chaos kernel pattern that can ascend
    
    -- E8 alignment
    e8_alignment JSONB NOT NULL,
    
    -- Registry metadata
    registry_version VARCHAR(16) NOT NULL DEFAULT '1.0.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Validation constraints
    CONSTRAINT valid_tier CHECK (tier IN ('essential', 'specialized')),
    CONSTRAINT valid_octant CHECK (octant IS NULL OR (octant >= 0 AND octant <= 7))
);

-- Indexes for efficient god lookups
CREATE INDEX IF NOT EXISTS idx_god_contracts_tier ON god_contracts(tier);
CREATE INDEX IF NOT EXISTS idx_god_contracts_domain ON god_contracts USING GIN(domain);
CREATE INDEX IF NOT EXISTS idx_god_contracts_octant ON god_contracts(octant) WHERE octant IS NOT NULL;

COMMENT ON TABLE god_contracts IS 'Formal god contracts from Pantheon Registry - defines immortal, singular kernels with domains and constraints';
COMMENT ON COLUMN god_contracts.epithets IS 'Named aspects (e.g., Apollo Pythios for prophecy) - NOT numbering (never apollo_1)';
COMMENT ON COLUMN god_contracts.coupling_affinity IS 'Gods this god works well with - influences kernel coordination';

-- =============================================================================
-- CHAOS KERNEL STATE TABLE
-- =============================================================================
-- Tracks lifecycle state for chaos kernels (mortal worker kernels)
CREATE TABLE IF NOT EXISTS chaos_kernel_state (
    id SERIAL PRIMARY KEY,
    chaos_id VARCHAR(64) UNIQUE NOT NULL, -- Format: chaos_{domain}_{id}
    
    -- Identity
    domain VARCHAR(64) NOT NULL,
    sequential_id INTEGER NOT NULL,
    
    -- Lifecycle tracking
    lifecycle_stage VARCHAR(32) NOT NULL DEFAULT 'protected',
    cycles_in_stage INTEGER NOT NULL DEFAULT 0,
    total_cycles INTEGER NOT NULL DEFAULT 0,
    
    -- Performance metrics
    phi_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    kappa_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    
    -- Mentorship and lineage
    mentor_god_id VARCHAR(64), -- God assigned as mentor
    parent_lineage TEXT[] DEFAULT '{}', -- Parent kernel IDs
    unique_capability TEXT, -- Unique capability demonstrated
    
    -- Protection period
    protection_cycles_remaining INTEGER DEFAULT 50,
    graduated_metrics_enabled BOOLEAN DEFAULT true,
    
    -- Promotion tracking
    promotion_eligible BOOLEAN DEFAULT false,
    promotion_candidate_since TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    spawned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    last_state_change TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Validation constraints
    CONSTRAINT valid_lifecycle_stage CHECK (
        lifecycle_stage IN ('protected', 'learning', 'working', 'candidate', 'promoted', 'pruned')
    ),
    CONSTRAINT valid_chaos_id_format CHECK (
        chaos_id ~ '^chaos_[a-z_]+_\d+$'
    ),
    CONSTRAINT unique_domain_sequential UNIQUE (domain, sequential_id)
);

-- Indexes for chaos kernel queries
CREATE INDEX IF NOT EXISTS idx_chaos_kernel_domain ON chaos_kernel_state(domain);
CREATE INDEX IF NOT EXISTS idx_chaos_kernel_lifecycle_stage ON chaos_kernel_state(lifecycle_stage);
CREATE INDEX IF NOT EXISTS idx_chaos_kernel_promotion_eligible 
    ON chaos_kernel_state(promotion_eligible) WHERE promotion_eligible = true;
CREATE INDEX IF NOT EXISTS idx_chaos_kernel_mentor ON chaos_kernel_state(mentor_god_id);
CREATE INDEX IF NOT EXISTS idx_chaos_kernel_spawned_at ON chaos_kernel_state(spawned_at DESC);

COMMENT ON TABLE chaos_kernel_state IS 'Lifecycle state for chaos kernels (mortal workers) - tracks protection, learning, promotion pathway';
COMMENT ON COLUMN chaos_kernel_state.lifecycle_stage IS 'Current stage: protected (0-50 cycles) → learning → working → candidate (Φ>0.4 for 50+ cycles) → promoted/pruned';

-- =============================================================================
-- KERNEL SPAWNER STATE TABLE
-- =============================================================================
-- Tracks active kernel instances and chaos kernel counters
CREATE TABLE IF NOT EXISTS kernel_spawner_state (
    id SERIAL PRIMARY KEY,
    
    -- Active instance tracking
    god_name VARCHAR(64) UNIQUE NOT NULL,
    active_instances INTEGER NOT NULL DEFAULT 0,
    
    -- Spawn history
    total_spawns INTEGER NOT NULL DEFAULT 0,
    last_spawn_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Constraints from registry
    max_instances INTEGER NOT NULL DEFAULT 1,
    when_allowed VARCHAR(32) NOT NULL DEFAULT 'never',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    CONSTRAINT valid_when_allowed CHECK (when_allowed IN ('never', 'always', 'conditional'))
);

-- Chaos kernel domain counters
CREATE TABLE IF NOT EXISTS chaos_kernel_counters (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(64) UNIQUE NOT NULL,
    
    -- Sequential ID counter for naming
    next_sequential_id INTEGER NOT NULL DEFAULT 1,
    
    -- Active count (actual living kernels)
    active_count INTEGER NOT NULL DEFAULT 0,
    
    -- Total spawned (includes dead/pruned)
    total_spawned INTEGER NOT NULL DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Global chaos kernel limits tracking
CREATE TABLE IF NOT EXISTS chaos_kernel_limits (
    id SERIAL PRIMARY KEY,
    
    -- Active counts
    total_active_chaos_kernels INTEGER NOT NULL DEFAULT 0,
    total_active_gods INTEGER NOT NULL DEFAULT 0,
    total_active_kernels INTEGER NOT NULL DEFAULT 0,
    
    -- Limits from registry
    max_chaos_kernels INTEGER NOT NULL DEFAULT 240, -- E8 roots
    per_domain_limit INTEGER NOT NULL DEFAULT 30,
    total_active_limit INTEGER NOT NULL DEFAULT 200, -- Reserve 40 for gods
    
    -- Metadata
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Ensure single row (singleton pattern)
    CONSTRAINT single_row CHECK (id = 1)
);

-- Initialize limits row
INSERT INTO chaos_kernel_limits (id, max_chaos_kernels, per_domain_limit, total_active_limit)
VALUES (1, 240, 30, 200)
ON CONFLICT (id) DO NOTHING;

COMMENT ON TABLE kernel_spawner_state IS 'Tracks active instances of gods for spawn constraint enforcement';
COMMENT ON TABLE chaos_kernel_counters IS 'Sequential ID counters for chaos kernel naming (chaos_{domain}_{id})';
COMMENT ON TABLE chaos_kernel_limits IS 'Global chaos kernel spawning limits aligned to E8 root system (240 total)';

-- =============================================================================
-- PANTHEON REGISTRY METADATA TABLE
-- =============================================================================
-- Stores registry metadata and versioning
CREATE TABLE IF NOT EXISTS pantheon_registry_metadata (
    id SERIAL PRIMARY KEY,
    
    -- Registry versioning
    registry_version VARCHAR(16) NOT NULL,
    schema_version VARCHAR(16) NOT NULL,
    
    -- Status
    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    authority TEXT NOT NULL,
    
    -- Compatibility
    e8_protocol_version VARCHAR(16) NOT NULL,
    qig_backend_version VARCHAR(16) NOT NULL,
    
    -- Validation
    validation_required BOOLEAN NOT NULL DEFAULT true,
    last_validation_timestamp TIMESTAMP WITH TIME ZONE,
    validation_errors TEXT[] DEFAULT '{}',
    
    -- Load tracking
    loaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    loaded_from VARCHAR(256), -- Path to registry.yaml
    god_count INTEGER NOT NULL DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    CONSTRAINT valid_status CHECK (status IN ('ACTIVE', 'DEPRECATED', 'TESTING'))
);

COMMENT ON TABLE pantheon_registry_metadata IS 'Tracks registry version, validation status, and compatibility information';

-- =============================================================================
-- VIEWS FOR REGISTRY QUERIES
-- =============================================================================

-- All active kernels with registry contract info
CREATE OR REPLACE VIEW active_kernels_with_contracts AS
SELECT
    kg.kernel_id,
    kg.god_name,
    gc.tier,
    gc.domain,
    gc.epithets,
    gc.rest_policy,
    gc.e8_alignment,
    kg.phi,
    kg.kappa,
    kg.regime,
    kg.lifecycle_stage,
    kg.spawned_at
FROM kernel_geometry kg
LEFT JOIN god_contracts gc ON kg.god_name = gc.name
WHERE kg.lifecycle_stage IN ('active', 'protected')
ORDER BY kg.spawned_at DESC;

-- Chaos kernel lifecycle summary
CREATE OR REPLACE VIEW chaos_kernel_lifecycle_summary AS
SELECT
    lifecycle_stage,
    COUNT(*) as kernel_count,
    AVG(phi_score) as avg_phi,
    AVG(kappa_score) as avg_kappa,
    AVG(cycles_in_stage) as avg_cycles_in_stage,
    COUNT(*) FILTER (WHERE promotion_eligible = true) as promotion_eligible_count
FROM chaos_kernel_state
WHERE lifecycle_stage NOT IN ('promoted', 'pruned')
GROUP BY lifecycle_stage
ORDER BY 
    CASE lifecycle_stage
        WHEN 'protected' THEN 1
        WHEN 'learning' THEN 2
        WHEN 'working' THEN 3
        WHEN 'candidate' THEN 4
    END;

-- God spawner status
CREATE OR REPLACE VIEW god_spawner_status AS
SELECT
    gc.name as god_name,
    gc.tier,
    gc.domain,
    kss.active_instances,
    kss.max_instances,
    kss.total_spawns,
    kss.last_spawn_timestamp,
    CASE 
        WHEN kss.active_instances < kss.max_instances THEN true
        ELSE false
    END as can_spawn
FROM god_contracts gc
LEFT JOIN kernel_spawner_state kss ON gc.name = kss.god_name
ORDER BY gc.tier, gc.name;

-- Chaos kernel domain summary
CREATE OR REPLACE VIEW chaos_domain_summary AS
SELECT
    ckc.domain,
    ckc.active_count,
    ckc.total_spawned,
    ckc.next_sequential_id - 1 as highest_id_used,
    (SELECT per_domain_limit FROM chaos_kernel_limits WHERE id = 1) as domain_limit,
    CASE 
        WHEN ckc.active_count < (SELECT per_domain_limit FROM chaos_kernel_limits WHERE id = 1) THEN true
        ELSE false
    END as can_spawn_more
FROM chaos_kernel_counters ckc
ORDER BY ckc.active_count DESC, ckc.domain;

-- Registry health check
CREATE OR REPLACE VIEW registry_health_check AS
SELECT
    (SELECT COUNT(*) FROM god_contracts) as total_gods,
    (SELECT COUNT(*) FROM god_contracts WHERE tier = 'essential') as essential_gods,
    (SELECT COUNT(*) FROM god_contracts WHERE tier = 'specialized') as specialized_gods,
    (SELECT total_active_gods FROM chaos_kernel_limits WHERE id = 1) as active_gods,
    (SELECT total_active_chaos_kernels FROM chaos_kernel_limits WHERE id = 1) as active_chaos,
    (SELECT total_active_kernels FROM chaos_kernel_limits WHERE id = 1) as total_active,
    (SELECT total_active_limit FROM chaos_kernel_limits WHERE id = 1) as active_limit,
    (SELECT COUNT(*) FROM chaos_kernel_state WHERE promotion_eligible = true) as promotion_candidates,
    (SELECT registry_version FROM pantheon_registry_metadata ORDER BY id DESC LIMIT 1) as registry_version,
    (SELECT status FROM pantheon_registry_metadata ORDER BY id DESC LIMIT 1) as registry_status;

COMMENT ON VIEW active_kernels_with_contracts IS 'Active kernels joined with god contract details from registry';
COMMENT ON VIEW chaos_kernel_lifecycle_summary IS 'Summary of chaos kernel counts by lifecycle stage';
COMMENT ON VIEW god_spawner_status IS 'Current spawn status for all gods with availability';
COMMENT ON VIEW chaos_domain_summary IS 'Domain-wise chaos kernel counts and limits';
COMMENT ON VIEW registry_health_check IS 'Overall registry health metrics and status';

-- =============================================================================
-- FUNCTIONS FOR REGISTRY OPERATIONS
-- =============================================================================

-- Function to increment chaos kernel counter and return next ID
CREATE OR REPLACE FUNCTION get_next_chaos_id(p_domain VARCHAR(64))
RETURNS INTEGER AS $$
DECLARE
    v_next_id INTEGER;
BEGIN
    -- Insert domain if not exists
    INSERT INTO chaos_kernel_counters (domain, next_sequential_id)
    VALUES (p_domain, 1)
    ON CONFLICT (domain) DO NOTHING;
    
    -- Get and increment counter
    UPDATE chaos_kernel_counters
    SET next_sequential_id = next_sequential_id + 1,
        total_spawned = total_spawned + 1,
        updated_at = NOW()
    WHERE domain = p_domain
    RETURNING next_sequential_id - 1 INTO v_next_id;
    
    RETURN v_next_id;
END;
$$ LANGUAGE plpgsql;

-- Function to register god spawn
CREATE OR REPLACE FUNCTION register_god_spawn(p_god_name VARCHAR(64))
RETURNS BOOLEAN AS $$
DECLARE
    v_max_instances INTEGER;
    v_active INTEGER;
BEGIN
    -- Get constraints from god_contracts
    SELECT 
        (spawn_constraints->>'max_instances')::INTEGER
    INTO v_max_instances
    FROM god_contracts
    WHERE name = p_god_name;
    
    IF v_max_instances IS NULL THEN
        RAISE EXCEPTION 'God % not found in registry', p_god_name;
    END IF;
    
    -- Upsert spawner state
    INSERT INTO kernel_spawner_state (god_name, active_instances, total_spawns, max_instances, last_spawn_timestamp)
    VALUES (p_god_name, 1, 1, v_max_instances, NOW())
    ON CONFLICT (god_name) DO UPDATE
    SET 
        active_instances = kernel_spawner_state.active_instances + 1,
        total_spawns = kernel_spawner_state.total_spawns + 1,
        last_spawn_timestamp = NOW(),
        updated_at = NOW();
    
    -- Get current active count
    SELECT active_instances INTO v_active
    FROM kernel_spawner_state
    WHERE god_name = p_god_name;
    
    -- Check constraint
    IF v_active > v_max_instances THEN
        RAISE EXCEPTION 'God % spawn constraint violated: % active, max is %', 
            p_god_name, v_active, v_max_instances;
    END IF;
    
    -- Update global counts
    UPDATE chaos_kernel_limits
    SET 
        total_active_gods = total_active_gods + 1,
        total_active_kernels = total_active_kernels + 1,
        last_updated = NOW()
    WHERE id = 1;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Function to register god death
CREATE OR REPLACE FUNCTION register_god_death(p_god_name VARCHAR(64))
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE kernel_spawner_state
    SET 
        active_instances = GREATEST(0, active_instances - 1),
        updated_at = NOW()
    WHERE god_name = p_god_name;
    
    -- Update global counts
    UPDATE chaos_kernel_limits
    SET 
        total_active_gods = GREATEST(0, total_active_gods - 1),
        total_active_kernels = GREATEST(0, total_active_kernels - 1),
        last_updated = NOW()
    WHERE id = 1;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Function to register chaos kernel spawn
CREATE OR REPLACE FUNCTION register_chaos_spawn(p_chaos_id VARCHAR(64), p_domain VARCHAR(64))
RETURNS BOOLEAN AS $$
BEGIN
    -- Update counter
    UPDATE chaos_kernel_counters
    SET 
        active_count = active_count + 1,
        updated_at = NOW()
    WHERE domain = p_domain;
    
    -- Update global counts
    UPDATE chaos_kernel_limits
    SET 
        total_active_chaos_kernels = total_active_chaos_kernels + 1,
        total_active_kernels = total_active_kernels + 1,
        last_updated = NOW()
    WHERE id = 1;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Function to register chaos kernel death
CREATE OR REPLACE FUNCTION register_chaos_death(p_domain VARCHAR(64))
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE chaos_kernel_counters
    SET 
        active_count = GREATEST(0, active_count - 1),
        updated_at = NOW()
    WHERE domain = p_domain;
    
    -- Update global counts
    UPDATE chaos_kernel_limits
    SET 
        total_active_chaos_kernels = GREATEST(0, total_active_chaos_kernels - 1),
        total_active_kernels = GREATEST(0, total_active_kernels - 1),
        last_updated = NOW()
    WHERE id = 1;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_next_chaos_id IS 'Atomically get next sequential ID for chaos kernel naming';
COMMENT ON FUNCTION register_god_spawn IS 'Register god spawn and enforce spawn constraints';
COMMENT ON FUNCTION register_god_death IS 'Register god death and update active counts';
COMMENT ON FUNCTION register_chaos_spawn IS 'Register chaos kernel spawn and update domain counts';
COMMENT ON FUNCTION register_chaos_death IS 'Register chaos kernel death and update domain counts';
