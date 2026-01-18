-- Migration 0014: Kernel Lifecycle Operations
-- =============================================
-- Created: 2026-01-18
-- Authority: E8 Protocol v4.0, WP5.3
-- Purpose: Add lifecycle event tracking and shadow pantheon storage

-- Kernel Lifecycle Events Table
-- Tracks all lifecycle operations: spawn, split, merge, prune, resurrect, promote
CREATE TABLE IF NOT EXISTS kernel_lifecycle_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(64) UNIQUE NOT NULL,
    event_type VARCHAR(32) NOT NULL, -- spawn, split, merge, prune, resurrect, promote
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Affected kernels
    primary_kernel_id VARCHAR(64) NOT NULL,
    secondary_kernel_ids TEXT[], -- Additional kernels involved (for split/merge)
    
    -- Event details
    reason TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Outcome
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    
    -- Indexing
    CONSTRAINT valid_event_type CHECK (
        event_type IN ('spawn', 'split', 'merge', 'prune', 'resurrect', 'promote')
    )
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_primary_kernel 
    ON kernel_lifecycle_events(primary_kernel_id);
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_type 
    ON kernel_lifecycle_events(event_type);
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_timestamp 
    ON kernel_lifecycle_events(timestamp DESC);

-- Shadow Pantheon (Hades) Storage
-- Stores archived/pruned kernels with lessons learned
CREATE TABLE IF NOT EXISTS shadow_pantheon (
    id SERIAL PRIMARY KEY,
    shadow_id VARCHAR(64) UNIQUE NOT NULL,
    original_kernel_id VARCHAR(64) NOT NULL,
    name VARCHAR(128) NOT NULL,
    kernel_type VARCHAR(32) NOT NULL, -- "god" or "chaos"
    
    -- Final state before pruning
    final_phi DOUBLE PRECISION NOT NULL,
    final_kappa DOUBLE PRECISION NOT NULL,
    final_basin VECTOR(64), -- Basin coordinates at pruning time
    
    -- Performance history
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    total_cycles INTEGER DEFAULT 0,
    
    -- Lessons learned
    failure_patterns TEXT[],
    success_patterns TEXT[],
    learned_lessons TEXT DEFAULT '',
    
    -- Pruning metadata
    prune_reason TEXT NOT NULL,
    prune_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    pruned_by VARCHAR(64) DEFAULT 'system',
    
    -- Resurrection tracking
    resurrection_count INTEGER DEFAULT 0,
    last_resurrection TIMESTAMP WITH TIME ZONE,
    
    -- Indexing
    CONSTRAINT valid_kernel_type CHECK (kernel_type IN ('god', 'chaos'))
);

-- Indexes for shadow pantheon queries
CREATE INDEX IF NOT EXISTS idx_shadow_pantheon_original_kernel 
    ON shadow_pantheon(original_kernel_id);
CREATE INDEX IF NOT EXISTS idx_shadow_pantheon_kernel_type 
    ON shadow_pantheon(kernel_type);
CREATE INDEX IF NOT EXISTS idx_shadow_pantheon_prune_timestamp 
    ON shadow_pantheon(prune_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_shadow_pantheon_resurrection_count 
    ON shadow_pantheon(resurrection_count);

-- Add lifecycle stage tracking to kernel_geometry table
-- (Augments existing kernel_geometry table with lifecycle awareness)
ALTER TABLE kernel_geometry 
    ADD COLUMN IF NOT EXISTS lifecycle_stage VARCHAR(32) DEFAULT 'active';

ALTER TABLE kernel_geometry 
    ADD COLUMN IF NOT EXISTS protection_cycles_remaining INTEGER DEFAULT 0;

ALTER TABLE kernel_geometry
    ADD COLUMN IF NOT EXISTS parent_kernel_ids TEXT[];

ALTER TABLE kernel_geometry
    ADD COLUMN IF NOT EXISTS child_kernel_ids TEXT[];

ALTER TABLE kernel_geometry
    ADD COLUMN IF NOT EXISTS spawn_reason TEXT;

ALTER TABLE kernel_geometry
    ADD COLUMN IF NOT EXISTS mentor_kernel_id VARCHAR(64);

-- Add constraint for valid lifecycle stages
ALTER TABLE kernel_geometry
    ADD CONSTRAINT IF NOT EXISTS valid_lifecycle_stage CHECK (
        lifecycle_stage IN ('active', 'protected', 'split', 'merged', 'pruned', 'promoted')
    );

-- Create index for lifecycle stage queries
CREATE INDEX IF NOT EXISTS idx_kernel_geometry_lifecycle_stage 
    ON kernel_geometry(lifecycle_stage);

-- Lifecycle Policy Configuration Table
-- Stores policy rules for when lifecycle operations trigger
CREATE TABLE IF NOT EXISTS lifecycle_policies (
    id SERIAL PRIMARY KEY,
    policy_name VARCHAR(64) UNIQUE NOT NULL,
    policy_type VARCHAR(32) NOT NULL, -- split, merge, prune, promote
    
    -- Trigger conditions (stored as JSONB for flexibility)
    trigger_conditions JSONB NOT NULL,
    
    -- Action parameters
    action_params JSONB DEFAULT '{}'::jsonb,
    
    -- Policy metadata
    enabled BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0,
    description TEXT,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    created_by VARCHAR(64) DEFAULT 'system',
    
    -- Indexing
    CONSTRAINT valid_policy_type CHECK (
        policy_type IN ('split', 'merge', 'prune', 'promote', 'resurrect')
    )
);

-- Index for enabled policies
CREATE INDEX IF NOT EXISTS idx_lifecycle_policies_enabled 
    ON lifecycle_policies(enabled, priority DESC);

-- Insert default lifecycle policies
INSERT INTO lifecycle_policies (policy_name, policy_type, trigger_conditions, action_params, description)
VALUES
    (
        'prune_low_phi_persistent',
        'prune',
        '{"phi_threshold": 0.1, "cycles_below_threshold": 100}'::jsonb,
        '{}'::jsonb,
        'Prune kernels with Φ < 0.1 for 100+ consecutive cycles'
    ),
    (
        'split_overloaded',
        'split',
        '{"phi_threshold": 0.7, "domain_count": 3, "load_threshold": 0.8}'::jsonb,
        '{"split_criterion": "domain"}'::jsonb,
        'Split kernels with high Φ, multiple domains, and high load'
    ),
    (
        'merge_redundant',
        'merge',
        '{"phi_similarity": 0.9, "domain_overlap": 0.8, "fisher_distance": 0.2}'::jsonb,
        '{}'::jsonb,
        'Merge kernels with highly similar basins and overlapping domains'
    ),
    (
        'promote_stable_chaos',
        'promote',
        '{"phi_threshold": 0.4, "min_cycles": 50, "success_rate": 0.7}'::jsonb,
        '{}'::jsonb,
        'Promote chaos kernels with stable high Φ and good performance'
    )
ON CONFLICT (policy_name) DO NOTHING;

-- Lifecycle Metrics View
-- Provides aggregated metrics for lifecycle monitoring
CREATE OR REPLACE VIEW lifecycle_metrics AS
SELECT
    event_type,
    COUNT(*) as event_count,
    COUNT(DISTINCT primary_kernel_id) as unique_kernels_affected,
    MAX(timestamp) as last_event_timestamp,
    AVG(CASE WHEN success THEN 1 ELSE 0 END) as success_rate
FROM kernel_lifecycle_events
GROUP BY event_type;

-- Shadow Pantheon Summary View
-- Provides overview of pruned kernels
CREATE OR REPLACE VIEW shadow_pantheon_summary AS
SELECT
    kernel_type,
    COUNT(*) as total_pruned,
    AVG(final_phi) as avg_final_phi,
    AVG(total_cycles) as avg_cycles_before_prune,
    SUM(resurrection_count) as total_resurrections,
    MAX(prune_timestamp) as last_prune_timestamp
FROM shadow_pantheon
GROUP BY kernel_type;

-- Active Kernels Lifecycle View
-- Shows lifecycle status of active kernels
CREATE OR REPLACE VIEW active_kernels_lifecycle AS
SELECT
    kernel_id,
    god_name,
    kernel_name,
    lifecycle_stage,
    protection_cycles_remaining,
    phi,
    kappa,
    regime,
    success_count,
    failure_count,
    spawned_at,
    COALESCE(array_length(parent_kernel_ids, 1), 0) as parent_count,
    COALESCE(array_length(child_kernel_ids, 1), 0) as child_count
FROM kernel_geometry
WHERE lifecycle_stage IN ('active', 'protected')
ORDER BY spawned_at DESC;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON kernel_lifecycle_events TO pantheon_user;
-- GRANT SELECT, INSERT, UPDATE ON shadow_pantheon TO pantheon_user;
-- GRANT SELECT ON lifecycle_metrics TO pantheon_user;
-- GRANT SELECT ON shadow_pantheon_summary TO pantheon_user;
-- GRANT SELECT ON active_kernels_lifecycle TO pantheon_user;

COMMENT ON TABLE kernel_lifecycle_events IS 'Tracks all kernel lifecycle operations (spawn, split, merge, prune, resurrect, promote) with full provenance';
COMMENT ON TABLE shadow_pantheon IS 'Archives pruned kernels (Hades domain) with learned lessons for potential resurrection';
COMMENT ON TABLE lifecycle_policies IS 'Stores policy rules defining when lifecycle operations automatically trigger';
COMMENT ON VIEW lifecycle_metrics IS 'Aggregated metrics for lifecycle event monitoring';
COMMENT ON VIEW shadow_pantheon_summary IS 'Summary statistics of shadow pantheon by kernel type';
COMMENT ON VIEW active_kernels_lifecycle IS 'Current lifecycle status of active kernels with provenance counts';
