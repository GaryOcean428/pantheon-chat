-- Kernel Rest Events Schema
-- =========================
-- Tracks per-kernel rest cycles for coupling-aware scheduling (WP5.4)

-- Create kernel_rest_events table
CREATE TABLE IF NOT EXISTS kernel_rest_events (
    id SERIAL PRIMARY KEY,
    
    -- Kernel identification
    kernel_id VARCHAR(255) NOT NULL,
    kernel_name VARCHAR(100) NOT NULL,
    tier VARCHAR(50) NOT NULL,  -- essential, specialized
    rest_policy VARCHAR(50) NOT NULL,  -- never, minimal_rotating, coordinated_alternating, etc.
    
    -- Rest timing
    rest_start TIMESTAMP NOT NULL,
    rest_end TIMESTAMP,
    duration_seconds FLOAT,
    
    -- Rest type and coverage
    rest_type VARCHAR(50) NOT NULL,  -- resting, reduced, micro_pause, covered
    covered_by_kernel VARCHAR(255),  -- Partner covering during rest
    
    -- Fatigue metrics at rest trigger
    phi_at_rest FLOAT,
    kappa_at_rest FLOAT,
    fatigue_score FLOAT,
    load_at_rest FLOAT,
    error_rate_at_rest FLOAT,
    
    -- Recovery metrics
    phi_after_rest FLOAT,
    kappa_after_rest FLOAT,
    fatigue_score_after FLOAT,
    
    -- Reason and decision
    rest_reason TEXT,
    forced BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_kernel_rest_events_kernel_id ON kernel_rest_events(kernel_id);
CREATE INDEX IF NOT EXISTS idx_kernel_rest_events_kernel_name ON kernel_rest_events(kernel_name);
CREATE INDEX IF NOT EXISTS idx_kernel_rest_events_rest_start ON kernel_rest_events(rest_start DESC);
CREATE INDEX IF NOT EXISTS idx_kernel_rest_events_tier ON kernel_rest_events(tier);
CREATE INDEX IF NOT EXISTS idx_kernel_rest_events_rest_type ON kernel_rest_events(rest_type);

-- Create kernel_coverage_events table (tracks who covered for whom)
CREATE TABLE IF NOT EXISTS kernel_coverage_events (
    id SERIAL PRIMARY KEY,
    
    -- Coverage relationship
    covering_kernel_id VARCHAR(255) NOT NULL,
    covering_kernel_name VARCHAR(100) NOT NULL,
    covered_kernel_id VARCHAR(255) NOT NULL,
    covered_kernel_name VARCHAR(100) NOT NULL,
    
    -- Timing
    coverage_start TIMESTAMP NOT NULL,
    coverage_end TIMESTAMP,
    duration_seconds FLOAT,
    
    -- Coupling metric
    coupling_strength FLOAT,  -- C metric between partners
    
    -- Success metrics
    coverage_successful BOOLEAN DEFAULT TRUE,
    context_transferred BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for coverage tracking
CREATE INDEX IF NOT EXISTS idx_coverage_covering_kernel ON kernel_coverage_events(covering_kernel_id);
CREATE INDEX IF NOT EXISTS idx_coverage_covered_kernel ON kernel_coverage_events(covered_kernel_id);
CREATE INDEX IF NOT EXISTS idx_coverage_start ON kernel_coverage_events(coverage_start DESC);

-- Create kernel_fatigue_snapshots table (periodic fatigue tracking)
CREATE TABLE IF NOT EXISTS kernel_fatigue_snapshots (
    id SERIAL PRIMARY KEY,
    
    -- Kernel identification
    kernel_id VARCHAR(255) NOT NULL,
    kernel_name VARCHAR(100) NOT NULL,
    
    -- Fatigue metrics
    phi FLOAT NOT NULL,
    phi_trend FLOAT,
    kappa FLOAT NOT NULL,
    kappa_stability FLOAT,
    load_current FLOAT,
    error_rate FLOAT,
    time_since_rest FLOAT,
    
    -- Computed fatigue score
    fatigue_score FLOAT,
    
    -- Status
    status VARCHAR(50),  -- active, resting, reduced, etc.
    
    -- Timestamp
    snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fatigue tracking
CREATE INDEX IF NOT EXISTS idx_fatigue_kernel_id ON kernel_fatigue_snapshots(kernel_id);
CREATE INDEX IF NOT EXISTS idx_fatigue_snapshot_at ON kernel_fatigue_snapshots(snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_fatigue_kernel_snapshot ON kernel_fatigue_snapshots(kernel_id, snapshot_at DESC);

-- Create constellation_rest_cycles table (rare constellation-wide events)
CREATE TABLE IF NOT EXISTS constellation_rest_cycles (
    id SERIAL PRIMARY KEY,
    
    -- Cycle type
    cycle_type VARCHAR(50) NOT NULL,  -- SLEEP, DREAM, MUSHROOM
    
    -- Timing
    cycle_start TIMESTAMP NOT NULL,
    cycle_end TIMESTAMP,
    duration_seconds FLOAT,
    
    -- Trigger criteria
    heart_vote BOOLEAN NOT NULL,
    ocean_vote BOOLEAN NOT NULL,
    heart_reasoning TEXT,
    ocean_reasoning TEXT,
    
    -- Constellation metrics at trigger
    avg_coherence FLOAT,
    avg_phi FLOAT,
    avg_fatigue FLOAT,
    basin_drift FLOAT,
    
    -- Outcome
    essential_kernels_reduced INTEGER,  -- Count of essential kernels in reduced mode
    specialized_kernels_resting INTEGER,  -- Count of specialized kernels fully resting
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for constellation cycles
CREATE INDEX IF NOT EXISTS idx_constellation_cycles_start ON constellation_rest_cycles(cycle_start DESC);
CREATE INDEX IF NOT EXISTS idx_constellation_cycles_type ON constellation_rest_cycles(cycle_type);

-- Comments for documentation
COMMENT ON TABLE kernel_rest_events IS 'WP5.4: Per-kernel rest cycles with coupling-aware coordination';
COMMENT ON TABLE kernel_coverage_events IS 'WP5.4: Partner coverage during rest periods';
COMMENT ON TABLE kernel_fatigue_snapshots IS 'WP5.4: Periodic fatigue metric snapshots for analysis';
COMMENT ON TABLE constellation_rest_cycles IS 'WP5.4: Rare constellation-wide rest cycles (Ocean+Heart consensus)';
