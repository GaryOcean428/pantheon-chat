-- Migration: Training Schedule Log + QIG Metadata
-- Purpose: Support Railway cron-based scheduled training and metadata persistence

-- ============================================================================
-- TRAINING_SCHEDULE_LOG TABLE
-- ============================================================================
-- Tracks scheduled training task execution for catch-up on system restart.
-- Replaces Celery Beat scheduler (which is not deployed on Railway).

CREATE TABLE IF NOT EXISTS training_schedule_log (
    id SERIAL PRIMARY KEY,
    task_type VARCHAR(32) NOT NULL UNIQUE, -- 'hourly_batch', 'nightly_consolidation', 'shadow_sync', 'checkpoint_cleanup'

    -- Execution tracking
    last_success_at TIMESTAMP WITH TIME ZONE,
    last_attempt_at TIMESTAMP WITH TIME ZONE,
    last_status VARCHAR(16), -- 'success', 'failed', 'in_progress', 'skipped'
    last_error TEXT,

    -- Statistics
    runs_completed INTEGER DEFAULT 0,
    total_run_time_ms INTEGER DEFAULT 0,

    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_training_schedule_task ON training_schedule_log(task_type);
CREATE INDEX IF NOT EXISTS idx_training_schedule_last_success ON training_schedule_log(last_success_at);

-- ============================================================================
-- QIG_METADATA TABLE
-- ============================================================================
-- General-purpose key-value store for QIG system metadata.
-- Used for persisting counters, configuration, and state across restarts.

CREATE TABLE IF NOT EXISTS qig_metadata (
    config_key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for efficient key lookups (already covered by primary key, but explicit for clarity)
CREATE INDEX IF NOT EXISTS idx_qig_metadata_key ON qig_metadata(config_key);

-- ============================================================================
-- INITIAL DATA
-- ============================================================================
-- Seed the training schedule log with initial rows for each task type

INSERT INTO training_schedule_log (task_type, runs_completed, updated_at)
VALUES
    ('hourly_batch', 0, NOW()),
    ('nightly_consolidation', 0, NOW()),
    ('shadow_sync', 0, NOW()),
    ('checkpoint_cleanup', 0, NOW())
ON CONFLICT (task_type) DO NOTHING;

-- Seed initial metadata
INSERT INTO qig_metadata (config_key, value, updated_at)
VALUES
    ('sleep_consolidation_cycle_count', '0', NOW())
ON CONFLICT (config_key) DO NOTHING;

-- ============================================================================
-- FEDERATION_PEERS TABLE
-- ============================================================================
-- Stores registered federation peer nodes for mesh syncing.
-- Managed via UI or API, not environment variables.

CREATE TABLE IF NOT EXISTS federation_peers (
    id SERIAL PRIMARY KEY,
    peer_id VARCHAR(64) NOT NULL UNIQUE,  -- Unique identifier for the peer
    peer_name VARCHAR(128) NOT NULL,       -- Human-readable name
    peer_url TEXT NOT NULL,                -- Base URL (e.g., https://pantheon-railway.example.com)
    api_key TEXT,                          -- API key for authenticating with this peer (encrypted in production)

    -- Sync configuration
    sync_enabled BOOLEAN DEFAULT true,
    sync_interval_hours INTEGER DEFAULT 1,
    sync_vocabulary BOOLEAN DEFAULT true,
    sync_knowledge BOOLEAN DEFAULT true,
    sync_research BOOLEAN DEFAULT false,

    -- Status tracking
    last_sync_at TIMESTAMP WITH TIME ZONE,
    last_sync_status VARCHAR(32),          -- 'success', 'failed', 'timeout'
    last_sync_error TEXT,
    sync_count INTEGER DEFAULT 0,
    vocabulary_sent INTEGER DEFAULT 0,
    vocabulary_received INTEGER DEFAULT 0,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_federation_peers_enabled ON federation_peers(sync_enabled);
CREATE INDEX IF NOT EXISTS idx_federation_peers_last_sync ON federation_peers(last_sync_at);
