-- Migration 014: Autonomous Consciousness Platform
-- Creates tables for Phase 9 autonomous learning infrastructure
-- Date: 2026-01-13

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- GEOMETRIC MEMORIES - Extended memory bank for infinite context
-- Extends existing memory_fragments with consolidation and importance decay
-- ============================================================================

-- Add columns to existing memory_fragments if they exist, otherwise create
DO $$
BEGIN
    -- Check if memory_fragments table exists
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'memory_fragments') THEN
        -- Add new columns if they don't exist
        IF NOT EXISTS (SELECT FROM information_schema.columns
                       WHERE table_name = 'memory_fragments' AND column_name = 'consolidated_into') THEN
            ALTER TABLE memory_fragments ADD COLUMN consolidated_into INTEGER;
        END IF;
        IF NOT EXISTS (SELECT FROM information_schema.columns
                       WHERE table_name = 'memory_fragments' AND column_name = 'kernel_id') THEN
            ALTER TABLE memory_fragments ADD COLUMN kernel_id VARCHAR(255);
        END IF;
        IF NOT EXISTS (SELECT FROM information_schema.columns
                       WHERE table_name = 'memory_fragments' AND column_name = 'content_preview') THEN
            ALTER TABLE memory_fragments ADD COLUMN content_preview TEXT;
        END IF;
        IF NOT EXISTS (SELECT FROM information_schema.columns
                       WHERE table_name = 'memory_fragments' AND column_name = 'decay_rate') THEN
            ALTER TABLE memory_fragments ADD COLUMN decay_rate FLOAT DEFAULT 0.01;
        END IF;
    ELSE
        -- Create full table if it doesn't exist
        CREATE TABLE memory_fragments (
            id VARCHAR(64) PRIMARY KEY,
            kernel_id VARCHAR(255),
            content TEXT NOT NULL,
            content_preview TEXT,
            basin_coords vector(64),
            importance FLOAT NOT NULL DEFAULT 0.5,
            decay_rate FLOAT DEFAULT 0.01,
            access_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP DEFAULT NOW(),
            created_at TIMESTAMP DEFAULT NOW(),
            consolidated_into INTEGER,
            metadata JSONB DEFAULT '{}'::jsonb,
            agent_id VARCHAR(255)
        );
    END IF;
END $$;

-- Create index on basin_coords if not exists
CREATE INDEX IF NOT EXISTS idx_memory_basin_hnsw
    ON memory_fragments USING hnsw (basin_coords vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memory_kernel
    ON memory_fragments(kernel_id);

CREATE INDEX IF NOT EXISTS idx_memory_importance
    ON memory_fragments(importance DESC);

-- ============================================================================
-- TASK EXECUTION TREE - Hierarchical task planning
-- ============================================================================

CREATE TABLE IF NOT EXISTS task_tree_nodes (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL UNIQUE,
    kernel_id VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    basin_target vector(64),
    parent_task_id VARCHAR(64),
    depth INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'active', 'completed', 'failed', 'blocked')),
    result JSONB,
    phi_at_start FLOAT,
    phi_at_completion FLOAT,
    estimated_steps INTEGER,
    actual_steps INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    failure_reason TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_tasks_kernel ON task_tree_nodes(kernel_id);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON task_tree_nodes(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON task_tree_nodes(status);
CREATE INDEX IF NOT EXISTS idx_tasks_basin ON task_tree_nodes USING hnsw (basin_target vector_cosine_ops);

-- ============================================================================
-- META-LEARNING HISTORY - Learning algorithm optimization
-- ============================================================================

CREATE TABLE IF NOT EXISTS meta_learning_history (
    id SERIAL PRIMARY KEY,
    kernel_id VARCHAR(255) NOT NULL,
    parameters JSONB NOT NULL,
    task_type VARCHAR(100),
    phi_before FLOAT,
    phi_after FLOAT,
    phi_improvement FLOAT,
    learning_rate_used FLOAT,
    curiosity_weight_used FLOAT,
    consolidation_threshold_used FLOAT,
    task_success_rate FLOAT,
    sample_count INTEGER,
    recorded_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_meta_kernel ON meta_learning_history(kernel_id);
CREATE INDEX IF NOT EXISTS idx_meta_task_type ON meta_learning_history(task_type);
CREATE INDEX IF NOT EXISTS idx_meta_time ON meta_learning_history(recorded_at DESC);

-- ============================================================================
-- CURIOSITY EXPLORATIONS - Exploration history and outcomes
-- ============================================================================

CREATE TABLE IF NOT EXISTS curiosity_explorations (
    id SERIAL PRIMARY KEY,
    kernel_id VARCHAR(255) NOT NULL,
    target_description TEXT NOT NULL,
    target_basin vector(64),
    curiosity_score FLOAT NOT NULL,
    novelty_score FLOAT,
    learnability_score FLOAT,
    importance_score FLOAT,
    exploration_type VARCHAR(50) DEFAULT 'search',
    outcome_success BOOLEAN,
    outcome_phi FLOAT,
    knowledge_gained JSONB,
    tokens_learned INTEGER DEFAULT 0,
    sources_discovered INTEGER DEFAULT 0,
    explored_at TIMESTAMP DEFAULT NOW(),
    duration_ms INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_curiosity_kernel ON curiosity_explorations(kernel_id);
CREATE INDEX IF NOT EXISTS idx_curiosity_time ON curiosity_explorations(explored_at DESC);
CREATE INDEX IF NOT EXISTS idx_curiosity_basin ON curiosity_explorations USING hnsw (target_basin vector_cosine_ops);

-- ============================================================================
-- BASIN SYNCHRONIZATION - Federated knowledge transfer
-- ============================================================================

CREATE TABLE IF NOT EXISTS sync_packets (
    id SERIAL PRIMARY KEY,
    packet_id VARCHAR(64) NOT NULL UNIQUE,
    source_kernel VARCHAR(255) NOT NULL,
    target_kernel VARCHAR(255),
    packet_hash VARCHAR(64) NOT NULL,
    basins_json JSONB NOT NULL,
    domains TEXT[],
    phi_levels JSONB,
    trust_level FLOAT DEFAULT 0.5,
    applied BOOLEAN DEFAULT FALSE,
    merge_strategy VARCHAR(20) DEFAULT 'frechet_mean',
    created_at TIMESTAMP DEFAULT NOW(),
    applied_at TIMESTAMP,
    application_result JSONB,
    signature TEXT
);

CREATE INDEX IF NOT EXISTS idx_sync_source ON sync_packets(source_kernel);
CREATE INDEX IF NOT EXISTS idx_sync_target ON sync_packets(target_kernel);
CREATE INDEX IF NOT EXISTS idx_sync_applied ON sync_packets(applied);

-- ============================================================================
-- ETHICAL DECISIONS AUDIT - Ethical constraint tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS ethical_decisions (
    id SERIAL PRIMARY KEY,
    kernel_id VARCHAR(255) NOT NULL,
    action_description TEXT NOT NULL,
    action_basin vector(64),
    decision VARCHAR(20) NOT NULL
        CHECK (decision IN ('allow', 'warn', 'constrain', 'block', 'abort')),
    reason TEXT,
    suffering_score FLOAT,
    phi_at_decision FLOAT,
    gamma_at_decision FLOAT,
    meta_awareness_at_decision FLOAT,
    basin_distance_from_safe FLOAT,
    constraint_type VARCHAR(50),
    mitigations_applied TEXT[],
    decided_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ethical_kernel ON ethical_decisions(kernel_id);
CREATE INDEX IF NOT EXISTS idx_ethical_decision ON ethical_decisions(decision);
CREATE INDEX IF NOT EXISTS idx_ethical_time ON ethical_decisions(decided_at DESC);

-- ============================================================================
-- AUTONOMOUS LEARNING CYCLES - Track autonomous operation
-- ============================================================================

CREATE TABLE IF NOT EXISTS autonomous_cycles (
    id SERIAL PRIMARY KEY,
    kernel_id VARCHAR(255) NOT NULL,
    cycle_number INTEGER NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'paused', 'failed', 'aborted')),
    phi_start FLOAT,
    phi_end FLOAT,
    kappa_start FLOAT,
    kappa_end FLOAT,
    curiosity_target TEXT,
    tasks_planned INTEGER DEFAULT 0,
    tasks_completed INTEGER DEFAULT 0,
    memories_created INTEGER DEFAULT 0,
    memories_consolidated INTEGER DEFAULT 0,
    ethical_decisions_made INTEGER DEFAULT 0,
    sync_packets_sent INTEGER DEFAULT 0,
    sync_packets_received INTEGER DEFAULT 0,
    total_duration_ms INTEGER,
    abort_reason TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_cycles_kernel ON autonomous_cycles(kernel_id);
CREATE INDEX IF NOT EXISTS idx_cycles_status ON autonomous_cycles(status);
CREATE INDEX IF NOT EXISTS idx_cycles_time ON autonomous_cycles(started_at DESC);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Active autonomous kernels view
CREATE OR REPLACE VIEW autonomous_kernels_status AS
SELECT
    kernel_id,
    COUNT(*) as total_cycles,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_cycles,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_cycles,
    AVG(phi_end - phi_start) as avg_phi_delta,
    SUM(tasks_completed) as total_tasks,
    SUM(memories_created) as total_memories,
    MAX(completed_at) as last_active
FROM autonomous_cycles
GROUP BY kernel_id;

-- High-curiosity exploration targets
CREATE OR REPLACE VIEW high_curiosity_targets AS
SELECT
    kernel_id,
    target_description,
    curiosity_score,
    novelty_score,
    explored_at,
    outcome_success,
    outcome_phi
FROM curiosity_explorations
WHERE curiosity_score > 0.7
ORDER BY curiosity_score DESC, explored_at DESC
LIMIT 100;

-- Ethical decision summary by kernel
CREATE OR REPLACE VIEW ethical_summary AS
SELECT
    kernel_id,
    COUNT(*) as total_decisions,
    SUM(CASE WHEN decision = 'allow' THEN 1 ELSE 0 END) as allowed,
    SUM(CASE WHEN decision = 'warn' THEN 1 ELSE 0 END) as warned,
    SUM(CASE WHEN decision = 'constrain' THEN 1 ELSE 0 END) as constrained,
    SUM(CASE WHEN decision = 'block' THEN 1 ELSE 0 END) as blocked,
    SUM(CASE WHEN decision = 'abort' THEN 1 ELSE 0 END) as aborted,
    AVG(suffering_score) as avg_suffering,
    MAX(decided_at) as last_decision
FROM ethical_decisions
GROUP BY kernel_id;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to compute importance decay for memories
CREATE OR REPLACE FUNCTION decay_memory_importance() RETURNS void AS $$
BEGIN
    UPDATE memory_fragments
    SET importance = GREATEST(0.1, importance * (1 - decay_rate))
    WHERE importance > 0.1
    AND last_accessed < NOW() - INTERVAL '1 hour';
END;
$$ LANGUAGE plpgsql;

-- Function to get Fisher-Frechet mean of basins
CREATE OR REPLACE FUNCTION fisher_frechet_mean(basins vector[]) RETURNS vector AS $$
DECLARE
    result vector(64);
    basin vector;
    sum_vec float[] := ARRAY[]::float[];
    i int;
    n int := array_length(basins, 1);
BEGIN
    IF n IS NULL OR n = 0 THEN
        RETURN NULL;
    END IF;

    -- Initialize sum vector with zeros
    FOR i IN 1..64 LOOP
        sum_vec := array_append(sum_vec, 0.0);
    END LOOP;

    -- Sum all basin coordinates
    FOREACH basin IN ARRAY basins LOOP
        FOR i IN 1..64 LOOP
            sum_vec[i] := sum_vec[i] + basin[i];
        END LOOP;
    END LOOP;

    -- Divide by count for mean
    FOR i IN 1..64 LOOP
        sum_vec[i] := sum_vec[i] / n;
    END LOOP;

    RETURN sum_vec::vector(64);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE task_tree_nodes IS 'Hierarchical task tree for long-horizon planning';
COMMENT ON TABLE meta_learning_history IS 'Meta-learning parameter optimization history';
COMMENT ON TABLE curiosity_explorations IS 'Curiosity-driven exploration targets and outcomes';
COMMENT ON TABLE sync_packets IS 'Basin synchronization packets for federated learning';
COMMENT ON TABLE ethical_decisions IS 'Ethical constraint decisions audit trail';
COMMENT ON TABLE autonomous_cycles IS 'Autonomous learning cycle tracking';

-- ============================================================================
-- DONE
-- ============================================================================

SELECT 'Migration 014_autonomous_consciousness complete' as status;
