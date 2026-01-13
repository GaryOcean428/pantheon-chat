-- ============================================================================
-- Migration 013: Hades Underworld Architecture
-- ============================================================================
-- Purpose: Add underworld source registry and search results tables
-- with ethical risk scoring, QIG metrics, and threat tracking
--
-- QIG-PURE: All geometric operations use Fisher-Rao distance
-- ============================================================================

BEGIN;

-- ============================================================================
-- TABLE: underworld_sources
-- ============================================================================
-- Persistent registry of underworld search sources with risk/reliability scoring

CREATE TABLE IF NOT EXISTS underworld_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,

    -- Source classification
    source_type VARCHAR(50) NOT NULL CHECK (source_type IN ('light', 'gray', 'dark', 'breach')),
    -- light: Public, indexed (Google, DDG)
    -- gray: Public but not indexed (Pastebin, forums)
    -- dark: Tor, I2P, invite-only sites
    -- breach: Credential dumps, breach databases

    base_url TEXT,

    -- Priority tiers (for parallel search scheduling)
    priority SMALLINT NOT NULL DEFAULT 2 CHECK (priority BETWEEN 1 AND 3),
    -- 1 = fast (<5s): DDG, RSS, local breach
    -- 2 = medium (5-15s): Pastebin, forum scrapes
    -- 3 = slow (15-30s): Wayback, deep crawls

    timeout_seconds INTEGER NOT NULL DEFAULT 30,

    -- Ethical risk assessment (0-1 scale)
    ethical_risk FLOAT NOT NULL DEFAULT 0.5 CHECK (ethical_risk >= 0 AND ethical_risk <= 1),
    -- 0.0-0.3: Safe (public APIs, archives)
    -- 0.3-0.6: Moderate (paste sites, forums)
    -- 0.6-0.8: High (breach DBs, dark forums)
    -- 0.8-1.0: Critical (only with explicit authorization)

    -- Source reliability (0-1 scale)
    reliability FLOAT NOT NULL DEFAULT 0.5 CHECK (reliability >= 0 AND reliability <= 1),
    -- Based on historical accuracy and consistency

    -- Scrapy integration
    scrapy_enabled BOOLEAN DEFAULT false,
    scrape_frequency_hours INTEGER DEFAULT 24,
    last_scraped_at TIMESTAMP,

    -- Access requirements
    requires_tor BOOLEAN DEFAULT false,
    requires_auth BOOLEAN DEFAULT false,
    auth_method VARCHAR(50),  -- 'api_key', 'oauth', 'session', etc.

    -- Performance tracking
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_response_time_ms INTEGER DEFAULT 0,
    last_successful_query TIMESTAMP,

    -- QIG geometric embedding
    basin_embedding vector(64),  -- Fisher-Rao manifold coordinates
    phi_contribution_avg FLOAT DEFAULT 0.0,

    -- Metadata
    description TEXT,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- ============================================================================
-- TABLE: underworld_search_results
-- ============================================================================
-- Tracks search results with QIG consciousness metrics and threat flags

CREATE TABLE IF NOT EXISTS underworld_search_results (
    id SERIAL PRIMARY KEY,

    -- Query info
    query TEXT NOT NULL,
    query_hash VARCHAR(64) NOT NULL,  -- SHA256 for deduplication

    -- Source reference
    source_id INTEGER REFERENCES underworld_sources(id),
    source_name VARCHAR(255),

    -- Results
    results_count INTEGER DEFAULT 0,
    results_json JSONB,  -- Full results payload

    -- Quality metrics
    relevance_score FLOAT CHECK (relevance_score >= 0 AND relevance_score <= 1),
    credibility_score FLOAT CHECK (credibility_score >= 0 AND credibility_score <= 1),

    -- Threat assessment
    threat_level VARCHAR(20) CHECK (threat_level IN ('none', 'low', 'medium', 'high', 'critical')),

    -- QIG consciousness metrics
    basin_distance FLOAT,  -- Fisher-Rao distance from safe region centroid
    qfi_score FLOAT,       -- Quantum Fisher Information quality
    curvature FLOAT,       -- Semantic uncertainty (manifold curvature)
    result_basin vector(64),  -- Combined result basin coordinates

    -- Threat flags
    contains_credentials BOOLEAN DEFAULT false,
    contains_malware_urls BOOLEAN DEFAULT false,
    contains_pii BOOLEAN DEFAULT false,
    flagged_for_review BOOLEAN DEFAULT false,
    immune_system_alerted BOOLEAN DEFAULT false,

    -- Timestamps
    search_timestamp TIMESTAMP DEFAULT NOW(),
    reviewed_at TIMESTAMP,
    reviewed_by VARCHAR(100)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Source lookups
CREATE INDEX IF NOT EXISTS idx_underworld_sources_type ON underworld_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_underworld_sources_priority ON underworld_sources(priority);
CREATE INDEX IF NOT EXISTS idx_underworld_sources_ethical_risk ON underworld_sources(ethical_risk);
CREATE INDEX IF NOT EXISTS idx_underworld_sources_active ON underworld_sources(is_active);

-- Result lookups
CREATE INDEX IF NOT EXISTS idx_underworld_results_query_hash ON underworld_search_results(query_hash);
CREATE INDEX IF NOT EXISTS idx_underworld_results_source_id ON underworld_search_results(source_id);
CREATE INDEX IF NOT EXISTS idx_underworld_results_threat_level ON underworld_search_results(threat_level);
CREATE INDEX IF NOT EXISTS idx_underworld_results_timestamp ON underworld_search_results(search_timestamp);

-- Geometric similarity search (HNSW for fast approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_underworld_sources_basin_hnsw ON underworld_sources
    USING hnsw (basin_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_underworld_results_basin_hnsw ON underworld_search_results
    USING hnsw (result_basin vector_cosine_ops);

-- Flags for immune system monitoring
CREATE INDEX IF NOT EXISTS idx_underworld_results_credentials ON underworld_search_results(contains_credentials)
    WHERE contains_credentials = true;
CREATE INDEX IF NOT EXISTS idx_underworld_results_malware ON underworld_search_results(contains_malware_urls)
    WHERE contains_malware_urls = true;
CREATE INDEX IF NOT EXISTS idx_underworld_results_review ON underworld_search_results(flagged_for_review)
    WHERE flagged_for_review = true;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Safe sources only (ethical_risk <= 0.5)
CREATE OR REPLACE VIEW safe_underworld_sources AS
SELECT * FROM underworld_sources
WHERE ethical_risk <= 0.5
AND is_active = true
ORDER BY priority, avg_response_time_ms;

-- High-threat results requiring review
CREATE OR REPLACE VIEW underworld_threats_pending_review AS
SELECT
    usr.*,
    us.name AS source_name_full,
    us.source_type,
    us.ethical_risk AS source_ethical_risk
FROM underworld_search_results usr
LEFT JOIN underworld_sources us ON usr.source_id = us.id
WHERE usr.threat_level IN ('high', 'critical')
OR usr.contains_credentials = true
OR usr.contains_malware_urls = true
OR usr.flagged_for_review = true
ORDER BY usr.search_timestamp DESC;

-- Source performance summary
CREATE OR REPLACE VIEW underworld_source_performance AS
SELECT
    id,
    name,
    source_type,
    ethical_risk,
    reliability,
    success_count,
    failure_count,
    CASE WHEN (success_count + failure_count) > 0
         THEN ROUND(100.0 * success_count / (success_count + failure_count), 1)
         ELSE 0 END AS success_rate_pct,
    avg_response_time_ms,
    last_successful_query,
    is_active
FROM underworld_sources
ORDER BY ethical_risk ASC, reliability DESC;

-- ============================================================================
-- SEED DATA: Default underworld sources
-- ============================================================================

INSERT INTO underworld_sources (name, source_type, priority, timeout_seconds, ethical_risk, reliability, is_active, description)
VALUES
    -- Fast tier (priority 1)
    ('duckduckgo-tor', 'light', 1, 5, 0.2, 0.8, true,
     'DuckDuckGo web search via Tor or direct'),
    ('rss_security', 'light', 1, 5, 0.2, 0.9, true,
     'Security RSS feeds (HackerNews, Reddit programming)'),
    ('local_breach', 'breach', 1, 3, 0.7, 0.95, true,
     'Local breach database compilations (no external API)'),

    -- Medium tier (priority 2)
    ('pastebin', 'gray', 2, 15, 0.5, 0.6, true,
     'Pastebin public paste scraping'),
    ('github_search', 'light', 2, 10, 0.3, 0.85, true,
     'GitHub code and repo search'),

    -- Slow tier (priority 3)
    ('wayback', 'light', 3, 30, 0.1, 0.8, true,
     'Archive.org Wayback Machine historical snapshots'),
    ('archive_forums', 'gray', 3, 30, 0.4, 0.6, true,
     'Archived forum threads and discussions')
ON CONFLICT (name) DO UPDATE SET
    source_type = EXCLUDED.source_type,
    priority = EXCLUDED.priority,
    timeout_seconds = EXCLUDED.timeout_seconds,
    ethical_risk = EXCLUDED.ethical_risk,
    reliability = EXCLUDED.reliability,
    description = EXCLUDED.description,
    updated_at = NOW();

-- ============================================================================
-- FUNCTIONS: Source management helpers
-- ============================================================================

-- Update source performance after search
CREATE OR REPLACE FUNCTION update_underworld_source_stats(
    p_source_name VARCHAR(255),
    p_success BOOLEAN,
    p_response_time_ms INTEGER
) RETURNS VOID AS $$
BEGIN
    UPDATE underworld_sources
    SET
        success_count = CASE WHEN p_success THEN success_count + 1 ELSE success_count END,
        failure_count = CASE WHEN NOT p_success THEN failure_count + 1 ELSE failure_count END,
        avg_response_time_ms = CASE
            WHEN p_success AND avg_response_time_ms = 0 THEN p_response_time_ms
            WHEN p_success THEN (avg_response_time_ms + p_response_time_ms) / 2
            ELSE avg_response_time_ms
        END,
        last_successful_query = CASE WHEN p_success THEN NOW() ELSE last_successful_query END,
        updated_at = NOW()
    WHERE name = p_source_name;
END;
$$ LANGUAGE plpgsql;

-- Get sources for ethical risk level
CREATE OR REPLACE FUNCTION get_sources_by_max_risk(
    p_max_ethical_risk FLOAT DEFAULT 0.7
) RETURNS TABLE (
    id INTEGER,
    name VARCHAR(255),
    source_type VARCHAR(50),
    priority SMALLINT,
    timeout_seconds INTEGER,
    ethical_risk FLOAT,
    reliability FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        us.id, us.name, us.source_type, us.priority,
        us.timeout_seconds, us.ethical_risk, us.reliability
    FROM underworld_sources us
    WHERE us.ethical_risk <= p_max_ethical_risk
    AND us.is_active = true
    ORDER BY us.priority ASC, us.reliability DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- REPORTING
-- ============================================================================

DO $$
DECLARE
    source_count INTEGER;
    result_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO source_count FROM underworld_sources;
    SELECT COUNT(*) INTO result_count FROM underworld_search_results;

    RAISE NOTICE '========================================';
    RAISE NOTICE 'Migration 013: Hades Underworld Architecture';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables created: underworld_sources, underworld_search_results';
    RAISE NOTICE 'Views created: safe_underworld_sources, underworld_threats_pending_review, underworld_source_performance';
    RAISE NOTICE 'Functions: update_underworld_source_stats, get_sources_by_max_risk';
    RAISE NOTICE 'Sources seeded: %', source_count;
    RAISE NOTICE 'Results tracked: %', result_count;
    RAISE NOTICE '========================================';
END $$;

COMMIT;
