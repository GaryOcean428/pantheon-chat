-- ============================================================================
-- CREATE USER_TARGET_ADDRESSES TABLE
-- Date: 2026-01-09
-- Purpose: Store user target addresses for AutonomousPantheon debate scanning
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_target_addresses (
    id SERIAL PRIMARY KEY,
    address TEXT NOT NULL UNIQUE,
    added_at TIMESTAMP DEFAULT NOW(),
    last_scanned_at TIMESTAMP,
    scan_count INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT TRUE
);

-- Index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_user_target_addresses_active
ON user_target_addresses(active, added_at DESC);

-- Usage: Run with psql or pg_query
-- psql $DATABASE_URL -f migrations/010_create_user_target_addresses.sql
