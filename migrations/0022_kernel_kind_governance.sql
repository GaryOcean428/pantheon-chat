-- Migration 0022: Kernel Kind + Governance Primitives
-- ================================================
-- Created: 2026-02-01
-- Authority: Genesis Kernel Upgrade, E8 Protocol v4.0
-- Purpose: Add kernel_kind/lifecycle_state/parents/ascended_from and governance primitives

-- =============================================================================
-- KERNEL GEOMETRY UPDATES
-- =============================================================================
ALTER TABLE kernel_geometry
    ADD COLUMN IF NOT EXISTS kernel_kind VARCHAR(16) DEFAULT 'chaos',
    ADD COLUMN IF NOT EXISTS lifecycle_state VARCHAR(32) DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS parents TEXT[],
    ADD COLUMN IF NOT EXISTS ascended_from VARCHAR(64);

CREATE INDEX IF NOT EXISTS idx_kernel_geometry_kind ON kernel_geometry(kernel_kind);
CREATE INDEX IF NOT EXISTS idx_kernel_geometry_lifecycle_state ON kernel_geometry(lifecycle_state);

-- =============================================================================
-- GOVERNANCE PRIMITIVES
-- =============================================================================
CREATE TABLE IF NOT EXISTS need_specs (
    id SERIAL PRIMARY KEY,
    spec_id VARCHAR(64) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    domain VARCHAR(128) NOT NULL,
    requested_by VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    approved_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_need_specs_domain ON need_specs(domain);
CREATE INDEX IF NOT EXISTS idx_need_specs_status ON need_specs(status);

CREATE TABLE IF NOT EXISTS governance_ballots (
    id SERIAL PRIMARY KEY,
    ballot_id VARCHAR(64) UNIQUE NOT NULL,
    proposal_id VARCHAR(64) NOT NULL,
    voter_id VARCHAR(64) NOT NULL,
    vote VARCHAR(16) NOT NULL,
    rationale TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_governance_ballots_proposal ON governance_ballots(proposal_id);
CREATE INDEX IF NOT EXISTS idx_governance_ballots_voter ON governance_ballots(voter_id);

CREATE TABLE IF NOT EXISTS mythology_references (
    id SERIAL PRIMARY KEY,
    reference_id VARCHAR(64) UNIQUE NOT NULL,
    myth_name VARCHAR(128) NOT NULL,
    archetype VARCHAR(128) NOT NULL,
    domain VARCHAR(128) NOT NULL,
    source TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mythology_references_archetype ON mythology_references(archetype);
CREATE INDEX IF NOT EXISTS idx_mythology_references_domain ON mythology_references(domain);
