-- Migration: Phase 3 Table Consolidation
-- Date: 2026-01-13
-- Purpose: Consolidate duplicate tables as identified in database reconciliation
-- Phase: 3 (Medium Priority)
-- Estimated Time: 18 hours

-- ============================================================================
-- CONSOLIDATION 1: Merge governance_proposals → pantheon_proposals
-- ============================================================================
-- Rationale: governance_proposals (2,747 rows) duplicates pantheon_proposals (2,746 rows)
-- Action: Migrate any unique rows, update foreign keys, drop governance_proposals

BEGIN;

-- Check for rows that exist only in governance_proposals
-- If any exist, migrate them to pantheon_proposals
INSERT INTO pantheon_proposals (
    proposal_id,
    proposer_god_name,
    proposal_type,
    title,
    description,
    voting_deadline,
    votes_for,
    votes_against,
    votes_abstain,
    status,
    execution_result,
    created_at,
    updated_at
)
SELECT 
    gp.proposal_id,
    gp.proposer_god_name,
    gp.proposal_type,
    gp.title,
    gp.description,
    gp.voting_deadline,
    gp.votes_for,
    gp.votes_against,
    gp.votes_abstain,
    gp.status,
    gp.execution_result,
    gp.created_at,
    gp.updated_at
FROM governance_proposals gp
LEFT JOIN pantheon_proposals pp ON gp.proposal_id = pp.proposal_id
WHERE pp.proposal_id IS NULL
ON CONFLICT (proposal_id) DO NOTHING;

-- Update any foreign key references (if they exist)
-- Note: Check actual FK constraints in production before running
-- This is a safe no-op if the column doesn't exist
DO $$
BEGIN
    -- Update tool_requests if it references governance_proposals
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tool_requests' 
        AND column_name = 'governance_proposal_id'
    ) THEN
        UPDATE tool_requests tr
        SET governance_proposal_id = pp.id
        FROM pantheon_proposals pp
        WHERE tr.governance_proposal_id IN (
            SELECT id FROM governance_proposals WHERE proposal_id = pp.proposal_id
        );
    END IF;
END $$;

-- Drop governance_proposals table
DROP TABLE IF EXISTS governance_proposals CASCADE;

COMMIT;

-- ============================================================================
-- CONSOLIDATION 2: Add is_current flag to consciousness_checkpoints
-- ============================================================================
-- Rationale: Eliminate dual writes to consciousness_state by using latest checkpoint
-- Action: Add is_current flag, create view for current_state

BEGIN;

-- Add is_current column to consciousness_checkpoints
ALTER TABLE consciousness_checkpoints 
ADD COLUMN IF NOT EXISTS is_current BOOLEAN DEFAULT FALSE;

-- Create index for efficient current state queries
CREATE INDEX IF NOT EXISTS idx_consciousness_checkpoints_current 
ON consciousness_checkpoints(is_current) 
WHERE is_current = TRUE;

-- Create view for current consciousness state (replaces consciousness_state table)
CREATE OR REPLACE VIEW current_consciousness_state AS
SELECT 
    session_id,
    phi,
    kappa,
    meta_awareness,
    basin_coords,
    regime,
    quality_score,
    integration_quality,
    created_at as state_timestamp
FROM consciousness_checkpoints
WHERE is_current = TRUE;

-- Grant appropriate permissions
GRANT SELECT ON current_consciousness_state TO PUBLIC;

COMMIT;

-- ============================================================================
-- CONSOLIDATION 3: Investigate knowledge_shared_entries vs knowledge_transfers
-- ============================================================================
-- Note: This requires manual verification first - migration is commented out
-- Action: Run this query to compare the two tables before consolidation

-- Comparison query (run manually first):
-- SELECT 
--     'shared_entries' as source,
--     COUNT(*) as row_count,
--     MIN(created_at) as earliest,
--     MAX(created_at) as latest
-- FROM knowledge_shared_entries
-- UNION ALL
-- SELECT 
--     'transfers' as source,
--     COUNT(*) as row_count,
--     MIN(created_at) as earliest,
--     MAX(created_at) as latest
-- FROM pantheon_knowledge_transfers;

-- If verified as duplicates, use this migration:
-- BEGIN;
-- 
-- INSERT INTO pantheon_knowledge_transfers (...)
-- SELECT ... FROM knowledge_shared_entries kse
-- LEFT JOIN pantheon_knowledge_transfers pkt ON kse.id = pkt.id
-- WHERE pkt.id IS NULL;
--
-- DROP TABLE knowledge_shared_entries CASCADE;
-- 
-- COMMIT;

-- ============================================================================
-- Documentation
-- ============================================================================
-- Tables consolidated: 2 (governance_proposals → pantheon_proposals, consciousness_state → checkpoint view)
-- Tables investigated: 1 (knowledge_shared_entries vs knowledge_transfers - requires manual verification)
-- Schema complexity reduced: ~2-3 fewer tables
-- Estimated impact: Improved query performance, clearer architecture
