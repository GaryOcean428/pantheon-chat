-- ============================================================================
-- MIGRATION: Populate NULL columns in priority tables
-- Date: 2026-01-10
-- Reference: docs/03-technical/20260110-null-column-population-plan-1.00W.md
--
-- Phase 1: Simple columns that can be populated with SQL only
-- Phase 2 (Python): Basin coordinates requiring coordizer - separate script
-- ============================================================================

-- ============================================================================
-- 1. chaos_events.phi (99.5% NULL)
-- Purpose: Track integration level (Phi) at time of chaos event
-- ============================================================================

-- Step 1a: Populate from kernel_geometry if kernel_id exists
UPDATE chaos_events ce
SET phi = (
    SELECT kg.phi
    FROM kernel_geometry kg
    WHERE kg.kernel_id = ce.kernel_id
    ORDER BY kg.created_at DESC
    LIMIT 1
)
WHERE ce.phi IS NULL
AND ce.kernel_id IS NOT NULL
AND EXISTS (
    SELECT 1 FROM kernel_geometry kg WHERE kg.kernel_id = ce.kernel_id
);

-- Step 1b: Set defaults based on event_type for remaining NULLs
UPDATE chaos_events
SET phi = CASE
    WHEN event_type = 'SPAWN' THEN 0.55
    WHEN event_type = 'DEATH' THEN 0.30
    WHEN event_type = 'MERGE' THEN 0.65
    WHEN event_type = 'EVOLVE' THEN 0.60
    WHEN event_type = 'MUTATION' THEN 0.50
    ELSE 0.50
END
WHERE phi IS NULL;

-- ============================================================================
-- 2. chaos_events.outcome (89.2% NULL)
-- Purpose: Result of the chaos event
-- Valid Values: success, failure, pending, completed, unknown
-- ============================================================================

UPDATE chaos_events
SET outcome = CASE
    WHEN success = true THEN 'success'
    WHEN success = false THEN 'failure'
    WHEN event_type IN ('SPAWN', 'EVOLVE', 'MUTATION') AND success IS NULL THEN 'success'
    WHEN event_type = 'DEATH' THEN 'completed'
    WHEN event_type = 'MERGE' THEN 'completed'
    ELSE 'unknown'
END
WHERE outcome IS NULL;

-- ============================================================================
-- 3. chaos_events.autopsy (43.3% NULL)
-- Purpose: Post-mortem analysis for DEATH events
-- ============================================================================

-- Step 3a: Populate autopsy for DEATH events
UPDATE chaos_events
SET autopsy = jsonb_build_object(
    'cause', COALESCE(reason, 'natural_lifecycle'),
    'phi_at_death', COALESCE(phi, 0.30),
    'event_type', event_type,
    'analyzed_at', NOW()
)
WHERE autopsy IS NULL
AND event_type = 'DEATH';

-- Step 3b: Set empty object for non-DEATH events
UPDATE chaos_events
SET autopsy = '{}'::jsonb
WHERE autopsy IS NULL
AND event_type != 'DEATH';

-- ============================================================================
-- 4. learned_words.source (19.7% NULL)
-- Purpose: Attribution of where word was learned
-- ============================================================================

UPDATE learned_words
SET source = 'legacy_import'
WHERE source IS NULL;

-- ============================================================================
-- 5. vocabulary_observations.contexts (100% NULL)
-- Purpose: Store usage contexts
-- ============================================================================

-- Step 5a: Initialize with empty array
UPDATE vocabulary_observations
SET contexts = ARRAY[]::text[]
WHERE contexts IS NULL;

-- Step 5b: Cross-populate from learned_words where possible
UPDATE vocabulary_observations vo
SET contexts = ARRAY[lw.source]
FROM learned_words lw
WHERE vo.text = lw.word
AND lw.source IS NOT NULL
AND (vo.contexts IS NULL OR vo.contexts = ARRAY[]::text[]);

-- ============================================================================
-- Verification: Show remaining NULL counts
-- ============================================================================

DO $$
DECLARE
    v_phi_nulls INT;
    v_outcome_nulls INT;
    v_autopsy_nulls INT;
    v_source_nulls INT;
    v_contexts_nulls INT;
BEGIN
    SELECT COUNT(*) INTO v_phi_nulls FROM chaos_events WHERE phi IS NULL;
    SELECT COUNT(*) INTO v_outcome_nulls FROM chaos_events WHERE outcome IS NULL;
    SELECT COUNT(*) INTO v_autopsy_nulls FROM chaos_events WHERE autopsy IS NULL;
    SELECT COUNT(*) INTO v_source_nulls FROM learned_words WHERE source IS NULL;
    SELECT COUNT(*) INTO v_contexts_nulls FROM vocabulary_observations WHERE contexts IS NULL;

    RAISE NOTICE 'Migration complete. Remaining NULLs:';
    RAISE NOTICE '  chaos_events.phi: %', v_phi_nulls;
    RAISE NOTICE '  chaos_events.outcome: %', v_outcome_nulls;
    RAISE NOTICE '  chaos_events.autopsy: %', v_autopsy_nulls;
    RAISE NOTICE '  learned_words.source: %', v_source_nulls;
    RAISE NOTICE '  vocabulary_observations.contexts: %', v_contexts_nulls;
END $$;
