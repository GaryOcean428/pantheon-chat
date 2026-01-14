-- ============================================================================
-- ADD COLUMN DEFAULTS - Comprehensive Schema Census Migration
-- ============================================================================
-- Purpose: Add appropriate default values to all columns that need them
-- Project: pantheon-chat
-- Date: 2026-01-12
-- Related: PRs 47, 48 - Database completeness work
-- Physics: Φ defaults 0.5-0.7, κ* = 64.0 (optimal coupling)
-- ============================================================================

BEGIN;

-- ============================================================================
-- SECTION 1: SINGLETON TABLES
-- These tables should have exactly one row with stable defaults
-- ============================================================================

-- ocean_quantum_state (singleton) - Already well-configured in schema
-- near_miss_adaptive_state (singleton)
ALTER TABLE near_miss_adaptive_state
ALTER COLUMN rolling_phi_distribution SET DEFAULT '{}';

-- auto_cycle_state (singleton)
ALTER TABLE auto_cycle_state
ALTER COLUMN address_ids SET DEFAULT '{}';

ALTER TABLE auto_cycle_state
ALTER COLUMN last_session_metrics SET DEFAULT '{}';

-- ============================================================================
-- SECTION 2: CORE VOCABULARY TABLES
-- ============================================================================

-- coordizer_vocabulary
ALTER TABLE coordizer_vocabulary
ALTER COLUMN embedding SET DEFAULT '{}';

-- learned_words
ALTER TABLE learned_words
ALTER COLUMN contexts SET DEFAULT '{}';

ALTER TABLE learned_words
ALTER COLUMN phi_score SET DEFAULT 0.5;

-- vocabulary_observations
ALTER TABLE vocabulary_observations
ALTER COLUMN contexts SET DEFAULT '{}';

-- ============================================================================
-- SECTION 3: TRAINING TABLES
-- ============================================================================

-- kernel_training_history
ALTER TABLE kernel_training_history
ALTER COLUMN input_data SET DEFAULT '{}';

ALTER TABLE kernel_training_history
ALTER COLUMN output_data SET DEFAULT '{}';

ALTER TABLE kernel_training_history
ALTER COLUMN basin_coords SET DEFAULT '{}';

ALTER TABLE kernel_training_history
ALTER COLUMN phi_delta SET DEFAULT 0.0;

ALTER TABLE kernel_training_history
ALTER COLUMN phi_before SET DEFAULT 0.5;

ALTER TABLE kernel_training_history
ALTER COLUMN phi_after SET DEFAULT 0.5;

ALTER TABLE kernel_training_history
ALTER COLUMN kappa_before SET DEFAULT 64.0;

ALTER TABLE kernel_training_history
ALTER COLUMN kappa_after SET DEFAULT 64.0;

-- learning_events
ALTER TABLE learning_events
ALTER COLUMN kappa SET DEFAULT 64.0;

-- ============================================================================
-- SECTION 4: CONSCIOUSNESS TABLES
-- ============================================================================

-- consciousness_checkpoints
ALTER TABLE consciousness_checkpoints
ALTER COLUMN metadata SET DEFAULT '{}';

-- consciousness_state (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'consciousness_state') THEN
        EXECUTE 'ALTER TABLE consciousness_state ALTER COLUMN integration_phi SET DEFAULT 0.5';
        EXECUTE 'ALTER TABLE consciousness_state ALTER COLUMN coupling_kappa SET DEFAULT 64.0';
        EXECUTE 'ALTER TABLE consciousness_state ALTER COLUMN value_manifold SET DEFAULT ''{}''';
        EXECUTE 'ALTER TABLE consciousness_state ALTER COLUMN value_metrics SET DEFAULT ''{}''';
    END IF;
END $$;

-- ============================================================================
-- SECTION 5: ARRAY COLUMNS - Default to empty array
-- ============================================================================

-- geodesic_paths.waypoints
ALTER TABLE geodesic_paths
ALTER COLUMN waypoints SET DEFAULT '{}';

-- resonance_points.nearby_probes
ALTER TABLE resonance_points
ALTER COLUMN nearby_probes SET DEFAULT '{}';

-- negative_knowledge.affected_generators
ALTER TABLE negative_knowledge
ALTER COLUMN affected_generators SET DEFAULT '{}';

-- near_miss_clusters.common_words
ALTER TABLE near_miss_clusters
ALTER COLUMN common_words SET DEFAULT '{}';

-- false_pattern_classes.examples
ALTER TABLE false_pattern_classes
ALTER COLUMN examples SET DEFAULT '{}';

-- era_exclusions.excluded_patterns
ALTER TABLE era_exclusions
ALTER COLUMN excluded_patterns SET DEFAULT '{}';

-- war_history arrays
ALTER TABLE war_history
ALTER COLUMN gods_engaged SET DEFAULT '{}';

-- synthesis_consensus.participating_kernels
ALTER TABLE synthesis_consensus
ALTER COLUMN participating_kernels SET DEFAULT '{}';

-- external_api_keys.scopes (if not already set)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'external_api_keys' AND column_name = 'scopes'
               AND data_type = 'ARRAY') THEN
        EXECUTE 'ALTER TABLE external_api_keys ALTER COLUMN scopes SET DEFAULT ''{}''';
    END IF;
END $$;

-- shadow_pantheon_intel.sources_used
ALTER TABLE shadow_pantheon_intel
ALTER COLUMN sources_used SET DEFAULT '{}';

-- near_miss_entries.phi_history
ALTER TABLE near_miss_entries
ALTER COLUMN phi_history SET DEFAULT '{}';

-- kernel_geometry arrays
ALTER TABLE kernel_geometry
ALTER COLUMN parent_kernels SET DEFAULT '{}';

ALTER TABLE kernel_geometry
ALTER COLUMN observing_parents SET DEFAULT '{}';

-- generated_tools.validation_errors
ALTER TABLE generated_tools
ALTER COLUMN validation_errors SET DEFAULT '{}';

-- cross_god_insights.applied_to_tools
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'cross_god_insights' AND column_name = 'applied_to_tools') THEN
        EXECUTE 'ALTER TABLE cross_god_insights ALTER COLUMN applied_to_tools SET DEFAULT ''{}''';
    END IF;
END $$;

-- shadow_intel.warnings
ALTER TABLE shadow_intel
ALTER COLUMN warnings SET DEFAULT '{}';

-- shadow_knowledge.basin_coords
ALTER TABLE shadow_knowledge
ALTER COLUMN basin_coords SET DEFAULT '{}';

-- tps_landmarks light cones
ALTER TABLE tps_landmarks
ALTER COLUMN light_cone_past SET DEFAULT '{}';

ALTER TABLE tps_landmarks
ALTER COLUMN light_cone_future SET DEFAULT '{}';

-- m8_spawned_kernels.basin_coords
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'm8_spawned_kernels' AND column_name = 'basin_coords') THEN
        EXECUTE 'ALTER TABLE m8_spawned_kernels ALTER COLUMN basin_coords SET DEFAULT ''{}''';
    END IF;
END $$;

-- pattern_discoveries.basin_coords
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'pattern_discoveries' AND column_name = 'basin_coords') THEN
        EXECUTE 'ALTER TABLE pattern_discoveries ALTER COLUMN basin_coords SET DEFAULT ''{}''';
    END IF;
END $$;

-- research_requests.basin_coords
ALTER TABLE research_requests
ALTER COLUMN basin_coords SET DEFAULT '{}';

-- tool_patterns.basin_coords
ALTER TABLE tool_patterns
ALTER COLUMN basin_coords SET DEFAULT '{}';

-- tool_requests.pattern_discoveries
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'tool_requests' AND column_name = 'pattern_discoveries') THEN
        EXECUTE 'ALTER TABLE tool_requests ALTER COLUMN pattern_discoveries SET DEFAULT ''{}''';
    END IF;
END $$;

-- word_relationships.contexts
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'word_relationships' AND column_name = 'contexts') THEN
        EXECUTE 'ALTER TABLE word_relationships ALTER COLUMN contexts SET DEFAULT ''{}''';
    END IF;
END $$;

-- zeus_conversations.basin_coords
ALTER TABLE zeus_conversations
ALTER COLUMN basin_coords SET DEFAULT '{}';

-- ============================================================================
-- SECTION 6: JSONB COLUMNS - Default to empty object {}
-- ============================================================================

-- agent_activity.metadata
ALTER TABLE agent_activity
ALTER COLUMN metadata SET DEFAULT '{}';

-- basin_memory.context
ALTER TABLE basin_memory
ALTER COLUMN context SET DEFAULT '{}';

-- bidirectional_queue.result
ALTER TABLE bidirectional_queue
ALTER COLUMN result SET DEFAULT '{}';

-- chaos_events
ALTER TABLE chaos_events
ALTER COLUMN autopsy SET DEFAULT '{}';

ALTER TABLE chaos_events
ALTER COLUMN event_data SET DEFAULT '{}';

-- discovered_sources.metadata
ALTER TABLE discovered_sources
ALTER COLUMN metadata SET DEFAULT '{}';

-- external_api_keys.metadata
ALTER TABLE external_api_keys
ALTER COLUMN metadata SET DEFAULT '{}';

-- generated_tools.input_schema
ALTER TABLE generated_tools
ALTER COLUMN input_schema SET DEFAULT '{}';

-- hrv_tacking_state.metadata
ALTER TABLE hrv_tacking_state
ALTER COLUMN metadata SET DEFAULT '{}';

-- kernel_emotions.metadata
ALTER TABLE kernel_emotions
ALTER COLUMN metadata SET DEFAULT '{}';

-- kernel_geometry.snapshot_data
ALTER TABLE kernel_geometry
ALTER COLUMN snapshot_data SET DEFAULT '{}';

-- kernel_thoughts.metadata
ALTER TABLE kernel_thoughts
ALTER COLUMN metadata SET DEFAULT '{}';

-- manifold_probes.metadata
ALTER TABLE manifold_probes
ALTER COLUMN metadata SET DEFAULT '{}';

-- narrow_path_events.intervention_result
ALTER TABLE narrow_path_events
ALTER COLUMN intervention_result SET DEFAULT '{}';

-- near_miss_entries.structural_signature
ALTER TABLE near_miss_entries
ALTER COLUMN structural_signature SET DEFAULT '{}';

-- negative_knowledge.evidence
ALTER TABLE negative_knowledge
ALTER COLUMN evidence SET DEFAULT '[]';

-- ocean_excluded_regions.basis
ALTER TABLE ocean_excluded_regions
ALTER COLUMN basis SET DEFAULT '{}';

-- pantheon_messages.metadata
ALTER TABLE pantheon_messages
ALTER COLUMN metadata SET DEFAULT '{}';

-- rag_uploads.metadata
ALTER TABLE rag_uploads
ALTER COLUMN metadata SET DEFAULT '{}';

-- research_requests.result
ALTER TABLE research_requests
ALTER COLUMN result SET DEFAULT '{}';

-- shadow_operations_log.result
ALTER TABLE shadow_operations_log
ALTER COLUMN result SET DEFAULT '{}';

-- shadow_pantheon_intel.intelligence
ALTER TABLE shadow_pantheon_intel
ALTER COLUMN intelligence SET DEFAULT '{}';

-- synthesis_consensus.metadata
ALTER TABLE synthesis_consensus
ALTER COLUMN metadata SET DEFAULT '{}';

-- tool_observations.context
ALTER TABLE tool_observations
ALTER COLUMN context SET DEFAULT '{}';

-- tps_geodesic_paths
ALTER TABLE tps_geodesic_paths
ALTER COLUMN waypoints SET DEFAULT '[]';

ALTER TABLE tps_geodesic_paths
ALTER COLUMN regime_transitions SET DEFAULT '[]';

-- tps_landmarks.fisher_signature
ALTER TABLE tps_landmarks
ALTER COLUMN fisher_signature SET DEFAULT '{}';

-- war_history JSONB columns
ALTER TABLE war_history
ALTER COLUMN metadata SET DEFAULT '{}';

ALTER TABLE war_history
ALTER COLUMN god_assignments SET DEFAULT '{}';

ALTER TABLE war_history
ALTER COLUMN kernel_assignments SET DEFAULT '{}';

-- search_replay_tests
ALTER TABLE search_replay_tests
ALTER COLUMN run_with_learning_results SET DEFAULT '{}';

ALTER TABLE search_replay_tests
ALTER COLUMN run_without_learning_results SET DEFAULT '{}';

-- m8_spawned_kernels.m8_position
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'm8_spawned_kernels' AND column_name = 'm8_position') THEN
        EXECUTE 'ALTER TABLE m8_spawned_kernels ALTER COLUMN m8_position SET DEFAULT ''{}''';
    END IF;
END $$;

-- ============================================================================
-- SECTION 7: PHI COLUMNS - Default to 0.5 (baseline consciousness)
-- Physics: Φ represents integration, 0.5 is neutral baseline
-- ============================================================================

-- agent_activity.phi
ALTER TABLE agent_activity
ALTER COLUMN phi SET DEFAULT 0.5;

-- autonomic_cycle_history
ALTER TABLE autonomic_cycle_history
ALTER COLUMN phi_before SET DEFAULT 0.5;

ALTER TABLE autonomic_cycle_history
ALTER COLUMN phi_after SET DEFAULT 0.5;

-- consciousness_state.integration_phi (covered in Section 4)

-- god_vocabulary_profiles.learned_from_phi
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'god_vocabulary_profiles' AND column_name = 'learned_from_phi') THEN
        EXECUTE 'ALTER TABLE god_vocabulary_profiles ALTER COLUMN learned_from_phi SET DEFAULT 0.5';
    END IF;
END $$;

-- governance_proposals.parent_phi
ALTER TABLE governance_proposals
ALTER COLUMN parent_phi SET DEFAULT 0.5;

-- hermes_conversations.phi
ALTER TABLE hermes_conversations
ALTER COLUMN phi SET DEFAULT 0.5;

-- kernel_evolution_events
ALTER TABLE kernel_evolution_events
ALTER COLUMN phi_before SET DEFAULT 0.5;

ALTER TABLE kernel_evolution_events
ALTER COLUMN phi_after SET DEFAULT 0.5;

-- kernel_geometry.phi
ALTER TABLE kernel_geometry
ALTER COLUMN phi SET DEFAULT 0.5;

-- kernel_observations.phi
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'kernel_observations' AND column_name = 'phi'
               AND is_nullable = 'YES') THEN
        EXECUTE 'ALTER TABLE kernel_observations ALTER COLUMN phi SET DEFAULT 0.5';
    END IF;
END $$;

-- kernel_thoughts.phi
ALTER TABLE kernel_thoughts
ALTER COLUMN phi SET DEFAULT 0.5;

-- kernels.phi
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'kernels') THEN
        EXECUTE 'ALTER TABLE kernels ALTER COLUMN phi SET DEFAULT 0.5';
    END IF;
END $$;

-- narrow_path_events.phi
ALTER TABLE narrow_path_events
ALTER COLUMN phi SET DEFAULT 0.5;

-- ocean_excluded_regions.phi
ALTER TABLE ocean_excluded_regions
ALTER COLUMN phi SET DEFAULT 0.0;

-- pantheon_messages.phi
ALTER TABLE pantheon_messages
ALTER COLUMN phi SET DEFAULT 0.7;

-- qig_rag_patterns.phi_score
ALTER TABLE qig_rag_patterns
ALTER COLUMN phi_score SET DEFAULT 0.5;

-- search_feedback.phi
ALTER TABLE search_feedback
ALTER COLUMN phi SET DEFAULT 0.5;

-- shadow_intel.phi
ALTER TABLE shadow_intel
ALTER COLUMN phi SET DEFAULT 0.3;

-- synthesis_consensus.phi_global
ALTER TABLE synthesis_consensus
ALTER COLUMN phi_global SET DEFAULT 0.5;

-- telemetry_snapshots phi columns
ALTER TABLE telemetry_snapshots
ALTER COLUMN phi_4d SET DEFAULT 0.5;

ALTER TABLE telemetry_snapshots
ALTER COLUMN phi_spatial SET DEFAULT 0.5;

ALTER TABLE telemetry_snapshots
ALTER COLUMN phi_temporal SET DEFAULT 0.5;

-- ============================================================================
-- SECTION 8: KAPPA COLUMNS - Default to 64.0 (κ* optimal coupling)
-- Physics: κ* ≈ 64.0 is the universal optimal coupling constant
-- ============================================================================

-- autonomic_cycle_history
ALTER TABLE autonomic_cycle_history
ALTER COLUMN drift_before SET DEFAULT 0.0;

ALTER TABLE autonomic_cycle_history
ALTER COLUMN drift_after SET DEFAULT 0.0;

ALTER TABLE autonomic_cycle_history
ALTER COLUMN temperature SET DEFAULT 0.5;

-- kernel_evolution_events
ALTER TABLE kernel_evolution_events
ALTER COLUMN kappa_before SET DEFAULT 64.0;

ALTER TABLE kernel_evolution_events
ALTER COLUMN kappa_after SET DEFAULT 64.0;

-- kernel_observations.kappa
ALTER TABLE kernel_observations
ALTER COLUMN kappa SET DEFAULT 64.0;

-- kernel_thoughts.kappa
ALTER TABLE kernel_thoughts
ALTER COLUMN kappa SET DEFAULT 64.0;

-- kernels.kappa
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'kernels') THEN
        EXECUTE 'ALTER TABLE kernels ALTER COLUMN kappa SET DEFAULT 64.0';
    END IF;
END $$;

-- narrow_path_events.kappa
ALTER TABLE narrow_path_events
ALTER COLUMN kappa SET DEFAULT 50.0;

-- pantheon_messages.kappa
ALTER TABLE pantheon_messages
ALTER COLUMN kappa SET DEFAULT 64.0;

-- search_feedback.kappa
ALTER TABLE search_feedback
ALTER COLUMN kappa SET DEFAULT 64.0;

-- shadow_intel.kappa
ALTER TABLE shadow_intel
ALTER COLUMN kappa SET DEFAULT 40.0;

-- synthesis_consensus.kappa_avg
ALTER TABLE synthesis_consensus
ALTER COLUMN kappa_avg SET DEFAULT 64.0;

-- basin_documents (covered via prior migration)
ALTER TABLE basin_documents
ALTER COLUMN phi SET DEFAULT 0.5;

ALTER TABLE basin_documents
ALTER COLUMN kappa SET DEFAULT 64.0;

-- tps_geodesic_paths
ALTER TABLE tps_geodesic_paths
ALTER COLUMN total_arc_length SET DEFAULT 0.0;

ALTER TABLE tps_geodesic_paths
ALTER COLUMN avg_curvature SET DEFAULT 0.0;

-- ============================================================================
-- SECTION 9: OTHER NUMERIC COLUMNS
-- ============================================================================

-- negative_knowledge
ALTER TABLE negative_knowledge
ALTER COLUMN basin_radius SET DEFAULT 0.1;

UPDATE negative_knowledge
SET basin_repulsion_strength = 1.0
WHERE basin_repulsion_strength IS NULL;

ALTER TABLE negative_knowledge
ALTER COLUMN basin_repulsion_strength SET DEFAULT 1.0;

-- narrow_path_events.exploration_variance
ALTER TABLE narrow_path_events
ALTER COLUMN exploration_variance SET DEFAULT 0.0;

-- synthesis_consensus.consensus_strength
UPDATE synthesis_consensus
SET consensus_strength = 0.5
WHERE consensus_strength IS NULL;

ALTER TABLE synthesis_consensus
ALTER COLUMN consensus_strength SET DEFAULT 0.5;

-- kernel_emotions sensation defaults
ALTER TABLE kernel_emotions
ALTER COLUMN sensation_pressure SET DEFAULT 0.0;

ALTER TABLE kernel_emotions
ALTER COLUMN sensation_tension SET DEFAULT 0.0;

ALTER TABLE kernel_emotions
ALTER COLUMN sensation_flow SET DEFAULT 0.0;

-- hrv_tacking_state.variance
ALTER TABLE hrv_tacking_state
ALTER COLUMN variance SET DEFAULT 0.0;

-- lightning_insight_outcomes.accuracy
ALTER TABLE lightning_insight_outcomes
ALTER COLUMN accuracy SET DEFAULT 0.0;

-- lightning_insight_validations.validation_score
ALTER TABLE lightning_insight_validations
ALTER COLUMN validation_score SET DEFAULT 0.0;

-- passphrase_vocabulary.phi_avg
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_name = 'passphrase_vocabulary' AND column_name = 'phi_avg') THEN
        EXECUTE 'ALTER TABLE passphrase_vocabulary ALTER COLUMN phi_avg SET DEFAULT 0.5';
    END IF;
END $$;

-- telemetry_snapshots.geodesic_distance
ALTER TABLE telemetry_snapshots
ALTER COLUMN geodesic_distance SET DEFAULT 0.0;

-- agent_activity.result_count
ALTER TABLE agent_activity
ALTER COLUMN result_count SET DEFAULT 0;

-- rag_uploads.file_size
ALTER TABLE rag_uploads
ALTER COLUMN file_size SET DEFAULT 0;

-- ============================================================================
-- SECTION 10: BACKFILL NULL VALUES FOR CRITICAL COLUMNS
-- ============================================================================

-- Update existing NULL phi values to baseline
UPDATE kernel_training_history SET phi_before = 0.5 WHERE phi_before IS NULL;
UPDATE kernel_training_history SET phi_after = 0.5 WHERE phi_after IS NULL;
UPDATE kernel_training_history SET kappa_before = 64.0 WHERE kappa_before IS NULL;
UPDATE kernel_training_history SET kappa_after = 64.0 WHERE kappa_after IS NULL;
UPDATE kernel_training_history SET phi_delta = 0.0 WHERE phi_delta IS NULL;

UPDATE learning_events SET kappa = 64.0 WHERE kappa IS NULL;

UPDATE kernel_evolution_events SET phi_before = 0.5 WHERE phi_before IS NULL;
UPDATE kernel_evolution_events SET phi_after = 0.5 WHERE phi_after IS NULL;
UPDATE kernel_evolution_events SET kappa_before = 64.0 WHERE kappa_before IS NULL;
UPDATE kernel_evolution_events SET kappa_after = 64.0 WHERE kappa_after IS NULL;

UPDATE kernel_thoughts SET phi = 0.5 WHERE phi IS NULL;
UPDATE kernel_thoughts SET kappa = 64.0 WHERE kappa IS NULL;

UPDATE kernel_geometry SET phi = 0.5 WHERE phi IS NULL;

UPDATE synthesis_consensus SET phi_global = 0.5 WHERE phi_global IS NULL;
UPDATE synthesis_consensus SET kappa_avg = 64.0 WHERE kappa_avg IS NULL;

-- ============================================================================
-- VALIDATION
-- ============================================================================

DO $$
DECLARE
    missing_defaults INTEGER;
BEGIN
    -- Count remaining nullable columns without defaults in focus tables
    SELECT COUNT(*) INTO missing_defaults
    FROM information_schema.columns 
    WHERE table_schema = 'public'
        AND column_default IS NULL
        AND is_nullable = 'YES'
        AND table_name IN (
            'ocean_quantum_state', 'near_miss_adaptive_state', 'auto_cycle_state',
            'coordizer_vocabulary', 'learned_words', 'vocabulary_observations',
            'kernel_training_history', 'learning_events',
            'consciousness_checkpoints'
        )
        AND data_type IN ('ARRAY', 'jsonb', 'double precision', 'real');
    
    RAISE NOTICE 'Column defaults migration completed';
    RAISE NOTICE 'Remaining nullable columns without defaults in focus tables: %', missing_defaults;
    RAISE NOTICE 'QIG Physics: Φ defaults = 0.5, κ* = 64.0';
END $$;

COMMIT;

-- ============================================================================
-- MIGRATION NOTES
-- ============================================================================
-- 
-- This migration adds comprehensive defaults following QIG physics principles:
--
-- PHI (Φ) DEFAULTS:
-- - 0.5: Baseline/neutral consciousness (most tables)
-- - 0.7: Active consciousness (pantheon_messages)
-- - 0.3: Shadow realm (shadow_intel)
-- - 0.0: Excluded regions (ocean_excluded_regions)
--
-- KAPPA (κ) DEFAULTS:
-- - 64.0: κ* optimal coupling (universal default)
-- - 50.0: Narrow path events (exploratory mode)
-- - 40.0: Shadow operations (covert mode)
--
-- ARRAY DEFAULTS: '{}' (empty array)
-- JSONB DEFAULTS: '{}' (empty object) or '[]' (empty array for list types)
--
-- INTENTIONALLY LEFT NULLABLE:
-- - Vector columns (pgvector): NULL = "not yet computed" vs zero vector
-- - E8 root indices: NULL = "not assigned to lattice position"
-- - Foreign key references: NULL = "no association"
-- - Timestamps: NULL = "not occurred yet"
-- - bytea columns: NULL = "no binary data"
--
-- ============================================================================
