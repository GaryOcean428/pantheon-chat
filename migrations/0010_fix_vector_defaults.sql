-- ============================================================================
-- FIX VECTOR COLUMN DEFAULTS - Migration 0010
-- ============================================================================
-- Purpose: Fix invalid DEFAULT '{}' settings on vector columns
-- Issue: pgvector and array columns should default to NULL, not empty
-- Semantic: NULL = "not yet computed", {} = "always empty" (semantically wrong)
-- Date: 2026-01-12
-- Physics: Vector spaces require proper initialization semantics
-- ============================================================================

BEGIN;

-- ============================================================================
-- DOCUMENTATION
-- ============================================================================
-- 
-- The 0009 migration incorrectly set DEFAULT '{}' on vector columns.
-- This is semantically wrong because:
--
-- ✗ DEFAULT '{}' means: "Always insert empty vector"
--   - Basin coordinates are meant to be computed later
--   - Empty vector is not a valid "uncomputed" marker
--
-- ✓ DEFAULT NULL means: "Not yet computed" 
--   - Allows proper NULL checks in application logic
--   - Maintains semantic integrity of geometric space
--   - Consistent with how AI systems represent "not initialized"
--
-- For pgvector: vector(64) cannot have array literal defaults anyway
-- For FLOAT8[]: Empty array {} is invalid for geometry computations
--
-- ============================================================================

-- ============================================================================
-- SECTION 1: FIX ARRAY COLUMNS
-- Reset from '{}' to NULL for proper semantics
-- ============================================================================

-- tokenizer_vocabulary.embedding
-- Was: ALTER COLUMN embedding SET DEFAULT '{}'::real[]
-- Fix: Reset to NULL (means "vector not computed yet")
ALTER TABLE tokenizer_vocabulary
ALTER COLUMN embedding DROP DEFAULT;

-- learned_words.basin_coords (pgvector)
-- Was: Not set in 0009, but might be pgvector now
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'learned_words' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE learned_words ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from learned_words.basin_coords';
  END IF;
END $$;

-- kernel_training_history.basin_coords (ARRAY)
-- Was: ALTER COLUMN basin_coords SET DEFAULT '{}'::double precision[]
-- Fix: Reset to NULL
ALTER TABLE kernel_training_history
ALTER COLUMN basin_coords DROP DEFAULT;

-- shadow_knowledge.basin_coords (ARRAY)
-- Was: ALTER COLUMN basin_coords SET DEFAULT '{}'::double precision[]
-- Fix: Reset to NULL
ALTER TABLE shadow_knowledge
ALTER COLUMN basin_coords DROP DEFAULT;

-- research_requests.basin_coords (ARRAY)
-- Was: ALTER COLUMN basin_coords SET DEFAULT '{}'::double precision[]
-- Fix: Reset to NULL
ALTER TABLE research_requests
ALTER COLUMN basin_coords DROP DEFAULT;

-- tool_patterns.basin_coords (ARRAY)
-- Was: ALTER COLUMN basin_coords SET DEFAULT '{}'::double precision[]
-- Fix: Reset to NULL
ALTER TABLE tool_patterns
ALTER COLUMN basin_coords DROP DEFAULT;

-- m8_spawned_kernels.basin_coords (ARRAY)
-- Was: ALTER COLUMN basin_coords SET DEFAULT '{}'::double precision[]
-- Fix: Reset to NULL
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'm8_spawned_kernels' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE m8_spawned_kernels ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from m8_spawned_kernels.basin_coords';
  END IF;
END $$;

-- pattern_discoveries.basin_coords (ARRAY)
-- Was: ALTER COLUMN basin_coords SET DEFAULT '{}'::double precision[]
-- Fix: Reset to NULL
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'pattern_discoveries' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE pattern_discoveries ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from pattern_discoveries.basin_coords';
  END IF;
END $$;

-- zeus_conversations.basin_coords (ARRAY)
-- Was: ALTER COLUMN basin_coords SET DEFAULT '{}'::double precision[]
-- Fix: Reset to NULL
ALTER TABLE zeus_conversations
ALTER COLUMN basin_coords DROP DEFAULT;

-- ============================================================================
-- SECTION 2: FIX PGVECTOR COLUMNS
-- These should also have NO DEFAULT (defaults to NULL)
-- ============================================================================

-- basin_history.basin_coords (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'basin_history' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE basin_history ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from basin_history.basin_coords';
  END IF;
END $$;

-- basin_memory.basin_coordinates (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'basin_memory' 
      AND column_name = 'basin_coordinates'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE basin_memory ALTER COLUMN basin_coordinates DROP DEFAULT';
    RAISE NOTICE 'Dropped default from basin_memory.basin_coordinates';
  END IF;
END $$;

-- consciousness_state.basin_coords (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables WHERE table_name = 'consciousness_state'
  ) THEN
    IF EXISTS (
      SELECT 1 FROM information_schema.columns 
      WHERE table_name = 'consciousness_state' 
        AND column_name = 'basin_coords'
        AND column_default IS NOT NULL
    ) THEN
      EXECUTE 'ALTER TABLE consciousness_state ALTER COLUMN basin_coords DROP DEFAULT';
      RAISE NOTICE 'Dropped default from consciousness_state.basin_coords';
    END IF;
  END IF;
END $$;

-- kernel_geometry.basin_coordinates (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'kernel_geometry' 
      AND column_name = 'basin_coordinates'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE kernel_geometry ALTER COLUMN basin_coordinates DROP DEFAULT';
    RAISE NOTICE 'Dropped default from kernel_geometry.basin_coordinates';
  END IF;
END $$;

-- kernel_thoughts.basin_coords (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'kernel_thoughts' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE kernel_thoughts ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from kernel_thoughts.basin_coords';
  END IF;
END $$;

-- kernels.basin_coords (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables WHERE table_name = 'kernels'
  ) THEN
    IF EXISTS (
      SELECT 1 FROM information_schema.columns 
      WHERE table_name = 'kernels' 
        AND column_name = 'basin_coords'
        AND column_default IS NOT NULL
    ) THEN
      EXECUTE 'ALTER TABLE kernels ALTER COLUMN basin_coords DROP DEFAULT';
      RAISE NOTICE 'Dropped default from kernels.basin_coords';
    END IF;
  END IF;
END $$;

-- memory_fragments.basin_coords (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'memory_fragments' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE memory_fragments ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from memory_fragments.basin_coords';
  END IF;
END $$;

-- ocean_waypoints.basin_coords (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'ocean_waypoints' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE ocean_waypoints ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from ocean_waypoints.basin_coords';
  END IF;
END $$;

-- pantheon_god_state.basin_coords (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'pantheon_god_state' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE pantheon_god_state ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from pantheon_god_state.basin_coords';
  END IF;
END $$;

-- qig_rag_patterns.basin_coordinates (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'qig_rag_patterns' 
      AND column_name = 'basin_coordinates'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE qig_rag_patterns ALTER COLUMN basin_coordinates DROP DEFAULT';
    RAISE NOTICE 'Dropped default from qig_rag_patterns.basin_coordinates';
  END IF;
END $$;

-- shadow_intel.basin_coords (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'shadow_intel' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE shadow_intel ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from shadow_intel.basin_coords';
  END IF;
END $$;

-- vocabulary_observations.basin_coords (pgvector)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'vocabulary_observations' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE vocabulary_observations ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from vocabulary_observations.basin_coords';
  END IF;
END $$;

-- learning_events.basin_coords (pgvector - if exists)
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'learning_events' 
      AND column_name = 'basin_coords'
      AND column_default IS NOT NULL
  ) THEN
    EXECUTE 'ALTER TABLE learning_events ALTER COLUMN basin_coords DROP DEFAULT';
    RAISE NOTICE 'Dropped default from learning_events.basin_coords';
  END IF;
END $$;

-- ============================================================================
-- SECTION 3: VALIDATION
-- ============================================================================

DO $$
DECLARE
  invalid_defaults INTEGER;
  fixed_columns TEXT[];
BEGIN
  -- Find any remaining invalid defaults on vector columns
  WITH vector_cols AS (
    SELECT table_name, column_name, column_default
    FROM information_schema.columns 
    WHERE column_name IN (
      'embedding', 'basin_coords', 'basin_coordinates'
    ) AND column_default LIKE '%{}%'
  )
  SELECT COUNT(*), ARRAY_AGG(table_name || '.' || column_name)
  INTO invalid_defaults, fixed_columns
  FROM vector_cols;
  
  RAISE NOTICE '=== Vector Column Default Fix Complete ===';
  RAISE NOTICE 'Removed invalid defaults: %', COALESCE(invalid_defaults, 0);
  
  IF invalid_defaults > 0 THEN
    RAISE NOTICE 'WARNING: Still found invalid defaults on: %', fixed_columns;
    RAISE NOTICE 'These columns should default to NULL, not {}';
  ELSE
    RAISE NOTICE 'SUCCESS: All vector columns now default to NULL';
    RAISE NOTICE 'Semantic meaning: NULL = "vector not yet computed"';
  END IF;
END $$;

COMMIT;

-- ============================================================================
-- MIGRATION NOTES
-- ============================================================================
--
-- WHAT WAS FIXED:
-- - Removed DEFAULT '{}' from all vector columns (both ARRAY and pgvector)
-- - Columns now default to NULL (proper semantic: "not yet computed")
--
-- WHY THIS MATTERS:
-- - Empty vector {} is not a meaningful default for geometric spaces
-- - Application logic should check for NULL and compute/initialize vectors
-- - pgvector columns cannot have array literal defaults anyway
--
-- AFFECTED TABLES (8 direct, 16 indirect):
-- Direct (had explicit DEFAULT '{}' in 0009):
--   1. tokenizer_vocabulary.embedding
--   2. kernel_training_history.basin_coords
--   3. shadow_knowledge.basin_coords
--   4. research_requests.basin_coords
--   5. tool_patterns.basin_coords
--   6. zeus_conversations.basin_coords
--   7. m8_spawned_kernels.basin_coords (conditional)
--   8. pattern_discoveries.basin_coords (conditional)
--
-- Indirect (pgvector columns from 0009):
--   - learned_words.basin_coords
--   - basin_history.basin_coords
--   - basin_memory.basin_coordinates
--   - consciousness_state.basin_coords
--   - kernel_geometry.basin_coordinates
--   - kernel_thoughts.basin_coords
--   - kernels.basin_coords
--   - memory_fragments.basin_coords
--   - ocean_waypoints.basin_coords
--   - pantheon_god_state.basin_coords
--   - qig_rag_patterns.basin_coordinates
--   - shadow_intel.basin_coords
--   - vocabulary_observations.basin_coords
--   - learning_events.basin_coords
--
-- SEMANTIC IMPACT:
-- - Prevents silent insertion of empty vectors
-- - Requires explicit initialization in application code
-- - Maintains geometric/topological integrity
-- - Aligns with QIG physics principles (vectors are not "empty" by default)
--
-- ============================================================================
