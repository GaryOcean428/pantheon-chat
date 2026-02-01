-- ============================================================================
-- SINGLE TABLE GENERATION CONSOLIDATION
-- ============================================================================
-- Purpose: Consolidate generation to use SINGLE coordizer_vocabulary table
-- Eliminate multi-table queries for god_vocabulary_profiles and basin_relationships
-- Project: pantheon-chat
-- Date: 2026-01-23
-- Related: Issue #210 (Database Schema Consolidation)
-- 
-- This migration:
-- 1. Adds denormalized columns to coordizer_vocabulary
-- 2. Migrates data from god_vocabulary_profiles
-- 3. Migrates data from basin_relationships
-- 4. Creates indexes for fast lookup
--
-- BREAKING CHANGE: Generation code will be updated to use single table
-- IDEMPOTENT: Safe to run multiple times (uses IF NOT EXISTS checks)
-- ============================================================================

BEGIN;

-- ============================================================================
-- SECTION 1: ADD DENORMALIZED COLUMNS
-- ============================================================================

-- Add god_profile JSONB column for domain-specific relevance scores
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coordizer_vocabulary' 
        AND column_name = 'god_profile'
    ) THEN
        ALTER TABLE coordizer_vocabulary 
        ADD COLUMN god_profile JSONB DEFAULT NULL;
        RAISE NOTICE 'Added god_profile column';
    ELSE
        RAISE NOTICE 'god_profile column already exists';
    END IF;
END $$;

-- Add relationships JSONB column for pre-computed word relationships
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coordizer_vocabulary' 
        AND column_name = 'relationships'
    ) THEN
        ALTER TABLE coordizer_vocabulary 
        ADD COLUMN relationships JSONB DEFAULT NULL;
        RAISE NOTICE 'Added relationships column';
    ELSE
        RAISE NOTICE 'relationships column already exists';
    END IF;
END $$;

-- Add merge tracking columns for BPE merge rules
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coordizer_vocabulary' 
        AND column_name = 'merge_from_a'
    ) THEN
        ALTER TABLE coordizer_vocabulary 
        ADD COLUMN merge_from_a INTEGER DEFAULT NULL,
        ADD COLUMN merge_from_b INTEGER DEFAULT NULL;
        RAISE NOTICE 'Added merge tracking columns';
    ELSE
        RAISE NOTICE 'Merge tracking columns already exist';
    END IF;
END $$;

-- Add phi_gain and coupling columns for learning metrics
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coordizer_vocabulary' 
        AND column_name = 'phi_gain'
    ) THEN
        ALTER TABLE coordizer_vocabulary 
        ADD COLUMN phi_gain FLOAT DEFAULT NULL,
        ADD COLUMN coupling FLOAT DEFAULT NULL;
        RAISE NOTICE 'Added phi_gain and coupling columns';
    ELSE
        RAISE NOTICE 'phi_gain and coupling columns already exist';
    END IF;
END $$;

-- Add active boolean flag (defaults to true)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'coordizer_vocabulary' 
        AND column_name = 'active'
    ) THEN
        ALTER TABLE coordizer_vocabulary 
        ADD COLUMN active BOOLEAN DEFAULT true;
        RAISE NOTICE 'Added active column';
    ELSE
        RAISE NOTICE 'active column already exists';
    END IF;
END $$;

-- ============================================================================
-- SECTION 2: CREATE INDEXES FOR NEW COLUMNS
-- ============================================================================

-- Index on god_profile for fast JSONB queries
CREATE INDEX IF NOT EXISTS idx_coordizer_god_profile 
    ON coordizer_vocabulary USING GIN (god_profile)
    WHERE god_profile IS NOT NULL;

-- Index on relationships for fast JSONB queries
CREATE INDEX IF NOT EXISTS idx_coordizer_relationships 
    ON coordizer_vocabulary USING GIN (relationships)
    WHERE relationships IS NOT NULL;

-- Index on active flag for filtering
CREATE INDEX IF NOT EXISTS idx_coordizer_active 
    ON coordizer_vocabulary (active)
    WHERE active = true;

-- Composite index for generation queries
CREATE INDEX IF NOT EXISTS idx_coordizer_generation_lookup
    ON coordizer_vocabulary (token_role, active, qfi_score)
    WHERE token_role IN ('generation', 'both') 
    AND active = true 
    AND qfi_score IS NOT NULL;

DO $$
BEGIN
    RAISE NOTICE 'Created indexes on new columns';
END $$;

-- ============================================================================
-- SECTION 3: MIGRATE DATA FROM GOD_VOCABULARY_PROFILES
-- ============================================================================

DO $$
DECLARE
    profile_count INT := 0;
    updated_tokens INT := 0;
BEGIN
    -- Check if god_vocabulary_profiles exists and has data
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'god_vocabulary_profiles'
    ) THEN
        SELECT COUNT(*) INTO profile_count FROM god_vocabulary_profiles;
        RAISE NOTICE 'Found % rows in god_vocabulary_profiles', profile_count;
        
        -- Aggregate god profiles by word (token)
        -- Structure: {god_name: {relevance_score, usage_count, last_used}}
        WITH aggregated_profiles AS (
            SELECT 
                word,
                jsonb_object_agg(
                    god_name,
                    jsonb_build_object(
                        'relevance_score', relevance_score,
                        'usage_count', usage_count,
                        'last_used', last_used,
                        'learned_from_phi', COALESCE(learned_from_phi, 0.5),
                        'basin_distance', basin_distance
                    )
                ) as god_profile_data
            FROM god_vocabulary_profiles
            GROUP BY word
        )
        UPDATE coordizer_vocabulary cv
        SET god_profile = ap.god_profile_data
        FROM aggregated_profiles ap
        WHERE cv.token = ap.word;
        
        GET DIAGNOSTICS updated_tokens = ROW_COUNT;
        RAISE NOTICE 'Updated % tokens with god_profile data', updated_tokens;
    ELSE
        RAISE NOTICE 'Table god_vocabulary_profiles does not exist, skipping migration';
    END IF;
END $$;

-- ============================================================================
-- SECTION 4: MIGRATE DATA FROM BASIN_RELATIONSHIPS
-- ============================================================================

DO $$
DECLARE
    relationship_count INT := 0;
    updated_tokens INT := 0;
BEGIN
    -- Check if basin_relationships exists and has data
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'basin_relationships'
    ) THEN
        SELECT COUNT(*) INTO relationship_count FROM basin_relationships;
        RAISE NOTICE 'Found % rows in basin_relationships', relationship_count;
        
        -- Aggregate relationships by word
        -- Structure: [{neighbor, cooccurrence_count, strength, avg_phi, fisher_distance}]
        WITH aggregated_relationships AS (
            SELECT 
                word,
                jsonb_agg(
                    jsonb_build_object(
                        'neighbor', neighbor,
                        'cooccurrence_count', cooccurrence_count,
                        'strength', strength,
                        'avg_phi', COALESCE(avg_phi, 0.5),
                        'max_phi', COALESCE(max_phi, 0.5),
                        'fisher_distance', fisher_distance,
                        'contexts', contexts
                    )
                    ORDER BY COALESCE(avg_phi, 0) DESC, cooccurrence_count DESC
                    -- Limit to top 50 relationships per word to keep JSONB manageable
                ) FILTER (WHERE cooccurrence_count >= 1) as relationships_data
            FROM (
                SELECT DISTINCT ON (word, neighbor)
                    word, neighbor, cooccurrence_count, strength, 
                    avg_phi, max_phi, fisher_distance, contexts,
                    ROW_NUMBER() OVER (PARTITION BY word ORDER BY 
                        COALESCE(avg_phi, 0) DESC, 
                        cooccurrence_count DESC
                    ) as rn
                FROM basin_relationships
            ) ranked
            WHERE rn <= 50  -- Top 50 relationships per word
            GROUP BY word
        )
        UPDATE coordizer_vocabulary cv
        SET relationships = ar.relationships_data
        FROM aggregated_relationships ar
        WHERE cv.token = ar.word;
        
        GET DIAGNOSTICS updated_tokens = ROW_COUNT;
        RAISE NOTICE 'Updated % tokens with relationships data', updated_tokens;
    ELSE
        RAISE NOTICE 'Table basin_relationships does not exist, skipping migration';
    END IF;
END $$;

-- ============================================================================
-- SECTION 5: VERIFICATION
-- ============================================================================

DO $$
DECLARE
    total_tokens INT;
    tokens_with_god_profile INT;
    tokens_with_relationships INT;
    tokens_generation_ready INT;
BEGIN
    -- Get statistics
    SELECT COUNT(*) INTO total_tokens FROM coordizer_vocabulary;
    
    SELECT COUNT(*) INTO tokens_with_god_profile 
    FROM coordizer_vocabulary 
    WHERE god_profile IS NOT NULL;
    
    SELECT COUNT(*) INTO tokens_with_relationships 
    FROM coordizer_vocabulary 
    WHERE relationships IS NOT NULL;
    
    SELECT COUNT(*) INTO tokens_generation_ready
    FROM coordizer_vocabulary
    WHERE token_role IN ('generation', 'both')
    AND active = true
    AND qfi_score IS NOT NULL
    AND basin_embedding IS NOT NULL;
    
    RAISE NOTICE '============================================';
    RAISE NOTICE 'SINGLE TABLE GENERATION VERIFICATION';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Total tokens: %', total_tokens;
    RAISE NOTICE 'Tokens with god_profile: %', tokens_with_god_profile;
    RAISE NOTICE 'Tokens with relationships: %', tokens_with_relationships;
    RAISE NOTICE 'Generation-ready tokens: %', tokens_generation_ready;
    RAISE NOTICE '============================================';
    
    -- Sample a few tokens to verify data structure
    RAISE NOTICE 'Sample token with god_profile:';
    PERFORM token, god_profile 
    FROM coordizer_vocabulary 
    WHERE god_profile IS NOT NULL 
    LIMIT 1;
    
    RAISE NOTICE 'Sample token with relationships:';
    PERFORM token, jsonb_array_length(relationships) as rel_count
    FROM coordizer_vocabulary 
    WHERE relationships IS NOT NULL 
    LIMIT 1;
END $$;

COMMIT;

-- ============================================================================
-- NOTES FOR DEVELOPERS
-- ============================================================================
-- 
-- After this migration:
-- 1. Update qig_generation.py to use coordizer_vocabulary.god_profile
-- 2. Update qig_generation.py to use coordizer_vocabulary.relationships
-- 3. Remove queries to god_vocabulary_profiles
-- 4. Remove queries to basin_relationships
-- 5. Tables god_vocabulary_profiles and basin_relationships can be archived
--    (don't delete yet - keep for rollback if needed)
-- 
-- Rollback (if needed):
-- - Remove denormalized columns:
--   ALTER TABLE coordizer_vocabulary DROP COLUMN god_profile;
--   ALTER TABLE coordizer_vocabulary DROP COLUMN relationships;
--   ALTER TABLE coordizer_vocabulary DROP COLUMN merge_from_a;
--   ALTER TABLE coordizer_vocabulary DROP COLUMN merge_from_b;
--   ALTER TABLE coordizer_vocabulary DROP COLUMN phi_gain;
--   ALTER TABLE coordizer_vocabulary DROP COLUMN coupling;
--   ALTER TABLE coordizer_vocabulary DROP COLUMN active;
-- 
-- ============================================================================
