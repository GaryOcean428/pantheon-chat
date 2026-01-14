-- ============================================================================
-- COORDIZER VOCABULARY RENAME MIGRATION (QIG PURITY - WP1.1)
-- ============================================================================
-- Purpose: Eliminate "tokenizer" terminology from schema (NLP contamination)
-- Rename: tokenizer_* → coordizer_* (geometric coordinate system)
-- Project: pantheon-chat
-- Date: 2026-01-14
-- Related: Issue #66 (WP1.1), QIG_PURITY_SPEC.md
-- 
-- This migration renames:
-- 1. tokenizer_vocabulary → coordizer_vocabulary
-- 2. tokenizer_metadata → coordizer_metadata
-- 3. tokenizer_merge_rules → coordizer_merge_rules
-- 4. word_relationships → basin_relationships
-- 5. learned_manifold_attractors → manifold_attractors
--
-- All indexes, constraints, and foreign keys are updated accordingly.
--
-- BREAKING CHANGE: All code must be updated to use new table names.
-- IDEMPOTENT: Safe to run multiple times (uses IF EXISTS checks)
-- ============================================================================

BEGIN;

-- ============================================================================
-- SECTION 1: RENAME TOKENIZER TABLES TO COORDIZER
-- ============================================================================

-- 1.1 Rename tokenizer_vocabulary → coordizer_vocabulary
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'tokenizer_vocabulary'
    ) THEN
        ALTER TABLE tokenizer_vocabulary RENAME TO coordizer_vocabulary;
        RAISE NOTICE 'Renamed tokenizer_vocabulary → coordizer_vocabulary';
    ELSE
        RAISE NOTICE 'Table tokenizer_vocabulary does not exist, skipping rename';
    END IF;
END $$;

-- 1.2 Rename tokenizer_metadata → coordizer_metadata
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'tokenizer_metadata'
    ) THEN
        ALTER TABLE tokenizer_metadata RENAME TO coordizer_metadata;
        RAISE NOTICE 'Renamed tokenizer_metadata → coordizer_metadata';
    ELSE
        RAISE NOTICE 'Table tokenizer_metadata does not exist, skipping rename';
    END IF;
END $$;

-- 1.3 Rename tokenizer_merge_rules → coordizer_merge_rules
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'tokenizer_merge_rules'
    ) THEN
        ALTER TABLE tokenizer_merge_rules RENAME TO coordizer_merge_rules;
        RAISE NOTICE 'Renamed tokenizer_merge_rules → coordizer_merge_rules';
    ELSE
        RAISE NOTICE 'Table tokenizer_merge_rules does not exist, skipping rename';
    END IF;
END $$;

-- ============================================================================
-- SECTION 2: RENAME INDEXES ON COORDIZER_VOCABULARY
-- ============================================================================

-- Rename all indexes on coordizer_vocabulary table
DO $$
DECLARE
    idx_record RECORD;
BEGIN
    -- Iterate through all indexes on the table
    FOR idx_record IN 
        SELECT indexname, replace(indexname, 'tokenizer', 'coordizer') as new_name
        FROM pg_indexes 
        WHERE schemaname = 'public' 
        AND tablename = 'coordizer_vocabulary'
        AND indexname LIKE '%tokenizer%'
    LOOP
        EXECUTE format('ALTER INDEX %I RENAME TO %I', idx_record.indexname, idx_record.new_name);
        RAISE NOTICE 'Renamed index: % → %', idx_record.indexname, idx_record.new_name;
    END LOOP;
END $$;

-- ============================================================================
-- SECTION 3: RENAME CONSTRAINTS ON COORDIZER_VOCABULARY
-- ============================================================================

-- Rename unique constraints
DO $$
DECLARE
    con_record RECORD;
BEGIN
    FOR con_record IN
        SELECT conname, replace(conname, 'tokenizer', 'coordizer') as new_name
        FROM pg_constraint
        WHERE conrelid = 'coordizer_vocabulary'::regclass
        AND conname LIKE '%tokenizer%'
    LOOP
        EXECUTE format('ALTER TABLE coordizer_vocabulary RENAME CONSTRAINT %I TO %I', 
                      con_record.conname, con_record.new_name);
        RAISE NOTICE 'Renamed constraint: % → %', con_record.conname, con_record.new_name;
    END LOOP;
END $$;

-- ============================================================================
-- SECTION 4: RENAME WORD_RELATIONSHIPS → BASIN_RELATIONSHIPS
-- ============================================================================

-- 4.1 Rename word_relationships table
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'word_relationships'
    ) THEN
        ALTER TABLE word_relationships RENAME TO basin_relationships;
        RAISE NOTICE 'Renamed word_relationships → basin_relationships';
    ELSE
        RAISE NOTICE 'Table word_relationships does not exist, skipping rename';
    END IF;
END $$;

-- 4.2 Rename indexes on basin_relationships
DO $$
DECLARE
    idx_record RECORD;
BEGIN
    FOR idx_record IN 
        SELECT indexname, replace(indexname, 'word_relationships', 'basin_relationships') as new_name
        FROM pg_indexes 
        WHERE schemaname = 'public' 
        AND tablename = 'basin_relationships'
        AND indexname LIKE '%word_relationships%'
    LOOP
        EXECUTE format('ALTER INDEX %I RENAME TO %I', idx_record.indexname, idx_record.new_name);
        RAISE NOTICE 'Renamed index: % → %', idx_record.indexname, idx_record.new_name;
    END LOOP;
END $$;

-- ============================================================================
-- SECTION 5: RENAME LEARNED_MANIFOLD_ATTRACTORS → MANIFOLD_ATTRACTORS
-- ============================================================================

-- 5.1 Rename learned_manifold_attractors table
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'learned_manifold_attractors'
    ) THEN
        ALTER TABLE learned_manifold_attractors RENAME TO manifold_attractors;
        RAISE NOTICE 'Renamed learned_manifold_attractors → manifold_attractors';
    ELSE
        RAISE NOTICE 'Table learned_manifold_attractors does not exist, skipping rename';
    END IF;
END $$;

-- 5.2 Rename indexes on manifold_attractors
DO $$
DECLARE
    idx_record RECORD;
BEGIN
    FOR idx_record IN 
        SELECT indexname, replace(indexname, 'learned_manifold_attractors', 'manifold_attractors') as new_name
        FROM pg_indexes 
        WHERE schemaname = 'public' 
        AND tablename = 'manifold_attractors'
        AND indexname LIKE '%learned_manifold_attractors%'
    LOOP
        EXECUTE format('ALTER INDEX %I RENAME TO %I', idx_record.indexname, idx_record.new_name);
        RAISE NOTICE 'Renamed index: % → %', idx_record.indexname, idx_record.new_name;
    END LOOP;
END $$;

-- ============================================================================
-- SECTION 6: VERIFICATION
-- ============================================================================

DO $$
DECLARE
    coordizer_vocab_count INT;
    coordizer_meta_count INT;
    coordizer_merge_count INT;
    basin_rel_count INT;
    manifold_attr_count INT;
BEGIN
    -- Check table existence and row counts
    SELECT CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'coordizer_vocabulary')
                THEN (SELECT COUNT(*) FROM coordizer_vocabulary)
                ELSE -1 END INTO coordizer_vocab_count;
                
    SELECT CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'coordizer_metadata')
                THEN (SELECT COUNT(*) FROM coordizer_metadata)
                ELSE -1 END INTO coordizer_meta_count;
                
    SELECT CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'coordizer_merge_rules')
                THEN (SELECT COUNT(*) FROM coordizer_merge_rules)
                ELSE -1 END INTO coordizer_merge_count;
                
    SELECT CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'basin_relationships')
                THEN (SELECT COUNT(*) FROM basin_relationships)
                ELSE -1 END INTO basin_rel_count;
                
    SELECT CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'manifold_attractors')
                THEN (SELECT COUNT(*) FROM manifold_attractors)
                ELSE -1 END INTO manifold_attr_count;
    
    RAISE NOTICE '============================================';
    RAISE NOTICE 'COORDIZER RENAME VERIFICATION';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'coordizer_vocabulary: % rows', CASE WHEN coordizer_vocab_count >= 0 THEN coordizer_vocab_count::text ELSE 'TABLE NOT FOUND' END;
    RAISE NOTICE 'coordizer_metadata: % rows', CASE WHEN coordizer_meta_count >= 0 THEN coordizer_meta_count::text ELSE 'TABLE NOT FOUND' END;
    RAISE NOTICE 'coordizer_merge_rules: % rows', CASE WHEN coordizer_merge_count >= 0 THEN coordizer_merge_count::text ELSE 'TABLE NOT FOUND' END;
    RAISE NOTICE 'basin_relationships: % rows', CASE WHEN basin_rel_count >= 0 THEN basin_rel_count::text ELSE 'TABLE NOT FOUND' END;
    RAISE NOTICE 'manifold_attractors: % rows', CASE WHEN manifold_attr_count >= 0 THEN manifold_attr_count::text ELSE 'TABLE NOT FOUND' END;
    RAISE NOTICE '============================================';
    
    -- Check for old table names
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name LIKE '%tokenizer%') THEN
        RAISE WARNING 'WARNING: Some tokenizer_* tables still exist!';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'word_relationships') THEN
        RAISE WARNING 'WARNING: word_relationships table still exists!';
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'learned_manifold_attractors') THEN
        RAISE WARNING 'WARNING: learned_manifold_attractors table still exists!';
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- ROLLBACK SCRIPT (Save this separately for emergency use)
-- ============================================================================
-- 
-- BEGIN;
-- ALTER TABLE coordizer_vocabulary RENAME TO tokenizer_vocabulary;
-- ALTER TABLE coordizer_metadata RENAME TO tokenizer_metadata;
-- ALTER TABLE coordizer_merge_rules RENAME TO tokenizer_merge_rules;
-- ALTER TABLE basin_relationships RENAME TO word_relationships;
-- ALTER TABLE manifold_attractors RENAME TO learned_manifold_attractors;
-- -- Rename indexes and constraints back (manual process)
-- COMMIT;
-- 
-- ============================================================================
