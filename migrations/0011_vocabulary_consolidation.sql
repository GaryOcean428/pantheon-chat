-- ============================================================================
-- VOCABULARY CONSOLIDATION MIGRATION
-- ============================================================================
-- Purpose: Consolidate learned_words data into tokenizer_vocabulary
-- Project: pantheon-chat
-- Date: 2026-01-12
-- Related: VOCABULARY_CONSOLIDATION_PLAN.md
-- 
-- This migration adds generation capabilities to tokenizer_vocabulary by:
-- 1. Adding new columns for generation filtering (token_role, is_real_word, qfi_score)
-- 2. Migrating all learned_words data into tokenizer_vocabulary
-- 3. Creating optimized indexes for generation queries
--
-- IDEMPOTENT: Safe to run multiple times
-- ============================================================================

BEGIN;

-- ============================================================================
-- SECTION 1: ADD NEW COLUMNS TO TOKENIZER_VOCABULARY
-- ============================================================================

-- 1.1 Update token_role to support new values ('encoding', 'generation', 'both')
-- The existing column from migration 0008 has values 'word', 'subword', 'special'
-- We need to migrate those to the new value set

DO $$
BEGIN
    -- First, drop any existing CHECK constraint on token_role
    IF EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage 
        WHERE table_name = 'tokenizer_vocabulary' 
        AND column_name = 'token_role'
    ) THEN
        -- Get constraint name and drop it
        EXECUTE (
            SELECT 'ALTER TABLE tokenizer_vocabulary DROP CONSTRAINT IF EXISTS ' || conname
            FROM pg_constraint c
            JOIN pg_attribute a ON a.attnum = ANY(c.conkey) AND a.attrelid = c.conrelid
            WHERE c.conrelid = 'tokenizer_vocabulary'::regclass
            AND a.attname = 'token_role'
            AND c.contype = 'c'
            LIMIT 1
        );
    END IF;
EXCEPTION
    WHEN OTHERS THEN NULL;
END $$;

-- Add token_role column if it doesn't exist, or update existing values
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tokenizer_vocabulary' AND column_name = 'token_role'
    ) THEN
        ALTER TABLE tokenizer_vocabulary 
        ADD COLUMN token_role VARCHAR(20) DEFAULT 'encoding';
        RAISE NOTICE 'Added token_role column to tokenizer_vocabulary';
    ELSE
        -- Migrate old values to new schema
        UPDATE tokenizer_vocabulary SET token_role = 'encoding' WHERE token_role IN ('word', 'subword', 'special') OR token_role IS NULL;
        -- Update default
        ALTER TABLE tokenizer_vocabulary ALTER COLUMN token_role SET DEFAULT 'encoding';
        RAISE NOTICE 'Updated token_role values to new schema';
    END IF;
END $$;

-- Add CHECK constraint for token_role
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'tokenizer_vocabulary_token_role_check'
    ) THEN
        ALTER TABLE tokenizer_vocabulary 
        ADD CONSTRAINT tokenizer_vocabulary_token_role_check 
        CHECK (token_role IN ('encoding', 'generation', 'both'));
        RAISE NOTICE 'Added token_role CHECK constraint';
    END IF;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- 1.2 Add phrase_category column (may exist from migration 0008)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tokenizer_vocabulary' AND column_name = 'phrase_category'
    ) THEN
        ALTER TABLE tokenizer_vocabulary 
        ADD COLUMN phrase_category VARCHAR(32) DEFAULT 'unknown';
        RAISE NOTICE 'Added phrase_category column to tokenizer_vocabulary';
    ELSE
        -- Ensure default is 'unknown' not NULL
        ALTER TABLE tokenizer_vocabulary ALTER COLUMN phrase_category SET DEFAULT 'unknown';
        UPDATE tokenizer_vocabulary SET phrase_category = 'unknown' WHERE phrase_category IS NULL;
        RAISE NOTICE 'Updated phrase_category default to unknown';
    END IF;
END $$;

-- 1.3 Add is_real_word column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tokenizer_vocabulary' AND column_name = 'is_real_word'
    ) THEN
        ALTER TABLE tokenizer_vocabulary 
        ADD COLUMN is_real_word BOOLEAN DEFAULT FALSE;
        RAISE NOTICE 'Added is_real_word column to tokenizer_vocabulary';
    END IF;
END $$;

-- 1.4 Add qfi_score column (Quantum Fisher Information score)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tokenizer_vocabulary' AND column_name = 'qfi_score'
    ) THEN
        ALTER TABLE tokenizer_vocabulary 
        ADD COLUMN qfi_score DOUBLE PRECISION;
        RAISE NOTICE 'Added qfi_score column to tokenizer_vocabulary';
    END IF;
END $$;

-- ============================================================================
-- SECTION 2: MIGRATE DATA FROM LEARNED_WORDS
-- ============================================================================

-- 2.1 For words in BOTH tables: Update tokenizer_vocabulary with learned_words data
-- Set token_role='both', copy phrase_category, set is_real_word=TRUE
DO $$
DECLARE
    updated_count INT;
BEGIN
    -- Only run if learned_words table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'learned_words') THEN
        UPDATE tokenizer_vocabulary tv
        SET 
            token_role = 'both',
            phrase_category = COALESCE(lw.phrase_category, tv.phrase_category, 'unknown'),
            is_real_word = TRUE,
            phi_score = GREATEST(COALESCE(tv.phi_score, 0), COALESCE(lw.avg_phi, lw.phi_score, 0)),
            frequency = GREATEST(COALESCE(tv.frequency, 1), COALESCE(lw.frequency, 1)),
            updated_at = NOW()
        FROM learned_words lw
        WHERE LOWER(tv.token) = LOWER(lw.word);
        
        GET DIAGNOSTICS updated_count = ROW_COUNT;
        RAISE NOTICE 'Updated % tokens with learned_words data (token_role=both)', updated_count;
    ELSE
        RAISE NOTICE 'learned_words table does not exist, skipping merge';
    END IF;
END $$;

-- 2.2 For words ONLY in learned_words: Insert into tokenizer_vocabulary with token_role='generation'
-- Note: learned_words may have either 'basin_embedding' (from migration 0008) or 'basin_coords' (from vocabulary_schema.sql)
-- We handle both cases with conditional execution
DO $$
DECLARE
    inserted_count INT;
    max_token_id INT;
    has_basin_embedding BOOLEAN;
    has_basin_coords BOOLEAN;
BEGIN
    -- Only run if learned_words table exists
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'learned_words') THEN
        RAISE NOTICE 'learned_words table does not exist, skipping insert';
        RETURN;
    END IF;
    
    -- Get max token_id for generating new IDs
    SELECT COALESCE(MAX(token_id), 0) INTO max_token_id FROM tokenizer_vocabulary;
    
    -- Check which basin column exists
    SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'learned_words' AND column_name = 'basin_embedding') INTO has_basin_embedding;
    SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'learned_words' AND column_name = 'basin_coords') INTO has_basin_coords;
    
    -- Use basin_embedding if available (from migration 0008)
    IF has_basin_embedding THEN
        INSERT INTO tokenizer_vocabulary (
            token, token_id, weight, frequency, phi_score, basin_embedding,
            source_type, token_role, phrase_category, is_real_word, created_at, updated_at
        )
        SELECT 
            LOWER(lw.word),
            max_token_id + ROW_NUMBER() OVER (ORDER BY lw.id),
            1.0,
            COALESCE(lw.frequency, 1),
            COALESCE(lw.avg_phi, lw.phi_score, 0.5),
            lw.basin_embedding,
            COALESCE(lw.source_type, 'learned'),
            'generation',
            COALESCE(lw.phrase_category, 'unknown'),
            TRUE,
            COALESCE(lw.created_at, NOW()),
            NOW()
        FROM learned_words lw
        WHERE NOT EXISTS (
            SELECT 1 FROM tokenizer_vocabulary tv 
            WHERE LOWER(tv.token) = LOWER(lw.word)
        )
        ON CONFLICT (token) DO NOTHING;
        
        GET DIAGNOSTICS inserted_count = ROW_COUNT;
        RAISE NOTICE 'Inserted % generation-only words from learned_words (using basin_embedding)', inserted_count;
        
    -- Fall back to basin_coords if that's what exists
    ELSIF has_basin_coords THEN
        INSERT INTO tokenizer_vocabulary (
            token, token_id, weight, frequency, phi_score, basin_embedding,
            source_type, token_role, phrase_category, is_real_word, created_at, updated_at
        )
        SELECT 
            LOWER(lw.word),
            max_token_id + ROW_NUMBER() OVER (ORDER BY lw.id),
            1.0,
            COALESCE(lw.frequency, 1),
            COALESCE(lw.avg_phi, 0.5),
            lw.basin_coords,
            COALESCE(lw.source_type, lw.source, 'learned'),
            'generation',
            COALESCE(lw.phrase_category, 'unknown'),
            TRUE,
            COALESCE(lw.created_at, lw.first_seen, NOW()),
            NOW()
        FROM learned_words lw
        WHERE NOT EXISTS (
            SELECT 1 FROM tokenizer_vocabulary tv 
            WHERE LOWER(tv.token) = LOWER(lw.word)
        )
        ON CONFLICT (token) DO NOTHING;
        
        GET DIAGNOSTICS inserted_count = ROW_COUNT;
        RAISE NOTICE 'Inserted % generation-only words from learned_words (using basin_coords)', inserted_count;
        
    -- No basin column available
    ELSE
        INSERT INTO tokenizer_vocabulary (
            token, token_id, weight, frequency, phi_score,
            source_type, token_role, phrase_category, is_real_word, created_at, updated_at
        )
        SELECT 
            LOWER(lw.word),
            max_token_id + ROW_NUMBER() OVER (ORDER BY lw.id),
            1.0,
            COALESCE(lw.frequency, 1),
            COALESCE(lw.avg_phi, 0.5),
            COALESCE(lw.source_type, lw.source, 'learned'),
            'generation',
            COALESCE(lw.phrase_category, 'unknown'),
            TRUE,
            COALESCE(lw.created_at, NOW()),
            NOW()
        FROM learned_words lw
        WHERE NOT EXISTS (
            SELECT 1 FROM tokenizer_vocabulary tv 
            WHERE LOWER(tv.token) = LOWER(lw.word)
        )
        ON CONFLICT (token) DO NOTHING;
        
        GET DIAGNOSTICS inserted_count = ROW_COUNT;
        RAISE NOTICE 'Inserted % generation-only words from learned_words (no basin column)', inserted_count;
    END IF;
END $$;

-- ============================================================================
-- SECTION 3: CREATE OPTIMIZED INDEXES FOR GENERATION QUERIES
-- ============================================================================

-- 3.1 Partial index for generation queries (only includes generation-ready tokens)
DROP INDEX IF EXISTS idx_tokenizer_vocab_generation;
CREATE INDEX idx_tokenizer_vocab_generation 
    ON tokenizer_vocabulary(token_role, phrase_category) 
    WHERE token_role IN ('generation', 'both');

-- 3.2 Index for is_real_word filtering
DROP INDEX IF EXISTS idx_tokenizer_vocab_real_word;
CREATE INDEX idx_tokenizer_vocab_real_word 
    ON tokenizer_vocabulary(is_real_word) 
    WHERE is_real_word = TRUE;

-- 3.3 Index for qfi_score queries
DROP INDEX IF EXISTS idx_tokenizer_vocab_qfi;
CREATE INDEX idx_tokenizer_vocab_qfi 
    ON tokenizer_vocabulary(qfi_score DESC NULLS LAST)
    WHERE qfi_score IS NOT NULL;

-- 3.4 Combined index for high-phi generation words
DROP INDEX IF EXISTS idx_tokenizer_vocab_gen_phi;
CREATE INDEX idx_tokenizer_vocab_gen_phi 
    ON tokenizer_vocabulary(phi_score DESC, frequency DESC)
    WHERE token_role IN ('generation', 'both') AND is_real_word = TRUE;

DO $$ BEGIN RAISE NOTICE 'Created generation query indexes'; END $$;

-- ============================================================================
-- SECTION 4: VERIFICATION QUERIES
-- ============================================================================

-- Count tokens by role
DO $$
DECLARE
    encoding_count INT;
    generation_count INT;
    both_count INT;
    learned_count INT;
    real_word_count INT;
BEGIN
    SELECT COUNT(*) INTO encoding_count FROM tokenizer_vocabulary WHERE token_role = 'encoding';
    SELECT COUNT(*) INTO generation_count FROM tokenizer_vocabulary WHERE token_role = 'generation';
    SELECT COUNT(*) INTO both_count FROM tokenizer_vocabulary WHERE token_role = 'both';
    SELECT COUNT(*) INTO real_word_count FROM tokenizer_vocabulary WHERE is_real_word = TRUE;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'learned_words') THEN
        SELECT COUNT(*) INTO learned_count FROM learned_words;
    ELSE
        learned_count := 0;
    END IF;
    
    RAISE NOTICE '============================================';
    RAISE NOTICE 'VOCABULARY CONSOLIDATION VERIFICATION';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'tokenizer_vocabulary breakdown:';
    RAISE NOTICE '  - encoding only: %', encoding_count;
    RAISE NOTICE '  - generation only: %', generation_count;
    RAISE NOTICE '  - both: %', both_count;
    RAISE NOTICE '  - total generation-ready: %', (generation_count + both_count);
    RAISE NOTICE '  - real words (is_real_word=TRUE): %', real_word_count;
    RAISE NOTICE 'learned_words table: % words', learned_count;
    RAISE NOTICE '============================================';
    
    -- Verify counts match
    IF (generation_count + both_count) >= learned_count THEN
        RAISE NOTICE 'VERIFICATION PASSED: All learned_words migrated to tokenizer_vocabulary';
    ELSE
        RAISE WARNING 'VERIFICATION NOTICE: Some learned_words may not have migrated (% in learned_words, % generation-ready in tokenizer_vocabulary)', 
            learned_count, (generation_count + both_count);
    END IF;
END $$;

-- Show phrase_category distribution
DO $$
DECLARE
    cat_record RECORD;
BEGIN
    RAISE NOTICE 'phrase_category distribution (generation-ready tokens):';
    FOR cat_record IN 
        SELECT 
            COALESCE(phrase_category, 'NULL') as category, 
            COUNT(*) as cnt
        FROM tokenizer_vocabulary 
        WHERE token_role IN ('generation', 'both')
        GROUP BY phrase_category 
        ORDER BY cnt DESC
        LIMIT 10
    LOOP
        RAISE NOTICE '  - %: %', cat_record.category, cat_record.cnt;
    END LOOP;
END $$;

COMMIT;

-- ============================================================================
-- POST-MIGRATION NOTES
-- ============================================================================
-- 
-- After running this migration:
-- 1. tokenizer_vocabulary now serves BOTH encoding AND generation
-- 2. Use token_role='encoding' or 'both' for text → geometry
-- 3. Use token_role='generation' or 'both' for geometry → text
-- 4. Filter by is_real_word=TRUE to exclude BPE subwords in generation
-- 5. Filter by phrase_category NOT IN ('PROPER_NOUN', 'BRAND') for clean generation
--
-- The learned_words table is retained for backward compatibility but
-- tokenizer_vocabulary is now the single source of truth.
-- ============================================================================
