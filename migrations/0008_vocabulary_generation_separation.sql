-- Migration: Vocabulary Generation Separation
-- Separates tokenizer encoding vocabulary from generation vocabulary
-- Adds token_role, phrase_category filtering, and fixes shadow_operations_state

-- ============================================================================
-- PART 1: Add token_role column to tokenizer_vocabulary
-- ============================================================================
-- This distinguishes BPE subwords from actual words for future-proofing
-- Values: 'word', 'subword', 'special'

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tokenizer_vocabulary' AND column_name = 'token_role'
    ) THEN
        ALTER TABLE tokenizer_vocabulary 
        ADD COLUMN token_role VARCHAR(16) DEFAULT 'word';
        
        -- Mark special tokens
        UPDATE tokenizer_vocabulary 
        SET token_role = 'special' 
        WHERE source_type = 'special';
        
        -- Create index for filtering
        CREATE INDEX IF NOT EXISTS idx_tokenizer_vocabulary_role 
        ON tokenizer_vocabulary(token_role);
        
        RAISE NOTICE 'Added token_role column to tokenizer_vocabulary';
    END IF;
END $$;

-- ============================================================================
-- PART 2: Add phrase_category column to tokenizer_vocabulary
-- ============================================================================
-- This allows filtering out PROPER_NOUN, BRAND, etc. from generation

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tokenizer_vocabulary' AND column_name = 'phrase_category'
    ) THEN
        ALTER TABLE tokenizer_vocabulary 
        ADD COLUMN phrase_category VARCHAR(32) DEFAULT NULL;
        
        -- Create index for filtering
        CREATE INDEX IF NOT EXISTS idx_tokenizer_vocabulary_category 
        ON tokenizer_vocabulary(phrase_category);
        
        RAISE NOTICE 'Added phrase_category column to tokenizer_vocabulary';
    END IF;
END $$;

-- ============================================================================
-- PART 3: Delete garbage tokens from tokenizer_vocabulary
-- ============================================================================
-- Remove known garbage patterns: ffffff, fpdxwd, tysctnyzry, etc.

DELETE FROM tokenizer_vocabulary
WHERE token IN (
    'ffffff', 'fpdxwd', 'tysctnyzry', 'xxxx', 'zzzz', 'aaaa', 'bbbb', 'cccc',
    'dddd', 'eeee', 'ffff', 'gggg', 'hhhh', 'iiii', 'jjjj', 'kkkk', 'llll',
    'mmmm', 'nnnn', 'oooo', 'pppp', 'qqqq', 'rrrr', 'ssss', 'tttt', 'uuuu',
    'vvvv', 'wwww', 'yyyy'
);

-- Delete tokens with BPE markers
DELETE FROM tokenizer_vocabulary
WHERE token ~ '^[ĠġĊċ]'  -- Starts with BPE markers
   OR token LIKE '##%'     -- HuggingFace subword prefix
   OR token LIKE '▁%'      -- SentencePiece marker
   OR token LIKE '<%'      -- Special token markers
   OR token LIKE '[%'      -- Special token markers
   OR token ~ '^\d+$'      -- Pure numeric tokens
   OR token ~ '^[^a-zA-Z]+$'  -- No alphabetic characters
;

-- Delete very short tokens (likely fragments)
DELETE FROM tokenizer_vocabulary
WHERE LENGTH(token) < 2 AND source_type != 'special';

DO $$
BEGIN
    RAISE NOTICE 'Deleted garbage tokens from tokenizer_vocabulary';
END $$;

-- ============================================================================
-- PART 4: Fix shadow_operations_state PRIMARY KEY constraint
-- ============================================================================
-- Add composite PRIMARY KEY (god_name, state_type) if not exists

DO $$
BEGIN
    -- Check if primary key exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE table_name = 'shadow_operations_state' 
        AND constraint_type = 'PRIMARY KEY'
    ) THEN
        -- Add primary key constraint
        ALTER TABLE shadow_operations_state 
        ADD CONSTRAINT shadow_operations_state_pkey 
        PRIMARY KEY (god_name, state_type);
        
        RAISE NOTICE 'Added PRIMARY KEY constraint to shadow_operations_state';
    END IF;
END $$;

-- ============================================================================
-- PART 5: Create learned_words table for generation vocabulary
-- ============================================================================
-- This is the authoritative source for generation vocabulary
-- Separate from tokenizer_vocabulary which is for encoding only

CREATE TABLE IF NOT EXISTS learned_words (
    id SERIAL PRIMARY KEY,
    word TEXT NOT NULL UNIQUE,
    basin_embedding VECTOR(64) NOT NULL,
    phi_score DOUBLE PRECISION DEFAULT 0.5,
    frequency INTEGER DEFAULT 1,
    phrase_category VARCHAR(32) DEFAULT NULL,
    source_type VARCHAR(32) DEFAULT 'learned',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for learned_words
CREATE INDEX IF NOT EXISTS idx_learned_words_phi ON learned_words(phi_score DESC);
CREATE INDEX IF NOT EXISTS idx_learned_words_frequency ON learned_words(frequency DESC);
CREATE INDEX IF NOT EXISTS idx_learned_words_category ON learned_words(phrase_category);
CREATE INDEX IF NOT EXISTS idx_learned_words_last_used ON learned_words(last_used_at DESC);

-- Create pgvector index for fast nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_learned_words_basin_hnsw 
ON learned_words USING hnsw (basin_embedding vector_cosine_ops);

COMMENT ON TABLE learned_words IS 'Generation vocabulary - words used for text synthesis (separate from tokenizer_vocabulary encoding)';
COMMENT ON COLUMN learned_words.word IS 'Actual English word (validated, no BPE subwords)';
COMMENT ON COLUMN learned_words.basin_embedding IS '64D Fisher manifold coordinates for geometric operations';
COMMENT ON COLUMN learned_words.phi_score IS 'Integration score (Φ) - prefer high-Φ words in generation';
COMMENT ON COLUMN learned_words.phrase_category IS 'POS category from qig_phrase_classifier (NOUN, VERB, ADJECTIVE, etc.)';

-- ============================================================================
-- PART 6: Populate learned_words from tokenizer_vocabulary
-- ============================================================================
-- Copy valid words (not BPE subwords, not special tokens, not proper nouns)

INSERT INTO learned_words (word, basin_embedding, phi_score, frequency, phrase_category, source_type, created_at, updated_at)
SELECT 
    LOWER(token),  -- Normalize to lowercase for generation (case-insensitive matching)
    basin_embedding,
    phi_score,
    frequency,
    phrase_category,
    source_type,
    created_at,
    updated_at
FROM tokenizer_vocabulary
WHERE 
    -- Valid words only
    LENGTH(token) >= 2
    -- Match alphabetic tokens (both cases allowed in source, normalized to lowercase for generation)
    AND token ~ '^[a-zA-Z]+$'
    AND token_role = 'word'
    AND basin_embedding IS NOT NULL
    -- Exclude categories inappropriate for generation
    AND (phrase_category IS NULL OR phrase_category NOT IN ('PROPER_NOUN', 'BRAND'))
    -- Exclude special tokens
    AND source_type NOT IN ('special')
ON CONFLICT (word) DO UPDATE SET
    basin_embedding = EXCLUDED.basin_embedding,
    phi_score = GREATEST(learned_words.phi_score, EXCLUDED.phi_score),
    frequency = learned_words.frequency + EXCLUDED.frequency,
    updated_at = NOW();

DO $$
BEGIN
    RAISE NOTICE 'Populated learned_words table from tokenizer_vocabulary';
END $$;

-- ============================================================================
-- PART 7: Create view for generation-ready vocabulary
-- ============================================================================

CREATE OR REPLACE VIEW generation_vocabulary AS
SELECT 
    word,
    basin_embedding,
    phi_score,
    frequency,
    phrase_category,
    source_type
FROM learned_words
WHERE 
    phi_score > 0.0  -- Filter low-quality words
    AND LENGTH(word) >= 2
    AND (phrase_category IS NULL OR phrase_category NOT IN ('PROPER_NOUN', 'BRAND'))
ORDER BY phi_score DESC, frequency DESC;

COMMENT ON VIEW generation_vocabulary IS 'Filtered view of learned_words suitable for text generation';

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Count tokens by role
SELECT 
    token_role, 
    COUNT(*) as count,
    COUNT(CASE WHEN basin_embedding IS NOT NULL THEN 1 END) as with_embedding
FROM tokenizer_vocabulary 
GROUP BY token_role;

-- Count words by category
SELECT 
    phrase_category, 
    COUNT(*) as count
FROM learned_words 
GROUP BY phrase_category 
ORDER BY count DESC;

-- Shadow operations state verification
SELECT COUNT(*) as shadow_ops_records FROM shadow_operations_state;

-- Summary
DO $$
DECLARE
    tokenizer_count INT;
    learned_count INT;
    garbage_removed INT;
BEGIN
    SELECT COUNT(*) INTO tokenizer_count FROM tokenizer_vocabulary;
    SELECT COUNT(*) INTO learned_count FROM learned_words;
    
    RAISE NOTICE 'Migration complete:';
    RAISE NOTICE '  - tokenizer_vocabulary: % tokens (encoding)', tokenizer_count;
    RAISE NOTICE '  - learned_words: % words (generation)', learned_count;
    RAISE NOTICE '  - shadow_operations_state: PRIMARY KEY added';
END $$;
