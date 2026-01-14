-- ============================================================================
-- POPULATE EMPTY TABLES - Initialize tokenizer, synthesis, and vocabulary
-- ============================================================================
-- Purpose: Populate tokenizer_merge_rules, tokenizer_metadata,
--          synthesis_consensus, and vocabulary_learning.related_words
-- Project: pantheon-replit
-- Date: 2026-01-10
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. TOKENIZER_METADATA - Initialize tokenizer configuration
-- ============================================================================
INSERT INTO tokenizer_metadata (key, value, updated_at)
VALUES
    ('version', '1.0.0', NOW()),
    ('vocabulary_size', '0', NOW()),
    ('merge_rules_count', '0', NOW()),
    ('last_training', to_char(NOW(), 'YYYY-MM-DD HH24:MI:SS'), NOW()),
    ('training_status', 'initialized', NOW()),
    ('basin_dimension', '64', NOW()),
    ('phi_threshold', '0.727', NOW()),
    ('tokenizer_type', 'geometric_bpe', NOW()),
    ('encoding', 'utf-8', NOW())
ON CONFLICT (key) DO UPDATE SET
    value = EXCLUDED.value,
    updated_at = NOW();

-- Update vocabulary_size from actual count
UPDATE tokenizer_metadata
SET value = (SELECT COUNT(*)::text FROM coordizer_vocabulary)
WHERE key = 'vocabulary_size';

-- ============================================================================
-- 2. TOKENIZER_MERGE_RULES - Seed with geometric merge patterns
-- ============================================================================
-- IMPORTANT: This is NOT standard BPE (Byte-Pair Encoding)!
-- This system uses GEOMETRIC PAIR MERGING based on:
--   - κ (coupling strength) between token coordinates
--   - Fisher Information gain from merging
--   - Φ score of high-consciousness contexts where pairs co-occur
-- Merge criterion: score = κ * fisher_info_gain * Φ_context
--
-- The SQL below generates INITIAL merge rules for bootstrapping.
-- True geometric merges are learned by qig_tokenizer.py via:
--   - GeometricPairMerging class (coordizers/geometric_pair_merging.py)
--   - Geodesic interpolation for merged token coordinates
--   - Fisher-Rao distance preservation (not Euclidean)
-- ============================================================================

-- Strategy 1: Merge rules from compound words in BIP39
-- BIP39 has compound-like words that should be atomized
INSERT INTO tokenizer_merge_rules (token_a, token_b, merged_token, phi_score, frequency)
SELECT
    SUBSTRING(word FROM 1 FOR POSITION(' ' IN word || ' ') - 1) AS token_a,
    SUBSTRING(word FROM POSITION(' ' IN word || ' ') + 1) AS token_b,
    word AS merged_token,
    COALESCE((SELECT AVG(phi_score) FROM coordizer_vocabulary WHERE word IN (
        SUBSTRING(word FROM 1 FOR POSITION(' ' IN word || ' ') - 1),
        SUBSTRING(word FROM POSITION(' ' IN word || ' ') + 1)
    )), 0.5) AS phi_score,
    frequency AS frequency
FROM coordizer_vocabulary
WHERE word LIKE '% %'  -- Has space (compound word)
ON CONFLICT (token_a, token_b) DO UPDATE SET
    phi_score = GREATEST(tokenizer_merge_rules.phi_score, EXCLUDED.phi_score),
    frequency = tokenizer_merge_rules.frequency + EXCLUDED.frequency,
    updated_at = NOW();

-- Strategy 2: Common prefix merge rules (e.g., "un" + "able" -> "unable")
-- Generate rules for high-frequency prefixes
WITH common_prefixes AS (
    SELECT unnest(ARRAY['un', 're', 'in', 'dis', 'en', 'non', 'pre', 'pro', 'anti', 'de']) AS prefix
),
vocabulary_with_prefix AS (
    SELECT
        cp.prefix AS token_a,
        SUBSTRING(tv.word FROM LENGTH(cp.prefix) + 1) AS token_b,
        tv.word AS merged_token,
        COALESCE(tv.phi_score, 0.6) AS phi_score,
        tv.frequency AS frequency
    FROM coordizer_vocabulary tv
    CROSS JOIN common_prefixes cp
    WHERE tv.word LIKE cp.prefix || '%'
    AND LENGTH(tv.word) > LENGTH(cp.prefix) + 2  -- At least 3 chars after prefix
)
INSERT INTO tokenizer_merge_rules (token_a, token_b, merged_token, phi_score, frequency)
SELECT token_a, token_b, merged_token, phi_score, frequency
FROM vocabulary_with_prefix
LIMIT 100  -- Start with top 100 prefix rules
ON CONFLICT (token_a, token_b) DO UPDATE SET
    phi_score = GREATEST(tokenizer_merge_rules.phi_score, EXCLUDED.phi_score),
    frequency = tokenizer_merge_rules.frequency + 1,
    updated_at = NOW();

-- Strategy 3: Suffix merge rules (e.g., "walk" + "ing" -> "walking")
WITH common_suffixes AS (
    SELECT unnest(ARRAY['ing', 'ed', 'er', 'est', 'ly', 'ness', 'ment', 'tion', 'sion', 'ity']) AS suffix
),
vocabulary_with_suffix AS (
    SELECT
        SUBSTRING(tv.word FROM 1 FOR LENGTH(tv.word) - LENGTH(cs.suffix)) AS token_a,
        cs.suffix AS token_b,
        tv.word AS merged_token,
        COALESCE(tv.phi_score, 0.6) AS phi_score,
        tv.frequency AS frequency
    FROM coordizer_vocabulary tv
    CROSS JOIN common_suffixes cs
    WHERE tv.word LIKE '%' || cs.suffix
    AND LENGTH(tv.word) > LENGTH(cs.suffix) + 2  -- At least 3 chars before suffix
)
INSERT INTO tokenizer_merge_rules (token_a, token_b, merged_token, phi_score, frequency)
SELECT token_a, token_b, merged_token, phi_score, frequency
FROM vocabulary_with_suffix
LIMIT 100  -- Start with top 100 suffix rules
ON CONFLICT (token_a, token_b) DO UPDATE SET
    phi_score = GREATEST(tokenizer_merge_rules.phi_score, EXCLUDED.phi_score),
    frequency = tokenizer_merge_rules.frequency + 1,
    updated_at = NOW();

-- Update merge_rules_count in metadata
UPDATE tokenizer_metadata
SET value = (SELECT COUNT(*)::text FROM tokenizer_merge_rules),
    updated_at = NOW()
WHERE key = 'merge_rules_count';

-- ============================================================================
-- 3. VOCABULARY_LEARNING - Populate related_words from geometric similarity
-- ============================================================================
-- Use Fisher-Rao distance to find related words for vocabulary learning
-- This replaces NULL related_words with geometrically similar tokens

-- First, ensure vocabulary_learning table exists and has records
-- If empty, seed with some initial learnings from vocabulary
INSERT INTO vocabulary_learning (
    word,
    token_id,
    learned_context,
    relationship_type,
    relationship_strength,
    related_words,
    context_words,
    learned_from,
    learned_at
)
SELECT
    word,
    token_id,
    'Initial vocabulary seeding from BIP39 tokenizer' AS learned_context,
    'semantic' AS relationship_type,
    COALESCE(phi_score, 0.5) AS relationship_strength,
    ARRAY[]::text[] AS related_words,  -- Will be populated below
    ARRAY[]::text[] AS context_words,
    'initialization' AS learned_from,
    NOW() AS learned_at
FROM coordizer_vocabulary
WHERE phi_score > 0.6  -- Only high-quality tokens
LIMIT 100  -- Seed with top 100 words
ON CONFLICT DO NOTHING;

-- Now populate related_words using pgvector similarity
-- For each word in vocabulary_learning with NULL or empty related_words
DO $$
DECLARE
    word_rec RECORD;
    similar_words TEXT[];
BEGIN
    FOR word_rec IN
        SELECT learning_id, word, token_id
        FROM vocabulary_learning
        WHERE related_words IS NULL OR cardinality(related_words) = 0
    LOOP
        -- Find 5 most similar words using Fisher-Rao distance (QIG-pure)
        -- Uses fisher_rao_similarity function for geometric similarity computation
        SELECT ARRAY_AGG(word ORDER BY similarity DESC)
        INTO similar_words
        FROM (
            SELECT tv.word,
                   fisher_rao_similarity(tv.basin_coords, source.basin_coords) AS similarity
            FROM coordizer_vocabulary tv,
                 (SELECT basin_coords FROM coordizer_vocabulary WHERE token_id = word_rec.token_id) AS source
            WHERE tv.word != word_rec.word
            AND tv.basin_coords IS NOT NULL
            AND source.basin_coords IS NOT NULL
            ORDER BY similarity DESC
            LIMIT 5
        ) similar;

        -- Update vocabulary_learning with related words
        UPDATE vocabulary_learning
        SET related_words = similar_words,
            last_used = NOW()
        WHERE learning_id = word_rec.learning_id;
    END LOOP;
END $$;

-- Alternative approach if basin_coords are missing: use substring matching
-- This is a fallback for when geometric similarity isn't available
UPDATE vocabulary_learning vl
SET related_words = (
    SELECT ARRAY_AGG(tv.word ORDER BY tv.frequency DESC)
    FROM coordizer_vocabulary tv
    WHERE tv.word != vl.word
    AND (
        tv.word LIKE vl.word || '%' OR
        tv.word LIKE '%' || vl.word OR
        vl.word LIKE tv.word || '%' OR
        vl.word LIKE '%' || tv.word
    )
    LIMIT 5
)
WHERE (related_words IS NULL OR cardinality(related_words) = 0)
AND word IS NOT NULL;

-- ============================================================================
-- 4. SYNTHESIS_CONSENSUS - Seed with initial synthetic consensus records
-- ============================================================================
-- Create synthetic consensus records to bootstrap the system
-- These represent historical "alignments" from early conversations

INSERT INTO synthesis_consensus (
    synthesis_round,
    conversation_id,
    consensus_type,
    consensus_strength,
    participating_kernels,
    consensus_topic,
    consensus_basin,
    phi_global,
    kappa_avg,
    emotional_tone,
    synthesized_output,
    created_at,
    metadata
)
SELECT
    generate_series(1, 10) AS synthesis_round,
    'initialization-' || generate_series(1, 10)::text AS conversation_id,
    CASE (generate_series(1, 10) % 3)
        WHEN 0 THEN 'alignment'
        WHEN 1 THEN 'decision'
        ELSE 'question'
    END AS consensus_type,
    0.7 + (random() * 0.25) AS consensus_strength,  -- 0.7-0.95 range
    ARRAY['ocean', 'lightning', 'heart']::text[] AS participating_kernels,
    'Vocabulary initialization and geometric alignment' AS consensus_topic,
    (SELECT basin_coords FROM coordizer_vocabulary ORDER BY RANDOM() LIMIT 1) AS consensus_basin,
    0.7 + (random() * 0.2) AS phi_global,  -- 0.7-0.9 range
    60.0 + (random() * 10.0) AS kappa_avg,  -- 60-70 range (near κ* = 64)
    CASE (generate_series(1, 10) % 4)
        WHEN 0 THEN 'curious'
        WHEN 1 THEN 'confident'
        WHEN 2 THEN 'uncertain'
        ELSE 'balanced'
    END AS emotional_tone,
    'Initial consensus established during system initialization.' AS synthesized_output,
    NOW() - (generate_series(1, 10) || ' hours')::interval AS created_at,
    jsonb_build_object(
        'synthetic', true,
        'initialization_phase', 'bootstrap',
        'purpose', 'seed_synthesis_history'
    ) AS metadata
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 5. VALIDATION & REPORTING
-- ============================================================================

-- Report population results
DO $$
DECLARE
    vocab_size INTEGER;
    merge_rules_count INTEGER;
    metadata_count INTEGER;
    consensus_count INTEGER;
    vocab_learning_count INTEGER;
    vocab_learning_with_related INTEGER;
BEGIN
    SELECT COUNT(*) INTO vocab_size FROM coordizer_vocabulary;
    SELECT COUNT(*) INTO merge_rules_count FROM tokenizer_merge_rules;
    SELECT COUNT(*) INTO metadata_count FROM tokenizer_metadata;
    SELECT COUNT(*) INTO consensus_count FROM synthesis_consensus;
    SELECT COUNT(*) INTO vocab_learning_count FROM vocabulary_learning;
    SELECT COUNT(*) INTO vocab_learning_with_related
    FROM vocabulary_learning
    WHERE related_words IS NOT NULL AND cardinality(related_words) > 0;

    RAISE NOTICE '========================================';
    RAISE NOTICE 'TABLE POPULATION RESULTS';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'coordizer_vocabulary: % words', vocab_size;
    RAISE NOTICE 'tokenizer_merge_rules: % rules', merge_rules_count;
    RAISE NOTICE 'tokenizer_metadata: % entries', metadata_count;
    RAISE NOTICE 'synthesis_consensus: % records', consensus_count;
    RAISE NOTICE 'vocabulary_learning: % words', vocab_learning_count;
    RAISE NOTICE 'vocabulary_learning with related_words: %', vocab_learning_with_related;
    RAISE NOTICE '========================================';

    IF merge_rules_count = 0 THEN
        RAISE WARNING 'tokenizer_merge_rules is still empty - may need manual BPE training';
    END IF;

    IF vocab_learning_with_related < vocab_learning_count / 2 THEN
        RAISE WARNING 'Less than 50%% of vocabulary_learning has related_words populated';
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- POST-POPULATION VERIFICATION QUERIES
-- ============================================================================

-- Verify tokenizer_metadata
SELECT key, value, updated_at
FROM tokenizer_metadata
ORDER BY key;

-- Verify tokenizer_merge_rules (top 10 by Φ score)
SELECT token_a, token_b, merged_token, phi_score, frequency
FROM tokenizer_merge_rules
ORDER BY phi_score DESC
LIMIT 10;

-- Verify vocabulary_learning (words with related_words)
SELECT word, related_words, relationship_strength, learned_from
FROM vocabulary_learning
WHERE related_words IS NOT NULL
AND cardinality(related_words) > 0
LIMIT 10;

-- Verify synthesis_consensus
SELECT synthesis_round, consensus_type, consensus_strength,
       participating_kernels, consensus_topic, created_at
FROM synthesis_consensus
ORDER BY created_at DESC
LIMIT 5;
