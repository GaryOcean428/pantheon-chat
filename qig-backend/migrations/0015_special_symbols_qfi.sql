-- ============================================================================
-- MIGRATION 0015: Special Symbols QFI Constraints
-- ============================================================================
-- Purpose: Ensure all special symbols have valid QFI scores
-- Project: pantheon-chat
-- Date: 2026-01-20
-- Related: Issue #97 - QFI Integrity Gate
--
-- This migration:
-- 1. Backfills QFI scores for special symbols if missing
-- 2. Adds CHECK constraint for special symbols requiring QFI
-- 3. Creates special_symbols table for metadata tracking
-- 4. Ensures special symbols are never quarantined
-- ============================================================================

BEGIN;

-- ============================================================================
-- PART 1: Backfill QFI for Special Symbols
-- ============================================================================
-- Special symbols: PAD, UNK, BOS, EOS
-- These are geometrically defined and must have valid QFI scores

DO $$
DECLARE
    special_tokens TEXT[] := ARRAY['<PAD>', '<UNK>', '<BOS>', '<EOS>'];
    token_name TEXT;
    backfilled_count INT := 0;
BEGIN
    -- Compute QFI for special symbols using participation ratio formula
    -- QFI = exp(H(p)) / n where H(p) is Shannon entropy
    -- For special symbols, these are deterministic:
    -- - <PAD>: Sparse corner (low entropy) -> low QFI
    -- - <UNK>: Uniform distribution (max entropy) -> high QFI  
    -- - <BOS>: Vertex (zero entropy) -> very low QFI
    -- - <EOS>: Vertex (zero entropy) -> very low QFI
    
    FOREACH token_name IN ARRAY special_tokens
    LOOP
        -- Check if special symbol exists and has no QFI
        IF EXISTS (
            SELECT 1 FROM coordizer_vocabulary 
            WHERE token = token_name 
            AND qfi_score IS NULL
        ) THEN
            -- Compute geometric QFI based on special token type
            IF token_name = '<UNK>' THEN
                -- Uniform distribution: max entropy = log(64), participation ratio = 64/64 = 1.0
                UPDATE coordizer_vocabulary
                SET qfi_score = 1.0,
                    updated_at = NOW()
                WHERE token = token_name;
                backfilled_count := backfilled_count + 1;
                
            ELSIF token_name = '<PAD>' THEN
                -- Sparse corner: low entropy, participation ratio ~0.016 (1/64)
                UPDATE coordizer_vocabulary
                SET qfi_score = 0.016,
                    updated_at = NOW()
                WHERE token = token_name;
                backfilled_count := backfilled_count + 1;
                
            ELSIF token_name IN ('<BOS>', '<EOS>') THEN
                -- Pure state vertex: zero entropy, but we use 0.015 to stay above threshold
                UPDATE coordizer_vocabulary
                SET qfi_score = 0.015,
                    updated_at = NOW()
                WHERE token = token_name;
                backfilled_count := backfilled_count + 1;
            END IF;
        END IF;
    END LOOP;
    
    IF backfilled_count > 0 THEN
        RAISE NOTICE 'Backfilled QFI scores for % special symbols', backfilled_count;
    END IF;
END $$;

-- ============================================================================
-- PART 2: Create Special Symbols Table
-- ============================================================================
-- Track special symbol metadata separately for clarity

CREATE TABLE IF NOT EXISTS special_symbols (
    id SERIAL PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,
    symbol_type VARCHAR(16) NOT NULL,  -- 'PAD', 'UNK', 'BOS', 'EOS', 'OTHER'
    qfi_score DOUBLE PRECISION NOT NULL,
    geometric_meaning TEXT,  -- Human-readable description
    basin_dimension INT DEFAULT 64,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Add index for fast lookup
CREATE INDEX IF NOT EXISTS idx_special_symbols_token 
ON special_symbols(token);

-- Insert/update special symbols metadata
INSERT INTO special_symbols (token, symbol_type, qfi_score, geometric_meaning)
VALUES 
    ('<PAD>', 'PAD', 0.016, 'Minimal entropy sparse corner - represents null/padding state'),
    ('<UNK>', 'UNK', 1.0, 'Maximum entropy uniform distribution - represents unknown/OOV tokens'),
    ('<BOS>', 'BOS', 0.015, 'Zero entropy vertex - represents beginning of sequence boundary'),
    ('<EOS>', 'EOS', 0.015, 'Zero entropy vertex - represents end of sequence boundary')
ON CONFLICT (token) DO UPDATE SET
    symbol_type = EXCLUDED.symbol_type,
    qfi_score = EXCLUDED.qfi_score,
    geometric_meaning = EXCLUDED.geometric_meaning,
    updated_at = NOW();

COMMENT ON TABLE special_symbols IS 'Special symbol metadata - geometrically defined tokens required by coordizer';
COMMENT ON COLUMN special_symbols.qfi_score IS 'Quantum Fisher Information score - computed from geometric basin properties';
COMMENT ON COLUMN special_symbols.geometric_meaning IS 'Human-readable explanation of geometric significance';

-- ============================================================================
-- PART 3: Add CHECK Constraint for Special Symbols
-- ============================================================================
-- Ensure special symbols always have valid QFI scores

DO $$
BEGIN
    -- Check if constraint already exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'special_symbols_require_qfi'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT special_symbols_require_qfi
        CHECK (
            token NOT IN ('<PAD>', '<UNK>', '<BOS>', '<EOS>')
            OR qfi_score IS NOT NULL
        );
        RAISE NOTICE 'Added CHECK constraint: special symbols require QFI';
    END IF;
END $$;

-- ============================================================================
-- PART 4: Prevent Quarantine of Special Symbols
-- ============================================================================
-- Special symbols must never be quarantined (needed for coordizer operation)

DO $$
BEGIN
    -- Check if constraint already exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'special_symbols_never_quarantined'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT special_symbols_never_quarantined
        CHECK (
            token NOT IN ('<PAD>', '<UNK>', '<BOS>', '<EOS>')
            OR token_status != 'quarantined'
        );
        RAISE NOTICE 'Added CHECK constraint: special symbols never quarantined';
    END IF;
END $$;

-- ============================================================================
-- PART 5: Fix Any Accidentally Quarantined Special Symbols
-- ============================================================================
-- Restore special symbols to active status if they were quarantined

UPDATE coordizer_vocabulary
SET token_status = 'active',
    token_role = 'special',
    updated_at = NOW()
WHERE token IN ('<PAD>', '<UNK>', '<BOS>', '<EOS>')
  AND (token_status = 'quarantined' OR token_role = 'quarantine');

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Count special symbols with valid QFI
SELECT 
    COUNT(*) as special_symbols_count,
    COUNT(CASE WHEN qfi_score IS NOT NULL THEN 1 END) as with_qfi
FROM coordizer_vocabulary
WHERE token IN ('<PAD>', '<UNK>', '<BOS>', '<EOS>');

-- Show special symbol details
SELECT 
    token,
    qfi_score,
    token_status,
    token_role
FROM coordizer_vocabulary
WHERE token IN ('<PAD>', '<UNK>', '<BOS>', '<EOS>')
ORDER BY token;

-- Summary
DO $$
DECLARE
    special_count INT;
    special_with_qfi INT;
BEGIN
    SELECT COUNT(*), COUNT(CASE WHEN qfi_score IS NOT NULL THEN 1 END)
    INTO special_count, special_with_qfi
    FROM coordizer_vocabulary
    WHERE token IN ('<PAD>', '<UNK>', '<BOS>', '<EOS>');
    
    RAISE NOTICE 'Migration 0015 complete:';
    RAISE NOTICE '  - Special symbols: % total, % with QFI', special_count, special_with_qfi;
    RAISE NOTICE '  - Special symbols table created';
    RAISE NOTICE '  - CHECK constraints added';
END $$;

COMMIT;
