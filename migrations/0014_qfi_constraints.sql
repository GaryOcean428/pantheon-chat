-- ============================================================================
-- QFI CONSTRAINTS + TOKEN STATUS (QIG PURITY)
-- ============================================================================
-- Purpose: Enforce QFI validity and active token integrity.
-- Project: pantheon-chat
-- Date: 2026-01-16
--
-- IDEMPOTENT: Safe to run multiple times
-- ============================================================================

BEGIN;

-- Ensure token_status column exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'coordizer_vocabulary' AND column_name = 'token_status'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD COLUMN token_status VARCHAR(20) DEFAULT 'active';
        RAISE NOTICE 'Added token_status column to coordizer_vocabulary';
    END IF;
END $$;

-- Normalize null token_status values
UPDATE coordizer_vocabulary
SET token_status = 'active'
WHERE token_status IS NULL;

-- Add CHECK constraint for token_status values
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'coordizer_vocabulary_token_status_check'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT coordizer_vocabulary_token_status_check
        CHECK (token_status IN ('active', 'quarantined', 'deprecated'));
        RAISE NOTICE 'Added token_status CHECK constraint';
    END IF;
END $$;

-- Quarantine invalid QFI rows before adding constraints
DO $$
DECLARE
    range_quarantine_count INT;
    null_qfi_quarantine_count INT;
    basin_quarantine_count INT;
BEGIN
    UPDATE coordizer_vocabulary
    SET token_status = 'quarantined',
        qfi_score = NULL
    WHERE qfi_score IS NOT NULL
      AND (qfi_score < 0.0 OR qfi_score > 1.0);
    GET DIAGNOSTICS range_quarantine_count = ROW_COUNT;

    UPDATE coordizer_vocabulary
    SET token_status = 'quarantined'
    WHERE token_status = 'active'
      AND qfi_score IS NULL;
    GET DIAGNOSTICS null_qfi_quarantine_count = ROW_COUNT;

    UPDATE coordizer_vocabulary
    SET token_status = 'quarantined'
    WHERE token_status = 'active'
      AND basin_embedding IS NULL;
    GET DIAGNOSTICS basin_quarantine_count = ROW_COUNT;

    RAISE NOTICE 'Quarantined % rows with invalid qfi_score range', range_quarantine_count;
    RAISE NOTICE 'Quarantined % active rows with NULL qfi_score', null_qfi_quarantine_count;
    RAISE NOTICE 'Quarantined % active rows with NULL basin_embedding', basin_quarantine_count;
END $$;

-- Enforce QFI range validity
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'coordizer_qfi_range_check'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT coordizer_qfi_range_check
        CHECK (
            qfi_score IS NULL OR (qfi_score >= 0.0 AND qfi_score <= 1.0)
        );
        RAISE NOTICE 'Added coordizer_qfi_range_check';
    END IF;
END $$;

-- Enforce active tokens have QFI
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'coordizer_active_requires_qfi'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT coordizer_active_requires_qfi
        CHECK (
            token_status <> 'active' OR qfi_score IS NOT NULL
        );
        RAISE NOTICE 'Added coordizer_active_requires_qfi';
    END IF;
END $$;

-- Enforce active tokens have basin embeddings
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'coordizer_active_requires_basin'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT coordizer_active_requires_basin
        CHECK (
            token_status <> 'active' OR basin_embedding IS NOT NULL
        );
        RAISE NOTICE 'Added coordizer_active_requires_basin';
    END IF;
END $$;

COMMIT;
