-- ============================================================================
-- QFI INTEGRITY CONSTRAINTS
-- ============================================================================
-- Purpose: Enforce qfi_score bounds and active token integrity
-- Project: pantheon-chat
-- Date: 2026-01-21
--
-- Adds token_status column, validates existing data, and applies CHECK constraints.
-- ============================================================================

BEGIN;

-- 1) Ensure token_status exists and is normalized
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

    UPDATE coordizer_vocabulary
    SET token_status = 'active'
    WHERE token_status IS NULL;
END $$;

-- 2) Normalize invalid QFI data before constraints
DO $$
DECLARE
    invalid_qfi_count INT;
    missing_qfi_active_count INT;
    missing_basin_active_count INT;
BEGIN
    UPDATE coordizer_vocabulary
    SET token_status = 'quarantined',
        qfi_score = NULL
    WHERE qfi_score IS NOT NULL
      AND (qfi_score < 0 OR qfi_score > 1);
    GET DIAGNOSTICS invalid_qfi_count = ROW_COUNT;
    RAISE NOTICE 'Quarantined % tokens with invalid qfi_score', invalid_qfi_count;

    UPDATE coordizer_vocabulary
    SET token_status = 'quarantined'
    WHERE token_status = 'active'
      AND qfi_score IS NULL;
    GET DIAGNOSTICS missing_qfi_active_count = ROW_COUNT;
    RAISE NOTICE 'Quarantined % active tokens missing qfi_score', missing_qfi_active_count;

    UPDATE coordizer_vocabulary
    SET token_status = 'quarantined'
    WHERE token_status = 'active'
      AND basin_embedding IS NULL;
    GET DIAGNOSTICS missing_basin_active_count = ROW_COUNT;
    RAISE NOTICE 'Quarantined % active tokens missing basin_embedding', missing_basin_active_count;
END $$;

-- 3) Enforce token_status values
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

-- 4) Enforce QFI range constraint
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'coordizer_qfi_range'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT coordizer_qfi_range
        CHECK (
            qfi_score IS NULL OR (qfi_score >= 0.0 AND qfi_score <= 1.0)
        );
        RAISE NOTICE 'Added qfi_score range CHECK constraint';
    END IF;
END $$;

-- 5) Active tokens must have QFI
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
        RAISE NOTICE 'Added active requires qfi_score CHECK constraint';
    END IF;
END $$;

-- 6) Active tokens must have basin embedding
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
        RAISE NOTICE 'Added active requires basin_embedding CHECK constraint';
    END IF;
END $$;

COMMIT;
