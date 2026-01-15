-- ============================================================================
-- QFI CONSTRAINTS + TOKEN STATUS ENFORCEMENT
-- ============================================================================
-- Purpose: Enforce valid QFI ranges and active token integrity
-- Project: pantheon-chat
-- Date: 2026-01-18
--
-- This migration:
-- 1. Adds token_status if missing
-- 2. Quarantines invalid rows before constraints
-- 3. Adds CHECK constraints for qfi_score range and active requirements
-- ============================================================================

BEGIN;

-- Ensure token_status exists with safe default
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'coordizer_vocabulary' AND column_name = 'token_status'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD COLUMN token_status VARCHAR(16) DEFAULT 'active';
        RAISE NOTICE 'Added token_status column to coordizer_vocabulary';
    END IF;
END $$;

-- Ensure qfi_score exists (defensive for older envs)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'coordizer_vocabulary' AND column_name = 'qfi_score'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD COLUMN qfi_score DOUBLE PRECISION;
        RAISE NOTICE 'Added qfi_score column to coordizer_vocabulary';
    END IF;
END $$;

-- Default token_status for NULL values
UPDATE coordizer_vocabulary
SET token_status = 'active'
WHERE token_status IS NULL;

-- Quarantine invalid QFI values before constraints
DO $$
DECLARE
    invalid_qfi_count INT := 0;
    null_qfi_active_count INT := 0;
    null_basin_active_count INT := 0;
BEGIN
    UPDATE coordizer_vocabulary
    SET token_status = 'quarantined',
        qfi_score = NULL
    WHERE qfi_score IS NOT NULL
      AND (qfi_score < 0 OR qfi_score > 1);
    GET DIAGNOSTICS invalid_qfi_count = ROW_COUNT;
    RAISE NOTICE 'Quarantined % tokens with invalid qfi_score values', invalid_qfi_count;

    UPDATE coordizer_vocabulary
    SET token_status = 'quarantined'
    WHERE token_status = 'active'
      AND qfi_score IS NULL;
    GET DIAGNOSTICS null_qfi_active_count = ROW_COUNT;
    RAISE NOTICE 'Quarantined % active tokens with NULL qfi_score', null_qfi_active_count;

    UPDATE coordizer_vocabulary
    SET token_status = 'quarantined'
    WHERE token_status = 'active'
      AND basin_embedding IS NULL;
    GET DIAGNOSTICS null_basin_active_count = ROW_COUNT;
    RAISE NOTICE 'Quarantined % active tokens with NULL basin_embedding', null_basin_active_count;
END $$;

-- token_status constraint
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'coordizer_token_status_check'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT coordizer_token_status_check
        CHECK (token_status IN ('active', 'quarantined', 'deprecated'));
        RAISE NOTICE 'Added token_status CHECK constraint';
    END IF;
END $$;

-- QFI range constraint
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'coordizer_qfi_range'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT coordizer_qfi_range
        CHECK (qfi_score IS NULL OR (qfi_score >= 0.0 AND qfi_score <= 1.0));
        RAISE NOTICE 'Added qfi_score range CHECK constraint';
    END IF;
END $$;

-- Active tokens must have QFI
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'coordizer_active_requires_qfi'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT coordizer_active_requires_qfi
        CHECK (token_status <> 'active' OR qfi_score IS NOT NULL);
        RAISE NOTICE 'Added active requires qfi_score CHECK constraint';
    END IF;
END $$;

-- Active tokens must have basin_embedding
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'coordizer_active_requires_basin'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT coordizer_active_requires_basin
        CHECK (token_status <> 'active' OR basin_embedding IS NOT NULL);
        RAISE NOTICE 'Added active requires basin_embedding CHECK constraint';
    END IF;
END $$;

COMMIT;
