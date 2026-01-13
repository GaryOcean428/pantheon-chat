-- Migration: Fix qig_metadata column name inconsistency
-- Purpose: Ensure database schema matches code expectations (config_key instead of key)
-- Date: 2026-01-13
-- Issue: Database has 'key' column but code expects 'config_key'

-- Check if column 'key' exists and rename to 'config_key'
-- This migration is idempotent - safe to run multiple times

DO $$
BEGIN
    -- Check if 'key' column exists
    IF EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'qig_metadata' 
        AND column_name = 'key'
    ) THEN
        -- Rename 'key' to 'config_key'
        ALTER TABLE qig_metadata RENAME COLUMN key TO config_key;
        RAISE NOTICE 'Column renamed: key -> config_key';
    ELSE
        RAISE NOTICE 'Column key does not exist, assuming config_key already exists';
    END IF;
END $$;

-- Ensure the index exists with correct column name
DROP INDEX IF EXISTS idx_qig_metadata_key;
CREATE INDEX IF NOT EXISTS idx_qig_metadata_config_key ON qig_metadata(config_key);

-- Verify the fix
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'qig_metadata' 
        AND column_name = 'config_key'
    ) THEN
        RAISE NOTICE 'SUCCESS: qig_metadata.config_key column verified';
    ELSE
        RAISE EXCEPTION 'FAILED: qig_metadata.config_key column not found';
    END IF;
END $$;
