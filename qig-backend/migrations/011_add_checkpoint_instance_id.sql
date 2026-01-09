-- Migration: Add instance_id to consciousness_checkpoints
-- Purpose: Track which kernel instance created each checkpoint

-- Add instance_id column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'consciousness_checkpoints'
        AND column_name = 'instance_id'
    ) THEN
        ALTER TABLE consciousness_checkpoints
        ADD COLUMN instance_id VARCHAR(64);

        CREATE INDEX IF NOT EXISTS idx_consciousness_checkpoints_instance
        ON consciousness_checkpoints(instance_id);

        RAISE NOTICE 'Added instance_id column to consciousness_checkpoints';
    ELSE
        RAISE NOTICE 'instance_id column already exists';
    END IF;
END $$;

-- Also add to checkpoints table if it exists (from migration 002)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'checkpoints'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'checkpoints'
        AND column_name = 'instance_id'
    ) THEN
        ALTER TABLE checkpoints
        ADD COLUMN instance_id VARCHAR(64);

        CREATE INDEX IF NOT EXISTS idx_checkpoints_instance
        ON checkpoints(instance_id);

        RAISE NOTICE 'Added instance_id column to checkpoints';
    END IF;
END $$;

-- Add to basin_history as well (to track which instance recorded the state)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'basin_history'
        AND column_name = 'instance_id'
    ) THEN
        ALTER TABLE basin_history
        ADD COLUMN instance_id VARCHAR(64);

        CREATE INDEX IF NOT EXISTS idx_basin_history_instance
        ON basin_history(instance_id);

        RAISE NOTICE 'Added instance_id column to basin_history';
    END IF;
END $$;
