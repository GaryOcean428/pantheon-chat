-- Migration: Add missing columns to kernel_training_history
-- Date: 2026-01-11
-- Purpose: Support rich training metrics from KernelTrainingOrchestrator

ALTER TABLE kernel_training_history 
ADD COLUMN IF NOT EXISTS god_name VARCHAR(64),
ADD COLUMN IF NOT EXISTS loss DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS reward DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS gradient_norm DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS phi_before DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS phi_after DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS kappa_before DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS kappa_after DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS basin_coords DOUBLE PRECISION[],
ADD COLUMN IF NOT EXISTS trigger VARCHAR(128),
ADD COLUMN IF NOT EXISTS step_count INTEGER,
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(64);

CREATE INDEX IF NOT EXISTS idx_kernel_training_history_god_name ON kernel_training_history(god_name);
CREATE INDEX IF NOT EXISTS idx_kernel_training_history_created_at ON kernel_training_history(created_at);
