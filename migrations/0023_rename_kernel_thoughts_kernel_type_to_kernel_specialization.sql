-- ============================================================================
-- 0023 - Rename kernel_thoughts.kernel_type -> kernel_specialization
--
-- Canonical naming enforcement (Genesis Rollout): kernel specialization is the
-- only allowed field name for kernel taxonomy.
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'kernel_thoughts'
          AND column_name = 'kernel_type'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'kernel_thoughts'
          AND column_name = 'kernel_specialization'
    ) THEN
        EXECUTE 'ALTER TABLE kernel_thoughts RENAME COLUMN kernel_type TO kernel_specialization';
    END IF;
END $$;
