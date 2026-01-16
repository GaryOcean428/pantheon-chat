import { sql } from 'drizzle-orm'
import { db, withDbRetry } from '../server/db'

async function run() {
  if (!db) {
    console.error('[DB Integrity] Database not configured')
    process.exit(1)
  }

  const dbInstance = db

  const invalidQfi = await withDbRetry(
    async () =>
      dbInstance.execute<{ count: number }>(sql`
        SELECT COUNT(*)::int AS count
        FROM coordizer_vocabulary
        WHERE qfi_score IS NOT NULL
          AND (qfi_score < 0 OR qfi_score > 1)
      `),
    'verify-db-integrity-invalid-qfi'
  )

  // Check tokens with valid QFI but missing basin (should have both)
  const validQfiMissingBasin = await withDbRetry(
    async () =>
      dbInstance.execute<{ count: number }>(sql`
        SELECT COUNT(*)::int AS count
        FROM coordizer_vocabulary
        WHERE qfi_score IS NOT NULL
          AND qfi_score BETWEEN 0 AND 1
          AND basin_embedding IS NULL
      `),
    'verify-db-integrity-valid-qfi-missing-basin'
  )

  // Check tokens with basin but no QFI (generation candidates need QFI)
  const basinMissingQfi = await withDbRetry(
    async () =>
      dbInstance.execute<{ count: number }>(sql`
        SELECT COUNT(*)::int AS count
        FROM coordizer_vocabulary
        WHERE basin_embedding IS NOT NULL
          AND (qfi_score IS NULL OR qfi_score < 0 OR qfi_score > 1)
          AND token_role IN ('generation', 'both')
      `),
    'verify-db-integrity-basin-missing-qfi'
  )

  const invalidQfiCount = invalidQfi?.rows?.[0]?.count ?? 0
  const validQfiMissingBasinCount = validQfiMissingBasin?.rows?.[0]?.count ?? 0
  const basinMissingQfiCount = basinMissingQfi?.rows?.[0]?.count ?? 0

  console.log(`[DB Integrity] invalid_qfi=${invalidQfiCount}`)
  console.log(`[DB Integrity] valid_qfi_missing_basin=${validQfiMissingBasinCount}`)
  console.log(`[DB Integrity] basin_missing_qfi=${basinMissingQfiCount}`)

  if (invalidQfiCount > 0) {
    console.error('[DB Integrity] QFI violations detected')
    process.exit(1)
  }
  
  if (validQfiMissingBasinCount > 0 || basinMissingQfiCount > 0) {
    console.warn('[DB Integrity] Some tokens have inconsistent QFI/basin state (non-blocking)')
  }

  console.log('[DB Integrity] ✅ OK')
}

run().catch((error) => {
  console.error('[DB Integrity] Failed:', error)
  process.exit(1)
})
