import { sql } from 'drizzle-orm'
import { db, withDbRetry } from '../server/db'

async function run() {
  if (!db) {
    console.error('[DB Integrity] Database not configured')
    process.exit(1)
  }

  const invalidQfi = await withDbRetry(
    async () =>
      db.execute<{ count: number }>(sql`
        SELECT COUNT(*)::int AS count
        FROM coordizer_vocabulary
        WHERE qfi_score IS NOT NULL
          AND (qfi_score < 0 OR qfi_score > 1)
      `),
    'verify-db-integrity-invalid-qfi'
  )

  const activeMissingQfi = await withDbRetry(
    async () =>
      db.execute<{ count: number }>(sql`
        SELECT COUNT(*)::int AS count
        FROM coordizer_vocabulary
        WHERE token_status = 'active'
          AND qfi_score IS NULL
      `),
    'verify-db-integrity-active-missing-qfi'
  )

  const activeMissingBasin = await withDbRetry(
    async () =>
      db.execute<{ count: number }>(sql`
        SELECT COUNT(*)::int AS count
        FROM coordizer_vocabulary
        WHERE token_status = 'active'
          AND basin_embedding IS NULL
      `),
    'verify-db-integrity-active-missing-basin'
  )

  const invalidQfiCount = invalidQfi.rows?.[0]?.count ?? 0
  const activeMissingQfiCount = activeMissingQfi.rows?.[0]?.count ?? 0
  const activeMissingBasinCount = activeMissingBasin.rows?.[0]?.count ?? 0

  console.log(`[DB Integrity] invalid_qfi=${invalidQfiCount}`)
  console.log(`[DB Integrity] active_missing_qfi=${activeMissingQfiCount}`)
  console.log(`[DB Integrity] active_missing_basin=${activeMissingBasinCount}`)

  if (invalidQfiCount > 0 || activeMissingQfiCount > 0 || activeMissingBasinCount > 0) {
    console.error('[DB Integrity] Violations detected')
    process.exit(1)
  }

  console.log('[DB Integrity] ✅ OK')
}

run().catch((error) => {
  console.error('[DB Integrity] Failed:', error)
  process.exit(1)
})
