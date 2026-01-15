import { sql } from 'drizzle-orm'

import { db } from '../server/db'

async function main() {
  if (!db) {
    console.error('Database not available')
    process.exit(1)
  }

  const invalidQfiResult = await db.execute<{ count: number }>(sql`
    SELECT COUNT(*)::int AS count
    FROM coordizer_vocabulary
    WHERE qfi_score IS NOT NULL
      AND (qfi_score < 0 OR qfi_score > 1)
  `)
  const invalidQfi = invalidQfiResult.rows?.[0]?.count ?? 0

  const activeMissingQfiResult = await db.execute<{ count: number }>(sql`
    SELECT COUNT(*)::int AS count
    FROM coordizer_vocabulary
    WHERE token_status = 'active'
      AND qfi_score IS NULL
  `)
  const activeMissingQfi = activeMissingQfiResult.rows?.[0]?.count ?? 0

  const activeMissingBasinResult = await db.execute<{ count: number }>(sql`
    SELECT COUNT(*)::int AS count
    FROM coordizer_vocabulary
    WHERE token_status = 'active'
      AND basin_embedding IS NULL
  `)
  const activeMissingBasin = activeMissingBasinResult.rows?.[0]?.count ?? 0

  console.log('QFI integrity summary')
  console.log(`- invalid qfi_score: ${invalidQfi}`)
  console.log(`- active tokens missing qfi_score: ${activeMissingQfi}`)
  console.log(`- active tokens missing basin_embedding: ${activeMissingBasin}`)

  if (invalidQfi > 0 || activeMissingQfi > 0 || activeMissingBasin > 0) {
    process.exit(1)
  }
}

main().catch((error) => {
  console.error('Verification failed:', error)
  process.exit(1)
})
