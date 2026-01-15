#!/usr/bin/env tsx
import { sql } from 'drizzle-orm'
import { db } from '../server/db'

async function getCount(query: string): Promise<number> {
  const result = await db.execute(sql.raw(query))
  const count = parseInt(result[0]?.count as string, 10)
  return Number.isNaN(count) ? 0 : count
}

async function main() {
  if (!db) {
    console.error('Database unavailable')
    process.exit(1)
  }

  const invalidQfi = await getCount(
    `SELECT COUNT(*) as count FROM coordizer_vocabulary WHERE qfi_score < 0 OR qfi_score > 1`
  )
  const activeNullQfi = await getCount(
    `SELECT COUNT(*) as count FROM coordizer_vocabulary WHERE token_status = 'active' AND qfi_score IS NULL`
  )
  const activeNullBasin = await getCount(
    `SELECT COUNT(*) as count FROM coordizer_vocabulary WHERE token_status = 'active' AND basin_embedding IS NULL`
  )

  console.log('DB integrity check:')
  console.log(`- invalid qfi range: ${invalidQfi}`)
  console.log(`- active tokens with NULL qfi: ${activeNullQfi}`)
  console.log(`- active tokens with NULL basin: ${activeNullBasin}`)

  if (invalidQfi > 0 || activeNullQfi > 0 || activeNullBasin > 0) {
    process.exit(1)
  }
}

main().catch((error) => {
  console.error('DB integrity check failed:', error)
  process.exit(1)
})
