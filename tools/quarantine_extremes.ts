/**
 * Maintenance Tool: Quarantine QFI Extremes
 *
 * This tool performs direct SQL writes to coordizer_vocabulary outside the canonical
 * upsertToken path. This is intentional and acceptable because:
 *
 * 1. This is a maintenance/repair tool, not application logic
 * 2. It performs bulk operations that would be inefficient through upsertToken
 * 3. It's run manually by administrators, not automatically by the application
 * 4. It uses the same validation logic (QFI range checks) as the canonical path
 *
 * IMPORTANT: This tool should only be used for database maintenance and repair.
 * Regular application code MUST use the canonical upsertToken function from
 * server/persistence/coordizer-vocabulary.ts
 */

import { readFileSync } from 'fs'
import { sql } from 'drizzle-orm'
import { db, withDbRetry } from '../server/db'

const ALLOWLIST_PATH = 'tools/allowlists/qfi_extremes_allowlist.txt'

function loadAllowlist(): Set<string> {
  try {
    const content = readFileSync(ALLOWLIST_PATH, 'utf-8')
    return new Set(
      content
        .split('\n')
        .map((line) => line.trim())
        .filter((line) => line.length > 0 && !line.startsWith('#'))
    )
  } catch {
    return new Set()
  }
}

async function run() {
  if (!db) {
    console.error('[QFI Extremes] Database not configured')
    process.exit(1)
  }

  const allowlist = loadAllowlist()
  const allowlistTokens = Array.from(allowlist)
  const allowlistCondition =
    allowlistTokens.length > 0
      ? sql`AND token <> ALL(ARRAY[${sql.join(allowlistTokens.map(t => sql`${t}`), sql`, `)}])`
      : sql``

  const result = await withDbRetry(
    async () =>
      db.execute<{ count: number }>(sql`
        UPDATE coordizer_vocabulary
        SET token_status = 'quarantined'
        WHERE (qfi_score = 0 OR qfi_score >= 0.99)
          AND token_status = 'active'
          ${allowlistCondition}
        RETURNING 1
      `),
    'quarantine-extremes-update'
  )

  const quarantined = result.rows?.length ?? 0
  console.log(`[QFI Extremes] Quarantined ${quarantined} tokens`)
}

run().catch((error) => {
  console.error('[QFI Extremes] Failed:', error)
  process.exit(1)
})
