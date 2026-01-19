/**
 * Maintenance Tool: Quarantine QFI Extremes
 *
 * This tool nullifies QFI scores for tokens with extreme values (0 or >= 0.99)
 * which prevents them from being selected during generation.
 * 
 * Tokens are NOT deleted - just excluded from active generation by setting qfi_score = NULL.
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

  // Nullify QFI for extreme values (this "quarantines" them from generation)
  const result = await withDbRetry(
    async () =>
      db.execute<{ count: number }>(sql`
        UPDATE coordizer_vocabulary
        SET qfi_score = NULL
        WHERE (qfi_score = 0 OR qfi_score >= 0.99)
          AND qfi_score IS NOT NULL
          ${allowlistCondition}
        RETURNING 1
      `),
    'quarantine-extremes-update'
  )

  const quarantined = result.rows?.length ?? 0
  console.log(`[QFI Extremes] Nullified QFI for ${quarantined} tokens with extreme values`)
}

run().catch((error) => {
  console.error('[QFI Extremes] Failed:', error)
  process.exit(1)
})
