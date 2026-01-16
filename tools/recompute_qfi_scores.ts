/**
 * Maintenance Tool: Recompute QFI Scores
 *
 * This tool performs direct SQL writes to coordizer_vocabulary outside the canonical
 * upsertToken path. This is intentional and acceptable because:
 *
 * 1. This is a maintenance/repair tool for backfilling QFI scores, not application logic
 * 2. It performs bulk operations (batch updates) that would be inefficient through upsertToken
 * 3. It's run manually by administrators with explicit --dry-run or --apply flags
 * 4. It uses the same canonical QFI computation (compute_qfi_score_simplex) as upsertToken
 *
 * IMPORTANT: This tool should only be used for database maintenance and repair.
 * Regular application code MUST use the canonical upsertToken function from
 * server/persistence/coordizer-vocabulary.ts
 */

import { sql } from 'drizzle-orm'
import { compute_qfi_score_simplex, to_simplex_probabilities } from '@shared'
import { db, withDbRetry } from '../server/db'

interface TokenRow {
  id: number
  token: string
  basin_embedding: number[] | string | null
  qfi_score: number | null
  token_status: string | null
}

const BATCH_SIZE = 200

function parseVector(raw: number[] | string | null): number[] | null {
  if (!raw) return null
  if (Array.isArray(raw)) return raw

  const trimmed = (raw as string).trim()
  const jsonLike = trimmed.startsWith('[') ? trimmed : trimmed.replace(/^\{/, '[').replace(/\}$/, ']')

  try {
    const parsed = JSON.parse(jsonLike)
    if (Array.isArray(parsed)) {
      return parsed.map((value) => Number(value))
    }
  } catch {
    return null
  }

  return null
}

function isSameQfi(a: number | null, b: number | null, tolerance = 1e-8) {
  if (a === null || b === null) return false
  return Math.abs(a - b) < tolerance
}

async function run() {
  const dryRun = process.argv.includes('--dry-run')
  const apply = process.argv.includes('--apply')

  if (!dryRun && !apply) {
    console.error('Usage: tsx tools/recompute_qfi_scores.ts --dry-run|--apply')
    process.exit(1)
  }

  if (!db) {
    console.error('[QFI Backfill] Database not configured')
    process.exit(1)
  }

  const rows = await withDbRetry(
    async () =>
      db.execute<TokenRow>(sql`
        SELECT id, token, basin_embedding, qfi_score, token_status
        FROM coordizer_vocabulary
        WHERE basin_embedding IS NOT NULL
      `),
    'recompute-qfi-select'
  )

  const tokens = rows.rows ?? []
  let updated = 0
  let quarantined = 0
  let unchanged = 0
  let errors = 0

  for (let i = 0; i < tokens.length; i += BATCH_SIZE) {
    const batch = tokens.slice(i, i + BATCH_SIZE)
    const updates: Array<{ id: number; qfiScore: number | null; tokenStatus: string }> = []

    for (const token of batch) {
      const basin = parseVector(token.basin_embedding)

      if (!basin) {
        quarantined++
        updates.push({ id: token.id, qfiScore: null, tokenStatus: 'quarantined' })
        continue
      }

      try {
        const simplex = to_simplex_probabilities(basin)
        const qfiScore = compute_qfi_score_simplex(simplex)
        const nextStatus = token.token_status ?? 'active'

        if (isSameQfi(token.qfi_score, qfiScore) && token.token_status === nextStatus) {
          unchanged++
          continue
        }

        updates.push({ id: token.id, qfiScore, tokenStatus: nextStatus })
        updated++
      } catch {
        errors++
        quarantined++
        updates.push({ id: token.id, qfiScore: null, tokenStatus: 'quarantined' })
      }
    }

    if (apply && updates.length > 0) {
      await withDbRetry(
        async () => {
          const values = updates.map((update) =>
            sql`(${update.id}, ${update.qfiScore}, ${update.tokenStatus})`
          )

          await db.execute(sql`
            UPDATE coordizer_vocabulary AS cv
            SET qfi_score = updates.qfi_score,
                token_status = updates.token_status
            FROM (VALUES ${sql.join(values, sql`, `)}) AS updates(id, qfi_score, token_status)
            WHERE cv.id = updates.id
          `)
        },
        'recompute-qfi-update'
      )
    }
  }

  console.log('[QFI Backfill] Summary')
  console.log(`  total scanned: ${tokens.length}`)
  console.log(`  updated qfi: ${updated}`)
  console.log(`  quarantined: ${quarantined}`)
  console.log(`  unchanged: ${unchanged}`)
  console.log(`  errors: ${errors}`)

  if (dryRun) {
    console.log('[QFI Backfill] Dry-run complete (no updates applied)')
  }
}

run().catch((error) => {
  console.error('[QFI Backfill] Failed:', error)
  process.exit(1)
})
