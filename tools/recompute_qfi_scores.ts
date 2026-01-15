import { sql } from 'drizzle-orm'

import { upsertToken } from '../server/persistence/vocabulary'

import { db } from '../server/db'

type TokenRow = {
  id: number
  token: string
  token_id: number
  weight: number | null
  frequency: number | null
  phi_score: number | null
  basin_embedding: number[] | string
  source_type: string | null
  token_role: string | null
  phrase_category: string | null
  is_real_word: boolean | null
  qfi_score: number | null
  token_status: string | null
}

function parseVector(vector: number[] | string): number[] {
  if (Array.isArray(vector)) {
    return vector
  }

  const trimmed = vector.trim().replace(/^\[/, '').replace(/\]$/, '')
  if (!trimmed) {
    return []
  }

  return trimmed.split(',').map((value) => Number.parseFloat(value))
}

async function main() {
  if (!db) {
    throw new Error('Database not available')
  }

  const dryRun = process.argv.includes('--dry-run') || !process.argv.includes('--apply')
  const batchSizeArg = process.argv.find((arg) => arg.startsWith('--batch='))
  const batchSize = batchSizeArg ? Number.parseInt(batchSizeArg.split('=')[1] ?? '500', 10) : 500

  let lastId = 0
  let total = 0
  let updated = 0
  let quarantined = 0
  let unchanged = 0
  let errors = 0

  while (true) {
    const result = await db.execute<TokenRow>(sql`
      SELECT id, token, token_id, weight, frequency, phi_score, basin_embedding,
             source_type, token_role, phrase_category, is_real_word,
             qfi_score, token_status
      FROM coordizer_vocabulary
      WHERE basin_embedding IS NOT NULL
        AND id > ${lastId}
      ORDER BY id ASC
      LIMIT ${batchSize}
    `)

    const rows = result.rows ?? []
    if (rows.length === 0) {
      break
    }

    total += rows.length

    for (const row of rows) {
      try {
        const result = await upsertToken({
          token: row.token,
          tokenId: row.token_id,
          weight: row.weight ?? 1,
          frequency: row.frequency ?? 1,
          phiScore: row.phi_score ?? 0,
          basinEmbedding: parseVector(row.basin_embedding),
          sourceType: row.source_type ?? 'system',
          tokenRole: (row.token_role as 'encoding' | 'generation' | 'both') ?? 'encoding',
          phraseCategory: row.phrase_category ?? 'unknown',
          isRealWord: row.is_real_word ?? false,
          tokenStatus: (row.token_status as 'active' | 'quarantined' | 'deprecated') ?? 'active',
          source: 'system',
          dryRun
        })

        if (result.tokenStatus === 'quarantined') {
          quarantined += 1
        } else if (row.qfi_score !== result.qfiScore || row.token_status !== result.tokenStatus) {
          updated += 1
        } else {
          unchanged += 1
        }
      } catch (error) {
        quarantined += 1
        errors += 1
      }
    }

    lastId = rows[rows.length - 1].id
  }

  console.log('QFI recompute summary')
  console.log(`- total scanned: ${total}`)
  console.log(`- updated qfi: ${updated}`)
  console.log(`- quarantined: ${quarantined}`)
  console.log(`- unchanged: ${unchanged}`)
  console.log(`- errors: ${errors}`)

  if (dryRun) {
    console.log('Dry run complete (no updates applied)')
  }
}

main().catch((error) => {
  console.error('QFI recompute failed:', error)
  process.exit(1)
})
