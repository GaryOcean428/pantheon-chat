#!/usr/bin/env tsx
import { and, isNotNull } from 'drizzle-orm'
import { coordizerVocabulary } from '@shared/schema'
import { compute_qfi_score_simplex, isValidQfiScore } from '@shared/qfi'
import { db } from '../server/db'
import { upsertToken } from '../server/vocabulary-persistence'

interface Summary {
  total: number
  updated: number
  quarantined: number
  unchanged: number
  errors: number
}

function parseArgs() {
  const apply = process.argv.includes('--apply')
  const dryRun = process.argv.includes('--dry-run') || !apply
  return { apply, dryRun }
}

async function main() {
  if (!db) {
    throw new Error('Database unavailable')
  }

  const { apply, dryRun } = parseArgs()
  const summary: Summary = {
    total: 0,
    updated: 0,
    quarantined: 0,
    unchanged: 0,
    errors: 0,
  }

  const batchSize = 200
  let offset = 0
  let hasMore = true

  while (hasMore) {
    const rows = await db
      .select()
      .from(coordizerVocabulary)
      .where(and(isNotNull(coordizerVocabulary.basinEmbedding)))
      .limit(batchSize)
      .offset(offset)

    if (rows.length === 0) {
      hasMore = false
      break
    }

    for (const row of rows) {
      summary.total += 1
      try {
        const basin = row.basinEmbedding ?? []
        const qfiScore = compute_qfi_score_simplex(basin)
        const valid = isValidQfiScore(qfiScore)

        if (!valid) {
          summary.quarantined += 1
        }

        if (!dryRun && apply) {
          await upsertToken({
            token: row.token,
            tokenId: row.tokenId,
            weight: row.weight ?? null,
            frequency: row.frequency ?? null,
            phiScore: row.phiScore ?? null,
            basinEmbedding: row.basinEmbedding ?? null,
            scale: row.scale ?? null,
            sourceType: row.sourceType ?? null,
            tokenRole: row.tokenRole ?? null,
            phraseCategory: row.phraseCategory ?? null,
            isRealWord: row.isRealWord ?? null,
            source: 'system',
          })
          summary.updated += 1
        } else {
          summary.unchanged += 1
        }
      } catch (error) {
        summary.errors += 1
      }
    }

    offset += batchSize
  }

  console.log('QFI recompute summary:')
  console.log(`- total scanned: ${summary.total}`)
  console.log(`- updated qfi: ${summary.updated}`)
  console.log(`- quarantined: ${summary.quarantined}`)
  console.log(`- unchanged: ${summary.unchanged}`)
  console.log(`- errors: ${summary.errors}`)

  if (!dryRun && apply) {
    console.log('Updates applied.')
  } else {
    console.log('Dry run complete. Use --apply to write changes.')
  }
}

main().catch((error) => {
  console.error('Failed to recompute QFI scores:', error)
  process.exit(1)
})
