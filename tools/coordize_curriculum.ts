#!/usr/bin/env tsx
import { loadCurriculumTokens } from '../server/curriculum'
import { upsertToken } from '../server/vocabulary-persistence'
import { oceanQIGBackend } from '../server/ocean-qig-backend-adapter'
import { generateTokenIdFromHash } from '../server/lib/hash-utils'

interface Summary {
  total: number
  inserted: number
  quarantined: number
  failed: number
}

function parseArgs() {
  const apply = process.argv.includes('--apply')
  const dryRun = process.argv.includes('--dry-run') || !apply
  return { apply, dryRun }
}

async function main() {
  const { apply, dryRun } = parseArgs()
  const tokens = loadCurriculumTokens()

  const summary: Summary = {
    total: tokens.length,
    inserted: 0,
    quarantined: 0,
    failed: 0,
  }

  for (const entry of tokens) {
    try {
      if (dryRun) {
        console.log(`[DRY RUN] Would coordize ${entry.token}`)
        continue
      }

      const basinResult = await oceanQIGBackend.computeBasinCoords(entry.token)
      const tokenId = generateTokenIdFromHash(entry.token)
      const result = await upsertToken({
        token: entry.token,
        tokenId,
        frequency: entry.frequency ?? 1,
        basinEmbedding: basinResult.basinCoords,
        tokenRole: 'generation',
        phraseCategory: entry.role,
        isRealWord: entry.is_real_word,
        sourceType: 'curriculum',
        source: 'curriculum',
      })

      if (result.status === 'quarantined') {
        summary.quarantined += 1
      } else {
        summary.inserted += 1
      }
    } catch (error) {
      summary.failed += 1
      console.error(`Failed to coordize ${entry.token}:`, error)
    }
  }

  console.log('Curriculum coordization summary:')
  console.log(`- total tokens: ${summary.total}`)
  console.log(`- inserted/updated: ${summary.inserted}`)
  console.log(`- quarantined: ${summary.quarantined}`)
  console.log(`- failures: ${summary.failed}`)

  if (dryRun) {
    console.log('Dry run complete. Use --apply to write changes.')
  }
}

main().catch((error) => {
  console.error('Curriculum coordization failed:', error)
  process.exit(1)
})
