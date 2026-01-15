import { sql } from 'drizzle-orm'

import { upsertToken } from '../server/persistence/vocabulary'
import { loadCurriculumManifest } from '../server/curriculum'
import { db } from '../server/db'

const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001'

function extractBasinEmbedding(payload: Record<string, unknown>): number[] | null {
  const candidateKeys = [
    'basin_embedding',
    'basin',
    'basin_coords',
    'basin_coordinates',
    'embedding',
    'coordinates'
  ]

  for (const key of candidateKeys) {
    const value = payload[key]
    if (Array.isArray(value) && value.every((item) => typeof item === 'number')) {
      return value as number[]
    }
  }

  for (const value of Object.values(payload)) {
    if (Array.isArray(value) && value.every((item) => typeof item === 'number')) {
      return value as number[]
    }
  }

  return null
}

/**
 * Coordizes a single token by calling the Python QIG backend.
 * 
 * NOTE: Direct fetch call is used here as an exception to the centralized API client pattern.
 * This is a backend tool/script (not frontend code) that needs to call an external Python service.
 * The architectural pattern (§2 Centralized API Client) primarily applies to frontend components.
 * For backend-to-backend HTTP calls in tools, direct fetch is acceptable and simpler than
 * introducing a backend HTTP utility layer for a single coordize operation.
 */
async function coordizeToken(token: string): Promise<number[]> {
  const response = await fetch(`${backendUrl}/api/coordize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: token })
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Coordize failed: ${response.status} ${errorText}`)
  }

  const payload = await response.json() as Record<string, unknown>
  const embedding = extractBasinEmbedding(payload)

  if (!embedding) {
    throw new Error('Coordize response missing basin embedding')
  }

  return embedding
}

async function main() {
  if (!db) {
    throw new Error('Database not available')
  }

  const manifest = loadCurriculumManifest()
  if (manifest.length === 0) {
    throw new Error('Curriculum manifest is empty')
  }

  const tokens = manifest.map((entry) => entry.token.toLowerCase())

  const existingRows = await db.execute<{ token: string; token_id: number }>(sql`
    SELECT token, token_id
    FROM coordizer_vocabulary
    WHERE token = ANY(${tokens})
  `)

  const existingMap = new Map<string, number>()
  for (const row of existingRows.rows ?? []) {
    existingMap.set(row.token, row.token_id)
  }

  const maxIdResult = await db.execute<{ max_id: number }>(sql`
    SELECT COALESCE(MAX(token_id), 0)::int AS max_id
    FROM coordizer_vocabulary
  `)
  let nextId = (maxIdResult.rows?.[0]?.max_id ?? 0) + 1

  let success = 0
  let failures = 0
  let quarantined = 0

  for (const entry of manifest) {
    try {
      const embedding = await coordizeToken(entry.token)
      const tokenId = existingMap.get(entry.token) ?? nextId++

      const result = await upsertToken({
        token: entry.token.toLowerCase(),
        tokenId,
        weight: 1,
        frequency: entry.frequency ?? 1,
        phiScore: 0.5,
        basinEmbedding: embedding,
        sourceType: 'curriculum',
        tokenRole: 'generation',
        phraseCategory: entry.role ?? 'curriculum',
        isRealWord: entry.is_real_word ?? true,
        source: 'curriculum'
      })

      if (result.tokenStatus === 'quarantined') {
        quarantined += 1
      } else {
        success += 1
      }
    } catch (error) {
      console.error(`Failed to coordize ${entry.token}:`, error)
      failures += 1
    }
  }

  console.log('Curriculum coordization report')
  console.log(`- total tokens: ${manifest.length}`)
  console.log(`- inserted/updated: ${success}`)
  console.log(`- quarantined: ${quarantined}`)
  console.log(`- failures: ${failures}`)
}

main().catch((error) => {
  console.error('Curriculum coordization failed:', error)
  process.exit(1)
})
