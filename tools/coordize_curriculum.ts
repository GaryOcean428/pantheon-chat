import { readFileSync } from 'fs'
import { sql } from 'drizzle-orm'
import { db, withDbRetry } from '../server/db'
import { upsertToken } from '../server/persistence/coordizer-vocabulary'

interface CurriculumEntry {
  token: string
  role: string
  is_real_word: boolean
  frequency?: number
  notes?: string
}

const MANIFEST_PATH = 'data/curriculum/curriculum_tokens.jsonl'
const BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001'

function loadManifest(): CurriculumEntry[] {
  const content = readFileSync(MANIFEST_PATH, 'utf-8')
  return content
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as CurriculumEntry)
}

function extractBasinCoords(payload: Record<string, unknown>): number[] | null {
  const candidates = [
    payload?.basin_coords,
    payload?.basinCoordinates,
    payload?.basin_embedding,
    payload?.basin,
    payload?.coordinates,
    (payload?.data as Record<string, unknown>)?.basin_coords,
    (payload?.data as Record<string, unknown>)?.basinCoordinates,
  ]

  for (const candidate of candidates) {
    if (Array.isArray(candidate) && candidate.length === 64) {
      return candidate.map((value) => Number(value))
    }
  }

  return null
}

async function coordizeToken(token: string): Promise<number[] | null> {
  const response = await fetch(`${BACKEND_URL}/api/coordize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: token }),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Coordizer error: ${response.status} ${text}`)
  }

  const payload = await response.json() as Record<string, unknown>
  return extractBasinCoords(payload)
}

async function run() {
  if (!db) {
    console.error('[Curriculum] Database not configured')
    process.exit(1)
  }

  const entries = loadManifest()
  const tokens = entries.map((entry) => entry.token)

  const existing = await withDbRetry(
    async () =>
      db.execute<{ token: string; token_id: number }>(sql`
        SELECT token, token_id
        FROM coordizer_vocabulary
        WHERE token = ANY(${tokens})
      `),
    'curriculum-existing-tokens'
  )

  const existingMap = new Map(
    (existing.rows ?? []).map((row) => [row.token, row.token_id])
  )

  const maxTokenIdResult = await withDbRetry(
    async () =>
      db.execute<{ max_id: number }>(sql`
        SELECT COALESCE(MAX(token_id), 0)::int AS max_id
        FROM coordizer_vocabulary
      `),
    'curriculum-max-token-id'
  )

  let nextTokenId = (maxTokenIdResult.rows?.[0]?.max_id ?? 0) + 1
  let updated = 0
  let quarantined = 0
  let failures = 0

  for (const entry of entries) {
    try {
      const basinEmbedding = await coordizeToken(entry.token)
      if (!basinEmbedding) {
        failures++
        console.error(`[Curriculum] Missing basin for ${entry.token}`)
        continue
      }

      const tokenId = existingMap.get(entry.token) ?? nextTokenId++
      const result = await upsertToken({
        token: entry.token,
        tokenId,
        basinEmbedding,
        frequency: entry.frequency ?? 1,
        sourceType: 'curriculum',
        tokenRole: 'generation',
        phraseCategory: entry.role,
        isRealWord: entry.is_real_word,
        source: 'curriculum',
      })

      if (result?.tokenStatus === 'quarantined') {
        quarantined++
      } else {
        updated++
      }
    } catch (error) {
      failures++
      console.error(`[Curriculum] Failed to coordize ${entry.token}:`, error)
    }
  }

  console.log('[Curriculum] Completion report')
  console.log(`  total tokens: ${entries.length}`)
  console.log(`  successfully inserted/updated: ${updated}`)
  console.log(`  quarantined: ${quarantined}`)
  console.log(`  failures: ${failures}`)
}

run().catch((error) => {
  console.error('[Curriculum] Failed:', error)
  process.exit(1)
})
