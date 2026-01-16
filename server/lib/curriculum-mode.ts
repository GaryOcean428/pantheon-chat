import { readFileSync } from 'fs'
import path from 'path'
import { sql } from 'drizzle-orm'
import { db, withDbRetry } from '../db'

export interface CurriculumEntry {
  token: string
  role: string
  is_real_word: boolean
  frequency?: number
  notes?: string
}

const CURRICULUM_ONLY_FLAG = 'true'
const MANIFEST_PATH = path.resolve(process.cwd(), 'data/curriculum/curriculum_tokens.jsonl')
const STATUS_TTL_MS = 60_000

let cachedManifest: CurriculumEntry[] | null = null
let cachedStatus:
  | {
      checkedAt: number
      complete: boolean
      missing: string[]
      invalid: string[]
    }
  | null = null

export function isCurriculumOnlyEnabled(): boolean {
  return process.env.QIG_CURRICULUM_ONLY === CURRICULUM_ONLY_FLAG
}

export function loadCurriculumManifest(): CurriculumEntry[] {
  if (cachedManifest) {
    return cachedManifest
  }

  let content: string
  try {
    content = readFileSync(MANIFEST_PATH, 'utf-8')
  } catch {
    // File doesn't exist - return empty manifest
    cachedManifest = []
    return []
  }

  const entries = content
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as CurriculumEntry)

  cachedManifest = entries
  return entries
}

export async function getCurriculumStatus() {
  const now = Date.now()
  if (cachedStatus && now - cachedStatus.checkedAt < STATUS_TTL_MS) {
    return cachedStatus
  }

  const entries = loadCurriculumManifest()
  const tokens = entries.map((entry) => entry.token)

  // Helper to create incomplete status
  const createIncompleteStatus = () => ({
    checkedAt: now,
    complete: false,
    missing: tokens,
    invalid: [],
  })

  // Capture and validate db reference early
  const dbInstance = db
  if (!dbInstance || tokens.length === 0) {
    cachedStatus = createIncompleteStatus()
    return cachedStatus
  }

  const result = await withDbRetry(
    async () => {
      // Cast tokens to proper SQL array format
      const tokensArray = tokens as string[]
      return dbInstance.execute<{
        token: string
        qfi_score: number | null
      }>(sql`
        SELECT token, qfi_score
        FROM coordizer_vocabulary
        WHERE token = ANY(ARRAY[${sql.join(tokensArray.map(t => sql`${t}`), sql`, `)}]::text[])
      `)
    },
    'curriculum-status'
  )

  if (!result) {
    cachedStatus = createIncompleteStatus()
    return cachedStatus
  }

  const found = new Map(result.rows?.map((row) => [row.token, row]) ?? [])
  const missing = tokens.filter((token) => !found.has(token))
  // Token is invalid if QFI is null or outside valid range [0, 1]
  const invalid = Array.from(found.values())
    .filter(
      (row) =>
        row.qfi_score === null ||
        row.qfi_score < 0 ||
        row.qfi_score > 1
    )
    .map((row) => row.token)

  cachedStatus = {
    checkedAt: now,
    complete: missing.length === 0 && invalid.length === 0,
    missing,
    invalid,
  }

  return cachedStatus
}

export function resetCurriculumCache() {
  cachedManifest = null
  cachedStatus = null
}
