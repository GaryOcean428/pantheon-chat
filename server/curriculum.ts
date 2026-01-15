import { readFileSync } from 'fs'
import path from 'path'
import { inArray } from 'drizzle-orm'
import { coordizerVocabulary } from '@shared/schema'
import { isValidQfiScore } from '@shared/qfi'
import { db } from './db'

export interface CurriculumToken {
  token: string
  role: string
  is_real_word: boolean
  frequency?: number
  notes?: string
}

const curriculumPath = path.join(process.cwd(), 'data', 'curriculum', 'curriculum_tokens.jsonl')
let cachedTokens: CurriculumToken[] | null = null

export function loadCurriculumTokens(): CurriculumToken[] {
  if (cachedTokens) return cachedTokens

  const raw = readFileSync(curriculumPath, 'utf-8')
  const tokens = raw
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as CurriculumToken)

  cachedTokens = tokens
  return tokens
}

export function getCurriculumTokens(): string[] {
  return loadCurriculumTokens().map((entry) => entry.token)
}

export async function checkCurriculumCompleteness(): Promise<{
  complete: boolean
  missing: string[]
  quarantined: string[]
}> {
  const tokens = getCurriculumTokens()
  if (!db) {
    throw new Error('Database unavailable')
  }

  if (tokens.length === 0) {
    return { complete: false, missing: [], quarantined: [] }
  }

  const rows = await db
    .select({
      token: coordizerVocabulary.token,
      tokenStatus: coordizerVocabulary.tokenStatus,
      qfiScore: coordizerVocabulary.qfiScore,
      basinEmbedding: coordizerVocabulary.basinEmbedding,
    })
    .from(coordizerVocabulary)
    .where(inArray(coordizerVocabulary.token, tokens))

  const rowMap = new Map(rows.map((row) => [row.token, row]))
  const missing: string[] = []
  const quarantined: string[] = []

  for (const token of tokens) {
    const row = rowMap.get(token)
    if (!row) {
      missing.push(token)
      continue
    }

    if (row.tokenStatus !== 'active') {
      quarantined.push(token)
      continue
    }

    if (!row.basinEmbedding || !isValidQfiScore(row.qfiScore ?? null)) {
      quarantined.push(token)
    }
  }

  return {
    complete: missing.length === 0 && quarantined.length === 0,
    missing,
    quarantined,
  }
}

export async function assertCurriculumReady(): Promise<void> {
  const status = await checkCurriculumCompleteness()
  if (!status.complete) {
    const details = [
      status.missing.length ? `missing=${status.missing.length}` : null,
      status.quarantined.length ? `quarantined=${status.quarantined.length}` : null,
    ]
      .filter(Boolean)
      .join(', ')
    throw new Error(`Curriculum incomplete: ${details}`)
  }
}

export function assertTokensInCurriculum(tokens: string[]): void {
  const curriculumSet = new Set(getCurriculumTokens())
  const invalid = tokens.filter((token) => !curriculumSet.has(token))
  if (invalid.length > 0) {
    throw new Error(`Curriculum-only mode: tokens not in manifest: ${invalid.join(', ')}`)
  }
}
