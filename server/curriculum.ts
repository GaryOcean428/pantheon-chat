import fs from 'fs'
import path from 'path'

import { sql } from 'drizzle-orm'

import { isValidQfiScore } from '@shared/qfi'

import { db, withDbRetry } from './db'

export type CurriculumEntry = {
  token: string
  role?: string
  is_real_word?: boolean
  frequency?: number
  notes?: string
}

const manifestPath = path.join(process.cwd(), 'data', 'curriculum', 'curriculum_tokens.jsonl')
let cachedManifest: CurriculumEntry[] | null = null
let cachedManifestMtime = 0

export function isCurriculumOnlyMode(): boolean {
  return process.env.QIG_CURRICULUM_ONLY === 'true'
}

export function loadCurriculumManifest(): CurriculumEntry[] {
  try {
    const stats = fs.statSync(manifestPath)
    if (cachedManifest && stats.mtimeMs === cachedManifestMtime) {
      return cachedManifest
    }

    const content = fs.readFileSync(manifestPath, 'utf-8')
    const entries = content
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => JSON.parse(line) as CurriculumEntry)
      .filter((entry) => entry.token && entry.token.length > 0)

    cachedManifest = entries
    cachedManifestMtime = stats.mtimeMs
    return entries
  } catch (error) {
    if (isCurriculumOnlyMode()) {
      throw new Error(`Curriculum manifest missing or unreadable: ${manifestPath}`)
    }

    return []
  }
}

export function getCurriculumTokens(): string[] {
  return loadCurriculumManifest().map((entry) => entry.token.toLowerCase())
}

export type CurriculumCoverage = {
  total: number
  active: number
  missingTokens: string[]
  quarantinedTokens: string[]
}

export async function getCurriculumCoverage(): Promise<CurriculumCoverage> {
  const tokens = getCurriculumTokens()
  if (tokens.length === 0) {
    return {
      total: 0,
      active: 0,
      missingTokens: [],
      quarantinedTokens: [],
    }
  }

  if (!db) {
    throw new Error('Database unavailable while checking curriculum coverage')
  }

  const result = await withDbRetry(
    async () => db.execute(sql`
      SELECT token, token_status, qfi_score, basin_embedding
      FROM coordizer_vocabulary
      WHERE token = ANY(${tokens})
    `),
    'curriculum-coverage'
  )

  const rows = result.rows ?? []
  const foundTokens = new Set<string>()
  const quarantinedTokens: string[] = []
  let active = 0

  for (const row of rows) {
    const token = String(row.token)
    foundTokens.add(token)

    const status = row.token_status as string | null
    const qfiScore = row.qfi_score as number | null
    const hasBasin = !!row.basin_embedding

    const isActive = status === 'active' && isValidQfiScore(qfiScore) && hasBasin
    if (isActive) {
      active += 1
    } else {
      quarantinedTokens.push(token)
    }
  }

  const missingTokens = tokens.filter((token) => !foundTokens.has(token))

  return {
    total: tokens.length,
    active,
    missingTokens,
    quarantinedTokens,
  }
}

export async function assertCurriculumReady(): Promise<void> {
  if (!isCurriculumOnlyMode()) {
    return
  }

  const coverage = await getCurriculumCoverage()
  if (coverage.total === 0) {
    throw new Error('Curriculum-only mode requires a non-empty curriculum manifest')
  }

  if (coverage.missingTokens.length > 0 || coverage.quarantinedTokens.length > 0) {
    const missingPreview = coverage.missingTokens.slice(0, 5).join(', ')
    const quarantinedPreview = coverage.quarantinedTokens.slice(0, 5).join(', ')

    throw new Error(
      `Curriculum incomplete. Missing: ${missingPreview || 'none'}; ` +
      `quarantined: ${quarantinedPreview || 'none'}.`
    )
  }
}
