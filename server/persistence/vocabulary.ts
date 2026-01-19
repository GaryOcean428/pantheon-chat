import { sql } from 'drizzle-orm'

import { compute_qfi_score_simplex, isValidQfiScore, toSimplexProbabilities } from '@shared/qfi'

import { db, withDbRetry } from '../db'
import { isCurriculumOnlyMode } from '../curriculum'

export type TokenSource = 'curriculum' | 'sync' | 'seed' | 'system' | 'manual' | 'import'

export type UpsertTokenInput = {
  token: string
  tokenId: number
  weight?: number
  frequency?: number
  phiScore?: number
  basinEmbedding?: number[] | string | null
  sourceType?: string
  tokenRole?: 'encoding' | 'generation' | 'both'
  phraseCategory?: string
  isRealWord?: boolean
  source: TokenSource
  dryRun?: boolean
}

export type UpsertTokenResult = {
  qfiScore: number | null
  isActive: boolean
}

const vectorPrefix = '['
const vectorSuffix = ']'

function parseVectorString(vector: string): number[] {
  const trimmed = vector.trim()
  const content = trimmed.startsWith(vectorPrefix) && trimmed.endsWith(vectorSuffix)
    ? trimmed.slice(1, -1)
    : trimmed

  if (!content) {
    return []
  }

  return content.split(',').map((value) => Number.parseFloat(value))
}

function formatVector(values: number[]): string {
  return `${vectorPrefix}${values.join(',')}${vectorSuffix}`
}

/**
 * Normalizes basin embeddings to simplex probabilities before storage.
 * 
 * All basin embeddings are stored as simplex-normalized values
 * using the canonical toSimplexProbabilities transformation. This ensures:
 * - All coordinates are strictly positive (required for QFI computation)
 * - Vector sums to 1 (simplex constraint)
 * - Consistent geometric interpretation across the codebase
 */
function normalizeBasinEmbedding(basinEmbedding?: number[] | string | null): number[] | null {
  if (!basinEmbedding) {
    return null
  }

  const values = Array.isArray(basinEmbedding) ? basinEmbedding : parseVectorString(basinEmbedding)
  if (values.length === 0) {
    return null
  }

  return toSimplexProbabilities(values)
}

export async function upsertToken(input: UpsertTokenInput): Promise<UpsertTokenResult> {
  if (isCurriculumOnlyMode() && input.source !== 'curriculum') {
    throw new Error('Curriculum-only mode: token upsert rejected')
  }

  if (!db) {
    throw new Error('Database not available for token upsert')
  }

  const simplexEmbedding = normalizeBasinEmbedding(input.basinEmbedding)

  let qfiScore: number | null = null

  if (simplexEmbedding) {
    try {
      qfiScore = compute_qfi_score_simplex(simplexEmbedding)
    } catch {
      qfiScore = null
    }
  }

  if (!isValidQfiScore(qfiScore)) {
    qfiScore = null
  }

  const vectorValue = simplexEmbedding ? formatVector(simplexEmbedding) : null

  if (!input.dryRun) {
    await withDbRetry(
      async () => db!.execute(sql`
        INSERT INTO coordizer_vocabulary (
          token,
          token_id,
          weight,
          frequency,
          phi_score,
          basin_embedding,
          source_type,
          token_role,
          phrase_category,
          is_real_word,
          qfi_score,
          updated_at
        ) VALUES (
          ${input.token},
          ${input.tokenId},
          ${input.weight ?? 1},
          ${input.frequency ?? 1},
          ${input.phiScore ?? 0},
          ${vectorValue}::vector,
          ${input.sourceType ?? 'base'},
          ${input.tokenRole ?? 'encoding'},
          ${input.phraseCategory ?? 'unknown'},
          ${input.isRealWord ?? false},
          ${qfiScore},
          CURRENT_TIMESTAMP
        )
        ON CONFLICT (token) DO UPDATE SET
          weight = EXCLUDED.weight,
          frequency = EXCLUDED.frequency,
          phi_score = EXCLUDED.phi_score,
          basin_embedding = EXCLUDED.basin_embedding,
          source_type = EXCLUDED.source_type,
          token_role = EXCLUDED.token_role,
          phrase_category = EXCLUDED.phrase_category,
          is_real_word = EXCLUDED.is_real_word,
          qfi_score = EXCLUDED.qfi_score,
          updated_at = CURRENT_TIMESTAMP
      `),
      `upsert-token-${input.token}`
    )
  }

  // Token is "active" if it has a valid QFI score
  return { qfiScore, isActive: isValidQfiScore(qfiScore) }
}
