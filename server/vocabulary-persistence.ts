import type { CoordizerVocabularyRow } from '@shared/schema'
import { coordizerVocabulary } from '@shared/schema'
import { BASIN_DIMENSION } from '@shared/constants'
import { compute_qfi_score_simplex, isValidQfiScore, toSimplexProbabilities } from '@shared/qfi'
import { db, withDbRetry } from './db'

export type TokenStatus = 'active' | 'quarantined' | 'deprecated'

export interface UpsertTokenInput {
  token: string
  tokenId: number
  weight?: number | null
  frequency?: number | null
  phiScore?: number | null
  basinEmbedding?: ArrayLike<number> | null
  scale?: string | null
  sourceType?: string | null
  tokenRole?: 'encoding' | 'generation' | 'both' | null
  phraseCategory?: string | null
  isRealWord?: boolean | null
  source?: 'curriculum' | 'sync' | 'seed' | 'manual' | 'system' | 'unknown'
}

export interface UpsertTokenResult {
  status: TokenStatus
  qfiScore: number | null
  basinEmbedding: number[] | null
}

function getCurriculumOnlyFlag(): boolean {
  return process.env.QIG_CURRICULUM_ONLY === 'true'
}

function resolveTokenStatus(qfiScore: number | null): TokenStatus {
  return isValidQfiScore(qfiScore) ? 'active' : 'quarantined'
}

function normalizeBasin(basin: ArrayLike<number> | null): number[] | null {
  if (!basin) return null
  const values = Array.from(basin).map((value) => Number(value))
  if (values.length !== BASIN_DIMENSION) {
    throw new Error(`Invalid basin dimension: expected ${BASIN_DIMENSION}, got ${values.length}`)
  }
  return toSimplexProbabilities(values)
}

export async function upsertToken(input: UpsertTokenInput): Promise<UpsertTokenResult> {
  if (!db) {
    throw new Error('Database unavailable')
  }

  if (getCurriculumOnlyFlag() && input.source !== 'curriculum') {
    throw new Error('Curriculum-only mode: token upsert requires source=curriculum')
  }

  let normalizedBasin: number[] | null = null
  let qfiScore: number | null = null

  if (input.basinEmbedding && Array.from(input.basinEmbedding).length > 0) {
    normalizedBasin = normalizeBasin(input.basinEmbedding)
    if (normalizedBasin) {
      qfiScore = compute_qfi_score_simplex(normalizedBasin)
      if (!isValidQfiScore(qfiScore)) {
        qfiScore = null
      }
    }
  }

  const status = resolveTokenStatus(qfiScore)

  const record = {
    token: input.token,
    tokenId: input.tokenId,
    weight: input.weight ?? undefined,
    frequency: input.frequency ?? undefined,
    phiScore: input.phiScore ?? undefined,
    basinEmbedding: normalizedBasin ?? undefined,
    scale: input.scale ?? undefined,
    sourceType: input.sourceType ?? undefined,
    tokenRole: input.tokenRole ?? undefined,
    phraseCategory: input.phraseCategory ?? undefined,
    isRealWord: input.isRealWord ?? undefined,
    qfiScore,
    tokenStatus: status,
  }

  await withDbRetry(
    async () => {
      if (!db) throw new Error('Database unavailable')
      return db.insert(coordizerVocabulary)
        .values(record)
        .onConflictDoUpdate({
          target: coordizerVocabulary.token,
          set: {
            weight: record.weight,
            frequency: record.frequency,
            phiScore: record.phiScore,
            basinEmbedding: record.basinEmbedding,
            scale: record.scale,
            sourceType: record.sourceType,
            tokenRole: record.tokenRole,
            phraseCategory: record.phraseCategory,
            isRealWord: record.isRealWord,
            qfiScore: record.qfiScore,
            tokenStatus: record.tokenStatus,
            updatedAt: new Date(),
          },
        })
    },
    'upsert-token'
  )

  return {
    status,
    qfiScore,
    basinEmbedding: normalizedBasin,
  }
}

export function assertValidQfiForToken(token: CoordizerVocabularyRow): void {
  if (!isValidQfiScore(token.qfiScore)) {
    throw new Error(`Invalid qfi_score for token ${token.token}`)
  }
  if (!token.basinEmbedding) {
    throw new Error(`Missing basin_embedding for token ${token.token}`)
  }
}
