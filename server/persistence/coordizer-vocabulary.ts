import { sql } from 'drizzle-orm'
import { coordizerVocabulary } from '@shared/schema'
import { QIG_CONSTANTS, compute_qfi_score_simplex, to_simplex_probabilities } from '@shared'
import { db, withDbRetry } from '../db'
import { logger } from '../lib/logger'
import { isCurriculumOnlyEnabled } from '../lib/curriculum-mode'

const CURRICULUM_ONLY_FLAG = 'true'
const VALID_TOKEN_STATUSES = ['active', 'quarantined', 'deprecated'] as const

type TokenStatus = (typeof VALID_TOKEN_STATUSES)[number]

export interface UpsertTokenInput {
  token: string
  tokenId: number
  basinEmbedding?: number[] | null
  weight?: number | null
  frequency?: number | null
  phiScore?: number | null
  sourceType?: string | null
  tokenRole?: string | null
  phraseCategory?: string | null
  isRealWord?: boolean | null
  source?: string
}

interface PreparedTokenValues {
  basinEmbedding: number[] | null
  qfiScore: number | null
  tokenStatus: TokenStatus
}

export function isQfiScoreValid(value: number | null | undefined): boolean {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
}

export function assertCandidateTokensHaveValidQfi(
  tokens: Array<{ qfiScore?: number | null; tokenStatus?: string | null }>,
  context: string
): void {
  const invalid = tokens.filter(
    (token) =>
      token.tokenStatus === 'active' &&
      !isQfiScoreValid(token.qfiScore)
  )

  if (invalid.length > 0) {
    throw new Error(
      `[QFI] Invalid qfi_score detected in ${context}: ${invalid.length} token(s)`
    )
  }
}

function enforceCurriculumOnly(source?: string) {
  if (!isCurriculumOnlyEnabled()) {
    return
  }

  if (source !== 'curriculum') {
    throw new Error('Curriculum-only mode: upsert_token requires source=curriculum')
  }
}

export function prepareUpsertTokenValues(
  basinEmbedding?: number[] | null
): PreparedTokenValues {
  if (!basinEmbedding || basinEmbedding.length === 0) {
    return { basinEmbedding: null, qfiScore: null, tokenStatus: 'quarantined' }
  }

  if (basinEmbedding.length !== QIG_CONSTANTS.BASIN_DIMENSION) {
    return { basinEmbedding: null, qfiScore: null, tokenStatus: 'quarantined' }
  }

  try {
    const qfiScore = compute_qfi_score_simplex(basinEmbedding)
    const normalized = to_simplex_probabilities(basinEmbedding)

    if (!isQfiScoreValid(qfiScore)) {
      return { basinEmbedding: normalized, qfiScore: null, tokenStatus: 'quarantined' }
    }

    return { basinEmbedding: normalized, qfiScore, tokenStatus: 'active' }
  } catch (error) {
    logger.warn({ error }, '[QFI] Failed to compute qfi_score, quarantining token')
    return { basinEmbedding: null, qfiScore: null, tokenStatus: 'quarantined' }
  }
}

export async function upsertToken(input: UpsertTokenInput) {
  enforceCurriculumOnly(input.source)

  // Capture and validate db reference early
  const dbInstance = db
  if (!dbInstance) {
    return null
  }

  const prepared = prepareUpsertTokenValues(input.basinEmbedding ?? null)
  const now = new Date()

  return withDbRetry(
    async () => {
      const [result] = await dbInstance
        .insert(coordizerVocabulary)
        .values({
          token: input.token,
          tokenId: input.tokenId,
          weight: input.weight ?? 1,
          frequency: input.frequency ?? 1,
          phiScore: input.phiScore ?? 0,
          basinEmbedding: prepared.basinEmbedding,
          qfiScore: prepared.qfiScore,
          tokenRole: input.tokenRole ?? 'encoding',
          phraseCategory: input.phraseCategory ?? 'unknown',
          isRealWord: input.isRealWord ?? false,
          tokenStatus: prepared.tokenStatus,
          sourceType: input.sourceType ?? 'base',
          updatedAt: now,
        })
        .onConflictDoUpdate({
          target: coordizerVocabulary.token,
          set: {
            weight: sql`excluded.weight`,
            frequency: sql`excluded.frequency`,
            phiScore: sql`excluded.phiScore`,
            basinEmbedding: sql`excluded.basinEmbedding`,
            qfiScore: sql`excluded.qfiScore`,
            tokenRole: sql`excluded.tokenRole`,
            phraseCategory: sql`excluded.phraseCategory`,
            isRealWord: sql`excluded.isRealWord`,
            tokenStatus: sql`excluded.tokenStatus`,
            sourceType: sql`excluded.sourceType`,
            updatedAt: now,
          },
        })
        .returning()

      return result ?? null
    },
    'upsertToken'
  )
}

export function activeVocabularyFilter() {
  return sql`${coordizerVocabulary.tokenStatus} = 'active' AND ${coordizerVocabulary.qfiScore} IS NOT NULL AND ${coordizerVocabulary.qfiScore} BETWEEN 0 AND 1`
}
