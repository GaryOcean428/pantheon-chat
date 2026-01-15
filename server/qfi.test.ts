import { describe, expect, it, vi } from 'vitest'
import { compute_qfi_score_simplex } from '@shared/qfi'

vi.mock('./db', () => {
  const onConflictDoUpdate = vi.fn().mockResolvedValue(undefined)
  const values = vi.fn(() => ({ onConflictDoUpdate }))
  const insert = vi.fn(() => ({ values }))
  return {
    db: { insert },
    withDbRetry: (fn: () => Promise<void>) => fn(),
  }
})

describe('compute_qfi_score_simplex', () => {
  it('returns a finite score between 0 and 1 for random inputs', () => {
    for (let i = 0; i < 50; i += 1) {
      const basin = Array.from({ length: 64 }, () => Math.random() - 0.5)
      const score = compute_qfi_score_simplex(basin)
      expect(Number.isFinite(score)).toBe(true)
      expect(score).toBeGreaterThanOrEqual(0)
      expect(score).toBeLessThanOrEqual(1)
    }
  })
})

describe('upsertToken', () => {
  it('persists a computed qfi_score within range', async () => {
    const { upsertToken } = await import('./vocabulary-persistence')
    const result = await upsertToken({
      token: 'test-token',
      tokenId: 42,
      basinEmbedding: Array.from({ length: 64 }, () => 0.1),
      source: 'seed',
    })

    expect(result.qfiScore).not.toBeNull()
    expect(result.qfiScore ?? 0).toBeGreaterThanOrEqual(0)
    expect(result.qfiScore ?? 0).toBeLessThanOrEqual(1)
  })
})
