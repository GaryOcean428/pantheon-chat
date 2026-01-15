import fs from 'fs'
import path from 'path'
import { describe, expect, it, vi } from 'vitest'

import { compute_qfi_score_simplex } from '@shared/qfi'

vi.mock('../db', () => {
  return {
    db: {
      execute: vi.fn(async () => ({ rows: [] }))
    },
    withDbRetry: async <T>(fn: () => Promise<T>) => fn()
  }
})

import { upsertToken } from '../persistence/vocabulary'
describe('compute_qfi_score_simplex', () => {
  it('returns finite values in [0,1] for random inputs', () => {
    for (let i = 0; i < 50; i += 1) {
      const basin = Array.from({ length: 64 }, () => (Math.random() * 2) - 1)
      const score = compute_qfi_score_simplex(basin)
      expect(Number.isFinite(score)).toBe(true)
      expect(score).toBeGreaterThanOrEqual(0)
      expect(score).toBeLessThanOrEqual(1)
    }
  })
})

describe('upsertToken', () => {
  it('computes qfi_score in [0,1] for persisted basins', async () => {
    const result = await upsertToken({
      token: 'test-token',
      tokenId: 12345,
      weight: 1,
      frequency: 1,
      phiScore: 0.5,
      basinEmbedding: Array.from({ length: 64 }, (_, idx) => idx + 1),
      sourceType: 'test',
      tokenRole: 'generation',
      phraseCategory: 'test',
      isRealWord: true,
      source: 'curriculum'
    })

    expect(result.qfiScore).not.toBeNull()
    expect(result.qfiScore).toBeGreaterThanOrEqual(0)
    expect(result.qfiScore).toBeLessThanOrEqual(1)
    expect(result.tokenStatus).toBe('active')
  })
})

describe('QFI enforcement guards', () => {
  const repoRoot = path.resolve(__dirname, '..', '..')
  const scanDirs = ['server', 'scripts', 'tools', 'shared']
  const scanExtensions = new Set(['.ts', '.tsx', '.py'])
  const bannedFunctionPatterns = [
    /\bcompute_qfi\s*\(/,
    /\bcompute_qfi_for_basin\s*\(/,
    /\bdeterminant_qfi\s*\(/
  ]
  const allowedWriter = path.join('server', 'persistence', 'vocabulary.ts')

  function walk(dir: string, files: string[] = []) {
    const entries = fs.readdirSync(dir, { withFileTypes: true })
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        if (entry.name === 'node_modules' || entry.name === 'dist') {
          continue
        }
        walk(fullPath, files)
      } else {
        files.push(fullPath)
      }
    }
    return files
  }

  it('does not reference deprecated QFI function names', () => {
    const violations: string[] = []

    for (const dir of scanDirs) {
      const dirPath = path.join(repoRoot, dir)
      if (!fs.existsSync(dirPath)) {
        continue
      }

      const files = walk(dirPath)
      for (const filePath of files) {
        if (!scanExtensions.has(path.extname(filePath))) {
          continue
        }

        const content = fs.readFileSync(filePath, 'utf-8')
        for (const pattern of bannedFunctionPatterns) {
          if (pattern.test(content)) {
            violations.push(`${path.relative(repoRoot, filePath)} uses deprecated QFI naming`)
          }
        }
      }
    }

    expect(violations).toEqual([])
  })

  it('only writes qfi_score via upsertToken', () => {
    const violations: string[] = []
    const writePattern = /SET\\s+qfi_score|qfi_score\\s*=|INSERT INTO\\s+coordizer_vocabulary|UPDATE\\s+coordizer_vocabulary/i

    for (const dir of scanDirs) {
      const dirPath = path.join(repoRoot, dir)
      if (!fs.existsSync(dirPath)) {
        continue
      }

      const files = walk(dirPath)
      for (const filePath of files) {
        const relative = path.relative(repoRoot, filePath)
        if (!scanExtensions.has(path.extname(filePath))) {
          continue
        }

        if (relative === allowedWriter) {
          continue
        }

        const content = fs.readFileSync(filePath, 'utf-8')
        if (content.includes('qfi_score') && writePattern.test(content)) {
          violations.push(`${relative} writes qfi_score outside upsertToken`)
        }
      }
    }

    expect(violations).toEqual([])
  })
})
