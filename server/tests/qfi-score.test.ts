import { describe, expect, it } from 'vitest'
import path from 'path'
import { readdirSync, readFileSync, statSync } from 'fs'
import { compute_qfi_score_simplex } from '@shared'
import { prepareUpsertTokenValues } from '../persistence/coordizer-vocabulary'

const BASIN_DIMENSION = 64
const repoRoot = path.resolve(__dirname, '..', '..')

function randomBasin(length: number): number[] {
  return Array.from({ length }, () => (Math.random() - 0.5) * 4)
}

function collectFiles(root: string, extensions: string[], skipDirs: Set<string>): string[] {
  const results: string[] = []
  const entries = readdirSync(root)

  for (const entry of entries) {
    const fullPath = path.join(root, entry)
    const stat = statSync(fullPath)

    if (stat.isDirectory()) {
      if (skipDirs.has(entry)) {
        continue
      }
      results.push(...collectFiles(fullPath, extensions, skipDirs))
      continue
    }

    if (extensions.some((ext) => fullPath.endsWith(ext))) {
      results.push(fullPath)
    }
  }

  return results
}

describe('compute_qfi_score_simplex', () => {
  it('returns finite values in [0,1] for random inputs', () => {
    for (let i = 0; i < 50; i++) {
      const basin = randomBasin(BASIN_DIMENSION)
      const qfi = compute_qfi_score_simplex(basin)
      expect(Number.isFinite(qfi)).toBe(true)
      expect(qfi).toBeGreaterThanOrEqual(0)
      expect(qfi).toBeLessThanOrEqual(1)
    }
  })
})

describe('prepareUpsertTokenValues', () => {
  it('produces valid qfi_score for persistence', () => {
    const basin = randomBasin(BASIN_DIMENSION)
    const prepared = prepareUpsertTokenValues(basin)
    expect(prepared.tokenStatus).toBe('active')
    expect(prepared.qfiScore).not.toBeNull()
    expect(prepared.qfiScore).toBeGreaterThanOrEqual(0)
    expect(prepared.qfiScore).toBeLessThanOrEqual(1)
  })
})

describe('QFI regressions', () => {
  it('does not reference legacy compute_qfi helpers in canonical paths', () => {
    const targetDirs = ['server', 'shared', 'scripts', 'tools']
    const skipDirs = new Set(['node_modules', 'dist', 'docs'])
    // Only check TypeScript/JavaScript files - Python files can't import TS functions
    const files = targetDirs.flatMap((dir) =>
      collectFiles(path.join(repoRoot, dir), ['.ts', '.tsx', '.js'], skipDirs)
    )

    const legacyPattern = /\bcompute_qfi\s*\(/
    const allowedPattern = /compute_qfi_score_simplex/i
    const violations = files.filter((file) => {
      const content = readFileSync(file, 'utf-8')
      return legacyPattern.test(content) && !allowedPattern.test(content)
    })

    expect(violations).toEqual([])
  })

  it('does not write coordizer_vocabulary outside upsert_token in server code', () => {
    const serverDir = path.join(repoRoot, 'server')
    const serverFiles = collectFiles(serverDir, ['.ts', '.tsx', '.js'], new Set(['node_modules', 'dist']))
    const writePattern = /(INSERT INTO|UPDATE\s+\w+\s+SET|DELETE FROM)\s+coordizer_vocabulary/i
    // Allow both persistence files that handle canonical vocabulary writes
    const allowedPaths = new Set([
      path.join(repoRoot, 'server', 'persistence', 'coordizer-vocabulary.ts'),
      path.join(repoRoot, 'server', 'persistence', 'vocabulary.ts')
    ])

    const violations = serverFiles.filter((file) => {
      if (allowedPaths.has(file)) {
        return false
      }
      return writePattern.test(readFileSync(file, 'utf-8'))
    })

    expect(violations).toEqual([])
  })
})
