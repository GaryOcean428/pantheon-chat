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

describe('legacy qfi path guard', () => {
  it('does not reference deprecated compute_qfi_for_basin in canonical paths', async () => {
    const { readFile, readdir } = await import('fs/promises')
    const { join } = await import('path')

    const roots = ['server', 'shared', 'scripts', 'tools']
    const deprecated = 'compute_qfi_for_basin'
    const ignore = new Set(['node_modules', 'dist', 'docs', 'migrations'])

    async function scan(dir: string): Promise<void> {
      const entries = await readdir(dir, { withFileTypes: true })
      for (const entry of entries) {
        if (ignore.has(entry.name)) continue
        const fullPath = join(dir, entry.name)
        if (entry.isDirectory()) {
          await scan(fullPath)
        } else if (entry.isFile()) {
          const content = await readFile(fullPath, 'utf-8')
          expect(content.includes(deprecated)).toBe(false)
        }
      }
    }

    for (const root of roots) {
      await scan(root)
    }
  })
})

describe('qfi_score write guard', () => {
  it('only writes qfiScore through the canonical upsert path', async () => {
    const { readFile, readdir } = await import('fs/promises')
    const { join } = await import('path')

    const roots = ['server', 'scripts', 'tools']
    const allowedFiles = new Set(['server/vocabulary-persistence.ts'])
    const ignore = new Set(['node_modules', 'dist', 'docs', 'migrations'])

    async function scanFile(filePath: string) {
      const relPath = filePath.replace(`${process.cwd()}/`, '')
      const content = await readFile(filePath, 'utf-8')
      const lines = content.split('\n')

      for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i]
        if (line.includes('.values(') || line.includes('.set(')) {
          const slice = lines.slice(i, i + 25).join('\n')
          if (slice.includes('qfiScore:') && !allowedFiles.has(relPath)) {
            throw new Error(`qfiScore write detected in ${relPath}`)
          }
        }
      }
    }

    async function scan(dir: string): Promise<void> {
      const entries = await readdir(dir, { withFileTypes: true })
      for (const entry of entries) {
        if (ignore.has(entry.name)) continue
        const fullPath = join(dir, entry.name)
        if (entry.isDirectory()) {
          await scan(fullPath)
        } else if (entry.isFile() && fullPath.endsWith('.ts')) {
          await scanFile(fullPath)
        }
      }
    }

    for (const root of roots) {
      await scan(root)
    }
  })
})
