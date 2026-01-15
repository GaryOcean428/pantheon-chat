import fs from 'fs'
import path from 'path'

const repoRoot = process.cwd()

const skipDirs = new Set([
  '.git',
  'node_modules',
  'dist',
  'build',
  'docs/08-experiments',
  'docs/99-quarantine'
])

const scanExtensions = new Set(['.ts', '.tsx', '.py'])

const bannedPatterns = [
  { regex: /cosine_similarity\s*\(/i, label: 'cosine_similarity(' },
  { regex: /F\.normalize\s*\(/, label: 'F.normalize(' },
  { regex: /np\.linalg\.norm\s*\(/, label: 'np.linalg.norm(' },
  { regex: new RegExp(`unit\\s+sphere`, 'i'), label: 'unit-sphere' }
]

const directSqlRegex = /(INSERT INTO|UPDATE\s+\w+\s+SET|DELETE FROM).*coordizer_vocabulary/i
const allowedSqlWriters = new Set([
  path.join('server', 'persistence', 'vocabulary.ts')
])

function shouldSkipDir(entryPath: string): boolean {
  const relative = path.relative(repoRoot, entryPath)
  for (const skip of skipDirs) {
    if (relative === skip || relative.startsWith(`${skip}${path.sep}`)) {
      return true
    }
  }
  return false
}

function walk(dir: string, files: string[] = []): string[] {
  if (shouldSkipDir(dir)) {
    return files
  }

  const entries = fs.readdirSync(dir, { withFileTypes: true })
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      walk(fullPath, files)
    } else {
      files.push(fullPath)
    }
  }
  return files
}

function main() {
  const violations: string[] = []

  const files = walk(repoRoot)
  for (const filePath of files) {
    const ext = path.extname(filePath)
    if (!scanExtensions.has(ext)) {
      continue
    }

    const relative = path.relative(repoRoot, filePath)
    if (shouldSkipDir(filePath)) {
      continue
    }

    const content = fs.readFileSync(filePath, 'utf-8')

    for (const pattern of bannedPatterns) {
      if (pattern.regex.test(content)) {
        violations.push(`${relative}: banned pattern ${pattern.label}`)
      }
    }

    if (directSqlRegex.test(content)) {
      if (!allowedSqlWriters.has(relative)) {
        violations.push(`${relative}: direct SQL write to coordizer_vocabulary`)
      }
    }
  }

  if (violations.length > 0) {
    console.error('Purity validation failed:')
    for (const violation of violations) {
      console.error(`- ${violation}`)
    }
    process.exit(1)
  }

  console.log('Purity validation passed')
}

main()
