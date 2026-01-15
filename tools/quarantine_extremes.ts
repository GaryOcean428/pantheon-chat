import fs from 'fs'
import path from 'path'

import { sql } from 'drizzle-orm'

import { db } from '../server/db'

const allowlistPath = path.join(process.cwd(), 'tools', 'allowlists', 'qfi_extremes_allowlist.txt')

function loadAllowlist(): Set<string> {
  if (!fs.existsSync(allowlistPath)) {
    return new Set()
  }

  const lines = fs.readFileSync(allowlistPath, 'utf-8')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith('#'))

  return new Set(lines)
}

async function main() {
  if (!db) {
    throw new Error('Database not available')
  }

  const allowlist = loadAllowlist()

  const result = await db.execute<{ token: string }>(sql`
    SELECT token
    FROM coordizer_vocabulary
    WHERE token_status = 'active'
      AND (qfi_score = 0 OR qfi_score >= 0.99)
  `)

  const candidates = (result.rows ?? []).map((row) => row.token)
  const toQuarantine = candidates.filter((token) => !allowlist.has(token))

  if (toQuarantine.length === 0) {
    console.log('No extreme QFI tokens to quarantine')
    return
  }

  console.log(
    `Identified ${toQuarantine.length} tokens with extreme QFI scores to quarantine (no direct DB updates performed by this tool).`,
  )
  console.log('Tokens to quarantine:')
  for (const token of toQuarantine) {
    console.log(`- ${token}`)
  }
}

main().catch((error) => {
  console.error('Quarantine extremes failed:', error)
  process.exit(1)
})
