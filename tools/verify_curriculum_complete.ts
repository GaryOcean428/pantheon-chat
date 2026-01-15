#!/usr/bin/env tsx
import { checkCurriculumCompleteness } from '../server/curriculum'

async function main() {
  const status = await checkCurriculumCompleteness()

  if (status.complete) {
    console.log('Curriculum completeness: OK')
    return
  }

  console.error('Curriculum completeness: FAILED')
  if (status.missing.length > 0) {
    console.error(`Missing tokens: ${status.missing.join(', ')}`)
  }
  if (status.quarantined.length > 0) {
    console.error(`Quarantined tokens: ${status.quarantined.join(', ')}`)
  }
  process.exit(1)
}

main().catch((error) => {
  console.error('Curriculum verification failed:', error)
  process.exit(1)
})
