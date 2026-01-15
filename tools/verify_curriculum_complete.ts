import { getCurriculumCoverage } from '../server/curriculum'

async function main() {
  const coverage = await getCurriculumCoverage()

  console.log('Curriculum completeness')
  console.log(`- total tokens: ${coverage.total}`)
  console.log(`- active tokens: ${coverage.active}`)
  console.log(`- missing tokens: ${coverage.missingTokens.length}`)
  console.log(`- quarantined tokens: ${coverage.quarantinedTokens.length}`)

  if (coverage.total === 0 || coverage.missingTokens.length > 0 || coverage.quarantinedTokens.length > 0) {
    process.exit(1)
  }
}

main().catch((error) => {
  console.error('Curriculum verification failed:', error)
  process.exit(1)
})
