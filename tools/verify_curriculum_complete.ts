import { getCurriculumStatus } from '../server/lib/curriculum-mode'

async function run() {
  const status = await getCurriculumStatus()

  if (!status.complete) {
    console.error('[Curriculum] Curriculum incomplete')
    console.error(`  missing: ${status.missing.length}`)
    console.error(`  invalid: ${status.invalid.length}`)
    if (status.missing.length > 0) {
      console.error(`  missing tokens (sample): ${status.missing.slice(0, 20).join(', ')}`)
    }
    if (status.invalid.length > 0) {
      console.error(`  invalid tokens (sample): ${status.invalid.slice(0, 20).join(', ')}`)
    }
    process.exit(1)
  }

  console.log('[Curriculum] ✅ Curriculum complete')
}

run().catch((error) => {
  console.error('[Curriculum] Failed:', error)
  process.exit(1)
})
