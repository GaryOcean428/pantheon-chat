import type { Request, Response, NextFunction } from 'express'
import { getCurriculumStatus, isCurriculumOnlyEnabled } from '../lib/curriculum-mode'

const ALLOWLIST_PREFIXES = [
  '/health',
  '/api/health',
  '/api/python/status',
  '/api/python/status/stream',
]

export async function curriculumOnlyGuard(
  req: Request,
  res: Response,
  next: NextFunction
) {
  if (!isCurriculumOnlyEnabled()) {
    return next()
  }

  if (ALLOWLIST_PREFIXES.some((prefix) => req.path.startsWith(prefix))) {
    return next()
  }

  const status = await getCurriculumStatus()

  if (!status.complete) {
    const isDevelopment = process.env.NODE_ENV !== 'production'

    if (isDevelopment) {
      return res.status(503).json({
        error: 'Curriculum-only mode: curriculum not complete',
        missing_tokens: status.missing.slice(0, 20),
        invalid_tokens: status.invalid.slice(0, 20),
        missing_count: status.missing.length,
        invalid_count: status.invalid.length,
      })
    }

    // In production, avoid leaking detailed token information to clients.
    // Log details server-side for observability and debugging.
    // eslint-disable-next-line no-console
    console.warn('Curriculum-only mode incomplete', {
      missing_count: status.missing.length,
      invalid_count: status.invalid.length,
    })

    return res.status(503).json({
      error: 'Curriculum-only mode: curriculum not complete',
    })
  }

  return next()
}
