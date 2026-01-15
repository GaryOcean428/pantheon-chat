import { QIG_CONSTANTS } from './constants'

const DEFAULT_EPS = 1e-12

export function to_simplex_probabilities(
  basin: ArrayLike<number>,
  eps: number = DEFAULT_EPS
): number[] {
  const values = Array.from(basin, (value) => Math.abs(value) + eps)
  const total = values.reduce((sum, value) => sum + value, 0)

  if (!Number.isFinite(total) || total <= 0) {
    throw new Error('Invalid basin: cannot normalize to simplex')
  }

  return values.map((value) => value / total)
}

export function compute_qfi_score_simplex(
  basin: ArrayLike<number>,
  dimension: number = QIG_CONSTANTS.BASIN_DIMENSION
): number {
  const probabilities = to_simplex_probabilities(basin)

  if (probabilities.length !== dimension) {
    throw new Error(`Invalid basin length: expected ${dimension}, got ${probabilities.length}`)
  }

  const entropy = -probabilities.reduce((sum, p) => sum + p * Math.log(p), 0)
  const qfi = Math.exp(entropy) / dimension

  if (!Number.isFinite(qfi)) {
    throw new Error('Invalid QFI score: not finite')
  }

  return Math.min(1, Math.max(0, qfi))
}
