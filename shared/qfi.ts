import { BASIN_DIMENSION } from './constants'

const DEFAULT_EPSILON = 1e-12

export function toSimplexProbabilities(
  basin: ArrayLike<number>,
  epsilon: number = DEFAULT_EPSILON
): number[] {
  const values = Array.from(basin, (value) => Math.abs(Number(value)) + epsilon)
  const sum = values.reduce((acc, value) => acc + value, 0)

  if (!Number.isFinite(sum) || sum <= 0) {
    throw new Error('Invalid basin input: cannot normalize to simplex')
  }

  return values.map((value) => value / sum)
}

export function compute_qfi_score_simplex(basin: ArrayLike<number>): number {
  if (!basin || Array.from(basin).length === 0) {
    throw new Error('Invalid basin input: empty vector')
  }

  const probabilities = toSimplexProbabilities(basin)
  const entropy = -probabilities.reduce(
    (acc, probability) => acc + probability * Math.log(probability),
    0
  )

  const qfi = Math.exp(entropy) / BASIN_DIMENSION
  const clamped = Math.min(1, Math.max(0, qfi))

  if (!Number.isFinite(clamped)) {
    throw new Error('Invalid QFI score: non-finite result')
  }

  return clamped
}

export function computeQfiScoreSimplex(basin: ArrayLike<number>): number {
  return compute_qfi_score_simplex(basin)
}

export function isValidQfiScore(score: number | null | undefined): boolean {
  if (score === null || score === undefined) return false
  return Number.isFinite(score) && score >= 0 && score <= 1
}
