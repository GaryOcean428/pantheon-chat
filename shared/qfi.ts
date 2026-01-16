export const BASIN_DIM = 64
export const QFI_EPSILON = 1e-12

export type BasinInput = number[] | Float64Array

export function toSimplexProbabilities(basin: BasinInput, epsilon = QFI_EPSILON): number[] {
  // Convert potentially negative basin coordinates to non-negative values and add a small epsilon
  // so that all entries are strictly positive before normalization. This avoids zero probabilities
  // (which would cause issues for log/entropy calculations) while preserving the relative geometry.
  const values = Array.from(basin).map((value) => Math.abs(value) + epsilon)
  const total = values.reduce((sum, value) => sum + value, 0)

  if (!Number.isFinite(total) || total <= 0) {
    throw new Error('Invalid basin total for simplex normalization')
  }

  return values.map((value) => value / total)
}

export function compute_qfi_score_simplex(basin: BasinInput): number {
  const probabilities = toSimplexProbabilities(basin)

  let entropy = 0
  for (const probability of probabilities) {
    entropy -= probability * Math.log(probability)
  }

  const qfi = Math.exp(entropy) / BASIN_DIM
  const clamped = Math.min(1, Math.max(0, qfi))

  if (!Number.isFinite(clamped)) {
    throw new Error('QFI score is not finite')
  }

  return clamped
}

export function isValidQfiScore(score: number | null | undefined): boolean {
  if (score === null || score === undefined) {
    return false
  }

  return Number.isFinite(score) && score >= 0 && score <= 1
}
