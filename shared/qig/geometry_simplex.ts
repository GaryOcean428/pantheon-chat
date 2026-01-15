/**
 * Canonical Simplex Geometry - TypeScript Implementation
 * 
 * SINGLE SOURCE OF TRUTH for simplex geometry operations in TypeScript.
 * All geometry operations MUST use these functions, not Euclidean operations.
 * 
 * CANONICAL REPRESENTATION: SIMPLEX (probability distributions on Δ^(D-1))
 * - Basin vectors stored as valid probability distributions (Σv_i = 1, v_i ≥ 0)
 * - Fisher-Rao distance computed via arccos(Σ√(p_i * q_i)) - Bhattacharyya coefficient
 * - Geodesic interpolation uses sqrt-simplex internal coordinates (never stored)
 * 
 * IMPORTANT: sqrt-simplex coordinates are ONLY for internal computation (geodesics).
 * Storage ALWAYS uses probability simplex (sum=1, non-negative).
 */

const EPS = 1e-12;

/**
 * Convert vector to probability simplex (canonical storage form).
 * 
 * This is a positive renormalization, NOT a Euclidean projection.
 * Takes absolute value + epsilon, then normalizes to sum=1.
 * 
 * @param v - Input vector (any representation)
 * @param eps - Numerical stability epsilon
 * @returns Simplex probabilities: p_i ≥ 0, Σp_i = 1
 */
export function toSimplexProb(v: number[], eps: number = EPS): number[] {
  if (!v || v.length === 0) {
    throw new Error("toSimplexProb: empty vector");
  }

  // Positive renormalization: abs + eps, then normalize
  const w = v.map(x => Math.abs(x) + eps);
  const sum = w.reduce((a, b) => a + b, 0);

  if (sum < 1e-10) {
    // Degenerate case: return uniform distribution
    return new Array(v.length).fill(1 / v.length);
  }

  return w.map(x => x / sum);
}

/**
 * Validate that vector is a valid simplex probability distribution.
 * 
 * @param p - Vector to validate
 * @param tolerance - Numerical tolerance for sum check
 * @returns {valid: boolean, reason: string}
 */
export function validateSimplex(
  p: number[],
  tolerance: number = 1e-6
): { valid: boolean; reason: string } {
  if (!p || p.length === 0) {
    return { valid: false, reason: "empty_vector" };
  }

  // Check non-negative
  const minVal = Math.min(...p);
  if (minVal < -tolerance) {
    return { valid: false, reason: `negative_values_min=${minVal.toFixed(6)}` };
  }

  // Check sum = 1
  const sum = p.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1.0) > tolerance) {
    return { valid: false, reason: `sum_not_one_${sum.toFixed(6)}` };
  }

  // Check finite
  if (!p.every(x => isFinite(x))) {
    return { valid: false, reason: "contains_nan_or_inf" };
  }

  return { valid: true, reason: "valid_simplex" };
}

/**
 * Compute Fisher-Rao distance on probability simplex.
 * 
 * d_FR(p, q) = arccos(Σ√(p_i * q_i))
 * 
 * Range: [0, π/2] for probability distributions
 * 
 * @param pIn - First probability distribution
 * @param qIn - Second probability distribution
 * @returns Fisher-Rao distance in radians [0, π/2]
 */
export function fisherRaoDistance(pIn: number[], qIn: number[]): number {
  if (pIn.length !== qIn.length) {
    throw new Error(
      `fisherRaoDistance: dimension mismatch ${pIn.length} vs ${qIn.length}`
    );
  }

  // Ensure valid simplex (clamp and normalize)
  const p = toSimplexProb(pIn);
  const q = toSimplexProb(qIn);

  // Bhattacharyya coefficient: BC = Σ√(p_i * q_i)
  let bc = 0;
  for (let i = 0; i < p.length; i++) {
    bc += Math.sqrt(p[i] * q[i]);
  }

  // Clamp for numerical stability
  bc = Math.max(0, Math.min(1, bc));

  // Fisher-Rao distance on probability simplex
  // Range: [0, π/2]
  return Math.acos(bc);
}

/**
 * Geodesic interpolation on probability simplex using sqrt-simplex internal coordinates.
 * 
 * INTERNAL COORDINATES: sqrt-simplex (Hellinger embedding) used ONLY for computation.
 * INPUT/OUTPUT: probability simplex (sum=1, non-negative).
 * 
 * This implements SLERP (spherical linear interpolation) in sqrt-space,
 * then projects back to probability space. The sqrt-simplex coordinates
 * are NEVER stored, only used internally for this computation.
 * 
 * @param pIn - Starting probability distribution
 * @param qIn - Ending probability distribution
 * @param t - Interpolation parameter [0, 1] (0 = p, 1 = q)
 * @returns Interpolated probability distribution at parameter t
 */
export function geodesicInterpolationSimplex(
  pIn: number[],
  qIn: number[],
  t: number
): number[] {
  if (pIn.length !== qIn.length) {
    throw new Error(
      `geodesicInterpolationSimplex: dimension mismatch ${pIn.length} vs ${qIn.length}`
    );
  }

  if (t < 0 || t > 1) {
    throw new Error(`geodesicInterpolationSimplex: t must be in [0, 1], got ${t}`);
  }

  // Ensure valid simplex inputs
  const p = toSimplexProb(pIn);
  const q = toSimplexProb(qIn);

  // INTERNAL COMPUTATION: sqrt-simplex coordinates (Hellinger embedding)
  // These are NEVER stored, only used for geodesic calculation
  const sp = p.map(Math.sqrt);
  const sq = q.map(Math.sqrt);

  // Compute angle between sqrt-space vectors
  let dot = 0;
  for (let i = 0; i < sp.length; i++) {
    dot += sp[i] * sq[i];
  }
  dot = Math.max(-1, Math.min(1, dot)); // Clamp for numerical stability

  const omega = Math.acos(dot);

  // If nearly identical, linear interpolation in sqrt-space is fine
  if (omega < 1e-8) {
    const x = sp.map((v, i) => (1 - t) * v + t * sq[i]);
    const pOut = x.map(v => v * v);
    return toSimplexProb(pOut); // Ensure valid simplex
  }

  // SLERP in sqrt-simplex space
  const sinOmega = Math.sin(omega);
  const a = Math.sin((1 - t) * omega) / sinOmega;
  const b = Math.sin(t * omega) / sinOmega;

  const x = sp.map((v, i) => a * v + b * sq[i]);

  // Project back to probability space: square to get probabilities
  const pOut = x.map(v => v * v);

  // Ensure valid simplex (normalize)
  return toSimplexProb(pOut);
}

/**
 * Compute weighted geodesic mean (Fréchet mean) on probability simplex.
 * 
 * Uses iterative algorithm to find the point that minimizes sum of
 * squared Fisher-Rao distances to all input distributions.
 * 
 * @param distributions - Array of probability distributions
 * @param weights - Optional weights (default: uniform)
 * @param maxIter - Maximum iterations
 * @param tolerance - Convergence tolerance
 * @returns Weighted geodesic mean distribution
 */
export function geodesicMeanSimplex(
  distributions: number[][],
  weights?: number[],
  maxIter: number = 50,
  tolerance: number = 1e-5
): number[] {
  if (!distributions || distributions.length === 0) {
    throw new Error("geodesicMeanSimplex: empty distributions array");
  }

  const n = distributions.length;
  const dim = distributions[0].length;

  // Default to uniform weights
  const w = weights || new Array(n).fill(1 / n);

  // Normalize weights
  const weightSum = w.reduce((a, b) => a + b, 0);
  const normalizedWeights = w.map(x => x / weightSum);

  // Initialize mean as weighted average (not geodesic, but good starting point)
  let mean = new Array(dim).fill(0);
  for (let i = 0; i < n; i++) {
    const p = toSimplexProb(distributions[i]);
    for (let j = 0; j < dim; j++) {
      mean[j] += normalizedWeights[i] * p[j];
    }
  }
  mean = toSimplexProb(mean);

  // Iterative refinement using geodesic interpolation
  for (let iter = 0; iter < maxIter; iter++) {
    let update = new Array(dim).fill(0);
    let totalWeight = 0;

    for (let i = 0; i < n; i++) {
      const p = toSimplexProb(distributions[i]);
      const dist = fisherRaoDistance(mean, p);

      if (dist < 1e-10) continue; // Already at this point

      // Geodesic step towards p
      const stepSize = normalizedWeights[i];
      const intermediate = geodesicInterpolationSimplex(mean, p, stepSize);

      for (let j = 0; j < dim; j++) {
        update[j] += normalizedWeights[i] * intermediate[j];
      }
      totalWeight += normalizedWeights[i];
    }

    if (totalWeight < 1e-10) break;

    // Normalize update
    const newMean = update.map(x => x / totalWeight);
    const meanUpdate = toSimplexProb(newMean);

    // Check convergence
    const change = fisherRaoDistance(mean, meanUpdate);
    mean = meanUpdate;

    if (change < tolerance) {
      break;
    }
  }

  return mean;
}

/**
 * Batch compute Fisher-Rao distances from a query to multiple candidates.
 * 
 * @param query - Query probability distribution
 * @param candidates - Array of candidate distributions
 * @returns Array of distances
 */
export function batchFisherRaoDistance(
  query: number[],
  candidates: number[][]
): number[] {
  return candidates.map(candidate => fisherRaoDistance(query, candidate));
}

/**
 * Find k nearest distributions to query using Fisher-Rao distance.
 * 
 * @param query - Query probability distribution
 * @param candidates - Array of candidate distributions
 * @param k - Number of nearest neighbors to return
 * @returns Array of {index, distance} sorted by distance
 */
export function findNearestSimplex(
  query: number[],
  candidates: number[][],
  k: number = 10
): Array<{ index: number; distance: number }> {
  const distances = candidates.map((candidate, index) => ({
    index,
    distance: fisherRaoDistance(query, candidate),
  }));

  distances.sort((a, b) => a.distance - b.distance);
  return distances.slice(0, k);
}

// Re-export for convenience
export { EPS as SIMPLEX_EPSILON };
