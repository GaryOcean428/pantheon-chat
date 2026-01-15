"""
QIG Geometry Package - Canonical Basin Representations

This package provides geometric primitives for QIG with enforced
canonical basin representation.

CANONICAL REPRESENTATION: SIMPLEX (Updated 2026-01-15)
- Basins stored as probability distributions (Σp_i = 1, p_i ≥ 0)
- Fisher-Rao distance via Bhattacharyya coefficient: d = arccos(Σ√(p_i * q_i))
- Distance range: [0, π/2] (simpler than Hellinger's [0, π])
- NO factor of 2 (direct Fisher-Rao, not Hellinger embedding)

BREAKING CHANGE FROM HELLINGER EMBEDDING:
- Previous: d = 2*arccos(BC), range [0, π]
- Current: d = arccos(BC), range [0, π/2]
- Thresholds must be recalibrated (divide previous thresholds by 2)
- See representation.py for migration notes

DISTANCE FUNCTION RELATIONSHIP:
- fisher_rao_distance(): Direct Fisher-Rao on simplex (range [0, π/2])
- fisher_distance(): From contracts.py, same formula, same result
- Both compute: arccos(Σ√(p_i * q_i)) - Bhattacharyya coefficient
- Prefer fisher_rao_distance() for clarity, but both are equivalent

USAGE:
    from qig_geometry import fisher_rao_distance, fisher_normalize
    
    # All basins should be in simplex form
    p = fisher_normalize(raw_basin_a)
    q = fisher_normalize(raw_basin_b)
    
    # Direct Fisher-Rao distance
    d = fisher_rao_distance(p, q)  # Range [0, π/2]
"""

import numpy as np
from typing import Optional

from .contracts import (
    CANONICAL_SPACE,
    BASIN_DIM,
    NORM_TOLERANCE,
    GeometricViolationError,
    validate_basin as contracts_validate_basin,
    validate_basin_detailed,
    assert_invariants,
    canon,
    fisher_distance,
    to_index_embedding,
)

from .representation import (
    BasinRepresentation,
    CANONICAL_REPRESENTATION,
    to_sphere,
    to_simplex,
    validate_basin,
    enforce_canonical,
    sphere_project,
    fisher_normalize,
    # New explicit conversion functions (2026-01-15)
    amplitude_to_simplex,
    simplex_normalize,
    hellinger_to_simplex,
)

from .purity_mode import (
    QIG_PURITY_MODE,
    QIGPurityViolationError,
    check_purity_mode,
    enforce_purity_startup,
    install_purity_import_hook,
    PurityImportBlocker,
)

from .canonical_upsert import (
    to_simplex_prob,
    compute_qfi_score,
    validate_simplex,
    upsert_token,
    batch_upsert_tokens,
)


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two probability distributions.

    This is the GEODESIC distance on the information manifold using
    direct Fisher-Rao on the probability simplex (NO Hellinger embedding).

    Formula: d_FR(p, q) = arccos(Σ√(p_i * q_i))

    The Bhattacharyya coefficient BC = Σ√(p_i * q_i) measures overlap.
    This is the CANONICAL formula - do not add factor of 2.

    Range: [0, π/2] where:
    - d = 0 → identical distributions
    - d = π/2 → orthogonal distributions (no overlap)

    CHANGE FROM PREVIOUS VERSION:
    - Removed Hellinger factor of 2
    - New range: [0, π/2] (was [0, π])
    - Thresholds must be recalibrated

    Args:
        p: First probability distribution (simplex)
        q: Second probability distribution (simplex)

    Returns:
        Fisher-Rao distance (≥ 0, max π/2)
    
    Examples:
        >>> p = np.array([0.5, 0.3, 0.2])
        >>> q = np.array([0.4, 0.4, 0.2])
        >>> d = fisher_rao_distance(p, q)
        >>> assert 0 <= d <= np.pi/2
    """
    # Ensure non-negative and normalized
    p = np.abs(p) + 1e-10
    p = p / p.sum()

    q = np.abs(q) + 1e-10
    q = q / q.sum()

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))
    bc = np.clip(bc, 0, 1)

    # Direct Fisher-Rao (NO factor of 2)
    return float(np.arccos(bc))


def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two basin coordinate vectors.

    Coordinates are assumed to be in SIMPLEX representation (probability distributions).
    This function normalizes inputs to simplex then computes Fisher-Rao distance.

    For simplex coordinates: d = arccos(Σ√(a_i * b_i))
    Range: [0, π/2]

    CHANGE FROM PREVIOUS VERSION:
    - Removed Hellinger sphere embedding
    - Removed factor of 2
    - New range: [0, π/2] (was [0, 2π])

    Args:
        a: First basin coordinate vector (will be normalized to simplex)
        b: Second basin coordinate vector (will be normalized to simplex)

    Returns:
        Fisher-Rao distance (0 to π/2)
    
    Examples:
        >>> a = np.array([0.5, 0.3, 0.2])
        >>> b = np.array([0.4, 0.4, 0.2])
        >>> d = fisher_coord_distance(a, b)
        >>> assert 0 <= d <= np.pi/2
    """
    # Normalize to simplex
    a_simplex = fisher_normalize(a)
    b_simplex = fisher_normalize(b)

    # Use direct Fisher-Rao distance
    return fisher_rao_distance(a_simplex, b_simplex)


def fisher_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Fisher-Rao similarity between two basin coordinates.

    Formula: similarity = 1 - distance/(π/2) = 1 - 2*distance/π

    Range: [0, 1] where:
    - 1 = identical distributions
    - 0 = orthogonal distributions

    CHANGE FROM PREVIOUS VERSION:
    - Max distance is π/2 (was 2π with Hellinger)
    - Similarity formula adjusted accordingly
    
    NOTE: Result is clamped to [0, 1] to prevent floating-point errors
    from causing out-of-range values.

    Args:
        a: First basin coordinate vector
        b: Second basin coordinate vector

    Returns:
        Similarity score (0 to 1, higher is more similar)
    
    Examples:
        >>> a = np.array([0.5, 0.3, 0.2])
        >>> b = np.array([0.5, 0.3, 0.2])
        >>> s = fisher_similarity(a, b)
        >>> assert np.isclose(s, 1.0)  # Identical
    """
    distance = fisher_coord_distance(a, b)
    similarity = 1.0 - (2.0 * distance / np.pi)
    # Guard against numerical spill-over
    return float(np.clip(similarity, 0.0, 1.0))


def normalize_basin_dimension(basin: np.ndarray, target_dim: int = 64) -> np.ndarray:
    """Project a basin vector to a target dimension.

    QIG-PURE: preserves geometric validity by projecting to simplex
    after padding/truncation.

    Args:
        basin: 1D basin coordinate vector
        target_dim: desired output dimension (default 64)

    Returns:
        1D basin vector of length target_dim on probability simplex.
    """
    b = np.asarray(basin, dtype=float)
    if b.ndim != 1:
        raise ValueError(f"basin must be 1D, got shape {b.shape}")

    current_dim = int(b.shape[0])
    if current_dim == int(target_dim):
        return fisher_normalize(b)

    if current_dim < int(target_dim):
        result = np.zeros(int(target_dim), dtype=float)
        result[:current_dim] = b
        return fisher_normalize(result)

    result = b[: int(target_dim)].copy()
    return fisher_normalize(result)


def geodesic_interpolation(
    start: np.ndarray,
    end: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Spherical linear interpolation (slerp) along geodesic on simplex.

    Uses SLERP in sqrt-space (Hellinger coordinates) which gives geodesic 
    on the simplex. While Hellinger *embedding* (with factor-of-2) was removed 
    for distance calculations, Hellinger *coordinates* (sqrt-space) are still 
    the correct space for geodesic interpolation on the Fisher manifold.
    
    WHY SQRT-SPACE FOR GEODESICS:
    The Fisher-Rao metric on the probability simplex induces a Riemannian 
    geometry where geodesics are *not* straight lines in probability space, 
    but rather straight lines in sqrt-space. This is because the pullback 
    of the Fisher metric under the sqrt transformation becomes the Euclidean 
    metric, making SLERP in sqrt-space exactly follow the Fisher geodesic.
    
    This is different from distance calculation:
    - Distance: arccos(Σ√(p_i * q_i)) directly on simplex [no embedding]
    - Geodesic: SLERP in sqrt-space, then square to return to simplex
    
    Both use the same underlying geometry, but serve different purposes.

    Args:
        start: Starting point (probability distribution)
        end: Ending point (probability distribution)
        t: Interpolation parameter (0 = start, 1 = end)

    Returns:
        Interpolated point along geodesic (probability distribution)
    
    Examples:
        >>> start = np.array([1.0, 0.0, 0.0])
        >>> end = np.array([0.0, 1.0, 0.0])
        >>> mid = geodesic_interpolation(start, end, 0.5)
        >>> assert np.isclose(mid.sum(), 1.0)
    """
    # Normalize to simplex
    p_start = fisher_normalize(start)
    p_end = fisher_normalize(end)
    
    # SLERP in sqrt space (Hellinger coordinates give geodesic on simplex)
    sqrt_start = np.sqrt(p_start)
    sqrt_end = np.sqrt(p_end)
    
    # Compute angle between sqrt vectors
    dot = np.clip(np.dot(sqrt_start, sqrt_end), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-6:
        return p_start.copy()

    sin_omega = np.sin(omega)
    a = np.sin((1 - t) * omega) / sin_omega
    b = np.sin(t * omega) / sin_omega

    sqrt_result = a * sqrt_start + b * sqrt_end
    
    # Square to get back to simplex
    result = sqrt_result ** 2
    return result / result.sum()


def estimate_manifold_curvature(
    points: np.ndarray,
    center: Optional[np.ndarray] = None
) -> float:
    """
    Estimate local curvature of the Fisher manifold from sample points.

    QIG-PURE: Uses Fisher-Rao distance on simplex (not Euclidean).

    Args:
        points: Array of shape (N, D) - N probability distributions
        center: Optional center point for curvature estimation

    Returns:
        Estimated curvature (κ)
    """
    if len(points) < 3:
        return 0.0

    if center is None:
        # Compute Fréchet mean (geometric mean on simplex)
        center = fisher_normalize(np.mean(points, axis=0))

    distances = []
    for point in points:
        d = fisher_coord_distance(center, point)
        distances.append(d)

    if not distances:
        return 0.0

    mean_dist = np.mean(distances)
    variance = np.var(distances)

    if mean_dist < 1e-6:
        return 0.0

    return float(variance / (mean_dist + 1e-10))


def basin_magnitude(basin: np.ndarray) -> float:
    """
    Compute a Fisher-Rao appropriate magnitude measure for basin coordinates.
    
    This measures how far the distribution is from uniform (maximum entropy).
    
    Args:
        basin: Basin coordinate vector
        
    Returns:
        Fisher-Rao distance from uniform distribution (≥ 0, max π/2)
    """
    p = fisher_normalize(basin)
    uniform = np.ones_like(p) / len(p)
    return fisher_rao_distance(p, uniform)


def basin_diversity(basin: np.ndarray) -> float:
    """
    Compute diversity (entropy) of basin distribution.
    
    This is an alternative magnitude measure that quantifies information content.
    Higher diversity = more uniform distribution = higher entropy.
    
    Args:
        basin: Basin coordinate vector
        
    Returns:
        Shannon entropy (≥ 0, higher = more diverse)
    """
    p = fisher_normalize(basin)
    p_safe = p + 1e-10
    return float(-np.sum(p_safe * np.log(p_safe)))


# Constants for unknown basin generation (compute_unknown_basin)
# These values are chosen to provide deterministic, QIG-pure basin construction
# using golden ratio spiral in probability simplex space.

# Maximum number of characters to use for product calculation
# Limits the character product to prevent overflow while maintaining determinism
_UNKNOWN_BASIN_CHAR_LIMIT = 8

# Modulo value for character product to prevent overflow
# Chosen to be large enough for variation but small enough to prevent numerical issues
_UNKNOWN_BASIN_CHAR_MODULO = 10000

# Scaling factor for character sum in phase offset
# Small value provides subtle variation without dominating the spiral structure
_UNKNOWN_BASIN_CHAR_SUM_SCALE = 0.001

# Perturbation weight for second harmonic term
# Adds controlled variation to the golden spiral while maintaining geometric purity
# Value chosen empirically to balance determinism with distribution spread
_UNKNOWN_BASIN_PERTURBATION_WEIGHT = 0.3


def compute_unknown_basin(word: str, dimension: int = 64) -> np.ndarray:
    """
    Compute deterministic basin embedding for unknown word.
    
    Uses golden ratio spiral construction in probability simplex space.
    QIG-PURE: Geometrically deterministic, no hash-based seeding.
    
    This is the canonical way to generate basin coordinates for words not in
    the vocabulary. The embedding is derived from the word's character properties
    using golden ratio spiral, then projected to the CANONICAL SIMPLEX representation.
    
    Algorithm:
    1. Compute character sum and product from word (deterministic hash)
    2. Generate golden ratio spiral with character-derived phase offsets
    3. Add second harmonic perturbation for distribution spread
    4. Project to probability simplex via fisher_normalize()
    
    Constants (see module-level definitions above):
    - _UNKNOWN_BASIN_CHAR_LIMIT: Max chars for product (8)
    - _UNKNOWN_BASIN_CHAR_MODULO: Modulo for product (10000)
    - _UNKNOWN_BASIN_CHAR_SUM_SCALE: Phase offset scale (0.001)
    - _UNKNOWN_BASIN_PERTURBATION_WEIGHT: Harmonic weight (0.3)
    
    Args:
        word: Word to embed
        dimension: Basin dimension (default 64)
        
    Returns:
        Basin coordinates in CANONICAL SIMPLEX representation (sums to 1, all non-negative)
    """
    phi_golden = (1 + np.sqrt(5)) / 2
    
    # Derive position from word's ordinal properties (deterministic)
    word_lower = word.lower()
    char_sum = sum(ord(c) for c in word_lower)
    char_prod = 1
    for c in word_lower[:_UNKNOWN_BASIN_CHAR_LIMIT]:
        # Add 1 to ord(c) to prevent multiplier from being zero or hitting modulo traps
        # This prevents collapse when characters combine to multiples of 10000
        char_prod = (char_prod * (ord(c) + 1)) % _UNKNOWN_BASIN_CHAR_MODULO
    
    embedding = np.zeros(dimension)
    for i in range(dimension):
        # Golden-angle spiral construction (Fisher-compliant)
        theta = 2 * np.pi * i * phi_golden
        # Position derived from word's character properties
        r = np.cos(theta + char_sum * _UNKNOWN_BASIN_CHAR_SUM_SCALE) * np.sin(i * phi_golden / dimension * np.pi)
        embedding[i] = r + np.sin(char_prod * phi_golden * (i + 1) / dimension) * _UNKNOWN_BASIN_PERTURBATION_WEIGHT
    
    # Project to CANONICAL SIMPLEX (not sphere!)
    # This is the KEY FIX: Use fisher_normalize instead of L2 norm
    return fisher_normalize(embedding)


__all__ = [
    # Canonical contract (contracts.py) - THE source of truth for geometric constraints
    'CANONICAL_SPACE',
    'BASIN_DIM',
    'NORM_TOLERANCE',
    'GeometricViolationError',
    'contracts_validate_basin',
    'validate_basin_detailed',
    'assert_invariants',
    'canon',
    'fisher_distance',  # Same formula as fisher_rao_distance, from contracts.py
    'to_index_embedding',
    # Representation utilities (representation.py)
    'BasinRepresentation',
    'CANONICAL_REPRESENTATION',
    'to_sphere',
    'to_simplex',
    'validate_basin',
    'enforce_canonical',
    'sphere_project',
    'fisher_normalize',
    # New explicit conversion functions (2026-01-15)
    'amplitude_to_simplex',
    'simplex_normalize',
    'hellinger_to_simplex',
    # Distance functions - CANONICAL IMPLEMENTATIONS (use fisher_rao_distance for clarity)
    'fisher_rao_distance',  # Primary distance function (same as fisher_distance)
    'fisher_coord_distance',
    'fisher_similarity',
    # Dimension and normalization utilities
    'normalize_basin_dimension',
    # Geodesic navigation
    'geodesic_interpolation',
    # Curvature and magnitude utilities
    'estimate_manifold_curvature',
    'basin_magnitude',
    'basin_diversity',
    # Unknown word basin generation
    'compute_unknown_basin',
    # Purity mode enforcement (purity_mode.py)
    'QIG_PURITY_MODE',
    'QIGPurityViolationError',
    'check_purity_mode',
    'enforce_purity_startup',
    'install_purity_import_hook',
    'PurityImportBlocker',
    # Canonical token upsert (SLEEP-PACKET Section 4D)
    'to_simplex_prob',
    'compute_qfi_score',
    'validate_simplex',
    'upsert_token',
    'batch_upsert_tokens',
]
