# 20260204 Geometry Doctrine: Simplex + Fisher–Rao Canon (1.00W)

This document defines the **single canonical geometry contract** for Pantheon-Chat: basins live on the probability simplex and all distance/mean/interpolation operations use the canonical Fisher–Rao manifold geometry.

## Canonical basin representation (Simplex)

- **State space:** probability simplex Δ⁶³ (64D stage)
- **Constraints:**
  - `p_i >= 0`
  - `Σ p_i = 1`
- **Terminology:** use `basin` / `basin_coordinates` (not “embedding”).

**Chart note:** sqrt-space (`√p`) may be used **internally** to compute Fisher–Rao operations. It must never be persisted or exchanged at module boundaries.

## Canonical distance (Fisher–Rao)

Let:

- `BC(p,q) = Σ_i sqrt(p_i q_i)` (Bhattacharyya coefficient)

**Canonical distance scaling for this repo:**

- `d_FR(p,q) = arccos(BC(p,q))`
- **Range:** `[0, π/2]`

### Explicitly non-canonical (forbidden in production paths)

- **Chord/Hellinger proxy** misnamed as Fisher distance:
  - `sqrt(2 - 2*BC(p,q))`
  - `np.linalg.norm(np.sqrt(p) - np.sqrt(q))`

These are Euclidean chord lengths in sqrt-space. They are monotone in `d_FR` but **not equal**, and using them breaks thresholds/governance.

## Canonical interpolation (geodesic)

- Use canonical geodesic interpolation (`geodesic_interpolation`) defined by SLERP in sqrt-space then squaring back to simplex.

Forbidden:
- Linear blending on basins: `(1-t)*p + t*q`.

## Canonical mean (Fréchet mean)

- Use canonical `frechet_mean(basins, weights=None)`.

Forbidden:
- Arithmetic mean on basins: `np.mean(basins, axis=0)`.

## Boundary validation (fail-closed)

All module boundaries that accept a basin must validate simplex invariants and **raise** on invalid input:

- trajectory ingest
- kernel lifecycle spawn / promote / resurrect
- persistence IO
- routing inputs
- training ingestion

## Canonical import surface

All runtime-critical code must import geometry from **one** canonical module surface:

- `from qig_geometry.canonical import fisher_rao_distance, frechet_mean, geodesic_interpolation, log_map, exp_map, assert_basin_valid`

No fallback imports (`try/except ImportError`) are allowed in geometry paths.

## Enforcement

- Static scan must flag:
  - Euclidean distance/norm patterns on basins
  - chord/Hellinger masquerading as Fisher–Rao
  - `np.mean(` on basin arrays
  - `embedding` used to refer to basins
- Runtime gate must fail closed before start/reset/rollback and before lifecycle operations.

## References

- `docs/10-e8-protocol/specs/CANONICAL_GEOMETRY_CONTRACT.md` (FROZEN authoritative inventory)
- `docs/11-Genesis-kernel-upgrade/20260201-genesis-kernel-upgrade-core-concepts-terms-1.00W.md`
- `docs/11-Genesis-kernel-upgrade/20260201-genesis-kernel-upgrade-purity-gate-validators-tests-1.00W.md`
