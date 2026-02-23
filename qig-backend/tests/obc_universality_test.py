#!/usr/bin/env python3
"""
Phase 1 Universality Stress Test: OBC vs PBC Bulk Response
===========================================================

Tests whether the canonical Einstein-like linear response (geometric deformation
observable) survives the transition from Periodic to Open Boundary Conditions,
provided we restrict measurements to the topological bulk.

Key physics: Under PBC every site is equivalent (torus symmetry). Under OBC,
boundary sites lack neighbors and cannot distribute stress symmetrically.
The canonical QIG pipeline extracts LOCAL geometry at the perturbation site,
so we must separate bulk from boundary.

Uses exact diagonalization (small L) with sparse operators.
Generators: local σ_x (transverse field operators) — canonical for TFIM QFI.

Convention: unscaled Fisher-Rao  d_FR(p,q) = arccos(Σ√(p_i·q_i))
            No Euclidean contamination. No cosine similarity. No Adam.

Protocol: Thermodynamic Consciousness Protocol v6.0
"""

import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import linregress
import time
import json


# ============================================================
# Spin operators
# ============================================================
def get_spin_ops():
    sz = csr_matrix([[1, 0], [0, -1]], dtype=np.float64)
    sx = csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
    id2 = identity(2, format='csr', dtype=np.float64)
    return sz, sx, id2


# ============================================================
# Build Hamiltonian: H = -J Σ σ_z^i σ_z^j - Σ h_i σ_x^i
# ============================================================
def build_hamiltonian(L, J, h_fields, pbc=True):
    """Build 2D TFIM Hamiltonian on L×L lattice."""
    sz, sx, id2 = get_spin_ops()
    N = L * L
    dim = 2**N

    def site_op(op, site):
        """Build N-site operator with `op` at `site`, identity elsewhere."""
        ops = [id2] * N
        ops[site] = op
        res = ops[0]
        for i in range(1, N):
            res = kron(res, ops[i], format='csr')
        return res

    H = csr_matrix((dim, dim), dtype=np.float64)

    # Transverse field terms
    for i in range(N):
        H -= h_fields[i] * site_op(sx, i)

    # Ising coupling (nearest-neighbor σ_z σ_z)
    for i in range(N):
        row, col = i // L, i % L

        # Right neighbor
        if col < L - 1:
            H -= J * site_op(sz, i) @ site_op(sz, i + 1)
        elif pbc:
            H -= J * site_op(sz, i) @ site_op(sz, i - (L - 1))

        # Down neighbor
        if row < L - 1:
            H -= J * site_op(sz, i) @ site_op(sz, i + L)
        elif pbc:
            H -= J * site_op(sz, i) @ site_op(sz, i % L)

    return H, site_op


# ============================================================
# Classify sites: bulk vs boundary (OBC)
# ============================================================
def classify_sites(L):
    """Return lists of bulk and boundary site indices for L×L OBC lattice."""
    bulk = []
    boundary = []
    for i in range(L * L):
        row, col = i // L, i % L
        if 0 < row < L - 1 and 0 < col < L - 1:
            bulk.append(i)
        else:
            boundary.append(i)
    return bulk, boundary


def get_neighbors(site, L, pbc):
    """Return list of neighbor indices for a site on L×L lattice."""
    row, col = site // L, site % L
    neighbors = []

    # Right
    if col < L - 1:
        neighbors.append(site + 1)
    elif pbc:
        neighbors.append(site - (L - 1))

    # Left
    if col > 0:
        neighbors.append(site - 1)
    elif pbc:
        neighbors.append(site + (L - 1))

    # Down
    if row < L - 1:
        neighbors.append(site + L)
    elif pbc:
        neighbors.append(site % L)  # wrap to top

    # Up
    if row > 0:
        neighbors.append(site - L)
    elif pbc:
        neighbors.append((L - 1) * L + col)  # wrap to bottom

    return neighbors


# ============================================================
# Local geometry proxy: sum of 4·Cov(σ_x^site, σ_x^neighbor)
# This mirrors the trace of the local QFI metric tensor
# ============================================================
def compute_local_geometry_proxy(psi, site, L, pbc, site_op_func, sx):
    """
    Compute D_local = Σ_neighbors 4·Cov(σ_x^site, σ_x^neighbor)
    This is the local geometric deformation observable.
    """
    neighbors = get_neighbors(site, L, pbc)
    sx_site = site_op_func(sx, site)
    exp_sx_site = float(np.real(psi.conj() @ sx_site @ psi))

    D_local = 0.0
    for n in neighbors:
        sx_n = site_op_func(sx, n)
        exp_sx_n = float(np.real(psi.conj() @ sx_n @ psi))
        exp_sx_site_n = float(np.real(psi.conj() @ (sx_site @ sx_n) @ psi))
        cov = exp_sx_site_n - (exp_sx_site * exp_sx_n)
        D_local += 4.0 * cov

    return D_local


# ============================================================
# Run a single experiment: one L, one BC type
# ============================================================
def run_experiment(L, pbc, n_perts_per_class, seed=42):
    """
    Run OBC/PBC universality test for given L.
    Under PBC: all sites are bulk (torus symmetry).
    Under OBC: separate bulk from boundary.
    """
    bc_str = "PBC" if pbc else "OBC"
    N = L * L
    J = 1.0
    rng = np.random.RandomState(seed)
    sz, sx, id2 = get_spin_ops()

    print(f"\n{'='*70}")
    print(f"  L={L} ({N} spins, dim=2^{N}={2**N})  |  Boundary: {bc_str}")
    print(f"{'='*70}")

    if pbc:
        bulk_sites = list(range(N))  # all sites equivalent under PBC
        boundary_sites = []
    else:
        bulk_sites, boundary_sites = classify_sites(L)

    print(f"  Bulk sites:     {bulk_sites}")
    print(f"  Boundary sites: {boundary_sites}")

    # Baseline ground state
    h_base = np.ones(N)
    t0 = time.time()
    H_base, site_op_func = build_hamiltonian(L, J, h_base, pbc=pbc)
    E0_base, psi_base = eigsh(H_base, k=1, which='SA')
    E0_base = E0_base[0]
    psi_base = psi_base[:, 0]
    print(f"  Baseline E₀ = {E0_base:.6f}  (solved in {time.time()-t0:.1f}s)")

    results = {"bulk": [], "boundary": []}

    def perturb_and_measure(sites, label, n_perts):
        if not sites:
            return
        chosen_sites = rng.choice(sites, size=n_perts, replace=True)
        delta_hs = rng.uniform(0.5, 0.7, size=n_perts)

        for idx, (site_idx, dh) in enumerate(zip(chosen_sites, delta_hs)):
            # Perturbed ground state
            h_pert = np.ones(N)
            h_pert[site_idx] += dh
            H_pert, _ = build_hamiltonian(L, J, h_pert, pbc=pbc)
            E0_pert, psi_pert = eigsh(H_pert, k=1, which='SA')
            E0_pert = E0_pert[0]
            psi_pert = psi_pert[:, 0]

            # Local observables
            D_base = compute_local_geometry_proxy(
                psi_base, site_idx, L, pbc, site_op_func, sx
            )
            D_pert = compute_local_geometry_proxy(
                psi_pert, site_idx, L, pbc, site_op_func, sx
            )

            dE = E0_pert - E0_base
            dD = D_pert - D_base

            results[label].append({
                "site": int(site_idx),
                "dh": float(dh),
                "dE": float(dE),
                "dD": float(dD),
            })

        print(f"  {label.capitalize()} perturbations: {n_perts} done")

    perturb_and_measure(bulk_sites, "bulk", n_perts_per_class)
    if not pbc and boundary_sites:
        perturb_and_measure(boundary_sites, "boundary", n_perts_per_class)

    # Fit results
    summary = {}
    for label in ["bulk", "boundary"]:
        data = results[label]
        if len(data) < 3:
            print(f"  {label.capitalize():12s} | N={len(data):2d} | Insufficient data for fit")
            summary[label] = {"n": len(data), "slope": None, "r2": None}
            continue

        dE_arr = np.array([r["dE"] for r in data])
        dD_arr = np.array([r["dD"] for r in data])
        slope, intercept, r_val, p_val, std_err = linregress(dE_arr, dD_arr)
        r2 = r_val**2

        print(f"  {label.capitalize():12s} | N={len(data):2d} | "
              f"Slope={slope:+.4f} ± {std_err:.4f} | R²={r2:.4f}")

        summary[label] = {
            "n": len(data),
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "std_err": float(std_err),
        }

    return {"L": L, "bc": bc_str, "results": results, "summary": summary}


# ============================================================
# Main: run the full Phase 1 stress test
# ============================================================
def main():
    print("\n" + "#" * 70)
    print("  PHASE 1: OBC vs PBC UNIVERSALITY STRESS TEST")
    print("  Geometric Deformation Observable — Local QFI Proxy")
    print("  Protocol: Thermodynamic Consciousness v6.0")
    print("  Convention: unscaled Fisher-Rao (no factor of 2)")
    print("#" * 70)

    all_results = {}

    # --- L=2 baseline (should show limited/no signal) ---
    print("\n" + "~" * 70)
    print("  L=2 BASELINE (null control — expect weak/no signal)")
    print("~" * 70)
    all_results["L2_PBC"] = run_experiment(L=2, pbc=True, n_perts_per_class=15, seed=42)
    all_results["L2_OBC"] = run_experiment(L=2, pbc=False, n_perts_per_class=15, seed=42)

    # --- L=3 (emergence threshold) ---
    print("\n" + "~" * 70)
    print("  L=3 EMERGENCE THRESHOLD")
    print("~" * 70)
    all_results["L3_PBC"] = run_experiment(L=3, pbc=True, n_perts_per_class=30, seed=42)
    all_results["L3_OBC"] = run_experiment(L=3, pbc=False, n_perts_per_class=30, seed=42)

    # --- L=4 (plateau onset — heavier, fewer perts) ---
    print("\n" + "~" * 70)
    print("  L=4 PLATEAU ONSET (16 spins, 65536-dim Hilbert space)")
    print("~" * 70)
    # L=4 is expensive (2^16 = 65536 dim). Run fewer perturbations.
    all_results["L4_PBC"] = run_experiment(L=4, pbc=True, n_perts_per_class=10, seed=42)
    all_results["L4_OBC"] = run_experiment(L=4, pbc=False, n_perts_per_class=10, seed=42)

    # --- Summary table ---
    print("\n\n" + "=" * 70)
    print("  UNIVERSALITY LEDGER SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<12s} {'Class':<12s} {'N':>4s} {'Slope':>10s} {'R²':>8s}")
    print(f"  {'-'*12} {'-'*12} {'-'*4} {'-'*10} {'-'*8}")

    for key in sorted(all_results.keys()):
        res = all_results[key]
        L_val = res["L"]
        bc = res["bc"]
        for label in ["bulk", "boundary"]:
            s = res["summary"].get(label, {})
            n = s.get("n", 0)
            if n == 0:
                continue
            slope = s.get("slope")
            r2 = s.get("r2")
            slope_str = f"{slope:+.4f}" if slope is not None else "N/A"
            r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
            print(f"  L={L_val} {bc:<6s} {label:<12s} {n:4d} {slope_str:>10s} {r2_str:>8s}")

    # --- Physics interpretation ---
    print("\n" + "-" * 70)
    print("  INTERPRETATION")
    print("-" * 70)

    l3_pbc = all_results.get("L3_PBC", {}).get("summary", {}).get("bulk", {})
    l3_obc_bulk = all_results.get("L3_OBC", {}).get("summary", {}).get("bulk", {})
    l3_obc_bdy = all_results.get("L3_OBC", {}).get("summary", {}).get("boundary", {})

    if l3_pbc.get("r2") is not None:
        r2_pbc = l3_pbc["r2"]
        if r2_pbc > 0.95:
            print(f"  ✅ L=3 PBC: Strong linear response (R²={r2_pbc:.4f})")
            print(f"     → Torus symmetry preserves Einstein-like deformation law")
        else:
            print(f"  ⚠️  L=3 PBC: Weaker than expected (R²={r2_pbc:.4f})")

    if l3_obc_bulk.get("r2") is not None and l3_obc_bulk["n"] >= 3:
        r2_obc = l3_obc_bulk["r2"]
        print(f"  {'✅' if r2_obc > 0.8 else '⚠️'} L=3 OBC Bulk: R²={r2_obc:.4f} (N={l3_obc_bulk['n']})")
    elif l3_obc_bulk.get("n", 0) < 3:
        print(f"  ℹ️  L=3 OBC Bulk: Only {l3_obc_bulk.get('n', 0)} samples (site 4 only)")
        print(f"     → Need targeted bulk-only perturbation for proper fit")

    if l3_obc_bdy.get("r2") is not None:
        r2_bdy = l3_obc_bdy["r2"]
        print(f"  {'⚠️' if r2_bdy < 0.8 else '✅'} L=3 OBC Boundary: R²={r2_bdy:.4f}")
        print(f"     → Boundary sites lack neighbor support → geometry frays")

    # Save results
    out_path = "/home/claude/obc_universality_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
