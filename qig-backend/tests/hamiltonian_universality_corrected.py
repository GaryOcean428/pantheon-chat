#!/usr/bin/env python3
"""
Phase 2b: Corrected Hamiltonian Universality Test
===================================================

Key insight from Phase 2: the generator must be TRANSVERSE to the ordering
direction. For TFIM (ordered along z, field along x), σ_x is correct.
For XXZ (ordered along z), we must ALSO use σ_x and perturb along x.

When generator aligns with the ordering direction, covariances saturate
and the geometry freezes — this is physically correct, not a bug.

The QFI measures QUANTUM fluctuations, which are maximal in the transverse
direction. This is the canonical choice for the Fisher information metric.

Also: test with generator-matched perturbations for ALL models.
"""

import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import linregress
import time
import json


def get_spin_ops():
    sz = csr_matrix([[1, 0], [0, -1]], dtype=np.float64)
    sx = csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
    sy_real = csr_matrix([[0, -1], [1, 0]], dtype=np.float64)  # -i·σ_y (real)
    id2 = identity(2, format='csr', dtype=np.float64)
    return sz, sx, sy_real, id2


def site_op_builder(N):
    def site_op(op, site):
        ops = [identity(2, format='csr', dtype=np.float64)] * N
        ops[site] = op
        res = ops[0]
        for i in range(1, N):
            res = kron(res, ops[i], format='csr')
        return res
    return site_op


def get_neighbors_2d(site, L, pbc):
    row, col = site // L, site % L
    neighbors = []
    if col < L - 1: neighbors.append(site + 1)
    elif pbc: neighbors.append(site - (L - 1))
    if col > 0: neighbors.append(site - 1)
    elif pbc: neighbors.append(site + (L - 1))
    if row < L - 1: neighbors.append(site + L)
    elif pbc: neighbors.append(site % L)
    if row > 0: neighbors.append(site - L)
    elif pbc: neighbors.append((L - 1) * L + col)
    return neighbors


def compute_local_D(psi, site, L, pbc, site_op_fn, generator):
    """Local QFI proxy: Σ_neighbors 4·Cov(gen_site, gen_neighbor)"""
    neighbors = get_neighbors_2d(site, L, pbc)
    gen_site = site_op_fn(generator, site)
    exp_s = float(psi @ gen_site @ psi)

    D = 0.0
    for n in neighbors:
        gen_n = site_op_fn(generator, n)
        exp_n = float(psi @ gen_n @ psi)
        exp_sn = float(psi @ (gen_site @ gen_n) @ psi)
        D += 4.0 * (exp_sn - exp_s * exp_n)
    return D


# ============================================================
# Build XXZ with TRANSVERSE field (h·σ_x) instead of longitudinal
# This ensures the perturbation and generator are both transverse
# ============================================================
def build_xxz_transverse(L, J, Delta, hx_fields, pbc=True):
    """
    XXZ with transverse field:
    H = -J Σ (σ_x^i σ_x^j + σ_y^i σ_y^j + Δ σ_z^i σ_z^j) - Σ hx_i σ_x^i

    Note: σ_y σ_y = (-iσ_y)(-iσ_y) · (-1) ... but actually σ_y^i σ_y^j
    For real-valued computation we use the identity:
    σ_x^i σ_x^j + σ_y^i σ_y^j = 2(|↑↓⟩⟨↓↑| + |↓↑⟩⟨↑↓|)
    i.e., the XX+YY part is the spin-flip exchange.
    We can compute this as: (σ_+^i σ_-^j + σ_-^i σ_+^j) where σ_± = (σ_x ± iσ_y)/2
    But for ED we can also use σ_x⊗σ_x + σ_y⊗σ_y directly.
    The σ_y⊗σ_y product of REAL operators gives a real result for the pair.
    """
    sz, sx, _, id2 = get_spin_ops()
    N = L * L
    dim = 2**N
    site_op = site_op_builder(N)

    # Build σ_+ and σ_- as real-valued ladder ops
    sp = csr_matrix([[0, 1], [0, 0]], dtype=np.float64)  # σ_+
    sm = csr_matrix([[0, 0], [1, 0]], dtype=np.float64)  # σ_-

    H = csr_matrix((dim, dim), dtype=np.float64)

    # Transverse field
    for i in range(N):
        H -= hx_fields[i] * site_op(sx, i)

    # XXZ coupling using σ_+σ_- + σ_-σ_+ + Δ·σ_z·σ_z
    # σ_x·σ_x + σ_y·σ_y = 2(σ_+·σ_- + σ_-·σ_+)
    for i in range(N):
        for j in get_neighbors_2d(i, L, pbc):
            if j > i:
                # XY part: 2(σ_+^i σ_-^j + σ_-^i σ_+^j)
                H -= J * 2.0 * (
                    site_op(sp, i) @ site_op(sm, j)
                    + site_op(sm, i) @ site_op(sp, j)
                )
                # Ising part
                H -= J * Delta * site_op(sz, i) @ site_op(sz, j)

    return H, site_op, sx  # Generator = σ_x (transverse)


def build_tfim(L, J, h_fields, pbc=True):
    """Standard TFIM: H = -J Σ σ_z·σ_z - h Σ σ_x"""
    sz, sx, _, _ = get_spin_ops()
    N = L * L
    dim = 2**N
    site_op = site_op_builder(N)
    H = csr_matrix((dim, dim), dtype=np.float64)
    for i in range(N):
        H -= h_fields[i] * site_op(sx, i)
    for i in range(N):
        for j in get_neighbors_2d(i, L, pbc):
            if j > i:
                H -= J * site_op(sz, i) @ site_op(sz, j)
    return H, site_op, sx


def build_disordered_tfim(L, J_mean, J_std, h_fields, pbc=True, rng=None):
    """Disordered TFIM with random J_ij"""
    if rng is None:
        rng = np.random.RandomState(99)
    sz, sx, _, _ = get_spin_ops()
    N = L * L
    dim = 2**N
    site_op = site_op_builder(N)
    H = csr_matrix((dim, dim), dtype=np.float64)
    for i in range(N):
        H -= h_fields[i] * site_op(sx, i)
    seen = set()
    for i in range(N):
        for j in get_neighbors_2d(i, L, pbc):
            bond = (min(i, j), max(i, j))
            if bond not in seen:
                seen.add(bond)
                J_ij = max(0.1, rng.normal(J_mean, J_std))
                H -= J_ij * site_op(sz, i) @ site_op(sz, j)
    return H, site_op, sx


def run_model(name, L, build_fn, build_kwargs, n_perts, seed=42, pbc=True):
    N = L * L
    rng = np.random.RandomState(seed)

    print(f"\n  {name} | L={L} {'PBC' if pbc else 'OBC'}")

    H_base, site_op_fn, generator = build_fn(**build_kwargs, pbc=pbc)
    t0 = time.time()
    E0, psi0 = eigsh(H_base, k=1, which='SA')
    E0 = E0[0]; psi0 = psi0[:, 0]
    print(f"    E₀={E0:.6f} ({time.time()-t0:.1f}s)")

    sites = list(range(N))  # all sites for PBC
    chosen = rng.choice(sites, size=n_perts, replace=True)
    dhs = rng.uniform(0.5, 0.7, size=n_perts)

    data = []
    for s, dh in zip(chosen, dhs):
        h_pert = build_kwargs["h_fields" if "h_fields" in build_kwargs else "hx_fields"].copy()
        h_pert[s] += dh

        pk = dict(build_kwargs)
        field_key = "h_fields" if "h_fields" in pk else "hx_fields"
        pk[field_key] = h_pert

        Hp, _, _ = build_fn(**pk, pbc=pbc)
        Ep, psip = eigsh(Hp, k=1, which='SA')
        Ep = Ep[0]; psip = psip[:, 0]

        D0 = compute_local_D(psi0, s, L, pbc, site_op_fn, generator)
        Dp = compute_local_D(psip, s, L, pbc, site_op_fn, generator)
        data.append({"dE": float(Ep - E0), "dD": float(Dp - D0)})

    dE = np.array([d["dE"] for d in data])
    dD = np.array([d["dD"] for d in data])
    sl, ic, rv, pv, se = linregress(dE, dD)
    r2 = rv**2

    print(f"    Slope={sl:+.4f} ± {se:.4f} | R²={r2:.4f}")
    return {"model": name, "L": L, "slope": float(sl), "r2": float(r2),
            "std_err": float(se), "n": n_perts}


def main():
    print("\n" + "#" * 70)
    print("  PHASE 2b: CORRECTED HAMILTONIAN UNIVERSALITY")
    print("  Generator = σ_x (transverse) for ALL models")
    print("  Perturbation = δh·σ_x for ALL models")
    print("#" * 70)

    L = 3
    N = L * L
    h = np.ones(N)
    ledger = []

    # TFIM baseline
    ledger.append(run_model(
        "TFIM", L, build_tfim,
        {"L": L, "J": 1.0, "h_fields": h}, n_perts=30
    ))

    # XXZ Δ=0.5 with transverse field
    ledger.append(run_model(
        "XXZ Δ=0.5 (transverse)", L, build_xxz_transverse,
        {"L": L, "J": 1.0, "Delta": 0.5, "hx_fields": h}, n_perts=30
    ))

    # Heisenberg Δ=1.0 with transverse field
    ledger.append(run_model(
        "Heisenberg (transverse)", L, build_xxz_transverse,
        {"L": L, "J": 1.0, "Delta": 1.0, "hx_fields": h}, n_perts=30
    ))

    # XXZ Δ=2.0 with transverse field
    ledger.append(run_model(
        "XXZ Δ=2.0 (transverse)", L, build_xxz_transverse,
        {"L": L, "J": 1.0, "Delta": 2.0, "hx_fields": h}, n_perts=30
    ))

    # XXZ Δ=5.0 (deep Ising limit) with transverse field
    ledger.append(run_model(
        "XXZ Δ=5.0 (deep Ising)", L, build_xxz_transverse,
        {"L": L, "J": 1.0, "Delta": 5.0, "hx_fields": h}, n_perts=30
    ))

    # Disordered TFIM mild
    rng1 = np.random.RandomState(77)
    ledger.append(run_model(
        "Disordered TFIM σ=0.3", L, build_disordered_tfim,
        {"L": L, "J_mean": 1.0, "J_std": 0.3, "h_fields": h, "rng": rng1},
        n_perts=30
    ))

    # Disordered TFIM strong
    rng2 = np.random.RandomState(88)
    ledger.append(run_model(
        "Disordered TFIM σ=0.7", L, build_disordered_tfim,
        {"L": L, "J_mean": 1.0, "J_std": 0.7, "h_fields": h, "rng": rng2},
        n_perts=30
    ))

    # L=4 checks
    print("\n  --- L=4 Scale Checks ---")
    h4 = np.ones(16)

    ledger.append(run_model(
        "TFIM L=4", 4, build_tfim,
        {"L": 4, "J": 1.0, "h_fields": h4}, n_perts=10
    ))

    ledger.append(run_model(
        "Heisenberg L=4 (transv)", 4, build_xxz_transverse,
        {"L": 4, "J": 1.0, "Delta": 1.0, "hx_fields": h4}, n_perts=10
    ))

    # Summary
    print("\n\n" + "=" * 70)
    print("  CORRECTED UNIVERSALITY LEDGER")
    print("=" * 70)
    print(f"  {'Model':<30s} {'L':>2s} {'N':>3s} {'Slope':>10s} {'R²':>8s}")
    print(f"  {'-'*30} {'-'*2} {'-'*3} {'-'*10} {'-'*8}")
    for r in ledger:
        print(f"  {r['model']:<30s} {r['L']:>2d} {r['n']:>3d} "
              f"{r['slope']:+.4f}    {r['r2']:.4f}")

    r2s = [r["r2"] for r in ledger]
    print(f"\n  Mean R²: {np.mean(r2s):.4f}")
    print(f"  Min  R²: {np.min(r2s):.4f} ({ledger[np.argmin(r2s)]['model']})")
    print(f"  Max  R²: {np.max(r2s):.4f} ({ledger[np.argmax(r2s)]['model']})")

    strong = [r for r in ledger if r["r2"] > 0.95]
    weak = [r for r in ledger if r["r2"] <= 0.95]

    if len(strong) == len(ledger):
        print(f"\n  ✅ ALL {len(ledger)} models show R²>0.95")
        print(f"     → Geometric deformation observable is UNIVERSAL")
    else:
        print(f"\n  Strong (R²>0.95): {len(strong)}/{len(ledger)}")
        print(f"  Weak:  {[r['model'] for r in weak]}")

    out_path = "/home/claude/hamiltonian_universality_corrected.json"
    with open(out_path, "w") as f:
        json.dump(ledger, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
