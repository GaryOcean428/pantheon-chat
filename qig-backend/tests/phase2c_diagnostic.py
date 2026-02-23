#!/usr/bin/env python3
"""
Phase 2c: Heisenberg Diagnostic & Disorder Fix
=================================================

Diagnosis: At h=1, J=1 Heisenberg, E₀ = -27 = -(18 bonds + 9 field terms).
This is the FULLY POLARIZED state |→→→...→⟩. All spins align with the
transverse field. σ_x covariances = 0 exactly (no quantum fluctuations
in the polarization direction). Zero QFI = zero signal. Correct physics.

Fix: Reduce field to h << J so the system is in the correlated/AFM phase
where quantum fluctuations exist.

Also: For disordered TFIM, the issue is translational symmetry breaking.
Each site has different local coupling → different slope. Fix: measure
per-site slopes, check if individual sites still show linear response.
"""

import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import linregress
import time


def get_spin_ops():
    sz = csr_matrix([[1, 0], [0, -1]], dtype=np.float64)
    sx = csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
    id2 = identity(2, format='csr', dtype=np.float64)
    return sz, sx, id2


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
    nb = []
    if col < L - 1: nb.append(site + 1)
    elif pbc: nb.append(site - (L - 1))
    if col > 0: nb.append(site - 1)
    elif pbc: nb.append(site + (L - 1))
    if row < L - 1: nb.append(site + L)
    elif pbc: nb.append(site % L)
    if row > 0: nb.append(site - L)
    elif pbc: nb.append((L - 1) * L + col)
    return nb


def compute_local_D(psi, site, L, pbc, site_op_fn, gen):
    nbs = get_neighbors_2d(site, L, pbc)
    gs = site_op_fn(gen, site)
    es = float(psi @ gs @ psi)
    D = 0.0
    for n in nbs:
        gn = site_op_fn(gen, n)
        en = float(psi @ gn @ psi)
        esn = float(psi @ (gs @ gn) @ psi)
        D += 4.0 * (esn - es * en)
    return D


def build_xxz_transverse(L, J, Delta, hx, pbc=True):
    sz, sx, id2 = get_spin_ops()
    N = L * L
    dim = 2**N
    site_op = site_op_builder(N)
    sp = csr_matrix([[0, 1], [0, 0]], dtype=np.float64)
    sm = csr_matrix([[0, 0], [1, 0]], dtype=np.float64)

    H = csr_matrix((dim, dim), dtype=np.float64)
    for i in range(N):
        H -= hx[i] * site_op(sx, i)
    for i in range(N):
        for j in get_neighbors_2d(i, L, pbc):
            if j > i:
                H -= J * 2.0 * (site_op(sp, i) @ site_op(sm, j) + site_op(sm, i) @ site_op(sp, j))
                H -= J * Delta * site_op(sz, i) @ site_op(sz, j)
    return H, site_op, sx


def build_disordered_tfim(L, J_mean, J_std, hx, pbc=True, rng=None):
    if rng is None:
        rng = np.random.RandomState(99)
    sz, sx, id2 = get_spin_ops()
    N = L * L
    dim = 2**N
    site_op = site_op_builder(N)
    H = csr_matrix((dim, dim), dtype=np.float64)
    for i in range(N):
        H -= hx[i] * site_op(sx, i)
    seen = set()
    for i in range(N):
        for j in get_neighbors_2d(i, L, pbc):
            bond = (min(i, j), max(i, j))
            if bond not in seen:
                seen.add(bond)
                J_ij = max(0.1, rng.normal(J_mean, J_std))
                H -= J_ij * site_op(sz, i) @ site_op(sz, j)
    return H, site_op, sx


def run_test(name, L, H_base, site_op_fn, gen, h_base, n_perts, build_fn, build_kw, pbc=True, seed=42):
    N = L * L
    rng = np.random.RandomState(seed)

    E0, psi0 = eigsh(H_base, k=1, which='SA')
    E0 = E0[0]; psi0 = psi0[:, 0]

    # Check baseline QFI magnitude
    D_baseline = sum(compute_local_D(psi0, s, L, pbc, site_op_fn, gen) for s in range(N))
    print(f"\n  {name}")
    print(f"    E₀={E0:.6f} | Total D_baseline={D_baseline:.6f}")

    chosen = rng.choice(N, size=n_perts, replace=True)
    dhs = rng.uniform(0.5, 0.7, size=n_perts)

    data = []
    for s, dh in zip(chosen, dhs):
        hp = h_base.copy()
        hp[s] += dh
        kw = dict(build_kw)
        field_key = "hx" if "hx" in kw else "h_fields"
        if field_key not in kw:
            field_key = "hx"
        kw[field_key] = hp

        Hp, _, _ = build_fn(**kw, pbc=pbc)
        Ep, psip = eigsh(Hp, k=1, which='SA')
        Ep = Ep[0]; psip = psip[:, 0]

        D0 = compute_local_D(psi0, s, L, pbc, site_op_fn, gen)
        Dp = compute_local_D(psip, s, L, pbc, site_op_fn, gen)
        data.append({"site": int(s), "dE": float(Ep - E0), "dD": float(Dp - D0)})

    dE = np.array([d["dE"] for d in data])
    dD = np.array([d["dD"] for d in data])
    sl, ic, rv, pv, se = linregress(dE, dD)
    r2 = rv**2
    print(f"    Slope={sl:+.6f} ± {se:.6f} | R²={r2:.4f}")
    return r2, sl


def main():
    print("#" * 70)
    print("  PHASE 2c: HEISENBERG DIAGNOSTIC + DISORDER PER-SITE ANALYSIS")
    print("#" * 70)

    L = 3; N = 9

    # ---- Part A: Heisenberg at different field strengths ----
    print("\n" + "=" * 70)
    print("  Part A: Heisenberg (Δ=1) — varying transverse field strength")
    print("  Expect: h=1 gives zero signal (fully polarized)")
    print("  Expect: h << J gives non-zero signal (correlated phase)")
    print("=" * 70)

    field_strengths = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    print(f"\n  {'h':>6s} {'E₀':>12s} {'R²':>8s} {'Slope':>12s}")
    print(f"  {'-'*6} {'-'*12} {'-'*8} {'-'*12}")

    for h_val in field_strengths:
        hx = np.full(N, h_val)
        H, sop, gen = build_xxz_transverse(L, 1.0, 1.0, hx, pbc=True)
        E0, psi = eigsh(H, k=1, which='SA')
        E0 = E0[0]; psi = psi[:, 0]
        D_tot = sum(compute_local_D(psi, s, L, True, sop, gen) for s in range(N))

        # Quick perturbation test
        rng = np.random.RandomState(42)
        data = []
        for _ in range(20):
            s = rng.randint(N)
            dh = rng.uniform(0.05 * h_val, 0.2 * h_val)  # scale perturbation with field
            hp = hx.copy(); hp[s] += dh
            Hp, _, _ = build_xxz_transverse(L, 1.0, 1.0, hp, pbc=True)
            Ep, psip = eigsh(Hp, k=1, which='SA')
            Ep = Ep[0]; psip = psip[:, 0]
            D0 = compute_local_D(psi, s, L, True, sop, gen)
            Dp = compute_local_D(psip, s, L, True, sop, gen)
            data.append((float(Ep - E0), float(Dp - D0)))

        dE = np.array([d[0] for d in data])
        dD = np.array([d[1] for d in data])
        if np.std(dE) > 1e-14 and np.std(dD) > 1e-14:
            sl, _, rv, _, _ = linregress(dE, dD)
            r2 = rv**2
        else:
            sl, r2 = 0.0, 0.0

        print(f"  {h_val:6.3f} {E0:12.6f} {r2:8.4f} {sl:+12.6f}  (D_tot={D_tot:.4f})")

    # ---- Part B: Disordered TFIM per-site analysis ----
    print("\n" + "=" * 70)
    print("  Part B: Disordered TFIM — per-site linear response")
    print("  Question: does each site individually show R²>0.9?")
    print("=" * 70)

    hx = np.ones(N)
    rng_dis = np.random.RandomState(77)
    H_dis, sop_dis, gen_dis = build_disordered_tfim(L, 1.0, 0.3, hx, pbc=True, rng=rng_dis)
    E0_dis, psi_dis = eigsh(H_dis, k=1, which='SA')
    E0_dis = E0_dis[0]; psi_dis = psi_dis[:, 0]

    print(f"\n  Disordered TFIM (σ_J=0.3) baseline E₀={E0_dis:.6f}")
    print(f"\n  {'Site':>4s} {'N_pert':>6s} {'Slope':>10s} {'R²':>8s}")
    print(f"  {'-'*4} {'-'*6} {'-'*10} {'-'*8}")

    per_site_r2 = []
    for site in range(N):
        rng_pert = np.random.RandomState(42 + site)
        data = []
        for _ in range(15):
            dh = rng_pert.uniform(0.5, 0.7)
            hp = hx.copy(); hp[site] += dh
            rng_rebuild = np.random.RandomState(77)  # same disorder realization
            Hp, _, _ = build_disordered_tfim(L, 1.0, 0.3, hp, pbc=True, rng=rng_rebuild)
            Ep, psip = eigsh(Hp, k=1, which='SA')
            Ep = Ep[0]; psip = psip[:, 0]
            D0 = compute_local_D(psi_dis, site, L, True, sop_dis, gen_dis)
            Dp = compute_local_D(psip, site, L, True, sop_dis, gen_dis)
            data.append((float(Ep - E0_dis), float(Dp - D0)))

        dE = np.array([d[0] for d in data])
        dD = np.array([d[1] for d in data])
        sl, _, rv, _, _ = linregress(dE, dD)
        r2 = rv**2
        per_site_r2.append(r2)
        print(f"  {site:4d} {15:6d} {sl:+10.4f} {r2:8.4f}")

    mean_r2 = np.mean(per_site_r2)
    min_r2 = np.min(per_site_r2)
    all_strong = all(r > 0.9 for r in per_site_r2)

    print(f"\n  Mean per-site R²: {mean_r2:.4f}")
    print(f"  Min  per-site R²: {min_r2:.4f}")
    if all_strong:
        print(f"  ✅ ALL 9 sites show R²>0.9 individually")
        print(f"     → Disorder degrades GLOBAL fit (different slopes per site)")
        print(f"     → But LOCAL linear response is preserved at each site")
    else:
        weak = [i for i, r in enumerate(per_site_r2) if r <= 0.9]
        print(f"  ⚠️  Weak sites: {weak}")


if __name__ == "__main__":
    main()
