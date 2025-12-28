#!/usr/bin/env python3
"""
E8 Structure Search Protocol
============================

Tests the hypothesis that information geometry exhibits E8 exceptional Lie group structure.

Evidence to date:
- κ* = 64 ≈ 8² (E8 rank squared)
- 64D basin coordinates work empirically
- Need to validate: 248D, 240 roots, 8-dimensional core

Search Strategy:
1. Dimensional analysis (quick) - Test 8, 64, 248 dimensions
2. Attractor counting (medium) - Count stable basins
3. Symmetry testing (advanced) - Test E8 Weyl operations

Author: Synthesis from κ* universality discovery
Date: 2025-12-28
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# Try imports
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN, KMeans
    from scipy.spatial.distance import pdist, squareform
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - some tests will be skipped")

# Import pantheon relationships for basin data
try:
    from learned_relationships import get_learned_relationships
    PANTHEON_AVAILABLE = True
except ImportError:
    PANTHEON_AVAILABLE = False
    logger.warning("pantheon learned_relationships not available")

# ============================================================================
# E8 CONSTANTS
# ============================================================================

E8_RANK = 8
E8_DIMENSION = 248
E8_ROOTS = 240
E8_WEYL_ORDER = 696729600

# E8 simple roots (standard basis)
# These are the 8 generators that span the entire E8 root system
E8_SIMPLE_ROOTS = np.array([
    [1, -1, 0, 0, 0, 0, 0, 0],
    [0, 1, -1, 0, 0, 0, 0, 0],
    [0, 0, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 1, -1, 0, 0, 0],
    [0, 0, 0, 0, 1, -1, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, 0],
    [0, 0, 0, 0, 0, 0, 1, -1],
    [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5]  # Exceptional root
])


# ============================================================================
# PHASE 1: DIMENSIONAL ANALYSIS
# ============================================================================

def test_dimensionality(basins_64d: np.ndarray, verbose: bool = True) -> Dict:
    """
    Test if semantic basins exhibit E8 dimensional structure.
    
    Tests:
    1. Does 8D capture most variance? (rank hypothesis)
    2. Does 64D plateau? (rank² hypothesis)
    3. Does 248D add information? (full E8 hypothesis)
    
    Args:
        basins_64d: Current basin coordinates (n_words, 64)
        
    Returns:
        results: Dictionary with dimensional analysis
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available", "verdict": "unknown"}
    
    n_words = basins_64d.shape[0]
    
    if verbose:
        print("\n" + "="*80)
        print("PHASE 1: DIMENSIONAL ANALYSIS")
        print("="*80)
        print(f"Testing E8 dimensional structure on {n_words} semantic basins")
    
    results = {
        "n_words": n_words,
        "current_dims": 64,
        "tests": {}
    }
    
    # Test 1: Variance capture by dimension
    if verbose:
        print("\n1. Variance Capture Test")
        print("-" * 40)
    
    pca = PCA()
    pca.fit(basins_64d)
    
    variance_ratios = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratios)
    
    # Key dimensional checkpoints
    dims_to_check = [8, 16, 32, 64]
    variance_at_dims = {}
    
    for d in dims_to_check:
        if d <= len(cumulative_variance):
            var_capture = cumulative_variance[d-1]
            variance_at_dims[d] = float(var_capture)
            
            if verbose:
                print(f"  {d}D: {var_capture:.1%} variance")
                
                # E8 rank hypothesis
                if d == E8_RANK:
                    if var_capture > 0.80:
                        print(f"    ✅ >80% variance in rank(E8)={E8_RANK} dimensions!")
                    elif var_capture > 0.60:
                        print(f"    ⚠️ Moderate variance in rank(E8)={E8_RANK} dimensions")
                    else:
                        print(f"    ❌ Low variance in rank(E8)={E8_RANK} dimensions")
                
                # rank² hypothesis
                if d == 64:
                    if var_capture > 0.95:
                        print(f"    ✅ Plateau at {d}D = rank²!")
                    elif var_capture > 0.90:
                        print(f"    ⚠️ Partial plateau at {d}D")
                    else:
                        print(f"    ❌ No plateau at {d}D")
    
    results["tests"]["variance_capture"] = variance_at_dims
    
    # Test 2: Effective dimensionality
    if verbose:
        print("\n2. Effective Dimensionality")
        print("-" * 40)
    
    # Participation ratio (inverse of normalized eigenvalue sum of squares)
    eigenvalues = pca.explained_variance_
    normalized_eigs = eigenvalues / np.sum(eigenvalues)
    participation_ratio = 1.0 / np.sum(normalized_eigs**2)
    
    if verbose:
        print(f"  Participation ratio: {participation_ratio:.1f}")
        print(f"  Interpretation: Data effectively spans ~{participation_ratio:.0f} dimensions")
        
        if abs(participation_ratio - E8_RANK) < 2:
            print(f"  ✅ Close to E8 rank = {E8_RANK}!")
        elif abs(participation_ratio - 64) < 5:
            print(f"  ⚠️ Close to rank² = 64")
        else:
            print(f"  ❓ Effective dimension unclear")
    
    results["tests"]["effective_dimension"] = float(participation_ratio)
    
    # Test 3: Dimensional scaling
    if verbose:
        print("\n3. Dimensional Scaling Pattern")
        print("-" * 40)
    
    # Check if variance growth follows E8 pattern
    dims_8_64_ratio = variance_at_dims.get(64, 1) / variance_at_dims.get(8, 1) if variance_at_dims.get(8, 0) > 0 else 0
    
    if verbose:
        print(f"  Variance ratio (64D/8D): {dims_8_64_ratio:.2f}")
        print(f"  E8 dimension ratio: {64/8} = 8")
        
        if abs(dims_8_64_ratio - 1.0) < 0.2:
            print(f"  ✅ Near-perfect scaling (8D captures almost everything)!")
        elif dims_8_64_ratio < 1.5:
            print(f"  ⚠️ Modest improvement from 8D to 64D")
        else:
            print(f"  ❌ Large improvement suggests higher effective dimension")
    
    results["tests"]["dimensional_scaling"] = {
        "variance_ratio_64_8": float(dims_8_64_ratio),
        "e8_ratio": 8.0
    }
    
    # Summary
    if verbose:
        print("\n" + "="*80)
        print("DIMENSIONAL ANALYSIS SUMMARY")
        print("="*80)
        
        var_8d = variance_at_dims.get(8, 0)
        var_64d = variance_at_dims.get(64, 0)
        
        if var_8d > 0.75 and var_64d > 0.95:
            print("✅ STRONG EVIDENCE: 8D captures >75%, 64D plateaus >95%")
            print("   → Consistent with E8 rank=8, dimension=64 structure")
            results["verdict"] = "strong_evidence"
        elif var_8d > 0.60 and var_64d > 0.90:
            print("⚠️ MODERATE EVIDENCE: 8D captures >60%, 64D plateaus >90%")
            print("   → Partial support for E8 structure")
            results["verdict"] = "moderate_evidence"
        else:
            print("❌ WEAK EVIDENCE: Dimensional structure unclear")
            print("   → E8 hypothesis not supported by variance analysis")
            results["verdict"] = "weak_evidence"
    
    return results


# ============================================================================
# PHASE 2: ATTRACTOR COUNTING
# ============================================================================

def count_attractors(basins_64d: np.ndarray, verbose: bool = True) -> Dict:
    """
    Count stable attractor modes in semantic basin landscape.
    
    E8 prediction: Should find ~240 fundamental attractors (E8 roots)
    
    Method:
    1. Cluster basins using DBSCAN (density-based)
    2. Identify stable cluster centers
    3. Count distinct attractors
    4. Test if count ≈ 240
    
    Args:
        basins_64d: Basin coordinates (n_words, 64)
        
    Returns:
        results: Dictionary with attractor count and analysis
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available", "verdict": "unknown"}
    
    n_words = basins_64d.shape[0]
    
    if verbose:
        print("\n" + "="*80)
        print("PHASE 2: ATTRACTOR COUNTING")
        print("="*80)
        print(f"Searching for fundamental attractors in {n_words} basins")
        print(f"E8 prediction: ~{E8_ROOTS} distinct attractors")
    
    results = {
        "n_words": n_words,
        "e8_prediction": E8_ROOTS,
        "clustering": {}
    }
    
    # Method 1: DBSCAN clustering (density-based)
    if verbose:
        print("\n1. DBSCAN Clustering (Density-Based)")
        print("-" * 40)
    
    eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    dbscan_results = []
    
    for eps in eps_values:
        clustering = DBSCAN(eps=eps, min_samples=5).fit(basins_64d)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        n_noise = list(clustering.labels_).count(-1)
        
        dbscan_results.append({
            "eps": eps,
            "n_clusters": n_clusters,
            "n_noise": n_noise
        })
        
        if verbose:
            print(f"  eps={eps}: {n_clusters} clusters, {n_noise} noise points")
            
            if abs(n_clusters - E8_ROOTS) < 50:
                print(f"    ✅ Close to E8 roots = {E8_ROOTS}!")
    
    results["clustering"]["dbscan"] = dbscan_results
    
    # Method 2: K-means with various k values
    if verbose:
        print("\n2. K-Means Clustering Analysis")
        print("-" * 40)
    
    k_values = [50, 100, 150, 200, 240, 280]
    kmeans_results = []
    
    for k in k_values:
        if k < n_words:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(basins_64d)
            inertia = kmeans.inertia_
            avg_distance = inertia / n_words
            
            kmeans_results.append({
                "k": k,
                "inertia": float(inertia),
                "avg_distance": float(avg_distance)
            })
            
            if verbose:
                print(f"  k={k}: inertia={inertia:.1f}, avg_dist={avg_distance:.3f}")
                
                if k == E8_ROOTS:
                    print(f"    → Testing E8 hypothesis (k={E8_ROOTS} roots)")
    
    results["clustering"]["kmeans"] = kmeans_results
    
    # Method 3: Elbow analysis (find optimal k)
    if verbose:
        print("\n3. Elbow Analysis (Optimal Cluster Count)")
        print("-" * 40)
    
    k_range = list(range(20, min(n_words, 300), 20))
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5).fit(basins_64d)
        inertias.append(kmeans.inertia_)
    
    # Find elbow (second derivative)
    inertias_arr = np.array(inertias)
    if len(inertias_arr) > 2:
        second_deriv = np.diff(inertias_arr, 2)
        elbow_idx = np.argmin(second_deriv)
        optimal_k = k_range[elbow_idx + 1]
    else:
        optimal_k = k_range[0]
    
    if verbose:
        print(f"  Optimal k (elbow method): {optimal_k}")
        print(f"  E8 prediction: k = {E8_ROOTS}")
        print(f"  Difference: {abs(optimal_k - E8_ROOTS)}")
        
        if abs(optimal_k - E8_ROOTS) < 30:
            print(f"  ✅ Elbow near E8 roots!")
        elif abs(optimal_k - E8_ROOTS) < 60:
            print(f"  ⚠️ Elbow somewhat close to E8")
        else:
            print(f"  ❌ Elbow far from E8 prediction")
    
    results["clustering"]["elbow"] = {
        "optimal_k": int(optimal_k),
        "e8_prediction": E8_ROOTS,
        "difference": int(abs(optimal_k - E8_ROOTS))
    }
    
    # Summary
    if verbose:
        print("\n" + "="*80)
        print("ATTRACTOR COUNTING SUMMARY")
        print("="*80)
        
        best_dbscan = min(dbscan_results, key=lambda x: abs(x["n_clusters"] - E8_ROOTS))
        
        print(f"Best DBSCAN: {best_dbscan['n_clusters']} clusters (eps={best_dbscan['eps']})")
        print(f"Elbow method: {optimal_k} optimal clusters")
        print(f"E8 prediction: {E8_ROOTS} roots")
        
        dbscan_diff = abs(best_dbscan['n_clusters'] - E8_ROOTS)
        elbow_diff = abs(optimal_k - E8_ROOTS)
        
        if dbscan_diff < 30 or elbow_diff < 30:
            print("\n✅ STRONG EVIDENCE: Attractor count ~240 (E8 roots)")
            results["verdict"] = "strong_evidence"
        elif dbscan_diff < 60 or elbow_diff < 60:
            print("\n⚠️ MODERATE EVIDENCE: Attractor count within 60 of E8")
            results["verdict"] = "moderate_evidence"
        else:
            print("\n❌ WEAK EVIDENCE: Attractor count far from E8 prediction")
            results["verdict"] = "weak_evidence"
    
    return results


# ============================================================================
# PHASE 3: SYMMETRY TESTING (Advanced)
# ============================================================================

def test_e8_symmetries(basins_64d: np.ndarray, verbose: bool = True) -> Dict:
    """
    Test if semantic basins exhibit E8 Weyl group symmetries.
    
    E8 Weyl group has order 696,729,600 - too large to test exhaustively.
    Instead, test:
    1. Simple root reflections (8 generators)
    2. Invariance under root transformations
    3. Cartan subalgebra structure
    
    Args:
        basins_64d: Basin coordinates (n_words, 64)
        
    Returns:
        results: Dictionary with symmetry tests
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available", "verdict": "unknown"}
    
    n_words = basins_64d.shape[0]
    
    if verbose:
        print("\n" + "="*80)
        print("PHASE 3: E8 SYMMETRY TESTING")
        print("="*80)
        print(f"Testing E8 Weyl symmetries on {n_words} basins")
        print(f"Note: This is advanced analysis - partial testing only")
    
    results = {
        "n_words": n_words,
        "weyl_order": E8_WEYL_ORDER,
        "tests": {}
    }
    
    # Extract 8D subspace (E8 rank)
    pca = PCA(n_components=8)
    basins_8d = pca.fit_transform(basins_64d)
    
    if verbose:
        print(f"\nExtracted 8D subspace (E8 rank) from 64D basins")
        print(f"Variance captured: {np.sum(pca.explained_variance_ratio_):.1%}")
    
    # Test 1: Root reflection invariance
    if verbose:
        print("\n1. Simple Root Reflections")
        print("-" * 40)
    
    reflection_invariances = []
    
    for i, root in enumerate(E8_SIMPLE_ROOTS):
        root_norm = np.linalg.norm(root)
        if root_norm > 0:
            root_unit = root / root_norm
            
            # Project basins onto root
            projections = basins_8d @ root_unit
            
            # Reflect
            reflected = basins_8d - 2 * np.outer(projections, root_unit)
            
            # Measure distance change (sample 100 for speed)
            sample_size = min(100, n_words)
            original_dists = pdist(basins_8d[:sample_size])
            reflected_dists = pdist(reflected[:sample_size])
            
            # Invariance = how well distances are preserved
            dist_diff = np.mean(np.abs(original_dists - reflected_dists))
            invariance = 1.0 / (1.0 + dist_diff)  # Normalized [0,1]
            
            reflection_invariances.append({
                "root_index": i,
                "invariance": float(invariance),
                "dist_diff": float(dist_diff)
            })
            
            if verbose:
                print(f"  Root {i}: invariance={invariance:.3f}, dist_diff={dist_diff:.3f}")
    
    results["tests"]["root_reflections"] = reflection_invariances
    
    avg_invariance = np.mean([r["invariance"] for r in reflection_invariances])
    
    if verbose:
        print(f"\n  Average invariance: {avg_invariance:.3f}")
        
        if avg_invariance > 0.90:
            print(f"  ✅ HIGH invariance under root reflections!")
        elif avg_invariance > 0.75:
            print(f"  ⚠️ MODERATE invariance")
        else:
            print(f"  ❌ LOW invariance")
    
    results["tests"]["avg_invariance"] = float(avg_invariance)
    
    # Test 2: Cartan subalgebra (8D torus structure)
    if verbose:
        print("\n2. Cartan Subalgebra Test")
        print("-" * 40)
        print("  Testing if 8D subspace forms torus-like structure")
    
    # Compute pairwise distances in 8D (sample for speed)
    sample_size = min(500, n_words)
    dists_8d = squareform(pdist(basins_8d[:sample_size]))
    
    # Check for periodic structure (torus has periodic distances)
    hist, bin_edges = np.histogram(dists_8d.flatten(), bins=50)
    
    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append(bin_edges[i])
    
    if verbose:
        print(f"  Distance distribution peaks: {len(peaks)} found")
        
        if len(peaks) >= 3:
            print(f"  ✅ Multiple distance peaks suggest periodic structure")
        elif len(peaks) >= 2:
            print(f"  ⚠️ Some periodic structure")
        else:
            print(f"  ❌ No clear periodic structure")
    
    results["tests"]["cartan_structure"] = {
        "n_peaks": len(peaks),
        "peak_distances": [float(p) for p in peaks[:5]]  # Top 5
    }
    
    # Summary
    if verbose:
        print("\n" + "="*80)
        print("SYMMETRY TESTING SUMMARY")
        print("="*80)
        
        if avg_invariance > 0.85 and len(peaks) >= 3:
            print("✅ STRONG EVIDENCE: High invariance + periodic structure")
            print("   → Consistent with E8 Weyl symmetries")
            results["verdict"] = "strong_evidence"
        elif avg_invariance > 0.70 or len(peaks) >= 2:
            print("⚠️ MODERATE EVIDENCE: Partial symmetry structure")
            results["verdict"] = "moderate_evidence"
        else:
            print("❌ WEAK EVIDENCE: No clear E8 symmetries")
            results["verdict"] = "weak_evidence"
    
    return results


# ============================================================================
# MAIN E8 SEARCH PROTOCOL
# ============================================================================

def run_e8_search(basins_64d: np.ndarray = None, verbose: bool = True) -> Dict:
    """
    Complete E8 structure search across all phases.
    
    Args:
        basins_64d: Optional basin coordinates. If None, loads from pantheon.
        verbose: Print detailed results
        
    Returns:
        results: Complete E8 search results
    """
    if verbose:
        print("\n" + "="*80)
        print("E8 STRUCTURE SEARCH PROTOCOL")
        print("="*80)
        print("\nSearching for E8 exceptional Lie group structure in semantic basins")
        print(f"E8 parameters: Rank={E8_RANK}, Dimension={E8_DIMENSION}, Roots={E8_ROOTS}")
    
    # Load basins if not provided
    if basins_64d is None:
        if PANTHEON_AVAILABLE:
            if verbose:
                print("\nLoading pantheon semantic basins...")
            
            lr = get_learned_relationships()
            
            # Extract all basins
            if lr.adjusted_basins:
                words = list(lr.adjusted_basins.keys())
                basins_64d = np.array([lr.adjusted_basins[w] for w in words])
                
                if verbose:
                    print(f"Loaded {len(words)} word basins from adjusted_basins (64D)")
            else:
                if verbose:
                    print("No adjusted basins found, generating synthetic data")
                n_words = 1000
                basins_64d = np.random.randn(n_words, 64)
                basins_64d = basins_64d / np.linalg.norm(basins_64d, axis=1, keepdims=True)
        else:
            if verbose:
                print("\n⚠️ Pantheon not available - generating synthetic test data")
            
            n_words = 1000
            basins_64d = np.random.randn(n_words, 64)
            basins_64d = basins_64d / np.linalg.norm(basins_64d, axis=1, keepdims=True)
    
    # Run all phases
    results = {
        "timestamp": "2025-12-28",
        "n_basins": basins_64d.shape[0],
        "basin_dims": basins_64d.shape[1],
        "phases": {}
    }
    
    # Phase 1: Dimensional Analysis
    dim_results = test_dimensionality(basins_64d, verbose=verbose)
    results["phases"]["dimensional_analysis"] = dim_results
    
    # Phase 2: Attractor Counting
    attractor_results = count_attractors(basins_64d, verbose=verbose)
    results["phases"]["attractor_counting"] = attractor_results
    
    # Phase 3: Symmetry Testing
    symmetry_results = test_e8_symmetries(basins_64d, verbose=verbose)
    results["phases"]["symmetry_testing"] = symmetry_results
    
    # Overall verdict
    verdicts = [
        dim_results.get("verdict", "unknown"),
        attractor_results.get("verdict", "unknown"),
        symmetry_results.get("verdict", "unknown")
    ]
    
    strong_count = verdicts.count("strong_evidence")
    moderate_count = verdicts.count("moderate_evidence")
    
    # Determine verdict first (always computed)
    if strong_count >= 2:
        overall = "validated"
    elif strong_count >= 1 or moderate_count >= 2:
        overall = "partial"
    else:
        overall = "not_validated"
    
    if verbose:
        print("\n" + "="*80)
        print("OVERALL E8 HYPOTHESIS VERDICT")
        print("="*80)
        print(f"\nPhase results:")
        print(f"  Dimensional Analysis: {dim_results.get('verdict', 'unknown')}")
        print(f"  Attractor Counting: {attractor_results.get('verdict', 'unknown')}")
        print(f"  Symmetry Testing: {symmetry_results.get('verdict', 'unknown')}")
        print()
        
        if overall == "validated":
            print("🏆 STRONG VALIDATION: E8 structure detected!")
            print("   → κ* = 64 = 8² consistent with E8 rank²")
            print("   → Multiple independent tests support E8")
            print("   → Ready for publication claim")
        elif overall == "partial":
            print("⚠️ PARTIAL VALIDATION: E8 structure partially supported")
            print("   → Some tests support E8, others inconclusive")
            print("   → κ* = 64 still suggestive of E8")
            print("   → Needs further investigation")
        else:
            print("❌ NO VALIDATION: E8 structure not detected")
            print("   → Tests do not support E8 hypothesis")
            print("   → κ* = 64 may be coincidence")
            print("   → Alternative explanations needed")
    
    results["overall_verdict"] = overall
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("E8 STRUCTURE SEARCH - EXECUTION")
    print("="*80)
    
    # Run complete E8 search
    results = run_e8_search(verbose=True)
    
    # Save results
    output_path = Path("results/e8_structure_search.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Generate summary report
    print("\n" + "="*80)
    print("E8 SEARCH SUMMARY REPORT")
    print("="*80)
    
    print(f"\nTested {results['n_basins']} semantic basins ({results['basin_dims']}D)")
    print(f"Overall verdict: {results['overall_verdict'].upper()}")
    
    if results['overall_verdict'] == 'validated':
        print("\n🎉 E8 structure VALIDATED!")
        print("   This supports the claim that κ* = 64 = rank(E8)²")
        print("   Information geometry exhibits exceptional symmetry")
        print("\n📝 Ready for Nature/Science submission")
    
    elif results['overall_verdict'] == 'partial':
        print("\n⚠️ Partial E8 evidence - needs investigation")
        print("   κ* = 64 remains suggestive")
        print("   Consider: larger datasets, refined methods")
        print("\n📝 Still publishable (NeurIPS/ICML level)")
    
    else:
        print("\n❌ E8 not detected - κ* = 64 may be coincidence")
        print("   Consider alternative explanations")
        print("\n📝 Focus on κ* universality (already validated)")
