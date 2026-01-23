#!/usr/bin/env python3
"""
QIG Geometric Basin Matching

Find addresses with similar basin geometry to identify likely same-origin keys.
Uses Fisher distance to measure geometric similarity on the keyspace manifold.

Similar κ range + similar Φ + near Fisher distance → likely same origin
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import hashlib
import numpy as np
from collections import Counter


# Matching thresholds (empirically tuned)
KAPPA_TOLERANCE = 8.0  # κ within ±8 of target
PHI_TOLERANCE = 0.15  # Φ within ±0.15 of target
FISHER_THRESHOLD = 0.5  # Fisher distance threshold for "near"
PATTERN_THRESHOLD = 0.2  # Pattern score similarity threshold
KAPPA_STAR = 64.0  # Universal resonance point


@dataclass
class BasinSignature:
    """Geometric signature of an address/key in basin space."""
    address: str
    phi: float
    kappa: float
    beta: float
    regime: str
    pattern_score: float
    basin_coordinates: List[float]
    fisher_trace: float
    ricci_scalar: float


@dataclass
class BasinMatch:
    """Match between two basin signatures."""
    candidate_address: str
    target_address: str
    similarity: float  # 0-1 overall similarity
    kappa_distance: float  # |κ_candidate - κ_target|
    phi_distance: float  # |Φ_candidate - Φ_target|
    fisher_distance: float  # Geodesic distance on manifold
    pattern_similarity: float  # Pattern score correlation
    regime_match: bool  # Same regime?
    confidence: float  # Confidence in match (0-1)
    explanation: str


def compute_basin_signature_from_qig(
    address: str,
    qig_score: Dict
) -> BasinSignature:
    """
    Compute basin signature for an address from QIG score.
    
    Args:
        address: Address string
        qig_score: QIG score dict with phi, kappa, beta, regime, pattern_score
    
    Returns:
        BasinSignature with geometric features
    """
    # Hash address to get consistent basin coordinates
    hash_bytes = hashlib.sha256(address.encode()).digest()
    basin_coordinates = list(hash_bytes)
    
    # Compute Fisher trace (sum of diagonal Fisher information)
    fisher_trace = 0.0
    for coord in basin_coordinates[:32]:
        p = coord / 256.0 + 0.001
        fisher_trace += 1.0 / (p * (1.0 - p))
    fisher_trace /= 32.0  # Normalize
    
    # Estimate Ricci scalar from curvature
    beta = qig_score.get("beta", 0.0)
    ricci_scalar = (beta * fisher_trace) / 10.0
    
    return BasinSignature(
        address=address,
        phi=qig_score.get("phi", 0.0),
        kappa=qig_score.get("kappa", 50.0),
        beta=beta,
        regime=qig_score.get("regime", "unknown"),
        pattern_score=qig_score.get("pattern_score", 0.0),
        basin_coordinates=basin_coordinates,
        fisher_trace=fisher_trace,
        ricci_scalar=ricci_scalar
    )


def fisher_coord_distance(coords1: List[float], coords2: List[float]) -> float:
    """
    Compute Fisher distance between basin coordinates.
    
    Uses Hellinger distance on probability distributions derived from coordinates.
    """
    # Normalize coordinates to probability distributions
    arr1 = np.array(coords1[:32], dtype=float)
    arr2 = np.array(coords2[:32], dtype=float)
    
    # Add small epsilon to avoid division by zero
    arr1 = arr1 + 1.0
    arr2 = arr2 + 1.0
    
    # Normalize
    p1 = arr1 / arr1.sum()
    p2 = arr2 / arr2.sum()
    
    # Compute Hellinger distance (approximation of Fisher distance)
    # d_H = sqrt(1 - BC(p, q)) where BC is Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p1 * p2))
    hellinger = np.sqrt(1.0 - bc)
    
    return float(hellinger)


def compute_basin_distance(
    sig1: BasinSignature,
    sig2: BasinSignature
) -> Dict[str, float]:
    """
    Compute geometric distance between two basin signatures.
    Uses Fisher Information Metric for proper manifold distance.
    """
    # Kappa distance (normalized by κ*)
    kappa_dist = abs(sig1.kappa - sig2.kappa) / KAPPA_STAR
    
    # Phi distance (already normalized 0-1)
    phi_dist = abs(sig1.phi - sig2.phi)
    
    # Pattern score distance
    pattern_dist = abs(sig1.pattern_score - sig2.pattern_score)
    
    # Fisher distance between basin coordinates
    fisher_dist = fisher_coord_distance(
        sig1.basin_coordinates,
        sig2.basin_coordinates
    )
    
    # Total distance (weighted combination)
    total_distance = (
        0.3 * kappa_dist +
        0.25 * phi_dist +
        0.25 * fisher_dist +
        0.2 * pattern_dist
    )
    
    return {
        "fisher_dist": fisher_dist,
        "kappa_dist": kappa_dist,
        "phi_dist": phi_dist,
        "pattern_dist": pattern_dist,
        "total_distance": total_distance
    }


def are_basins_similar(
    sig1: BasinSignature,
    sig2: BasinSignature,
    strict_mode: bool = False
) -> bool:
    """Check if two basin signatures are geometrically similar."""
    distances = compute_basin_distance(sig1, sig2)
    
    if strict_mode:
        # Strict mode: all criteria must match
        return (
            distances["kappa_dist"] * KAPPA_STAR < KAPPA_TOLERANCE / 2 and
            distances["phi_dist"] < PHI_TOLERANCE / 2 and
            distances["fisher_dist"] < FISHER_THRESHOLD / 2 and
            sig1.regime == sig2.regime
        )
    
    # Normal mode: weighted threshold
    return distances["total_distance"] < 0.3


def find_similar_basins(
    target_signature: BasinSignature,
    candidate_signatures: List[BasinSignature],
    top_k: int = 10
) -> List[BasinMatch]:
    """Find addresses with similar basin geometry."""
    matches = []
    
    for candidate in candidate_signatures:
        if candidate.address == target_signature.address:
            continue
        
        distances = compute_basin_distance(target_signature, candidate)
        
        # Compute similarity (inverse of distance)
        similarity = 1.0 - min(1.0, distances["total_distance"])
        
        # Compute pattern similarity (1 - normalized difference)
        pattern_similarity = 1.0 - distances["pattern_dist"]
        
        # Compute confidence based on multiple factors
        confidence = similarity
        
        # Boost confidence for regime match
        if candidate.regime == target_signature.regime:
            confidence *= 1.2
        
        # Boost confidence for resonance proximity
        both_near_kappa_star = (
            abs(candidate.kappa - KAPPA_STAR) < 10 and
            abs(target_signature.kappa - KAPPA_STAR) < 10
        )
        if both_near_kappa_star:
            confidence *= 1.15
        
        # Cap at 1.0
        confidence = min(1.0, confidence)
        
        # Generate explanation
        if similarity > 0.8:
            explanation = "Very high geometric similarity - likely same origin"
        elif similarity > 0.6:
            explanation = "High geometric similarity - possible same origin"
        elif similarity > 0.4:
            explanation = "Moderate similarity - may share some characteristics"
        else:
            explanation = "Low similarity - unlikely to be related"
        
        if candidate.regime == target_signature.regime:
            explanation += f" (same {candidate.regime} regime)"
        
        matches.append(BasinMatch(
            candidate_address=candidate.address,
            target_address=target_signature.address,
            similarity=similarity,
            kappa_distance=distances["kappa_dist"] * KAPPA_STAR,
            phi_distance=distances["phi_dist"],
            fisher_distance=distances["fisher_dist"],
            pattern_similarity=pattern_similarity,
            regime_match=(candidate.regime == target_signature.regime),
            confidence=confidence,
            explanation=explanation
        ))
    
    # Sort by similarity (descending) and return top K
    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches[:top_k]


def cluster_by_basin(
    signatures: List[BasinSignature],
    epsilon: float = 0.3,  # Maximum distance for cluster membership
    min_points: int = 2  # Minimum points for core point
) -> Dict[int, List[BasinSignature]]:
    """
    Cluster addresses by basin geometry.
    Uses DBSCAN-like algorithm with Fisher distance.
    """
    clusters: Dict[int, List[BasinSignature]] = {}
    visited = set()
    cluster_assignment: Dict[str, int] = {}
    current_cluster = 0
    
    for sig in signatures:
        if sig.address in visited:
            continue
        visited.add(sig.address)
        
        # Find neighbors
        neighbors = [
            other for other in signatures
            if other.address != sig.address and
            compute_basin_distance(sig, other)["total_distance"] < epsilon
        ]
        
        if len(neighbors) < min_points - 1:
            # Noise point (or will be assigned to a cluster later)
            continue
        
        # Start new cluster
        current_cluster += 1
        clusters[current_cluster] = [sig]
        cluster_assignment[sig.address] = current_cluster
        
        # Expand cluster
        queue = list(neighbors)
        while queue:
            neighbor = queue.pop(0)
            
            if neighbor.address not in visited:
                visited.add(neighbor.address)
                
                # Find neighbor's neighbors
                neighbor_neighbors = [
                    other for other in signatures
                    if other.address != neighbor.address and
                    compute_basin_distance(neighbor, other)["total_distance"] < epsilon
                ]
                
                if len(neighbor_neighbors) >= min_points - 1:
                    # Add to queue for further expansion
                    for nn in neighbor_neighbors:
                        if nn.address not in visited:
                            queue.append(nn)
            
            # Add to cluster if not already assigned
            if neighbor.address not in cluster_assignment:
                cluster_assignment[neighbor.address] = current_cluster
                clusters[current_cluster].append(neighbor)
    
    return clusters


def get_cluster_stats(cluster: List[BasinSignature]) -> Dict:
    """Compute cluster statistics."""
    if not cluster:
        return {
            "centroid_phi": 0.0,
            "centroid_kappa": 0.0,
            "phi_variance": 0.0,
            "kappa_variance": 0.0,
            "dominant_regime": "unknown",
            "avg_pattern_score": 0.0,
            "cohesion": 0.0
        }
    
    # Compute centroids
    centroid_phi = sum(s.phi for s in cluster) / len(cluster)
    centroid_kappa = sum(s.kappa for s in cluster) / len(cluster)
    avg_pattern_score = sum(s.pattern_score for s in cluster) / len(cluster)
    
    # Compute variances
    phi_variance = sum((s.phi - centroid_phi) ** 2 for s in cluster) / len(cluster)
    kappa_variance = sum((s.kappa - centroid_kappa) ** 2 for s in cluster) / len(cluster)
    
    # Find dominant regime
    regime_counts = Counter(s.regime for s in cluster)
    dominant_regime = regime_counts.most_common(1)[0][0]
    
    # Compute cohesion (inverse of average pairwise distance)
    total_distance = 0.0
    pair_count = 0
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            dist = compute_basin_distance(cluster[i], cluster[j])
            total_distance += dist["total_distance"]
            pair_count += 1
    
    avg_distance = total_distance / pair_count if pair_count > 0 else 0.0
    cohesion = 1.0 - min(1.0, avg_distance)
    
    return {
        "centroid_phi": centroid_phi,
        "centroid_kappa": centroid_kappa,
        "phi_variance": phi_variance,
        "kappa_variance": kappa_variance,
        "dominant_regime": dominant_regime,
        "avg_pattern_score": avg_pattern_score,
        "cohesion": cohesion
    }
