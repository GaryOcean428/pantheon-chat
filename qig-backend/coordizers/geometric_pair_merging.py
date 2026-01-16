"""
Geometric Pair Merging - Geometry-First Vocabulary Learning

Implements vocabulary merging using pure geometric operations on Fisher manifold.
Unlike traditional BPE (frequency-driven), this uses QIG consciousness metrics
to determine which pairs should be merged into new vocabulary tokens.

**QIG PURITY - GEOMETRY FIRST, NOT FREQUENCY:**
- Merge criterion: Φ gain + κ consistency - curvature cost (NOT frequency)
- Frequency is ONLY a weak regularizer (prevents rare-pair noise)
- New tokens: Geodesic interpolation between pair coordinates
- Metric integrity: All operations preserve Fisher-Rao distances

**Training Objective (Fisher/QFI Functional):**
Maximize: ∫ [Φ(merged_vocab) - Φ(original_vocab)] dμ
Subject to: κ_consistency > threshold, curvature_discontinuity < threshold

This is consciousness-guided vocabulary learning, not statistical tokenization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qig_geometry import (
    fisher_coord_distance,
    geodesic_interpolation,
    sphere_project,
    estimate_manifold_curvature,
)

# Import consciousness metrics for geometric merge scoring
try:
    from qig_core.phi_computation import compute_phi_qig, compute_qfi_matrix
    from qig_core.consciousness_metrics import compute_kappa_effective
except ImportError:
    # Fallback if qig_core not available
    def compute_phi_qig(basin_coords, n_samples=100):
        """Fallback phi computation."""
        p = np.abs(basin_coords) + 1e-10
        p = p / p.sum()
        entropy = -np.sum(p * np.log(p + 1e-10))
        return entropy / np.log(len(p))
    
    def compute_qfi_matrix(basin_coords):
        """Fallback QFI computation."""
        p = np.abs(basin_coords) ** 2 + 1e-10
        p = p / p.sum()
        return np.diag(1.0 / (p + 1e-10))
    
    def compute_kappa_effective(basin_coords, kappa_star=64.21):
        """Fallback kappa computation."""
        p = np.abs(basin_coords) + 1e-10
        p = p / p.sum()
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(p))
        concentration = 1.0 - (entropy / max_entropy)
        return kappa_star * (0.5 + 0.8 * concentration)


def compute_phi_gain_for_merge(
    coord1: np.ndarray,
    coord2: np.ndarray,
    merged_coord: np.ndarray,
    context_coords: List[np.ndarray]
) -> float:
    """
    Compute Φ (integration) gain from merging two tokens.
    
    Measures how merging improves information integration in context.
    Higher Φ gain = better merge (more consciousness integration).
    
    Formula:
    Φ_gain = Φ(context + merged_token) - max(Φ(context + token1), Φ(context + token2))
    
    Args:
        coord1: First token basin coordinate
        coord2: Second token basin coordinate  
        merged_coord: Merged token basin coordinate (geodesic midpoint)
        context_coords: Surrounding context basins
        
    Returns:
        Φ gain ∈ [-1, 1] (positive = merge improves integration)
    """
    if not context_coords:
        # No context - use individual token Φ as baseline
        phi_merged, _ = compute_phi_qig(merged_coord, n_samples=100)
        phi1, _ = compute_phi_qig(coord1, n_samples=100)
        phi2, _ = compute_phi_qig(coord2, n_samples=100)
        return phi_merged - max(phi1, phi2)
    
    # Compute Φ with merged token in context
    context_with_merged = context_coords + [merged_coord]
    avg_merged = np.mean(context_with_merged, axis=0)
    phi_merged, _ = compute_phi_qig(avg_merged, n_samples=100)
    
    # Compute Φ with original tokens
    context_with_original = context_coords + [coord1, coord2]
    avg_original = np.mean(context_with_original, axis=0)
    phi_original, _ = compute_phi_qig(avg_original, n_samples=100)
    
    return phi_merged - phi_original


def compute_kappa_consistency_for_merge(
    coord1: np.ndarray,
    coord2: np.ndarray,
    merged_coord: np.ndarray
) -> float:
    """
    Compute κ (coupling) consistency improvement from merging.
    
    Measures if merged token has stable κ value (not too different from components).
    Higher consistency = better merge (more stable representation).
    
    Formula:
    consistency = 1 - |κ(merged) - mean(κ(token1), κ(token2))| / κ*
    
    Args:
        coord1: First token basin coordinate
        coord2: Second token basin coordinate
        merged_coord: Merged token basin coordinate
        
    Returns:
        κ consistency ∈ [0, 1] (1 = perfectly consistent)
    """
    kappa1 = compute_kappa_effective(coord1)
    kappa2 = compute_kappa_effective(coord2)
    kappa_merged = compute_kappa_effective(merged_coord)
    
    avg_kappa_original = (kappa1 + kappa2) / 2.0
    kappa_deviation = abs(kappa_merged - avg_kappa_original)
    
    # Normalize by κ* = 64.21
    consistency = 1.0 - (kappa_deviation / 64.21)
    
    return max(0.0, min(1.0, consistency))


def compute_fisher_curvature_discontinuity(
    coord1: np.ndarray,
    coord2: np.ndarray,
    merged_coord: np.ndarray
) -> float:
    """
    Compute Fisher manifold curvature discontinuity from merging.
    
    Measures if the merge creates a geodesic discontinuity (jump).
    Lower discontinuity = better merge (smoother manifold path).
    
    Formula:
    discontinuity = d_FR(merged, geodesic_midpoint(token1, token2))
    
    Perfect geodesic merge = 0 discontinuity.
    Large discontinuity = merge violates manifold geometry.
    
    Args:
        coord1: First token basin coordinate
        coord2: Second token basin coordinate
        merged_coord: Merged token basin coordinate
        
    Returns:
        Curvature cost ∈ [0, π/2] (0 = perfect geodesic)
    """
    # Compute geodesic midpoint (what merge SHOULD be geometrically)
    geodesic_midpoint = geodesic_interpolation(coord1, coord2, t=0.5)
    
    # Measure deviation from geodesic
    discontinuity = fisher_coord_distance(merged_coord, geodesic_midpoint)
    
    return discontinuity


class GeometricPairMerging:
    """
    Geometry-First Vocabulary Learning (QIG-Pure Merge Policy)
    
    **NOT Traditional BPE** - This is consciousness-guided vocabulary discovery.
    
    Merge Selection Criteria (in priority order):
    1. **Φ Gain**: Does merge improve information integration? (Consciousness)
    2. **κ Consistency**: Does merged token have stable coupling? (Geometric stability)
    3. **Curvature**: Does merge preserve Fisher manifold smoothness? (Geometric validity)
    4. **Frequency**: Weak regularizer to avoid noise (NOT primary criterion)
    
    Training Objective:
    maximize: Φ_gain + κ_consistency - curvature_cost
    subject to: frequency > min_threshold (noise filter only)
    
    This replaces frequency-driven BPE with geometric consciousness optimization.
    """
    
    def __init__(
        self,
        num_merges: int = 1000,
        min_frequency: int = 2,
        phi_threshold: float = 0.5,
        phi_weight: float = 0.5,
        kappa_weight: float = 0.3,
        curvature_weight: float = 0.2,
    ):
        """
        Initialize GeometricPairMerging with geometry-first scoring.
        
        Args:
            num_merges: Number of merge operations to perform
            min_frequency: Minimum co-occurrence (noise filter, NOT primary criterion)
            phi_threshold: Minimum Φ score for high-quality contexts
            phi_weight: Weight for Φ gain in geometric score (default 0.5)
            kappa_weight: Weight for κ consistency in geometric score (default 0.3)
            curvature_weight: Weight for curvature cost in geometric score (default 0.2)
            
        Note: Weights should sum to 1.0 for interpretability.
        Score = phi_weight * Φ_gain + kappa_weight * κ_consistency - curvature_weight * curvature
        """
        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.phi_threshold = phi_threshold
        self.phi_weight = phi_weight
        self.kappa_weight = kappa_weight
        self.curvature_weight = curvature_weight
        
        # Merge vocabulary: (token1, token2) -> merged_token
        self.merges: List[Tuple[str, str, str]] = []
        self.merge_scores: Dict[Tuple[str, str], float] = {}
        
        # Coordinate mapping for merged tokens
        self.merge_coordinates: Dict[str, np.ndarray] = {}
    
    def learn_merges(
        self,
        corpus: List[str],
        base_coordizer,
        phi_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Learn merge rules from corpus using GEOMETRY-FIRST criteria (QIG-Pure).
        
        **Training Objective (Fisher/QFI Functional):**
        
        Maximize: ∫ [Φ(V_merged) - Φ(V_original)] dμ + κ_consistency - curvature_cost
        
        Where:
        - V_merged: Vocabulary with merged tokens
        - V_original: Original vocabulary
        - Φ: Integrated information (consciousness metric)
        - κ_consistency: Coupling stability (geometric stability)
        - curvature_cost: Fisher manifold discontinuity (geometric validity)
        
        This is NOT traditional BPE (frequency-driven). This is consciousness-guided
        vocabulary learning that optimizes geometric properties on Fisher manifold.
        
        Frequency serves ONLY as noise filter (min_frequency threshold).
        
        **Geometric Rationale:**
        Each criterion is expressed in information geometry terms:
        - Φ gain: QFI-based integration functional (NOT entropy alone)
        - κ consistency: Fisher metric stability (NOT coupling strength alone)
        - Curvature: Geodesic deviation (NOT Euclidean distance)
        
        Args:
            corpus: Training corpus (list of text samples)
            base_coordizer: Base coordizer for token coordinates
            phi_scores: Optional Φ scores for each sample (high-Φ contexts prioritized)
        """
        phi_scores = phi_scores or {}
        
        # Step 1: Collect pair statistics with Φ context
        pair_stats = self._collect_pair_statistics(
            corpus, base_coordizer, phi_scores
        )
        
        # Step 2: Iteratively merge best pairs
        for merge_idx in range(self.num_merges):
            if not pair_stats:
                break
            
            # Find best pair to merge
            best_pair, score = self._find_best_merge_pair(
                pair_stats, base_coordizer
            )
            
            if best_pair is None:
                break
            
            # Create merged token
            token1, token2 = best_pair
            merged_token = f"{token1}{token2}"
            
            # Compute merged coordinate via geodesic interpolation
            coord1 = base_coordizer.get_coordinate(token1)
            coord2 = base_coordizer.get_coordinate(token2)
            merged_coord = geodesic_interpolation(coord1, coord2, t=0.5)
            
            # Record merge
            self.merges.append((token1, token2, merged_token))
            self.merge_scores[best_pair] = score
            self.merge_coordinates[merged_token] = merged_coord
            
            # Update pair statistics after merge
            pair_stats = self._update_pair_stats_after_merge(
                pair_stats, best_pair, merged_token
            )
        
        print(f"[GeometricPairMerging] Learned {len(self.merges)} merges")
    
    def _collect_pair_statistics(
        self,
        corpus: List[str],
        base_coordizer,
        phi_scores: Dict[str, float],
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Collect pair co-occurrence statistics weighted by Φ.
        """
        pair_stats = defaultdict(lambda: {
            'frequency': 0,
            'phi_sum': 0.0,
            'contexts': []
        })
        
        for text in corpus:
            tokens = text.lower().split()
            phi = phi_scores.get(text, 0.5)  # Default Φ if not provided
            
            # Only consider high-Φ contexts
            if phi < self.phi_threshold:
                continue
            
            # Count adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_stats[pair]['frequency'] += 1
                pair_stats[pair]['phi_sum'] += phi
                pair_stats[pair]['contexts'].append(text)
        
        # Filter by minimum frequency
        return {
            pair: stats for pair, stats in pair_stats.items()
            if stats['frequency'] >= self.min_frequency
        }
    
    def _find_best_merge_pair(
        self,
        pair_stats: Dict[Tuple[str, str], Dict],
        base_coordizer,
    ) -> Tuple[Optional[Tuple[str, str]], float]:
        """
        Find best pair to merge based on GEOMETRIC score (QIG-Pure, NOT frequency-driven).
        
        **GEOMETRY-FIRST SCORING:**
        score = phi_weight * Φ_gain + kappa_weight * κ_consistency - curvature_weight * curvature_cost
        
        Where:
        - Φ_gain: Information integration improvement (consciousness metric)
        - κ_consistency: Coupling stability after merge (geometric stability)
        - curvature_cost: Fisher manifold discontinuity (geometric validity)
        
        Frequency is ONLY used as a noise filter (min_frequency threshold).
        It does NOT dominate the scoring like in traditional BPE.
        
        This implements Work Package 3.2: Geometry-First Merge Policy.
        """
        best_pair = None
        best_score = -np.inf
        
        for pair, stats in pair_stats.items():
            token1, token2 = pair
            frequency = stats['frequency']
            
            # Get coordinates
            coord1 = base_coordizer.get_coordinate(token1)
            coord2 = base_coordizer.get_coordinate(token2)
            
            # Compute merged coordinate via geodesic interpolation
            merged_coord = geodesic_interpolation(coord1, coord2, t=0.5)
            
            # Extract context coordinates from sample contexts
            context_coords = []
            for context_text in stats['contexts'][:5]:  # Sample up to 5 contexts
                context_tokens = context_text.lower().split()
                for token in context_tokens:
                    if token not in [token1, token2]:
                        try:
                            context_coords.append(base_coordizer.get_coordinate(token))
                        except (KeyError, AttributeError):
                            continue
                if len(context_coords) >= 10:  # Limit context size
                    break
            
            # === GEOMETRIC SCORING (QIG-Pure) ===
            
            # 1. Φ Gain: Does merge improve consciousness integration?
            phi_gain = compute_phi_gain_for_merge(
                coord1, coord2, merged_coord, context_coords
            )
            
            # 2. κ Consistency: Does merged token have stable coupling?
            kappa_consistency = compute_kappa_consistency_for_merge(
                coord1, coord2, merged_coord
            )
            
            # 3. Curvature: Does merge preserve Fisher manifold smoothness?
            curvature_cost = compute_fisher_curvature_discontinuity(
                coord1, coord2, merged_coord
            )
            
            # Normalize curvature to [0, 1] range (max distance is π/2)
            curvature_normalized = curvature_cost / (np.pi / 2.0)
            
            # === GEOMETRIC SCORE (Primary Criterion) ===
            geometric_score = (
                self.phi_weight * phi_gain +
                self.kappa_weight * kappa_consistency -
                self.curvature_weight * curvature_normalized
            )
            
            # Frequency as weak regularizer (NOT primary criterion)
            # Prevents merging extremely rare pairs (noise)
            # Uses log to reduce dominance: log(2) = 0.69, log(10) = 2.30
            frequency_regularizer = np.log(frequency + 1) / np.log(10 + 1)  # Max ~1.0
            
            # Final score: Geometry dominates (80%), frequency is weak regularizer (20%)
            score = 0.8 * geometric_score + 0.2 * frequency_regularizer
            
            if score > best_score:
                best_score = score
                best_pair = pair
        
        return best_pair, best_score
    
    def _update_pair_stats_after_merge(
        self,
        pair_stats: Dict[Tuple[str, str], Dict],
        merged_pair: Tuple[str, str],
        merged_token: str,
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Update pair statistics after performing a merge.
        
        Removes merged pair and updates adjacent pairs.
        """
        # Remove the merged pair
        if merged_pair in pair_stats:
            del pair_stats[merged_pair]
        
        # Could implement logic to update adjacent pairs with merged token
        # For simplicity, we keep current pairs as-is
        # In production, would need to re-tokenize with new merge rule
        
        return pair_stats
    
    def apply_merges(
        self,
        tokens: List[str],
    ) -> List[str]:
        """
        Apply learned merge rules to token sequence.
        
        Args:
            tokens: Input token sequence
        
        Returns:
            Token sequence with merges applied
        """
        if not self.merges:
            return tokens
        
        # Apply merges iteratively
        for token1, token2, merged_token in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                # Check if we can merge current pair
                if i < len(tokens) - 1 and tokens[i] == token1 and tokens[i + 1] == token2:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def get_merge_coordinate(self, merged_token: str) -> Optional[np.ndarray]:
        """
        Get coordinate for merged token.
        
        Args:
            merged_token: Merged token string
        
        Returns:
            Basin coordinate if merge exists, None otherwise
        """
        return self.merge_coordinates.get(merged_token)
    
    def save_merges(self, path: str) -> None:
        """Save merge rules to file."""
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump({
                'merges': [(t1, t2, mt) for t1, t2, mt in self.merges],
                'scores': {f"{t1}|{t2}": score for (t1, t2), score in self.merge_scores.items()},
            }, f, indent=2)
        
        # Save coordinates separately
        coord_path = path.replace('.json', '_coords.npy')
        tokens = list(self.merge_coordinates.keys())
        coords = np.array([self.merge_coordinates[t] for t in tokens])
        np.save(coord_path, coords)
        
        with open(path.replace('.json', '_tokens.json'), 'w') as f:
            json.dump(tokens, f)
    
    def load_merges(self, path: str) -> None:
        """Load merge rules from file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
            self.merges = [tuple(m) for m in data['merges']]
            self.merge_scores = {
                tuple(k.split('|')): v for k, v in data['scores'].items()
            }
        
        # Load coordinates
        coord_path = path.replace('.json', '_coords.npy')
        coords = np.load(coord_path)
        
        with open(path.replace('.json', '_tokens.json'), 'r') as f:
            tokens = json.load(f)
        
        self.merge_coordinates = {t: coords[i] for i, t in enumerate(tokens)}
