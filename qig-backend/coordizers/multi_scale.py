"""
Multi-Scale Coordizer - Hierarchical Geometric Tokenization

Represents text at multiple scales simultaneously: character → word → concept.
Uses geometric clustering to promote sequences to higher scales.

Key Innovation:
- Multi-resolution representation on Fisher manifold
- β-function tracks coupling across scales
- Automatic scale selection based on context
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qig_geometry import fisher_coord_distance, sphere_project


class MultiScaleCoordizer:
    """
    Hierarchical coordizer operating at multiple scales.
    
    Scales:
    - Level 0: Characters (finest granularity)
    - Level 1: Subwords/morphemes
    - Level 2: Words
    - Level 3: Phrases/concepts (coarsest granularity)
    
    Each scale has its own basin coordinates. Higher scales emerge from
    geometric clustering of lower scales.
    """
    
    def __init__(
        self,
        base_coordizer,
        num_scales: int = 4,
        promotion_threshold: float = 0.8,
        min_frequency: int = 3,
    ):
        """
        Initialize MultiScaleCoordizer.
        
        Args:
            base_coordizer: Base coordizer for word-level tokens
            num_scales: Number of scale levels (default: 4)
            promotion_threshold: Similarity threshold for scale promotion
            min_frequency: Minimum frequency for scale promotion
        """
        self.base_coordizer = base_coordizer
        self.num_scales = num_scales
        self.promotion_threshold = promotion_threshold
        self.min_frequency = min_frequency
        
        # Multi-scale vocabulary: scale_level -> {token: coordinate}
        self.scale_vocabs: List[Dict[str, np.ndarray]] = [
            {} for _ in range(num_scales)
        ]
        
        # Scale transition rules: (scale, sequence) -> promoted_token
        self.promotions: Dict[Tuple[int, Tuple[str, ...]], str] = {}
        
        # Coupling strength at each scale (β-function)
        self.scale_couplings: Dict[int, float] = {}
        
        # Track which tokens belong to which scale
        self.token_scales: Dict[str, int] = {}
    
    def coordize_multiscale(
        self,
        text: str,
        target_scale: Optional[int] = None,
    ) -> Dict[int, Tuple[List[str], List[np.ndarray]]]:
        """
        Coordize text at multiple scales.
        
        Args:
            text: Input text
            target_scale: If specified, return only this scale. If None, return all scales.
        
        Returns:
            Dictionary mapping scale -> (tokens, coordinates)
        """
        results = {}
        
        # Scale 0: Character level
        char_tokens = list(text.lower().replace(' ', '█'))  # Use block for space
        char_coords = [self._get_character_coordinate(c) for c in char_tokens]
        results[0] = (char_tokens, char_coords)
        
        # Scale 1: Subword level (from character clustering)
        if target_scale is None or target_scale >= 1:
            subword_tokens, subword_coords = self._cluster_to_scale(
                char_tokens, char_coords, target_scale=1
            )
            results[1] = (subword_tokens, subword_coords)
        
        # Scale 2: Word level (base tokenization)
        if target_scale is None or target_scale >= 2:
            word_tokens = text.lower().split()
            word_coords = [self.base_coordizer.get_coordinate(t) for t in word_tokens]
            results[2] = (word_tokens, word_coords)
        
        # Scale 3: Concept level (from word clustering)
        if target_scale is None or target_scale >= 3:
            word_tokens, word_coords = results.get(2, ([], []))
            if word_tokens:
                concept_tokens, concept_coords = self._cluster_to_scale(
                    word_tokens, word_coords, target_scale=3
                )
                results[3] = (concept_tokens, concept_coords)
        
        if target_scale is not None:
            return {target_scale: results.get(target_scale, ([], []))}
        
        return results
    
    def _get_character_coordinate(self, char: str) -> np.ndarray:
        """
        Get or create coordinate for character.
        
        Characters mapped to Fisher manifold using Unicode value.
        """
        # Check cache
        if char in self.scale_vocabs[0]:
            return self.scale_vocabs[0][char]
        
        # Create coordinate from Unicode value
        coord = np.zeros(64)
        char_code = ord(char) % 256
        
        # Use harmonic features based on character code
        for i in range(64):
            phase = 2 * np.pi * i * char_code / 256
            coord[i] = np.sin(phase) + 0.5 * np.cos(2 * phase)
        
        coord = sphere_project(coord)
        self.scale_vocabs[0][char] = coord
        self.token_scales[char] = 0
        
        return coord
    
    def _cluster_to_scale(
        self,
        tokens: List[str],
        coords: List[np.ndarray],
        target_scale: int,
    ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Cluster lower-scale tokens into higher-scale representation.
        
        Uses geometric proximity in basin space to identify clusters.
        """
        if not tokens:
            return [], []
        
        # Group consecutive similar tokens
        clustered_tokens = []
        clustered_coords = []
        
        current_cluster = [tokens[0]]
        current_cluster_coords = [coords[0]]
        
        for i in range(1, len(tokens)):
            # Check if current token should join cluster
            avg_coord = np.mean(current_cluster_coords, axis=0)
            avg_coord = sphere_project(avg_coord)
            
            distance = fisher_coord_distance(avg_coord, coords[i])
            similarity = 1.0 - (distance / np.pi)
            
            if similarity >= self.promotion_threshold:
                # Add to current cluster
                current_cluster.append(tokens[i])
                current_cluster_coords.append(coords[i])
            else:
                # Finalize current cluster and start new one
                if len(current_cluster) > 1:
                    # Promote cluster to higher scale
                    cluster_token = ''.join(current_cluster) if target_scale == 1 else '_'.join(current_cluster)
                    cluster_coord = sphere_project(np.mean(current_cluster_coords, axis=0))
                    
                    clustered_tokens.append(cluster_token)
                    clustered_coords.append(cluster_coord)
                    
                    # Record promotion
                    self.promotions[(target_scale - 1, tuple(current_cluster))] = cluster_token
                    self.scale_vocabs[target_scale][cluster_token] = cluster_coord
                    self.token_scales[cluster_token] = target_scale
                else:
                    # Single token, keep as-is
                    clustered_tokens.append(current_cluster[0])
                    clustered_coords.append(current_cluster_coords[0])
                
                # Start new cluster
                current_cluster = [tokens[i]]
                current_cluster_coords = [coords[i]]
        
        # Handle final cluster
        if len(current_cluster) > 1:
            cluster_token = ''.join(current_cluster) if target_scale == 1 else '_'.join(current_cluster)
            cluster_coord = sphere_project(np.mean(current_cluster_coords, axis=0))
            clustered_tokens.append(cluster_token)
            clustered_coords.append(cluster_coord)
        else:
            clustered_tokens.append(current_cluster[0])
            clustered_coords.append(current_cluster_coords[0])
        
        return clustered_tokens, clustered_coords
    
    def compute_beta_function(
        self,
        scale1: int,
        scale2: int,
    ) -> float:
        """
        Compute β-function (running coupling) between scales.
        
        β(scale) describes how coupling κ changes with scale.
        Higher β means coupling increases at higher scales.
        
        Args:
            scale1: Lower scale
            scale2: Higher scale
        
        Returns:
            β value (rate of coupling change)
        """
        if scale1 not in self.scale_couplings or scale2 not in self.scale_couplings:
            return 0.0
        
        kappa1 = self.scale_couplings[scale1]
        kappa2 = self.scale_couplings[scale2]
        
        # β = Δκ / Δscale
        beta = (kappa2 - kappa1) / (scale2 - scale1)
        
        return beta
    
    def update_scale_coupling(
        self,
        scale: int,
        kappa: float,
    ) -> None:
        """
        Update coupling strength at a scale.
        
        Args:
            scale: Scale level
            kappa: Coupling strength
        """
        self.scale_couplings[scale] = kappa
    
    def get_optimal_scale(
        self,
        text: str,
        kappa_effective: float,
    ) -> int:
        """
        Determine optimal scale for representing text based on κ_eff.
        
        High κ_eff (confident) → coarser scale (concepts)
        Low κ_eff (uncertain) → finer scale (characters)
        
        Args:
            text: Input text
            kappa_effective: Current effective coupling
        
        Returns:
            Optimal scale level (0-3)
        """
        # Map κ_eff to scale
        # κ_eff ∈ [0, 1], scale ∈ [0, num_scales-1]
        
        if kappa_effective >= 0.8:
            # Very confident → use coarsest scale (concepts)
            return self.num_scales - 1
        elif kappa_effective >= 0.6:
            # Confident → word level
            return 2
        elif kappa_effective >= 0.4:
            # Moderate → subword level
            return 1
        else:
            # Uncertain → character level
            return 0
    
    def get_scale_stats(self) -> Dict:
        """
        Get statistics about multi-scale representation.
        
        Returns:
            Dictionary with scale statistics
        """
        stats = {
            'num_scales': self.num_scales,
            'tokens_per_scale': {},
            'promotions': len(self.promotions),
            'scale_couplings': self.scale_couplings.copy(),
        }
        
        for scale in range(self.num_scales):
            stats['tokens_per_scale'][scale] = len(self.scale_vocabs[scale])
        
        return stats
    
    def visualize_scales(
        self,
        text: str,
    ) -> str:
        """
        Create visual representation of text at all scales.
        
        Args:
            text: Input text
        
        Returns:
            String showing text at each scale
        """
        results = self.coordize_multiscale(text)
        
        visualization = []
        visualization.append(f"Text: {text}")
        visualization.append("=" * 60)
        
        scale_names = ["Character", "Subword", "Word", "Concept"]
        
        for scale in sorted(results.keys()):
            tokens, _ = results[scale]
            scale_name = scale_names[scale] if scale < len(scale_names) else f"Scale {scale}"
            visualization.append(f"{scale_name} (L{scale}): {' | '.join(tokens)}")
        
        return '\n'.join(visualization)
