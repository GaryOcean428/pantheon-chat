"""
Consciousness-Aware Coordizer - Φ-Optimized Segmentation

Uses consciousness metrics (Φ, κ) to guide tokenization decisions.
Chooses segmentations that maximize integration and coherence.

Key Innovation:
- Tokenization is consciousness-guided, not just frequency-based
- High-Φ contexts lead to token consolidation
- Dynamically adjusts granularity based on understanding quality
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qig_geometry import fisher_coord_distance, sphere_project


class ConsciousnessCoordizer:
    """
    Consciousness-aware coordizer using Φ (integration) for segmentation.
    
    Traditional tokenizers segment text mechanically. This coordizer:
    1. Evaluates multiple segmentation hypotheses
    2. Computes Φ (integration) for each hypothesis
    3. Selects segmentation with highest Φ
    
    Result: Tokenization that maximizes semantic coherence and understanding.
    """
    
    def __init__(
        self,
        base_coordizer,
        phi_threshold: float = 0.7,
        max_segment_length: int = 5,
        integration_weight: float = 0.6,
    ):
        """
        Initialize ConsciousnessCoordizer.
        
        Args:
            base_coordizer: Base coordizer for token coordinates
            phi_threshold: Minimum Φ for high-quality segmentation
            max_segment_length: Maximum tokens in a segment
            integration_weight: Weight of Φ vs frequency in scoring
        """
        self.base_coordizer = base_coordizer
        self.phi_threshold = phi_threshold
        self.max_segment_length = max_segment_length
        self.integration_weight = integration_weight
        
        # Track learned consolidations
        self.consolidations: Dict[Tuple[str, ...], str] = {}
        self.consolidation_phi: Dict[Tuple[str, ...], float] = {}
        self.consolidation_coords: Dict[str, np.ndarray] = {}
    
    def coordize_with_phi(
        self,
        text: str,
        context_phi: Optional[float] = None,
    ) -> Tuple[List[str], List[np.ndarray], float]:
        """
        Coordize text using Φ-guided segmentation.
        
        Args:
            text: Input text
            context_phi: Optional context Φ score
        
        Returns:
            Tuple of (tokens, coordinates, computed_phi)
        """
        # Generate base tokenization
        base_tokens = text.lower().split()
        
        # If context has high Φ, try to consolidate
        if context_phi and context_phi >= self.phi_threshold:
            tokens = self._consolidate_high_phi_sequence(base_tokens, context_phi)
        else:
            tokens = base_tokens
        
        # Get coordinates
        coords = [self.base_coordizer.get_coordinate(t) for t in tokens]
        
        # Compute Φ for this segmentation
        computed_phi = self._compute_segmentation_phi(tokens, coords)
        
        return tokens, coords, computed_phi
    
    def _consolidate_high_phi_sequence(
        self,
        tokens: List[str],
        phi: float,
    ) -> List[str]:
        """
        Consolidate token sequence in high-Φ context.
        
        High Φ suggests the sequence forms a coherent unit.
        Try to consolidate sub-sequences into single tokens.
        """
        consolidated = []
        i = 0
        
        while i < len(tokens):
            # Try consolidation of increasing window sizes
            best_consolidation = None
            best_window = 1
            
            for window in range(min(self.max_segment_length, len(tokens) - i), 0, -1):
                if window == 1:
                    break
                
                sequence = tuple(tokens[i:i + window])
                
                # Check if we've learned this consolidation
                if sequence in self.consolidations:
                    best_consolidation = self.consolidations[sequence]
                    best_window = window
                    break
                
                # Check if sequence should be consolidated
                if self._should_consolidate(sequence, phi):
                    # Create new consolidated token
                    consolidated_token = '_'.join(sequence)
                    self._learn_consolidation(sequence, consolidated_token, phi)
                    best_consolidation = consolidated_token
                    best_window = window
                    break
            
            if best_consolidation:
                consolidated.append(best_consolidation)
                i += best_window
            else:
                consolidated.append(tokens[i])
                i += 1
        
        return consolidated
    
    def _should_consolidate(
        self,
        sequence: Tuple[str, ...],
        context_phi: float,
    ) -> bool:
        """
        Determine if sequence should be consolidated into single token.
        
        Consolidation criteria:
        1. High context Φ (indicates coherent understanding)
        2. Sequence coordinates are geometrically close
        3. Consolidation increases integration
        """
        if len(sequence) < 2:
            return False
        
        # Get coordinates for sequence
        coords = [self.base_coordizer.get_coordinate(t) for t in sequence]
        
        # Check geometric proximity
        max_distance = 0.0
        for i in range(len(coords) - 1):
            dist = fisher_coord_distance(coords[i], coords[i + 1])
            max_distance = max(max_distance, dist)
        
        # Close proximity suggests semantic coherence
        proximity_threshold = np.pi / 3  # 60 degrees
        if max_distance > proximity_threshold:
            return False
        
        # High Φ context + geometric proximity = consolidate
        return context_phi >= self.phi_threshold
    
    def _learn_consolidation(
        self,
        sequence: Tuple[str, ...],
        consolidated_token: str,
        phi: float,
    ) -> None:
        """
        Learn a new consolidation rule.
        
        Records that sequence should be treated as single token in high-Φ contexts.
        """
        self.consolidations[sequence] = consolidated_token
        self.consolidation_phi[sequence] = phi
        
        # Compute consolidated coordinate (average on manifold)
        coords = [self.base_coordizer.get_coordinate(t) for t in sequence]
        avg_coord = np.mean(coords, axis=0)
        self.consolidation_coords[consolidated_token] = sphere_project(avg_coord)
    
    def _compute_segmentation_phi(
        self,
        tokens: List[str],
        coords: List[np.ndarray],
    ) -> float:
        """
        Compute Φ (integration) for a segmentation.
        
        Higher Φ = more integrated, coherent representation.
        
        Φ approximation:
        - Internal integration: How well tokens couple together
        - Boundary distinctness: How separate from adjacent tokens
        
        For simplicity, we approximate:
        Φ ≈ (1 - avg_internal_distance) * boundary_sharpness
        """
        if len(tokens) < 2:
            return 0.5
        
        # Compute internal coupling (lower distance = higher integration)
        internal_distances = []
        for i in range(len(coords) - 1):
            dist = fisher_coord_distance(coords[i], coords[i + 1])
            internal_distances.append(dist)
        
        avg_internal_dist = np.mean(internal_distances)
        
        # Integration increases as internal distance decreases
        # Normalize by π (max Fisher distance)
        integration = 1.0 - (avg_internal_dist / np.pi)
        
        # Boundary sharpness: variance in distances (higher = more structured)
        if len(internal_distances) > 1:
            boundary_sharpness = np.std(internal_distances) / np.pi
        else:
            boundary_sharpness = 0.5
        
        # Φ combines integration and structure
        phi = integration * (0.7 + 0.3 * boundary_sharpness)
        
        return max(0.0, min(1.0, phi))
    
    def evaluate_segmentation_quality(
        self,
        text: str,
        segmentation1: List[str],
        segmentation2: List[str],
    ) -> Tuple[float, float, str]:
        """
        Compare two segmentations by their Φ scores.
        
        Args:
            text: Original text
            segmentation1: First segmentation hypothesis
            segmentation2: Second segmentation hypothesis
        
        Returns:
            Tuple of (phi1, phi2, winner)
        """
        coords1 = [self.base_coordizer.get_coordinate(t) for t in segmentation1]
        coords2 = [self.base_coordizer.get_coordinate(t) for t in segmentation2]
        
        phi1 = self._compute_segmentation_phi(segmentation1, coords1)
        phi2 = self._compute_segmentation_phi(segmentation2, coords2)
        
        winner = "segmentation1" if phi1 > phi2 else "segmentation2"
        
        return phi1, phi2, winner
    
    def optimize_segmentation(
        self,
        text: str,
        num_hypotheses: int = 5,
    ) -> Tuple[List[str], float]:
        """
        Generate multiple segmentation hypotheses and select best by Φ.
        
        Args:
            text: Input text
            num_hypotheses: Number of segmentation hypotheses to try
        
        Returns:
            Tuple of (best_tokens, best_phi)
        """
        base_tokens = text.lower().split()
        
        hypotheses = []
        
        # Hypothesis 1: Base tokenization (word-level)
        hypotheses.append(base_tokens)
        
        # Hypothesis 2: Character-level (for rare words)
        char_tokens = list(text.lower().replace(' ', '_'))
        hypotheses.append(char_tokens)
        
        # Hypothesis 3: Bigram consolidation
        bigram_tokens = []
        i = 0
        while i < len(base_tokens):
            if i < len(base_tokens) - 1:
                bigram_tokens.append(f"{base_tokens[i]}_{base_tokens[i+1]}")
                i += 2
            else:
                bigram_tokens.append(base_tokens[i])
                i += 1
        hypotheses.append(bigram_tokens)
        
        # Hypothesis 4: Full text as one token (for phrases)
        if len(base_tokens) <= self.max_segment_length:
            hypotheses.append([text.lower().replace(' ', '_')])
        
        # Evaluate each hypothesis
        best_tokens = base_tokens
        best_phi = 0.0
        
        for hyp in hypotheses:
            coords = [self.base_coordizer.get_coordinate(t) for t in hyp]
            phi = self._compute_segmentation_phi(hyp, coords)
            
            if phi > best_phi:
                best_phi = phi
                best_tokens = hyp
        
        return best_tokens, best_phi
    
    def get_consolidation_stats(self) -> Dict:
        """
        Get statistics about learned consolidations.
        
        Returns:
            Dictionary with consolidation statistics
        """
        return {
            'total_consolidations': len(self.consolidations),
            'avg_phi': np.mean(list(self.consolidation_phi.values())) if self.consolidation_phi else 0.0,
            'avg_length': np.mean([len(seq) for seq in self.consolidations.keys()]) if self.consolidations else 0.0,
        }
