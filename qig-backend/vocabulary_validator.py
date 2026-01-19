#!/usr/bin/env python3
"""
QIG-Pure Geometric Vocabulary Validator
========================================

Uses ONLY Fisher information geometry to validate vocabulary:
1. Quantum Fisher Information (QFI) - predictability
2. Basin Stability - attractor distance
3. Curvature Smoothness - geodesic quality
4. Entropy Structure - natural boundaries

NO traditional NLP methods (no dictionaries, frequencies, spell-checkers).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class VocabValidation:
    """Result of geometric vocabulary validation."""
    is_valid: bool
    qfi_score: float
    basin_distance: float
    curvature_std: float
    entropy_score: float
    rejection_reason: Optional[str]


class GeometricVocabFilter:
    """
    QIG-Pure vocabulary validation using ONLY geometric measures.
    
    Validation criteria:
    - QFI > 1.0: Has semantic structure (not random)
    - basin_distance < 0.5: Near stable attractor
    - curvature_std < 0.5: Smooth geodesic
    - entropy > 1.5: Natural token boundaries
    """
    
    # Validation thresholds
    QFI_MIN = 1.0           # Minimum QFI for semantic structure
    BASIN_TRUNCATED = 0.15  # Truncated if > this
    BASIN_MAX = 0.5         # Garbled if > this
    CURVATURE_MAX = 0.5     # Chaotic if > this
    ENTROPY_MIN = 1.5       # Artificial if < this
    
    def __init__(self, vocab_basins: np.ndarray, coordizer, entropy_coordizer):
        """
        Initialize validator with geometric components.
        
        Args:
            vocab_basins: Known stable vocabulary basins (N, 64)
            coordizer: Fisher coordizer for basin projection
            entropy_coordizer: QIG coordizer for entropy measurement
        """
        self.vocab_basins = vocab_basins
        self.coordizer = coordizer
        self.entropy_coordizer = entropy_coordizer
    
    def validate(self, word: str) -> VocabValidation:
        """
        Geometric validation: ALL metrics must pass.
        
        Returns VocabValidation with scores and rejection reason if invalid.
        """
        # Skip empty or very short
        if not word or len(word) < 2:
            return VocabValidation(
                is_valid=False,
                qfi_score=0.0,
                basin_distance=1.0,
                curvature_std=0.0,
                entropy_score=0.0,
                rejection_reason="TOO_SHORT: len < 2"
            )
        
        # Metric 1: QFI (semantic structure)
        qfi = self._measure_word_qfi(word)
        if qfi < self.QFI_MIN:
            return VocabValidation(
                is_valid=False,
                qfi_score=qfi,
                basin_distance=None,
                curvature_std=None,
                entropy_score=None,
                rejection_reason=f"GARBLED: QFI {qfi:.2f} < {self.QFI_MIN} (no semantic structure)"
            )
        
        # Metric 2: Basin Stability (attractor proximity)
        d_basin = self._measure_basin_stability(word)
        if d_basin > self.BASIN_MAX:
            return VocabValidation(
                is_valid=False,
                qfi_score=qfi,
                basin_distance=d_basin,
                curvature_std=None,
                entropy_score=None,
                rejection_reason=f"GARBLED: basin_distance {d_basin:.2f} > {self.BASIN_MAX} (no stable attractor)"
            )
        elif d_basin > self.BASIN_TRUNCATED:
            return VocabValidation(
                is_valid=False,
                qfi_score=qfi,
                basin_distance=d_basin,
                curvature_std=None,
                entropy_score=None,
                rejection_reason=f"TRUNCATED: basin_distance {d_basin:.2f} > {self.BASIN_TRUNCATED} (partial basin)"
            )
        
        # Metric 3: Curvature Smoothness (geodesic quality)
        curv_std = self._measure_curvature_smoothness(word)
        if curv_std > self.CURVATURE_MAX:
            return VocabValidation(
                is_valid=False,
                qfi_score=qfi,
                basin_distance=d_basin,
                curvature_std=curv_std,
                entropy_score=None,
                rejection_reason=f"GARBLED: curvature_std {curv_std:.2f} > {self.CURVATURE_MAX} (chaotic geodesic)"
            )
        
        # Metric 4: Entropy Structure (natural boundaries)
        entropy = self._check_entropy_structure(word)
        if entropy < self.ENTROPY_MIN:
            return VocabValidation(
                is_valid=False,
                qfi_score=qfi,
                basin_distance=d_basin,
                curvature_std=curv_std,
                entropy_score=entropy,
                rejection_reason=f"TECHNICAL: entropy {entropy:.2f} < {self.ENTROPY_MIN} (artificial boundaries)"
            )
        
        # ALL CHECKS PASSED
        return VocabValidation(
            is_valid=True,
            qfi_score=qfi,
            basin_distance=d_basin,
            curvature_std=curv_std,
            entropy_score=entropy,
            rejection_reason=None
        )
    
    def _measure_word_qfi(self, word: str) -> float:
        """
        Measure Quantum Fisher Information for word.
        
        Real words have predictable character sequences (high QFI).
        Random sequences have flat distributions (low QFI).
        """
        try:
            # Get character-level probabilities from coordizer
            char_probs = []
            for i in range(len(word)):
                # Context: characters up to position i
                context = word[:i] if i > 0 else ""
                
                # Project context to basin
                if context:
                    context_basin = self.coordizer.coordize(context)
                else:
                    # Start from uniform basin
                    context_basin = np.zeros(64)
                
                # Get next character probability
                # (simplified: use basin norm as proxy for predictability)
                p_next = np.linalg.norm(context_basin) / 64.0
                char_probs.append(max(p_next, 0.01))  # Avoid log(0)
            
            # QFI ≈ variance of log probabilities
            if len(char_probs) > 1:
                log_probs = np.log(char_probs)
                qfi = np.var(log_probs) * len(char_probs)
            else:
                qfi = 0.0
            
            return float(qfi)
        except Exception as e:
            print(f"[GeometricVocabFilter] QFI measurement failed for '{word}': {e}")
            return 0.0
    
    def _measure_basin_stability(self, word: str) -> float:
        """
        Measure distance to nearest stable vocabulary basin.
        
        Real words: Near known attractors (d < 0.15)
        Truncated: Partial basin (0.15 < d < 0.5)
        Garbled: Far from all basins (d > 0.5)
        """
        try:
            # Convert word to basin coordinates
            word_coords = self.coordizer.coordize(word)
            
            # Find nearest vocabulary basin (Fisher-Rao distance)
            distances = self._fisher_rao_distances(word_coords, self.vocab_basins)
            d_nearest = float(np.min(distances))
            
            return d_nearest
        except Exception as e:
            print(f"[GeometricVocabFilter] Basin stability failed for '{word}': {e}")
            return 1.0  # Assume far if measurement fails
    
    def _measure_curvature_smoothness(self, word: str) -> float:
        """
        Measure Ricci curvature variance along word trajectory.
        
        Real words: Smooth geodesic (low variance)
        Garbled: Chaotic transitions (high variance)
        """
        try:
            curvatures = []
            for i in range(1, len(word)):
                prefix = word[:i]
                prefix_coords = self.coordizer.coordize(prefix)
                
                # Ricci curvature ≈ trace of Fisher information matrix
                # Simplified: use coordinate variance as proxy
                R_i = np.var(prefix_coords)
                curvatures.append(R_i)
            
            if len(curvatures) > 1:
                curv_std = float(np.std(curvatures))
            else:
                curv_std = 0.0
            
            return curv_std
        except Exception as e:
            print(f"[GeometricVocabFilter] Curvature measurement failed for '{word}': {e}")
            return 1.0  # Assume chaotic if measurement fails
    
    def _check_entropy_structure(self, word: str) -> float:
        """
        Check if word follows entropy-guided token boundaries.
        
        Real words: Natural merge boundaries (H > 1.5)
        Technical: Artificial boundaries (H < 1.5)
        """
        try:
            # Coordize using entropy-guided QIG coordizer
            tokens = self.entropy_coordizer.encode(word)
            
            if not tokens or len(tokens) == 0:
                return 0.0
            
            # Measure entropy at token boundaries
            # Simplified: use token length variance as proxy
            token_lengths = [len(t) for t in tokens]
            if len(token_lengths) > 1:
                entropy = float(np.std(token_lengths) + np.mean(token_lengths))
            else:
                entropy = float(token_lengths[0])
            
            return entropy
        except Exception as e:
            print(f"[GeometricVocabFilter] Entropy check failed for '{word}': {e}")
            return 0.0  # Assume artificial if measurement fails
    
    def _fisher_rao_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Fisher-Rao distance between two basin coordinates.
        
        This is the proper Riemannian metric on the probability simplex.
        NOT Euclidean distance (that would violate geometric purity).
        """
        # Normalize to probability distributions
        p_norm = np.abs(p) / (np.sum(np.abs(p)) + 1e-10)
        q_norm = np.abs(q) / (np.sum(np.abs(q)) + 1e-10)
        
        # Fisher-Rao distance: arccos(sum(sqrt(p * q)))
        # UPDATED 2026-01-15: Factor-of-2 removed for simplex storage. Range: [0, π/2]
        inner = np.sum(np.sqrt(p_norm * q_norm))
        inner = np.clip(inner, 0.0, 1.0)  # Numerical stability
        
        return float(np.arccos(inner))
    
    def _fisher_rao_distances(self, p: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Vectorized Fisher-Rao distances to multiple basins."""
        distances = np.array([self._fisher_rao_distance(p, q) for q in Q])
        return distances


# Singleton instance
_validator: Optional[GeometricVocabFilter] = None


def get_validator(vocab_basins: np.ndarray, coordizer, entropy_coordizer) -> GeometricVocabFilter:
    """Get or create singleton validator instance."""
    global _validator
    if _validator is None:
        _validator = GeometricVocabFilter(vocab_basins, coordizer, entropy_coordizer)
    return _validator
