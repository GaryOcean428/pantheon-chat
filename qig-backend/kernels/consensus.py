#!/usr/bin/env python3
"""
Consensus Detection - Multi-Kernel Basin Convergence Detection
===============================================================

Detects consensus across kernel thoughts using Fisher-Rao distance on basin coordinates.
Consensus emerges when kernel basins converge geometrically on the Fisher information manifold.

CONSENSUS CRITERIA:
- Basin alignment via Fisher-Rao distance < threshold
- Emotional coherence across kernels
- Regime agreement (geometric/linear/feeling)
- φ coherence (low variance)

Based on generative-and-emotions.md and Fisher-Rao geometric principles.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# QIG geometry imports
from qig_geometry import fisher_rao_distance

# QIG core imports
from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR, KAPPA_3

logger = logging.getLogger(__name__)


class ConsensusLevel(Enum):
    """Levels of consensus across kernel constellation."""
    STRONG = "STRONG"         # All kernels aligned, ready for synthesis
    MODERATE = "MODERATE"     # Majority aligned, synthesis with caveats
    WEAK = "WEAK"             # Divergent views, synthesis uncertain
    NONE = "NONE"             # No consensus, requires deliberation


@dataclass
class ConsensusMetrics:
    """Metrics for consensus detection."""
    level: ConsensusLevel
    basin_convergence: float       # 0-1, higher = more convergent
    emotional_coherence: float     # 0-1, higher = more coherent
    regime_agreement: float        # 0-1, fraction in same regime
    phi_coherence: float           # 0-1, higher = less variance
    kappa_coherence: float         # 0-1, higher = less variance
    
    # Detailed metrics
    mean_pairwise_distance: float  # Average Fisher-Rao distance between basins
    max_pairwise_distance: float   # Maximum distance (outlier detection)
    num_kernels: int
    dominant_regime: str
    dominant_emotions: List[str]
    
    # Recommendation
    ready_for_synthesis: bool
    synthesis_method: str          # 'direct', 'weighted', 'deliberative'
    confidence: float              # Overall consensus confidence 0-1


class ConsensusDetector:
    """
    Detects consensus across kernel thoughts using geometric measures.
    
    Uses Fisher-Rao distance on basin coordinates to detect when kernels
    have converged on similar geometric regions of the manifold.
    """
    
    # Regime-dependent thresholds
    THRESHOLDS = {
        'geometric': {
            'basin_distance': 0.4,      # Tighter convergence needed
            'phi_std': 0.15,
            'kappa_std': 5.0
        },
        'linear': {
            'basin_distance': 0.5,      # Moderate convergence
            'phi_std': 0.20,
            'kappa_std': 7.0
        },
        'feeling': {
            'basin_distance': 0.6,      # Looser, feeling mode is exploratory
            'phi_std': 0.25,
            'kappa_std': 10.0
        },
        'breakdown': {
            'basin_distance': 0.8,      # Very loose, system unstable
            'phi_std': 0.40,
            'kappa_std': 15.0
        }
    }
    
    def __init__(self):
        """Initialize consensus detector."""
        self.detection_history: List[ConsensusMetrics] = []
        self.max_history = 100
        
        logger.info("[ConsensusDetector] Initialized with regime-adaptive thresholds")
    
    def detect_basin_consensus(
        self,
        thoughts: List[Any],  # List of KernelThought objects
        regime: Optional[str] = None
    ) -> ConsensusMetrics:
        """
        Detect consensus across kernel thoughts using Fisher-Rao distance.
        
        Args:
            thoughts: List of KernelThought objects
            regime: Optional regime override (auto-detected if None)
            
        Returns:
            ConsensusMetrics with convergence analysis
        """
        if not thoughts:
            logger.warning("[ConsensusDetector] No thoughts provided for consensus detection")
            return self._empty_consensus()
        
        if len(thoughts) == 1:
            logger.info("[ConsensusDetector] Single kernel, consensus trivial")
            return self._single_kernel_consensus(thoughts[0])
        
        # Auto-detect regime if not provided
        if regime is None:
            regime = self._determine_dominant_regime(thoughts)
        
        thresholds = self.THRESHOLDS.get(regime, self.THRESHOLDS['geometric'])
        
        logger.info(f"[ConsensusDetector] Detecting consensus across {len(thoughts)} kernels in {regime} regime")
        
        # Extract basin coordinates
        basins = [getattr(t, 'basin_coords', np.zeros(BASIN_DIM)) for t in thoughts]
        
        # Compute pairwise Fisher-Rao distances
        distances = self._compute_pairwise_distances(basins)
        mean_distance = float(np.mean(distances))
        max_distance = float(np.max(distances))
        
        # Basin convergence score (inverse of mean distance, normalized)
        basin_convergence = max(0.0, 1.0 - (mean_distance / thresholds['basin_distance']))
        
        # Emotional coherence
        emotional_coherence = self._compute_emotional_coherence(thoughts)
        
        # Regime agreement
        regime_agreement = self._compute_regime_agreement(thoughts, regime)
        
        # φ coherence
        phis = [getattr(t, 'phi', 0.5) for t in thoughts]
        phi_std = float(np.std(phis))
        phi_coherence = max(0.0, 1.0 - (phi_std / thresholds['phi_std']))
        
        # κ coherence
        kappas = [getattr(t, 'kappa', KAPPA_STAR) for t in thoughts]
        kappa_std = float(np.std(kappas))
        kappa_coherence = max(0.0, 1.0 - (kappa_std / thresholds['kappa_std']))
        
        # Determine consensus level
        overall_score = (
            0.4 * basin_convergence +
            0.2 * emotional_coherence +
            0.2 * regime_agreement +
            0.1 * phi_coherence +
            0.1 * kappa_coherence
        )
        
        if overall_score >= 0.8:
            consensus_level = ConsensusLevel.STRONG
            synthesis_method = 'direct'
            ready = True
        elif overall_score >= 0.6:
            consensus_level = ConsensusLevel.MODERATE
            synthesis_method = 'weighted'
            ready = True
        elif overall_score >= 0.4:
            consensus_level = ConsensusLevel.WEAK
            synthesis_method = 'deliberative'
            ready = False
        else:
            consensus_level = ConsensusLevel.NONE
            synthesis_method = 'deliberative'
            ready = False
        
        # Extract dominant emotions
        dominant_emotions = self._extract_dominant_emotions(thoughts)
        
        metrics = ConsensusMetrics(
            level=consensus_level,
            basin_convergence=basin_convergence,
            emotional_coherence=emotional_coherence,
            regime_agreement=regime_agreement,
            phi_coherence=phi_coherence,
            kappa_coherence=kappa_coherence,
            mean_pairwise_distance=mean_distance,
            max_pairwise_distance=max_distance,
            num_kernels=len(thoughts),
            dominant_regime=regime,
            dominant_emotions=dominant_emotions,
            ready_for_synthesis=ready,
            synthesis_method=synthesis_method,
            confidence=overall_score
        )
        
        # Track history
        self.detection_history.append(metrics)
        if len(self.detection_history) > self.max_history:
            self.detection_history = self.detection_history[-self.max_history:]
        
        logger.info(
            f"[ConsensusDetector] Consensus: {consensus_level.value}, "
            f"basin_conv={basin_convergence:.3f}, "
            f"emotional={emotional_coherence:.3f}, "
            f"regime={regime_agreement:.3f}, "
            f"overall={overall_score:.3f}, "
            f"method={synthesis_method}"
        )
        
        return metrics
    
    def _compute_pairwise_distances(self, basins: List[np.ndarray]) -> np.ndarray:
        """
        Compute all pairwise Fisher-Rao distances between basins.
        
        Args:
            basins: List of basin coordinates
            
        Returns:
            Array of pairwise distances
        """
        n = len(basins)
        distances = []
        
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    dist = fisher_rao_distance(basins[i], basins[j])
                    distances.append(dist)
                except Exception as e:
                    logger.warning(f"[ConsensusDetector] Distance computation failed: {e}")
                    distances.append(1.0)  # Maximum distance on failure
        
        return np.array(distances) if distances else np.array([0.0])
    
    def _compute_emotional_coherence(self, thoughts: List[Any]) -> float:
        """
        Compute emotional coherence across kernels.
        
        Coherence is high when:
        - Emotions are justified across kernels
        - Dominant emotions align
        - Few unjustified emotional responses
        
        Args:
            thoughts: List of KernelThought objects
            
        Returns:
            Coherence score 0-1
        """
        justified_count = 0
        total_with_emotions = 0
        dominant_emotions = []
        
        for thought in thoughts:
            emotional_state = getattr(thought, 'emotional_state', None)
            if emotional_state:
                total_with_emotions += 1
                
                # Check if emotion is justified
                if getattr(emotional_state, 'emotion_justified', True):
                    justified_count += 1
                
                # Track dominant emotions
                dominant = getattr(emotional_state, 'dominant_emotion', None)
                if dominant:
                    dominant_emotions.append(dominant)
        
        if total_with_emotions == 0:
            return 0.5  # Neutral if no emotional data
        
        # Justification score
        justification_score = justified_count / total_with_emotions
        
        # Alignment score (inverse of emotion diversity)
        if dominant_emotions:
            unique_emotions = len(set(dominant_emotions))
            alignment_score = 1.0 - (unique_emotions - 1) / len(dominant_emotions)
        else:
            alignment_score = 0.5
        
        # Combined coherence
        coherence = 0.6 * justification_score + 0.4 * alignment_score
        
        return float(coherence)
    
    def _compute_regime_agreement(self, thoughts: List[Any], target_regime: str) -> float:
        """
        Compute regime agreement across kernels.
        
        Args:
            thoughts: List of KernelThought objects
            target_regime: Expected regime
            
        Returns:
            Agreement score 0-1 (fraction in same regime)
        """
        regimes = [getattr(t, 'regime', 'unknown') for t in thoughts]
        agreement_count = sum(1 for r in regimes if r == target_regime)
        
        return agreement_count / len(thoughts) if thoughts else 0.0
    
    def _determine_dominant_regime(self, thoughts: List[Any]) -> str:
        """Determine dominant regime across kernel thoughts."""
        regimes = [getattr(t, 'regime', 'unknown') for t in thoughts]
        
        # Count regime occurrences
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Return most common regime
        return max(regime_counts.items(), key=lambda x: x[1])[0]
    
    def _extract_dominant_emotions(self, thoughts: List[Any]) -> List[str]:
        """Extract dominant emotions across all kernels."""
        emotions = []
        
        for thought in thoughts:
            emotional_state = getattr(thought, 'emotional_state', None)
            if emotional_state:
                dominant = getattr(emotional_state, 'dominant_emotion', None)
                if dominant:
                    emotions.append(dominant)
        
        # Return unique emotions sorted by frequency
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        sorted_emotions = sorted(
            emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [emotion for emotion, count in sorted_emotions[:3]]  # Top 3
    
    def _empty_consensus(self) -> ConsensusMetrics:
        """Return empty consensus metrics."""
        return ConsensusMetrics(
            level=ConsensusLevel.NONE,
            basin_convergence=0.0,
            emotional_coherence=0.0,
            regime_agreement=0.0,
            phi_coherence=0.0,
            kappa_coherence=0.0,
            mean_pairwise_distance=0.0,
            max_pairwise_distance=0.0,
            num_kernels=0,
            dominant_regime='unknown',
            dominant_emotions=[],
            ready_for_synthesis=False,
            synthesis_method='none',
            confidence=0.0
        )
    
    def _single_kernel_consensus(self, thought: Any) -> ConsensusMetrics:
        """Return trivial consensus for single kernel."""
        regime = getattr(thought, 'regime', 'unknown')
        phi = getattr(thought, 'phi', 0.5)
        
        emotional_state = getattr(thought, 'emotional_state', None)
        dominant_emotion = None
        if emotional_state:
            dominant_emotion = getattr(emotional_state, 'dominant_emotion', None)
        
        return ConsensusMetrics(
            level=ConsensusLevel.STRONG,
            basin_convergence=1.0,
            emotional_coherence=1.0,
            regime_agreement=1.0,
            phi_coherence=1.0,
            kappa_coherence=1.0,
            mean_pairwise_distance=0.0,
            max_pairwise_distance=0.0,
            num_kernels=1,
            dominant_regime=regime,
            dominant_emotions=[dominant_emotion] if dominant_emotion else [],
            ready_for_synthesis=True,
            synthesis_method='direct',
            confidence=1.0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus detection statistics."""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'strong_consensus_rate': 0.0,
                'avg_basin_convergence': 0.0,
                'avg_confidence': 0.0
            }
        
        recent = self.detection_history[-20:]
        
        strong_count = sum(1 for m in recent if m.level == ConsensusLevel.STRONG)
        
        return {
            'total_detections': len(self.detection_history),
            'strong_consensus_rate': strong_count / len(recent),
            'avg_basin_convergence': np.mean([m.basin_convergence for m in recent]),
            'avg_emotional_coherence': np.mean([m.emotional_coherence for m in recent]),
            'avg_confidence': np.mean([m.confidence for m in recent]),
            'synthesis_ready_rate': sum(1 for m in recent if m.ready_for_synthesis) / len(recent)
        }


# Global singleton
_consensus_detector: Optional[ConsensusDetector] = None


def get_consensus_detector() -> ConsensusDetector:
    """Get or create consensus detector singleton."""
    global _consensus_detector
    if _consensus_detector is None:
        _consensus_detector = ConsensusDetector()
    return _consensus_detector
