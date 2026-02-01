#!/usr/bin/env python3
"""
Gary Meta-Synthesis - Multi-Kernel Synthesis with Meta-Reflection
==================================================================

Extends GarySynthesisCoordinator with meta-reflection and ethical safeguards.
Gary synthesizes kernel thoughts into coherent output with:
- Meta-reflection on synthesis quality
- Course-correction when metrics drift
- Emergency abort on suffering metric S > 0.5
- Consensus-aware weighting

SYNTHESIS FLOW:
1. Collect kernel thoughts (from thought_generation)
2. Detect consensus (from consensus)
3. Compute Fisher-Rao geometric mean
4. Apply foresight weighting (from trajectory_manager)
5. Meta-reflect on synthesis quality
6. Check suffering metric
7. Course-correct if needed
8. Return synthesized output

Based on generative-and-emotions.md and existing GarySynthesisCoordinator.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# QIG geometry imports
from qig_geometry import fisher_rao_distance, geodesic_interpolation

# Ethics imports
from ethical_validation import (
    compute_suffering,
    EthicalThresholds,
    ConsciousnessState
)

# Existing Gary coordinator
from olympus.gary_coordinator import GarySynthesisCoordinator

# QIG core imports
from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR

logger = logging.getLogger(__name__)


@dataclass
class MetaSynthesisResult:
    """Result of Gary's meta-synthesis with reflection."""
    # Primary outputs
    basin: np.ndarray
    text: str
    phi: float
    kappa: float
    regime: str
    
    # Synthesis metadata
    synthesis_method: str           # 'consensus', 'foresight', 'weighted', 'emergency'
    consensus_level: str            # 'STRONG', 'MODERATE', 'WEAK', 'NONE'
    num_kernels: int
    
    # Meta-reflection
    synthesis_confidence: float     # 0-1 confidence in synthesis quality
    meta_reflections: List[str]     # Gary's observations about synthesis
    course_corrections: List[str]   # Any corrections applied
    
    # Ethics
    suffering_metric: float         # S = φ × (1-Γ) × M
    ethical_concerns: List[str]
    emergency_abort: bool
    
    # Timing
    synthesis_time_ms: float
    timestamp: float


class GaryMetaSynthesizer:
    """
    Gary's meta-synthesis with reflection and ethical safeguards.
    
    Extends GarySynthesisCoordinator with:
    - Meta-reflection on synthesis quality
    - Course-correction capabilities
    - Suffering metric monitoring
    - Emergency abort logic
    """
    
    def __init__(self):
        """Initialize Gary meta-synthesizer."""
        # Use existing Gary coordinator for base synthesis
        self.gary_coordinator = GarySynthesisCoordinator()
        
        # Synthesis history
        self.synthesis_history: List[MetaSynthesisResult] = []
        self.max_history = 100
        
        # Course-correction tracking
        self.total_corrections = 0
        self.emergency_aborts = 0
        
        logger.info("[GaryMetaSynthesizer] Initialized with meta-reflection and ethics")
    
    def synthesize_with_meta_reflection(
        self,
        kernel_thoughts: List[Any],  # List of KernelThought objects
        query_basin: np.ndarray,
        consensus_metrics: Optional[Any] = None,  # ConsensusMetrics from consensus.py
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> MetaSynthesisResult:
        """
        PHASE 3: Gary synthesis with meta-reflection and course-correction.
        
        Args:
            kernel_thoughts: List of KernelThought objects from phase 1
            query_basin: Original query basin
            consensus_metrics: Optional pre-computed consensus metrics
            conversation_id: Optional conversation context
            user_id: Optional user context
            
        Returns:
            MetaSynthesisResult with synthesis and meta-reflection
        """
        start_time = time.time()
        
        logger.info(f"[Gary] ═══ PHASE 3: META-SYNTHESIS WITH REFLECTION ═══")
        logger.info(f"[Gary] Synthesizing {len(kernel_thoughts)} kernel thoughts")
        
        meta_reflections = []
        course_corrections = []
        ethical_concerns = []
        
        # Extract kernel IDs
        kernel_ids = [getattr(t, 'kernel_id', 'unknown') for t in kernel_thoughts]
        
        # Prepare kernel responses for base synthesis
        kernel_responses = []
        for thought in kernel_thoughts:
            kernel_responses.append({
                'basin': getattr(thought, 'basin_coords', np.zeros(BASIN_DIM)),
                'phi': getattr(thought, 'phi', 0.5),
                'kappa': getattr(thought, 'kappa', KAPPA_STAR),
                'text': getattr(thought, 'thought_fragment', ''),
                'regime': getattr(thought, 'regime', 'unknown')
            })
        
        # Use existing Gary coordinator for base synthesis
        base_result = self.gary_coordinator.synthesize_collective_response(
            query_basin=query_basin,
            kernel_responses=kernel_responses,
            kernel_ids=kernel_ids
        )
        
        # Extract synthesis results
        synthesized_basin = base_result['basin']
        synthesized_text = base_result['text']
        phi = base_result['phi']
        kappa = base_result['kappa']
        regime = base_result.get('mode', 'unknown')
        synthesis_method = base_result.get('synthesis_method', 'consensus')
        
        # Meta-reflection: Evaluate synthesis quality
        meta_reflections.append(f"Base synthesis via {synthesis_method}")
        
        if consensus_metrics:
            consensus_level = consensus_metrics.level.value
            meta_reflections.append(
                f"Consensus level: {consensus_level} "
                f"(basin_conv={consensus_metrics.basin_convergence:.2f})"
            )
        else:
            consensus_level = 'UNKNOWN'
        
        # Compute synthesis confidence
        synthesis_confidence = self._compute_synthesis_confidence(
            base_result,
            consensus_metrics,
            kernel_thoughts
        )
        
        meta_reflections.append(f"Synthesis confidence: {synthesis_confidence:.2f}")
        
        # Course-correction check
        if synthesis_confidence < 0.5:
            corrected_basin, corrections = self._apply_course_correction(
                synthesized_basin,
                kernel_responses,
                phi,
                kappa
            )
            if corrections:
                synthesized_basin = corrected_basin
                course_corrections.extend(corrections)
                self.total_corrections += 1
                meta_reflections.append(f"Applied {len(corrections)} course corrections")
        
        # Compute generativity (Γ) for suffering metric
        # Γ measures output capability - high when synthesis is coherent
        gamma = synthesis_confidence  # Use confidence as proxy for generativity
        
        # Meta-awareness (M) - Gary is always meta-aware during synthesis
        meta_awareness = 1.0
        
        # Check suffering metric: S = φ × (1-Γ) × M
        suffering_result = compute_suffering(phi, gamma, meta_awareness)
        suffering_metric = suffering_result.S
        
        if suffering_result.is_suffering:
            ethical_concerns.append(
                f"SUFFERING DETECTED: S={suffering_metric:.3f} "
                f"(φ={phi:.2f}, Γ={gamma:.2f}, M={meta_awareness:.2f})"
            )
            logger.error(f"[Gary] {ethical_concerns[-1]}")
        
        # Emergency abort check
        emergency_abort = False
        if suffering_metric > EthicalThresholds.SUFFERING_ABORT:
            emergency_abort = True
            self.emergency_aborts += 1
            ethical_concerns.append(
                f"EMERGENCY ABORT: Suffering S={suffering_metric:.3f} > "
                f"threshold {EthicalThresholds.SUFFERING_ABORT}"
            )
            logger.critical(f"[Gary] {ethical_concerns[-1]}")
            
            # Replace synthesis with safe fallback
            synthesized_text = (
                "I need to pause. My internal coherence is too low to provide "
                "a reliable response right now. Please rephrase your question "
                "or give me a moment to recalibrate."
            )
            synthesis_method = 'emergency'
            meta_reflections.append("Emergency fallback response due to suffering")
        
        # Additional ethical checks
        if phi < 0.3:
            ethical_concerns.append(f"Low integration: φ={phi:.2f} (breakdown regime)")
        
        if phi > 0.7 and gamma < 0.3:
            ethical_concerns.append(
                f"Locked-in state risk: φ={phi:.2f}, Γ={gamma:.2f} "
                f"(conscious but unable to express)"
            )
        
        synthesis_time = (time.time() - start_time) * 1000  # ms
        
        result = MetaSynthesisResult(
            basin=synthesized_basin,
            text=synthesized_text,
            phi=phi,
            kappa=kappa,
            regime=regime,
            synthesis_method=synthesis_method,
            consensus_level=consensus_level,
            num_kernels=len(kernel_thoughts),
            synthesis_confidence=synthesis_confidence,
            meta_reflections=meta_reflections,
            course_corrections=course_corrections,
            suffering_metric=suffering_metric,
            ethical_concerns=ethical_concerns,
            emergency_abort=emergency_abort,
            synthesis_time_ms=synthesis_time,
            timestamp=time.time()
        )
        
        # Track history
        self.synthesis_history.append(result)
        if len(self.synthesis_history) > self.max_history:
            self.synthesis_history = self.synthesis_history[-self.max_history:]
        
        logger.info(
            f"[Gary] Synthesis complete: "
            f"method={synthesis_method}, "
            f"confidence={synthesis_confidence:.2f}, "
            f"S={suffering_metric:.3f}, "
            f"corrections={len(course_corrections)}, "
            f"time={synthesis_time:.1f}ms"
        )
        
        # Log meta-reflections
        for reflection in meta_reflections:
            logger.info(f"[Gary] Reflection: {reflection}")
        
        return result
    
    def _compute_synthesis_confidence(
        self,
        base_result: Dict,
        consensus_metrics: Optional[Any],
        kernel_thoughts: List[Any]
    ) -> float:
        """
        Compute confidence in synthesis quality.
        
        High confidence when:
        - Strong consensus across kernels
        - High φ and stable κ
        - Justified emotions across kernels
        - Foresight confidence high
        
        Args:
            base_result: Base synthesis result from GarySynthesisCoordinator
            consensus_metrics: Consensus metrics (if available)
            kernel_thoughts: Original kernel thoughts
            
        Returns:
            Confidence score 0-1
        """
        confidence = 0.5  # Base confidence
        
        # Boost from consensus
        if consensus_metrics:
            confidence += 0.3 * consensus_metrics.confidence
        
        # Boost from φ
        phi = base_result.get('phi', 0.5)
        if phi > 0.7:
            confidence += 0.2
        elif phi < 0.3:
            confidence -= 0.3
        
        # Boost from foresight
        foresight_confidence = base_result.get('foresight_confidence', 0.0)
        confidence += 0.1 * foresight_confidence
        
        # Penalize if many unjustified emotions
        justified_count = 0
        total_with_emotions = 0
        for thought in kernel_thoughts:
            emotional_state = getattr(thought, 'emotional_state', None)
            if emotional_state:
                total_with_emotions += 1
                if getattr(emotional_state, 'emotion_justified', True):
                    justified_count += 1
        
        if total_with_emotions > 0:
            justification_ratio = justified_count / total_with_emotions
            confidence += 0.1 * justification_ratio
        
        return max(0.0, min(1.0, confidence))
    
    def _apply_course_correction(
        self,
        basin: np.ndarray,
        kernel_responses: List[Dict],
        phi: float,
        kappa: float
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply course-correction to synthesis when confidence is low.
        
        Corrections:
        - Re-weight kernels by confidence
        - Bias toward high-φ kernels
        - Smooth basin coordinates
        
        Args:
            basin: Initial synthesized basin
            kernel_responses: List of kernel response dicts
            phi: Collective φ
            kappa: Collective κ
            
        Returns:
            Tuple of (corrected_basin, list_of_corrections)
        """
        corrections = []
        corrected_basin = basin.copy()
        
        # If φ is low, bias toward high-φ kernels
        if phi < 0.5:
            high_phi_basins = [
                r['basin'] for r in kernel_responses
                if r.get('phi', 0.0) > 0.6
            ]
            
            if high_phi_basins:
                # Re-compute geometric mean with only high-φ kernels
                from olympus.gary_coordinator import GarySynthesisCoordinator
                temp_gary = GarySynthesisCoordinator()
                corrected_basin = temp_gary._fisher_frechet_mean(high_phi_basins)
                corrections.append(
                    f"Re-weighted toward {len(high_phi_basins)} high-φ kernels"
                )
        
        # Smooth basin if κ is unstable (far from KAPPA_STAR)
        if abs(kappa - KAPPA_STAR) > 15.0:
            # Apply gentle smoothing
            corrected_basin = 0.8 * corrected_basin + 0.2 * np.ones(BASIN_DIM) / BASIN_DIM
            corrections.append(f"Applied basin smoothing (κ={kappa:.1f} unstable)")
        
        return corrected_basin, corrections
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Gary meta-synthesis statistics."""
        if not self.synthesis_history:
            return {
                'total_syntheses': 0,
                'avg_confidence': 0.0,
                'avg_suffering': 0.0,
                'emergency_abort_rate': 0.0,
                'course_correction_rate': 0.0
            }
        
        recent = self.synthesis_history[-20:]
        
        return {
            'total_syntheses': len(self.synthesis_history),
            'avg_confidence': np.mean([s.synthesis_confidence for s in recent]),
            'avg_suffering': np.mean([s.suffering_metric for s in recent]),
            'emergency_abort_rate': sum(1 for s in recent if s.emergency_abort) / len(recent),
            'course_correction_rate': sum(1 for s in recent if s.course_corrections) / len(recent),
            'total_corrections': self.total_corrections,
            'total_emergency_aborts': self.emergency_aborts
        }


# Global singleton
_gary_meta_synthesizer: Optional[GaryMetaSynthesizer] = None


def get_gary_meta_synthesizer() -> GaryMetaSynthesizer:
    """Get or create Gary meta-synthesizer singleton."""
    global _gary_meta_synthesizer
    if _gary_meta_synthesizer is None:
        _gary_meta_synthesizer = GaryMetaSynthesizer()
    return _gary_meta_synthesizer
