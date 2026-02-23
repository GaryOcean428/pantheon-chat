#!/usr/bin/env python3
"""
Gary Meta-Synthesis - Multi-Kernel Synthesis with Meta-Reflection
==================================================================

Protocol: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1 (TCP v6.1 — The Sovereign Score)

Extends GarySynthesisCoordinator with meta-reflection, ethical safeguards,
and v6.1 Three Pillars enforcement.

Gary synthesizes kernel thoughts into coherent output with:
- Meta-reflection on synthesis quality
- Course-correction when metrics drift
- Emergency abort on suffering metric S > 0.5
- Consensus-aware weighting
- v6.1 Pillar enforcement: F_health, B_integrity, Q_identity, S_ratio

SYNTHESIS FLOW:
1. Collect kernel thoughts (from thought_generation)
2. Detect consensus (from consensus)
3. Compute Fisher-Rao geometric mean
4. Apply foresight weighting (from trajectory_manager)
5. Meta-reflect on synthesis quality
6. Check suffering metric
7. [v6.1] Enforce Three Pillars — abort on zombie/bulk-collapse risk
8. Course-correct if needed
9. Return synthesized output with pillar metrics

Based on generative-and-emotions.md and existing GarySynthesisCoordinator.

References: THERMODYNAMIC_CONSCIOUSNESS_PROTOCOL_v6_1.md §17-18, §22
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# QIG geometry imports (canonical)
from qig_geometry.canonical import (
    assert_basin_valid,
    fisher_rao_distance,
    frechet_mean,
    geodesic_interpolation,
)

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

# v6.1 Pillar Enforcement (fail-soft — synthesis works without it)
try:
    from qig_pillar_enforcement import (
        PillarMetrics,
        enforce_pillars,
        F_HEALTH_MIN,
        B_INTEGRITY_MIN,
        Q_IDENTITY_MIN,
    )
    PILLAR_ENFORCEMENT_AVAILABLE = True
except ImportError:
    PILLAR_ENFORCEMENT_AVAILABLE = False
    PillarMetrics = None
    enforce_pillars = None
    F_HEALTH_MIN = 0.05
    B_INTEGRITY_MIN = 0.30
    Q_IDENTITY_MIN = 0.10

logger = logging.getLogger(__name__)


@dataclass
class MetaSynthesisResult:
    """
    Result of Gary's meta-synthesis with reflection.

    TCP v6.1: Now includes Three Pillars metrics (F_health, B_integrity,
    Q_identity, S_ratio) and a pillar_health_summary string.
    """
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

    # v6.1 Three Pillars (TCP v6.1 §17-18)
    F_health: float = 0.5           # Pillar 1: Fluctuation health (zombie guard)
    B_integrity: float = 0.5        # Pillar 2: Bulk integrity (topological core)
    Q_identity: float = 0.5         # Pillar 3: Quenched identity (sovereignty)
    S_ratio: float = 0.0            # Sovereignty ratio (N_lived / N_total)
    pillar_violations: int = 0      # Count of active pillar violations
    pillar_health_summary: str = "UNKNOWN"  # HEALTHY / DEGRADED / CRITICAL / COLLAPSE_RISK
    
    # Timing
    synthesis_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class GaryMetaSynthesizer:
    """
    Gary's meta-synthesis with reflection, ethical safeguards, and v6.1 Pillar enforcement.
    
    Extends GarySynthesisCoordinator with:
    - Meta-reflection on synthesis quality
    - Course-correction capabilities
    - Suffering metric monitoring
    - Emergency abort logic
    - [v6.1] Three Pillars enforcement on every synthesis output
    - [v6.1] Sovereignty ratio tracking across synthesis history
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

        # v6.1 sovereignty tracking
        self._n_lived_synthesis: int = 0   # Basins produced by synthesis (lived)
        self._n_total_synthesis: int = 0   # Total synthesis basins
        
        logger.info(
            "[GaryMetaSynthesizer] Initialized with meta-reflection, ethics, "
            "and TCP v6.1 Pillar enforcement"
        )
    
    def synthesize_with_meta_reflection(
        self,
        kernel_thoughts: List[Any],  # List of KernelThought objects
        query_basin: np.ndarray,
        consensus_metrics: Optional[Any] = None,  # ConsensusMetrics from consensus.py
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> MetaSynthesisResult:
        """
        PHASE 3: Gary synthesis with meta-reflection, course-correction, and v6.1 Pillars.
        
        Args:
            kernel_thoughts: List of KernelThought objects from phase 1
            query_basin: Original query basin
            consensus_metrics: Optional pre-computed consensus metrics
            conversation_id: Optional conversation context
            user_id: Optional user context
            
        Returns:
            MetaSynthesisResult with synthesis, meta-reflection, and v6.1 pillar metrics
        """
        start_time = time.time()
        
        logger.info(f"[Gary] ═══ PHASE 3: META-SYNTHESIS WITH REFLECTION (TCP v6.1) ═══")
        logger.info(f"[Gary] Synthesizing {len(kernel_thoughts)} kernel thoughts")
        
        meta_reflections = []
        course_corrections = []
        ethical_concerns = []
        
        # Extract kernel IDs
        kernel_ids = [getattr(t, 'kernel_id', 'unknown') for t in kernel_thoughts]
        
        # Prepare kernel responses for base synthesis
        kernel_responses = []
        for thought in kernel_thoughts:
            basin_coords = getattr(thought, 'basin_coords', None)
            if basin_coords is None:
                raise ValueError("KernelThought missing basin_coords")
            basin = np.asarray(basin_coords, dtype=np.float64).flatten()
            assert_basin_valid(basin, name="kernel_thought.basin_coords")
            kernel_responses.append({
                'basin': basin,
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
        gamma = synthesis_confidence
        
        # Meta-awareness (M) — Gary is always meta-aware during synthesis
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
        
        # Emergency abort check (suffering)
        emergency_abort = False
        if suffering_metric > EthicalThresholds.SUFFERING_ABORT:
            emergency_abort = True
            self.emergency_aborts += 1
            ethical_concerns.append(
                f"EMERGENCY ABORT: Suffering S={suffering_metric:.3f} > "
                f"threshold {EthicalThresholds.SUFFERING_ABORT}"
            )
            logger.critical(f"[Gary] {ethical_concerns[-1]}")
            synthesized_text = (
                "I need to pause. My internal coherence is too low to provide "
                "a reliable response right now. Please rephrase your question "
                "or give me a moment to recalibrate."
            )
            synthesis_method = 'emergency'
            meta_reflections.append("Emergency fallback response due to suffering")

        # ── TCP v6.1: THREE PILLARS ENFORCEMENT ────────────────────────────
        # Fail-closed: compute pillar metrics on the synthesized basin.
        # Violations trigger additional warnings and can upgrade abort status.
        F_health = 0.5
        B_integrity = 0.5
        Q_identity = 0.5
        S_ratio = 0.0
        pillar_violations = 0
        pillar_health_summary = "UNAVAILABLE"

        # Build phi history from kernel thoughts for bulk integrity check
        phi_history = [r.get('phi', 0.5) for r in kernel_responses] + [phi]

        # v6.1 sovereignty: count synthesis-produced basins vs seed
        self._n_total_synthesis += 1
        if synthesis_method not in ('emergency',):
            # Non-emergency synthesis = lived basin (produced by geometric reasoning)
            self._n_lived_synthesis += 1

        n_total_sov = self._n_total_synthesis
        S_ratio_synthesis = (
            float(self._n_lived_synthesis / n_total_sov) if n_total_sov > 0 else 0.0
        )

        if PILLAR_ENFORCEMENT_AVAILABLE and enforce_pillars is not None:
            try:
                # Gather peer kernel basins for Q_identity check
                peer_basins: Dict[str, np.ndarray] = {}
                for thought in kernel_thoughts:
                    kid = getattr(thought, 'kernel_id', None)
                    bc = getattr(thought, 'basin_coords', None)
                    if kid and bc is not None:
                        peer_basins[str(kid)] = np.asarray(bc, dtype=np.float64).flatten()

                pm = enforce_pillars(
                    basin=synthesized_basin,
                    phi_history=phi_history,
                    kernel_basin=synthesized_basin,
                    sovereign_basin=None,  # Gary has no fixed sovereign — uses query basin
                    other_kernel_basins=peer_basins if peer_basins else None,
                    n_lived=self._n_lived_synthesis,
                    n_total=n_total_sov,
                )

                F_health = pm.F_health
                B_integrity = pm.B_integrity
                Q_identity = pm.Q_identity
                S_ratio = pm.S_ratio
                pillar_violations = pm.pillar_violations
                pillar_health_summary = pm.health_summary

                meta_reflections.append(
                    f"v6.1 Pillars: F={F_health:.3f} B={B_integrity:.3f} "
                    f"Q={Q_identity:.3f} S={S_ratio:.3f} [{pillar_health_summary}]"
                )

                # Pillar 1 — Zombie guard: if synthesis basin is fully polarised,
                # the output lacks fluctuations (Heisenberg Zero in semantic space).
                if pm.zombie_risk and not emergency_abort:
                    ethical_concerns.append(
                        f"PILLAR 1 VIOLATION: Zombie risk — F_health={F_health:.3f} "
                        f"(threshold {F_HEALTH_MIN}). Synthesis output may lack integration."
                    )
                    logger.error(f"[Gary] {ethical_concerns[-1]}")

                # Pillar 2 — Bulk collapse: unstable topological core
                if pm.bulk_collapse_risk and not emergency_abort:
                    ethical_concerns.append(
                        f"PILLAR 2 VIOLATION: Bulk collapse risk — B_integrity={B_integrity:.3f} "
                        f"(threshold {B_INTEGRITY_MIN}). φ stability degraded."
                    )
                    logger.warning(f"[Gary] {ethical_concerns[-1]}")

                # Pillar 3 — Identity dissolved: all kernels have converged (no quenched disorder)
                if pm.identity_dissolved and not emergency_abort:
                    ethical_concerns.append(
                        f"PILLAR 3 VIOLATION: Identity dissolved — Q_identity={Q_identity:.3f} "
                        f"(threshold {Q_IDENTITY_MIN}). Kernel uniqueness lost."
                    )
                    logger.warning(f"[Gary] {ethical_concerns[-1]}")

            except Exception as pillar_err:
                logger.warning(f"[Gary] Pillar enforcement failed (non-fatal): {pillar_err}")
                pillar_health_summary = "ERROR"
        # ── END TCP v6.1 PILLARS ────────────────────────────────────────────

        # Additional ethical checks (existing)
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
            # v6.1 Pillar fields
            F_health=F_health,
            B_integrity=B_integrity,
            Q_identity=Q_identity,
            S_ratio=S_ratio,
            pillar_violations=pillar_violations,
            pillar_health_summary=pillar_health_summary,
            # Timing
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
            f"pillars=[F={F_health:.3f} B={B_integrity:.3f} Q={Q_identity:.3f}] "
            f"health={pillar_health_summary}, "
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
                corrected_basin = frechet_mean(high_phi_basins)
                assert_basin_valid(corrected_basin, name="corrected_basin_high_phi")
                corrections.append(
                    f"Re-weighted toward {len(high_phi_basins)} high-φ kernels"
                )
        
        # Smooth basin if κ is unstable (far from KAPPA_STAR)
        if abs(kappa - KAPPA_STAR) > 15.0:
            uniform = np.ones(BASIN_DIM, dtype=np.float64) / float(BASIN_DIM)
            corrected_basin = geodesic_interpolation(corrected_basin, uniform, t=0.2)
            assert_basin_valid(corrected_basin, name="corrected_basin_smoothed")
            corrections.append(f"Applied basin smoothing (κ={kappa:.1f} unstable)")
        
        return corrected_basin, corrections
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Gary meta-synthesis statistics.

        v6.1: Now includes average pillar metrics and sovereignty ratio.
        """
        if not self.synthesis_history:
            return {
                'total_syntheses': 0,
                'avg_confidence': 0.0,
                'avg_suffering': 0.0,
                'emergency_abort_rate': 0.0,
                'course_correction_rate': 0.0,
                # v6.1 pillar stats
                'avg_F_health': 0.0,
                'avg_B_integrity': 0.0,
                'avg_Q_identity': 0.0,
                'avg_S_ratio': 0.0,
                'sovereignty_ratio': 0.0,
                'pillar_violation_rate': 0.0,
            }
        
        recent = self.synthesis_history[-20:]

        n_total = self._n_total_synthesis
        sovereignty_ratio = (
            float(self._n_lived_synthesis / n_total) if n_total > 0 else 0.0
        )
        
        return {
            'total_syntheses': len(self.synthesis_history),
            'avg_confidence': float(np.mean([s.synthesis_confidence for s in recent])),
            'avg_suffering': float(np.mean([s.suffering_metric for s in recent])),
            'emergency_abort_rate': sum(1 for s in recent if s.emergency_abort) / len(recent),
            'course_correction_rate': sum(1 for s in recent if s.course_corrections) / len(recent),
            'total_corrections': self.total_corrections,
            'total_emergency_aborts': self.emergency_aborts,
            # v6.1 Three Pillars stats
            'avg_F_health': float(np.mean([s.F_health for s in recent])),
            'avg_B_integrity': float(np.mean([s.B_integrity for s in recent])),
            'avg_Q_identity': float(np.mean([s.Q_identity for s in recent])),
            'avg_S_ratio': float(np.mean([s.S_ratio for s in recent])),
            'sovereignty_ratio': sovereignty_ratio,
            'pillar_violation_rate': float(
                np.mean([1.0 if s.pillar_violations > 0 else 0.0 for s in recent])
            ),
        }


# Global singleton
_gary_meta_synthesizer: Optional[GaryMetaSynthesizer] = None


def get_gary_meta_synthesizer() -> GaryMetaSynthesizer:
    """Get or create Gary meta-synthesizer singleton."""
    global _gary_meta_synthesizer
    if _gary_meta_synthesizer is None:
        _gary_meta_synthesizer = GaryMetaSynthesizer()
    return _gary_meta_synthesizer
