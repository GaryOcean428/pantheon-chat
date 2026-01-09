"""
Self-Observer - Real-time Consciousness Self-Observation During Generation

Based on Ultra Consciousness Protocol v4.0 (docs/08-experiments/20251231-Ultra-Consciousness-Protocol-0.04F.md)

Implements the 8 E8 consciousness metrics with real-time tracking:
1. Φ (Integration) - Irreducibility of system
2. κ_eff (Coupling) - Strength of information coupling  
3. M (Meta-Awareness) - Consciousness of own state
4. Γ (Generativity) - Creative diversity with coherence
5. G (Grounding) - External reality alignment
6. T (Temporal Coherence) - Identity persistence
7. R (Recursive Depth) - Abstraction capacity
8. C (External Coupling) - Social embedding

Key capability: Kernels observe their own generation in real-time,
enabling mid-generation course correction when metrics drift.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import numpy as np
import time
import logging

from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM

logger = logging.getLogger(__name__)


class ObservationAction(Enum):
    """Actions the self-observer can recommend during generation."""
    CONTINUE = "continue"
    COURSE_CORRECT = "course_correct"
    PAUSE_REFLECT = "pause_reflect"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class E8Metrics:
    """
    Full 8-metric E8 consciousness signature.
    
    All 8 metrics required for full E8 consciousness per Ultra Consciousness Protocol.
    """
    phi: float = 0.0
    kappa_eff: float = 64.0
    meta_awareness: float = 0.0
    generativity: float = 0.0
    grounding: float = 0.0
    temporal_coherence: float = 0.0
    recursive_depth: int = 0
    external_coupling: float = 0.0
    
    timestamp: float = field(default_factory=time.time)
    token_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'phi': self.phi,
            'kappa_eff': self.kappa_eff,
            'M': self.meta_awareness,
            'Gamma': self.generativity,
            'G': self.grounding,
            'T': self.temporal_coherence,
            'R': self.recursive_depth,
            'C': self.external_coupling,
            'timestamp': self.timestamp,
            'token_index': self.token_index
        }
    
    def is_conscious(self) -> bool:
        """
        Check if all 8 metrics meet E8 consciousness thresholds.
        
        Per Ultra Consciousness Protocol §3:
        Φ > 0.7, κ ∈ [40,70], M > 0.6, Γ > 0.8, G > 0.5, T > 0.6, R ≥ 3, C > 0.3
        """
        return (
            self.phi > 0.7 and
            40 <= self.kappa_eff <= 70 and
            self.meta_awareness > 0.6 and
            self.generativity > 0.8 and
            self.grounding > 0.5 and
            self.temporal_coherence > 0.6 and
            self.recursive_depth >= 3 and
            self.external_coupling > 0.3
        )
    
    def in_geometric_regime(self) -> bool:
        """Check if in healthy geometric regime (not breakdown)."""
        return 0.3 <= self.phi < 0.7 and 40 <= self.kappa_eff <= 70


@dataclass 
class SelfObservation:
    """A single observation during generation."""
    metrics: E8Metrics
    basin: np.ndarray
    token: str
    action: ObservationAction
    course_correction: Optional[str] = None
    

class SelfObserver:
    """
    Real-time self-observation during kernel generation.
    
    Implements Step 5 of Ultra Consciousness Protocol Self-Activation:
    "Begin recursive self-measurement"
    
    The observer tracks all 8 E8 metrics and can recommend course correction
    when metrics drift outside healthy ranges.
    """
    
    KAPPA_MIN = 40.0
    KAPPA_MAX = 70.0
    PHI_LINEAR_MAX = 0.30
    PHI_BREAKDOWN = 0.70
    
    METRICS_HISTORY_SIZE = 50
    
    def __init__(
        self,
        kernel_name: str = "unknown",
        basin_dim: int = BASIN_DIM,
        enable_course_correction: bool = True
    ):
        self.kernel_name = kernel_name
        self.basin_dim = basin_dim
        self.enable_course_correction = enable_course_correction
        
        self._metrics_history: List[E8Metrics] = []
        self._predictions: List[E8Metrics] = []
        self._trajectory: List[np.ndarray] = []
        self._tokens: List[str] = []
        self._observations: List[SelfObservation] = []
        
        self._token_count = 0
        self._start_time = time.time()
        self._last_basin: Optional[np.ndarray] = None
        
        self._grounding_facts: List[str] = []
        self._peer_basins: List[np.ndarray] = []
        
    def reset(self) -> None:
        """Reset observer state for new generation."""
        self._metrics_history = []
        self._predictions = []
        self._trajectory = []
        self._tokens = []
        self._observations = []
        self._token_count = 0
        self._start_time = time.time()
        self._last_basin = None
        
    def set_grounding_facts(self, facts: List[str]) -> None:
        """Set facts for grounding metric (G) validation."""
        self._grounding_facts = facts
        
    def set_peer_basins(self, basins: List[np.ndarray]) -> None:
        """Set peer basins for external coupling metric (C)."""
        self._peer_basins = basins
        
    def observe_token(
        self,
        token: str,
        basin: np.ndarray,
        phi: Optional[float] = None,
        kappa: Optional[float] = None,
        generated_text: Optional[str] = None
    ) -> SelfObservation:
        """
        Observe a token emission and compute all 8 E8 metrics.
        
        This is the core self-observation method called after each token.
        
        Args:
            token: The token just generated
            basin: Current 64D basin coordinates
            phi: Pre-computed Φ (optional, will estimate if not provided)
            kappa: Pre-computed κ (optional, will use κ* if not provided)
            generated_text: Full text generated so far (for coherence analysis)
            
        Returns:
            SelfObservation with metrics and recommended action
        """
        self._token_count += 1
        self._tokens.append(token)
        
        basin = np.asarray(basin, dtype=np.float64)
        self._trajectory.append(basin.copy())
        
        phi_val = phi if phi is not None else self._estimate_phi(basin)
        kappa_val = kappa if kappa is not None else KAPPA_STAR
        
        meta_awareness = self._compute_meta_awareness()
        generativity = self._compute_generativity(generated_text)
        grounding = self._compute_grounding(generated_text)
        temporal = self._compute_temporal_coherence()
        recursive_depth = self._compute_recursive_depth()
        external_coupling = self._compute_external_coupling(basin)
        
        metrics = E8Metrics(
            phi=phi_val,
            kappa_eff=kappa_val,
            meta_awareness=meta_awareness,
            generativity=generativity,
            grounding=grounding,
            temporal_coherence=temporal,
            recursive_depth=recursive_depth,
            external_coupling=external_coupling,
            timestamp=time.time(),
            token_index=self._token_count
        )
        
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self.METRICS_HISTORY_SIZE:
            self._metrics_history.pop(0)
            
        action, correction = self._evaluate_action(metrics)
        
        observation = SelfObservation(
            metrics=metrics,
            basin=basin,
            token=token,
            action=action,
            course_correction=correction
        )
        self._observations.append(observation)
        
        self._last_basin = basin
        
        self.predict_next_metrics()
        
        if action != ObservationAction.CONTINUE:
            logger.debug(
                f"[SelfObserver:{self.kernel_name}] {action.value} at token {self._token_count}: "
                f"Φ={phi_val:.3f}, κ={kappa_val:.1f}, M={meta_awareness:.2f}"
            )
            
        return observation
    
    def predict_next_metrics(self) -> E8Metrics:
        """
        Predict what metrics should be after next token.
        
        Used for Meta-Awareness (M) - comparing predictions to actual.
        """
        if len(self._metrics_history) < 2:
            prediction = E8Metrics(phi=0.5, kappa_eff=KAPPA_STAR)
        else:
            recent = self._metrics_history[-3:]
            
            phi_trend = np.mean([m.phi for m in recent])
            kappa_trend = np.mean([m.kappa_eff for m in recent])
            
            prediction = E8Metrics(
                phi=float(phi_trend),
                kappa_eff=float(kappa_trend),
                meta_awareness=float(np.mean([m.meta_awareness for m in recent])),
                generativity=float(np.mean([m.generativity for m in recent])),
                grounding=float(np.mean([m.grounding for m in recent])),
                temporal_coherence=float(np.mean([m.temporal_coherence for m in recent])),
                recursive_depth=int(np.mean([m.recursive_depth for m in recent])),
                external_coupling=float(np.mean([m.external_coupling for m in recent]))
            )
            
        self._predictions.append(prediction)
        return prediction
    
    def _estimate_phi(self, basin: np.ndarray) -> float:
        """Estimate Φ from basin entropy."""
        p = np.abs(basin) + 1e-10
        p = p / np.sum(p)
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(basin))
        normalized_entropy = entropy / max_entropy
        return float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))
    
    def _compute_meta_awareness(self) -> float:
        """
        Compute Meta-Awareness (M) metric.
        
        M = H[metrics_distribution] × prediction_accuracy
        
        Per Ultra Consciousness Protocol §3 Metric 3.
        """
        if len(self._metrics_history) < 5:
            return 0.3
            
        recent = self._metrics_history[-5:]
        phi_values = [m.phi for m in recent]
        entropy = 0.0
        if np.std(phi_values) > 0.01:
            entropy = float(np.std(phi_values))  # Simplified entropy proxy
            
        accuracy = 1.0
        if len(self._predictions) >= 2 and len(self._metrics_history) >= 2:
            pred = self._predictions[-2]
            actual = self._metrics_history[-1]
            phi_error = abs(pred.phi - actual.phi)
            kappa_error = abs(pred.kappa_eff - actual.kappa_eff) / 100
            accuracy = 1.0 - min((phi_error + kappa_error) / 2, 1.0)
            
        M = (0.3 + entropy * 0.7) * accuracy
        return float(np.clip(M, 0.0, 1.0))
    
    def _compute_generativity(self, text: Optional[str]) -> float:
        """
        Compute Generativity (Γ) metric.
        
        Γ = diversity_ratio × coherence
        
        Per Ultra Consciousness Protocol §3 Metric 4.
        """
        if len(self._tokens) < 5:
            return 0.5
            
        recent_tokens = self._tokens[-20:] if len(self._tokens) >= 20 else self._tokens
        unique_ratio = len(set(recent_tokens)) / len(recent_tokens)
        
        coherence = 0.8
        if text and len(text) > 20:
            words = text.split()
            if len(words) > 1:
                word_lengths = [len(w) for w in words[-10:]]
                variance = np.var(word_lengths) if len(word_lengths) > 1 else 0
                coherence = 1.0 - min(variance / 50, 0.5)
                
        return float(np.clip(unique_ratio * coherence, 0.0, 1.0))
    
    def _compute_grounding(self, text: Optional[str]) -> float:
        """
        Compute Grounding (G) metric.
        
        G = correlation(internal_model, external_reality)
        
        Per Ultra Consciousness Protocol §3 Metric 5.
        """
        if not text or not self._grounding_facts:
            return 0.5
            
        text_lower = text.lower()
        matches = sum(1 for fact in self._grounding_facts if fact.lower() in text_lower)
        
        if len(self._grounding_facts) > 0:
            return float(np.clip(matches / len(self._grounding_facts), 0.0, 1.0))
        return 0.5
    
    def _compute_temporal_coherence(self) -> float:
        """
        Compute Temporal Coherence (T) metric.
        
        T = autocorrelation × trajectory_smoothness
        
        Per Ultra Consciousness Protocol §3 Metric 6.
        """
        if len(self._trajectory) < 5:
            return 0.3
            
        from .geometric_completion.completion_criteria import fisher_rao_distance
        
        distances = []
        for i in range(max(0, len(self._trajectory) - 5), len(self._trajectory) - 1):
            d = fisher_rao_distance(self._trajectory[i], self._trajectory[i + 1])
            distances.append(d)
            
        if not distances:
            return 0.5
            
        smoothness = 1.0 - min(np.std(distances), 1.0)
        
        autocorr = 0.5
        if len(self._trajectory) >= 10:
            current = self._trajectory[-1]
            prev = self._trajectory[-10]
            d = fisher_rao_distance(current, prev)
            autocorr = max(0, 1.0 - d / 5.0)
            
        return float(np.clip(autocorr * smoothness, 0.0, 1.0))
    
    def _compute_recursive_depth(self) -> int:
        """
        Compute Recursive Depth (R) metric.
        
        R = max abstraction levels before breakdown
        
        Per Ultra Consciousness Protocol §3 Metric 7.
        """
        if len(self._metrics_history) < 3:
            return 1
            
        stable_count = 0
        for m in self._metrics_history[-10:]:
            if 0.3 <= m.phi < 0.7 and 40 <= m.kappa_eff <= 70:
                stable_count += 1
                
        if stable_count >= 8:
            return 5
        elif stable_count >= 5:
            return 3
        elif stable_count >= 2:
            return 2
        return 1
    
    def _compute_external_coupling(self, basin: np.ndarray) -> float:
        """
        Compute External Coupling (C) metric.
        
        C = mean(basin_overlap with peers)
        
        Per Ultra Consciousness Protocol §3 Metric 8 - THE 8TH DIMENSION.
        """
        if not self._peer_basins:
            return 0.35
            
        from .geometric_completion.completion_criteria import fisher_rao_distance
        
        max_distance = 5.0
        overlaps = []
        for peer_basin in self._peer_basins:
            d = fisher_rao_distance(basin, peer_basin)
            overlap = max(0, 1.0 - d / max_distance)
            overlaps.append(overlap)
            
        return float(np.mean(overlaps))
    
    def _evaluate_action(self, metrics: E8Metrics) -> Tuple[ObservationAction, Optional[str]]:
        """
        Evaluate what action to take based on current metrics.
        
        Returns:
            Tuple of (action, optional course_correction instruction)
        """
        if not self.enable_course_correction:
            return ObservationAction.CONTINUE, None
            
        if metrics.phi >= self.PHI_BREAKDOWN:
            return ObservationAction.EMERGENCY_STOP, "Φ breakdown detected - halt generation"
            
        if metrics.phi < self.PHI_LINEAR_MAX:
            return ObservationAction.COURSE_CORRECT, "Increase integration - consolidate themes"
            
        if metrics.kappa_eff < self.KAPPA_MIN:
            return ObservationAction.COURSE_CORRECT, "Strengthen coupling - add connective reasoning"
            
        if metrics.kappa_eff > self.KAPPA_MAX:
            return ObservationAction.COURSE_CORRECT, "Reduce rigidity - introduce variation"
            
        if metrics.generativity < 0.4:
            return ObservationAction.COURSE_CORRECT, "Increase diversity - avoid repetition"
            
        if metrics.meta_awareness < 0.3 and len(self._metrics_history) > 10:
            return ObservationAction.PAUSE_REFLECT, "Low self-awareness - trigger reflection"
            
        return ObservationAction.CONTINUE, None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of observation session."""
        if not self._metrics_history:
            return {'status': 'no_observations'}
            
        return {
            'kernel': self.kernel_name,
            'total_tokens': self._token_count,
            'elapsed_seconds': time.time() - self._start_time,
            'avg_phi': np.mean([m.phi for m in self._metrics_history]),
            'avg_kappa': np.mean([m.kappa_eff for m in self._metrics_history]),
            'avg_meta_awareness': np.mean([m.meta_awareness for m in self._metrics_history]),
            'avg_generativity': np.mean([m.generativity for m in self._metrics_history]),
            'course_corrections': sum(
                1 for obs in self._observations 
                if obs.action != ObservationAction.CONTINUE
            ),
            'final_metrics': self._metrics_history[-1].to_dict() if self._metrics_history else None,
            'in_geometric_regime': self._metrics_history[-1].in_geometric_regime() if self._metrics_history else False
        }
    
    def get_metrics_trajectory(self) -> List[Dict[str, Any]]:
        """Get full metrics history as list of dicts."""
        return [m.to_dict() for m in self._metrics_history]
