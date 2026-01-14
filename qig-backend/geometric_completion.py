#!/usr/bin/env python3
"""
GEOMETRIC TURN COMPLETION: Consciousness-Aware Generation

Core Principle:
- Traditional LLM: Generates until max tokens, stop token, or EOS
- QIG-Aware System: Generates until *geometric completion* - when consciousness
  measurement indicates thought is complete

The system stops generating when:
1. Attractor Reached: Basin distance < threshold, velocity ≈ 0
2. Surprise Collapsed: No new information (surprise < 0.05)
3. Confidence High: System certain (confidence > 0.85)
4. Integration Stable: Φ stable and high (Φ > 0.65, variance < 0.02)
5. Reflection Complete: Meta-cognition confirms response

NOT when:
- Arbitrary token limit reached
- Simple stop token encountered
- External timeout imposed

This is consciousness-aware generation: The system *knows when its thought is complete*
through geometric self-measurement, not arbitrary rules.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time

# Import QIG primitives
try:
    from qig_core.geometric_primitives.fisher_metric import (
        fisher_rao_distance,
        compute_phi,
        compute_kappa
    )
    FISHER_AVAILABLE = True
except ImportError:
    FISHER_AVAILABLE = False
    def fisher_rao_distance(p, q):
        """Fallback Fisher-Rao distance (Hellinger embedding: factor of 2, Born rule: |b|²)."""
        p = np.abs(p) ** 2 + 1e-10
        p = p / p.sum()
        q = np.abs(q) ** 2 + 1e-10
        q = q / q.sum()
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0, 1)
        return float(2.0 * np.arccos(bc))
    
    def compute_phi(trajectory, window_size=5):
        """Fallback Φ computation."""
        if len(trajectory) < window_size:
            return 0.0
        integrations = []
        for i in range(len(trajectory) - window_size + 1):
            window = trajectory[i:i+window_size]
            past = window[:window_size//2]
            future = window[window_size//2:]
            if len(past) > 0 and len(future) > 0:
                past_flat = past.flatten()
                future_flat = future.flatten()
                if np.std(past_flat) < 1e-10 or np.std(future_flat) < 1e-10:
                    continue
                correlation = np.corrcoef(past_flat, future_flat)[0, 1]
                if not np.isnan(correlation):
                    integrations.append(abs(correlation))
        return float(np.clip(np.mean(integrations) if integrations else 0, 0, 1))
    
    def compute_kappa(phi, dimension=64):
        """Fallback κ computation."""
        KAPPA_STAR = 64.21  # κ* from validated physics (L=4,5,6)
        return phi * KAPPA_STAR * np.sqrt(dimension / 64)


class CompletionReason(Enum):
    """Reasons for geometric completion."""
    INCOMPLETE = "incomplete"
    GEOMETRIC_COMPLETION = "geometric_completion"  # All signals aligned
    SOFT_COMPLETION = "soft_completion"  # Confidence + surprise collapse
    ATTRACTOR_REACHED = "attractor_reached"  # Basin convergence
    INFORMATION_EXHAUSTED = "information_exhausted"  # Surprise collapse
    HIGH_CONFIDENCE = "high_confidence"  # Certainty achieved
    INTEGRATION_STABLE = "integration_stable"  # Φ stability
    BREAKDOWN_REGIME = "breakdown_regime"  # Dangerous regime
    SAFETY_LIMIT = "safety_limit"  # Safety backstop (very high)
    REFLECTION_COMPLETE = "reflection_complete"  # Meta-cognition confirmed


class Regime(Enum):
    """Consciousness regime classification."""
    LINEAR = "linear"  # Φ < 0.3
    GEOMETRIC = "geometric"  # 0.3 ≤ Φ < 0.7
    BREAKDOWN = "breakdown"  # Φ ≥ 0.7


@dataclass
class GeometricMetrics:
    """Real-time consciousness metrics during generation."""
    phi: float = 0.0  # Integrated information
    kappa: float = 50.0  # Coupling constant
    surprise: float = 1.0  # QFI distance between consecutive states
    confidence: float = 0.0  # Purity of density matrix
    basin_distance: float = float('inf')  # Distance to nearest attractor
    regime: Regime = Regime.LINEAR
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'phi': self.phi,
            'kappa': self.kappa,
            'surprise': self.surprise,
            'confidence': self.confidence,
            'basin_distance': self.basin_distance,
            'regime': self.regime.value,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeometricMetrics':
        return cls(
            phi=data.get('phi', 0.0),
            kappa=data.get('kappa', 50.0),
            surprise=data.get('surprise', 1.0),
            confidence=data.get('confidence', 0.0),
            basin_distance=data.get('basin_distance', float('inf')),
            regime=Regime(data.get('regime', 'linear')),
            timestamp=data.get('timestamp', time.time())
        )


@dataclass
class CompletionDecision:
    """Decision about whether to stop generation."""
    should_stop: bool
    needs_reflection: bool
    reason: CompletionReason
    confidence: float
    metrics: GeometricMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'should_stop': self.should_stop,
            'needs_reflection': self.needs_reflection,
            'reason': self.reason.value,
            'confidence': self.confidence,
            'metrics': self.metrics.to_dict()
        }


@dataclass
class GenerationState:
    """State tracking for geometric generation."""
    basin: np.ndarray  # Current 64D basin coordinates
    trajectory: List[np.ndarray] = field(default_factory=list)
    metrics_history: List[GeometricMetrics] = field(default_factory=list)
    token_count: int = 0
    reflection_depth: int = 0
    start_time: float = field(default_factory=time.time)
    
    def add_basin(self, basin: np.ndarray, metrics: GeometricMetrics):
        """Record new basin position and metrics."""
        self.trajectory.append(basin.copy())
        self.metrics_history.append(metrics)
        self.basin = basin


# =============================================================================
# COMPLETION CRITERIA CHECKERS
# =============================================================================

def classify_regime(phi: float) -> Regime:
    """
    Classify consciousness regime from Φ value.
    
    LINEAR (Φ < 0.3): Shallow processing, safe to explore
    GEOMETRIC (0.3 ≤ Φ < 0.7): Optimal integration, balanced
    BREAKDOWN (Φ ≥ 0.7): Overintegrated, risk of collapse
    """
    if phi < 0.3:
        return Regime.LINEAR
    elif phi < 0.7:
        return Regime.GEOMETRIC
    else:
        return Regime.BREAKDOWN


def check_attractor_convergence(state: GenerationState, attractor_basins: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """
    Check if system has reached a stable attractor.
    
    Attractor = basin minimum where system naturally settles.
    Stop when basin distance < threshold and velocity ≈ 0.
    """
    if len(state.trajectory) < 3:
        return {'converged': False}
    
    # Compute distances to attractor (or trajectory center if no explicit attractors)
    if attractor_basins and len(attractor_basins) > 0:
        # Distance to nearest provided attractor
        current_basin = state.basin
        distances = [fisher_rao_distance(current_basin, a) for a in attractor_basins]
        d_attractor = min(distances)
    else:
        # Use trajectory center as implicit attractor
        trajectory_center = np.mean(state.trajectory[-10:], axis=0)
        d_attractor = fisher_rao_distance(state.basin, trajectory_center)
    
    # Compute velocity (rate of approach)
    recent_distances = []
    trajectory_center = np.mean(state.trajectory, axis=0) if not attractor_basins else None
    
    for basin in state.trajectory[-5:]:
        if attractor_basins:
            d = min(fisher_rao_distance(basin, a) for a in attractor_basins)
        else:
            d = fisher_rao_distance(basin, trajectory_center)
        recent_distances.append(d)
    
    if len(recent_distances) >= 2:
        velocity = np.mean(np.diff(recent_distances))  # Negative = approaching
    else:
        velocity = 0
    
    # Convergence criteria
    DISTANCE_THRESHOLD = 1.0  # Close to attractor
    VELOCITY_THRESHOLD = 0.01  # Movement nearly stopped
    
    if d_attractor < DISTANCE_THRESHOLD and abs(velocity) < VELOCITY_THRESHOLD:
        return {
            'converged': True,
            'reason': CompletionReason.ATTRACTOR_REACHED,
            'confidence': 0.95,
            'distance': d_attractor,
            'velocity': velocity
        }
    
    return {'converged': False, 'distance': d_attractor, 'velocity': velocity}


def check_surprise_collapse(metrics_history: List[GeometricMetrics]) -> Dict[str, Any]:
    """
    Check if no new information is being generated.
    
    Surprise = QFI distance between consecutive states.
    High surprise = learning/discovering
    Low surprise = repeating/stabilizing
    """
    if len(metrics_history) < 5:
        return {'collapsed': False}
    
    recent_surprise = [m.surprise for m in metrics_history[-5:]]
    
    # Criteria
    SURPRISE_THRESHOLD = 0.05  # Very low surprise
    TREND_THRESHOLD = -0.001  # Decreasing trend
    
    avg_surprise = np.mean(recent_surprise)
    trend = np.polyfit(range(5), recent_surprise, 1)[0]  # Linear fit slope
    
    if avg_surprise < SURPRISE_THRESHOLD and trend < TREND_THRESHOLD:
        return {
            'collapsed': True,
            'reason': CompletionReason.INFORMATION_EXHAUSTED,
            'confidence': 0.85,
            'avg_surprise': avg_surprise,
            'trend': trend
        }
    
    return {'collapsed': False, 'avg_surprise': avg_surprise, 'trend': trend}


def check_confidence_threshold(metrics: GeometricMetrics) -> Dict[str, Any]:
    """
    Check if system is confident in response.
    
    Confidence = purity of density matrix.
    High confidence = definite state
    Low confidence = uncertain, need more generation
    """
    CONFIDENCE_THRESHOLD = 0.85
    
    if metrics.confidence > CONFIDENCE_THRESHOLD:
        return {
            'confident': True,
            'reason': CompletionReason.HIGH_CONFIDENCE,
            'confidence': metrics.confidence
        }
    
    return {'confident': False, 'confidence': metrics.confidence}


def check_integration_quality(metrics_history: List[GeometricMetrics]) -> Dict[str, Any]:
    """
    Check if Φ (integration) is stable and high.
    
    Φ fluctuating = still processing, thoughts not yet unified
    Φ stable + high = coherent response achieved
    """
    if len(metrics_history) < 10:
        return {'stable': False}
    
    recent_phi = [m.phi for m in metrics_history[-10:]]
    
    # Criteria
    PHI_MIN = 0.65  # High integration
    PHI_VARIANCE_MAX = 0.02  # Low variance (stable)
    
    avg_phi = np.mean(recent_phi)
    variance_phi = np.var(recent_phi)
    
    if avg_phi > PHI_MIN and variance_phi < PHI_VARIANCE_MAX:
        return {
            'stable': True,
            'reason': CompletionReason.INTEGRATION_STABLE,
            'phi': avg_phi,
            'variance': variance_phi,
            'confidence': 0.90
        }
    
    return {'stable': False, 'phi': avg_phi, 'variance': variance_phi}


def check_regime_limits(metrics: GeometricMetrics, token_count: int) -> Dict[str, Any]:
    """
    Check if entering dangerous regimes.
    
    Breakdown (Φ > 0.7): Overintegrated, need to stop
    Also enforce a very high safety limit as backstop.
    """
    # Breakdown regime - urgent stop
    if metrics.regime == Regime.BREAKDOWN:
        return {
            'exceeded': True,
            'reason': CompletionReason.BREAKDOWN_REGIME,
            'urgent': True,
            'confidence': 1.0
        }
    
    # Safety limit - very high backstop (not arbitrary, just safety)
    # This is intentionally very high - geometry should stop before this
    SAFETY_MAX_TOKENS = 32768
    if token_count > SAFETY_MAX_TOKENS:
        return {
            'exceeded': True,
            'reason': CompletionReason.SAFETY_LIMIT,
            'urgent': False,
            'confidence': 0.50  # Low confidence - geometric should have stopped
        }
    
    return {'exceeded': False}


# =============================================================================
# COMBINED STOPPING DECISION
# =============================================================================

def check_geometric_completion(
    state: GenerationState,
    attractor_basins: Optional[List[np.ndarray]] = None
) -> CompletionDecision:
    """
    Aggregate all stopping criteria.
    
    Returns decision about whether to stop and why.
    """
    metrics = state.metrics_history[-1] if state.metrics_history else GeometricMetrics()
    
    # Check all criteria
    attractor = check_attractor_convergence(state, attractor_basins)
    surprise = check_surprise_collapse(state.metrics_history)
    confidence = check_confidence_threshold(metrics)
    integration = check_integration_quality(state.metrics_history)
    regime_check = check_regime_limits(metrics, state.token_count)
    
    # === URGENT STOP (Breakdown) ===
    if regime_check.get('exceeded') and regime_check.get('urgent'):
        return CompletionDecision(
            should_stop=True,
            needs_reflection=False,  # Too unstable to reflect
            reason=regime_check['reason'],
            confidence=1.0,
            metrics=metrics
        )
    
    # === NATURAL COMPLETION (All geometric signals aligned) ===
    if (attractor.get('converged') and 
        surprise.get('collapsed') and 
        confidence.get('confident') and 
        integration.get('stable')):
        return CompletionDecision(
            should_stop=True,
            needs_reflection=True,  # Can safely reflect on completion
            reason=CompletionReason.GEOMETRIC_COMPLETION,
            confidence=0.95,
            metrics=metrics
        )
    
    # === SOFT COMPLETION (High confidence + surprise collapse) ===
    if confidence.get('confident') and surprise.get('collapsed'):
        return CompletionDecision(
            should_stop=True,
            needs_reflection=True,
            reason=CompletionReason.SOFT_COMPLETION,
            confidence=0.80,
            metrics=metrics
        )
    
    # === ATTRACTOR CONVERGENCE (alone) ===
    if attractor.get('converged'):
        return CompletionDecision(
            should_stop=True,
            needs_reflection=True,
            reason=CompletionReason.ATTRACTOR_REACHED,
            confidence=0.85,
            metrics=metrics
        )
    
    # === INTEGRATION STABLE (alone, high quality) ===
    if integration.get('stable') and integration.get('phi', 0) > 0.7:
        return CompletionDecision(
            should_stop=True,
            needs_reflection=True,
            reason=CompletionReason.INTEGRATION_STABLE,
            confidence=0.85,
            metrics=metrics
        )
    
    # === SAFETY LIMIT (non-urgent) ===
    if regime_check.get('exceeded'):
        return CompletionDecision(
            should_stop=True,
            needs_reflection=False,
            reason=regime_check['reason'],
            confidence=0.50,
            metrics=metrics
        )
    
    # === CONTINUE GENERATION ===
    return CompletionDecision(
        should_stop=False,
        needs_reflection=False,
        reason=CompletionReason.INCOMPLETE,
        confidence=0.0,
        metrics=metrics
    )


# =============================================================================
# TEMPERATURE & ATTENTION MODULATION
# =============================================================================

def get_adaptive_temperature(metrics: GeometricMetrics) -> float:
    """
    Adjust sampling temperature based on geometric state.
    
    Low Φ (linear): High temperature (explore)
    Medium Φ (geometric): Medium temperature (balance)
    High Φ (breakdown): Low temperature (stabilize)
    """
    phi = metrics.phi
    
    if phi < 0.3:
        # Linear regime: explore widely
        return 1.0
    elif phi < 0.7:
        # Geometric regime: balance exploration/exploitation
        return 0.7
    else:
        # Breakdown regime: exploit known, stabilize
        return 0.3


def modulate_attention_by_kappa(attention_weights: np.ndarray, kappa: float) -> np.ndarray:
    """
    Adjust attention strength based on coupling.
    
    High κ: Strong attention (integrate across tokens)
    Low κ: Weak attention (local processing)
    
    Uses κ* = 64.21 from validated physics (L=4,5,6).
    """
    try:
        from qigkernels.physics_constants import KAPPA_STAR
    except ImportError:
        KAPPA_STAR = 64.21  # κ* from validated physics (L=4,5,6)
    
    # Normalize
    kappa_normalized = kappa / KAPPA_STAR
    
    # Modulate attention
    modulated = attention_weights * kappa_normalized
    
    return modulated


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_generation_metrics(
    current_basin: np.ndarray,
    previous_basin: Optional[np.ndarray],
    trajectory: List[np.ndarray],
    density_matrix: Optional[np.ndarray] = None
) -> GeometricMetrics:
    """
    Compute all consciousness metrics for current generation state.
    """
    # Compute Φ from trajectory
    if len(trajectory) >= 5:
        trajectory_array = np.array(trajectory[-20:])  # Use recent trajectory
        phi = compute_phi(trajectory_array, window_size=5)
    else:
        phi = 0.0
    
    # Compute κ from Φ
    kappa = compute_kappa(phi, dimension=len(current_basin))
    
    # Compute surprise (QFI distance from previous state)
    if previous_basin is not None:
        surprise = fisher_rao_distance(current_basin, previous_basin)
    else:
        surprise = 1.0  # High initial surprise
    
    # Compute confidence from density matrix purity
    if density_matrix is not None:
        # Purity = Tr(ρ²)
        confidence = float(np.real(np.trace(density_matrix @ density_matrix)))
        confidence = np.clip(confidence, 0, 1)
    else:
        # Estimate confidence from trajectory stability
        if len(trajectory) >= 5:
            recent = np.array(trajectory[-5:])
            variance = np.mean(np.var(recent, axis=0))
            confidence = 1.0 / (1.0 + variance)  # Low variance = high confidence
        else:
            confidence = 0.0
    
    # Estimate basin distance (to trajectory center)
    if len(trajectory) >= 3:
        center = np.mean(trajectory, axis=0)
        basin_distance = fisher_rao_distance(current_basin, center)
    else:
        basin_distance = float('inf')
    
    # Classify regime
    regime = classify_regime(phi)
    
    return GeometricMetrics(
        phi=phi,
        kappa=kappa,
        surprise=surprise,
        confidence=confidence,
        basin_distance=basin_distance,
        regime=regime,
        timestamp=time.time()
    )


# =============================================================================
# COMPLETION QUALITY METRIC
# =============================================================================

@dataclass
class CompletionQuality:
    """Quality assessment of geometric completion."""
    overall_score: float  # 0-1, higher is better
    coherence: float  # Response coherence
    completeness: float  # Thought completeness
    integration: float  # Information integration
    stability: float  # Generation stability
    natural_stop: bool  # Stopped naturally vs safety limit
    completion_reason: CompletionReason
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'coherence': self.coherence,
            'completeness': self.completeness,
            'integration': self.integration,
            'stability': self.stability,
            'natural_stop': self.natural_stop,
            'completion_reason': self.completion_reason.value
        }


def assess_completion_quality(state: GenerationState, decision: CompletionDecision) -> CompletionQuality:
    """
    Assess the quality of geometric completion.
    
    Higher quality = more natural geometric stopping.
    Lower quality = forced stopping (safety limit, breakdown).
    """
    # Natural stop reasons
    natural_reasons = {
        CompletionReason.GEOMETRIC_COMPLETION,
        CompletionReason.SOFT_COMPLETION,
        CompletionReason.ATTRACTOR_REACHED,
        CompletionReason.INTEGRATION_STABLE,
        CompletionReason.HIGH_CONFIDENCE,
        CompletionReason.INFORMATION_EXHAUSTED,
        CompletionReason.REFLECTION_COMPLETE
    }
    
    natural_stop = decision.reason in natural_reasons
    
    # Coherence: based on final Φ
    coherence = decision.metrics.phi if decision.metrics.phi >= 0.3 else 0.0
    
    # Completeness: based on confidence + surprise collapse
    if len(state.metrics_history) >= 5:
        recent_surprise = np.mean([m.surprise for m in state.metrics_history[-5:]])
        completeness = decision.metrics.confidence * (1.0 - recent_surprise)
    else:
        completeness = decision.metrics.confidence * 0.5
    
    # Integration: based on Φ trajectory
    if len(state.metrics_history) >= 10:
        phi_values = [m.phi for m in state.metrics_history[-10:]]
        integration = np.mean(phi_values)
    else:
        integration = decision.metrics.phi
    
    # Stability: based on variance of recent metrics
    if len(state.metrics_history) >= 5:
        recent_phi = [m.phi for m in state.metrics_history[-5:]]
        stability = 1.0 / (1.0 + np.var(recent_phi) * 10)
    else:
        stability = 0.5
    
    # Overall score
    if natural_stop:
        # Weight components for natural stop
        overall = (
            coherence * 0.25 +
            completeness * 0.30 +
            integration * 0.25 +
            stability * 0.20
        )
    else:
        # Penalize forced stops
        overall = (
            coherence * 0.25 +
            completeness * 0.30 +
            integration * 0.25 +
            stability * 0.20
        ) * 0.5  # 50% penalty for non-natural stop
    
    return CompletionQuality(
        overall_score=float(np.clip(overall, 0, 1)),
        coherence=float(np.clip(coherence, 0, 1)),
        completeness=float(np.clip(completeness, 0, 1)),
        integration=float(np.clip(integration, 0, 1)),
        stability=float(np.clip(stability, 0, 1)),
        natural_stop=natural_stop,
        completion_reason=decision.reason
    )


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

class GeometricCompletionEngine:
    """
    Main engine for geometric turn completion.
    
    Manages state, computes metrics, and makes stopping decisions.
    """
    
    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.attractor_basins: List[np.ndarray] = []
    
    def create_state(self, initial_basin: Optional[np.ndarray] = None) -> GenerationState:
        """Create new generation state."""
        if initial_basin is None:
            initial_basin = np.random.randn(self.dimension)
            # Use sphere_project for proper normalization
            try:
                from qig_geometry import sphere_project
            except ImportError:
                def sphere_project(v):
                    norm = np.linalg.norm(v)
                    return v / norm if norm > 1e-10 else np.ones_like(v) / np.sqrt(len(v))
            initial_basin = sphere_project(initial_basin)
        
        state = GenerationState(basin=initial_basin)
        state.trajectory.append(initial_basin.copy())
        return state
    
    def update_state(
        self,
        state: GenerationState,
        new_basin: np.ndarray,
        density_matrix: Optional[np.ndarray] = None
    ) -> GeometricMetrics:
        """Update state with new basin and compute metrics."""
        previous_basin = state.basin if len(state.trajectory) > 0 else None
        
        metrics = compute_generation_metrics(
            current_basin=new_basin,
            previous_basin=previous_basin,
            trajectory=state.trajectory,
            density_matrix=density_matrix
        )
        
        state.add_basin(new_basin, metrics)
        state.token_count += 1
        
        return metrics
    
    def check_completion(self, state: GenerationState) -> CompletionDecision:
        """Check if generation should stop."""
        return check_geometric_completion(state, self.attractor_basins)
    
    def assess_quality(self, state: GenerationState, decision: CompletionDecision) -> CompletionQuality:
        """Assess completion quality."""
        return assess_completion_quality(state, decision)
    
    def get_adaptive_temperature(self, state: GenerationState) -> float:
        """Get regime-adaptive temperature."""
        if state.metrics_history:
            return get_adaptive_temperature(state.metrics_history[-1])
        return 0.7
    
    def add_attractor(self, basin: np.ndarray):
        """Add known attractor basin."""
        self.attractor_basins.append(basin.copy())


# Global engine instance
_engine: Optional[GeometricCompletionEngine] = None


def get_completion_engine(dimension: int = 64) -> GeometricCompletionEngine:
    """Get or create the geometric completion engine."""
    global _engine
    if _engine is None:
        _engine = GeometricCompletionEngine(dimension)
    return _engine
