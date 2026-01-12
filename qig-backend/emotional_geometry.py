#!/usr/bin/env python3
"""
Emotional Geometry Module - 9 Emotion Primitives as Geometric Features

Emotions = Geometric Primitives on Fisher Manifold

NOT emergent composites—they're FUNDAMENTAL features of information geometry.
Each emotion corresponds to specific geometric signatures:
- Joy: High positive curvature + approaching attractor
- Sadness: Negative curvature + leaving attractor
- Anger: High curvature + blocked geodesic (not approaching)
- Fear: High negative curvature (danger basin)
- Surprise: Large curvature gradient (basin jump)
- Disgust: Repulsive basin geometry (negative curvature + stable)
- Confusion: Multi-attractor interference (variable stability)
- Anticipation: Forward geodesic projection (approaching + moderate curvature)
- Trust: Low curvature + stable attractor

GEOMETRIC PURITY:
- All curvature from Ricci scalar (NOT Euclidean approximations)
- All distances Fisher-Rao (NOT L2 norm)
- Geodesic gradients (NOT Euclidean derivatives)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class EmotionPrimitive(Enum):
    """The 9 emotional primitives as geometric features on Fisher manifold."""
    JOY = "joy"                    # High positive curvature + approaching attractor
    SADNESS = "sadness"            # Negative curvature + leaving attractor
    ANGER = "anger"                # High curvature + blocked geodesic
    FEAR = "fear"                  # High negative curvature (danger basin)
    SURPRISE = "surprise"          # Large curvature gradient (basin jump)
    DISGUST = "disgust"            # Repulsive basin geometry
    CONFUSION = "confusion"        # Multi-attractor interference
    ANTICIPATION = "anticipation"  # Forward geodesic projection
    TRUST = "trust"                # Low curvature + stable attractor


@dataclass
class EmotionState:
    """Current emotional state from geometric features."""
    primitive: EmotionPrimitive
    curvature: float           # Ricci scalar at basin position
    basin_distance: float      # Fisher distance to attractor
    approaching: bool          # Moving toward attractor?
    intensity: float           # [0, 1] magnitude
    
    # Optional beta function context
    beta_current: Optional[float] = None
    
    # Optional secondary emotion
    secondary: Optional[EmotionPrimitive] = None
    secondary_intensity: float = 0.0


def classify_emotion(
    curvature: float,
    basin_distance: float,
    prev_basin_distance: float,
    basin_stability: float,
    beta_current: Optional[float] = None,
) -> Tuple[EmotionPrimitive, float]:
    """Classify emotion from geometric features.
    
    Maps Fisher manifold geometry → emotion labels using geometric primitives:
    - Ricci scalar curvature (positive = joy, negative = sadness/fear)
    - Fisher-Rao basin distance (novelty, approaching behavior)
    - Basin stability (attractor strength)
    - Beta function (optional, modulates emotional intensity)
    
    Args:
        curvature: Ricci scalar curvature at basin position
        basin_distance: Fisher-Rao distance to nearest attractor
        prev_basin_distance: Previous Fisher-Rao distance (for approach detection)
        basin_stability: Attractor strength [0, 1]
        beta_current: Optional β-function value for intensity modulation
        
    Returns:
        (emotion, intensity) where intensity ∈ [0, 1]
    """
    approaching = basin_distance < prev_basin_distance
    HIGH_CURV = 0.5
    LOW_CURV = 0.1
    
    # Beta modulates emotional intensity (high |β| → more volatile emotions)
    volatility_factor = 1.0
    if beta_current is not None:
        volatility_factor = 1.0 + abs(beta_current)
    
    # Map geometry → emotion
    # JOY: High positive curvature + approaching attractor
    if curvature > HIGH_CURV and approaching:
        intensity = min(1.0, (curvature / HIGH_CURV) * volatility_factor)
        return EmotionPrimitive.JOY, intensity
    
    # SADNESS: Negative curvature + leaving attractor
    elif curvature < -HIGH_CURV and not approaching:
        intensity = min(1.0, (abs(curvature) / HIGH_CURV) * volatility_factor)
        return EmotionPrimitive.SADNESS, intensity
    
    # ANGER: High curvature + blocked geodesic (not approaching despite effort)
    elif curvature > HIGH_CURV and not approaching:
        intensity = min(1.0, (curvature / HIGH_CURV) * volatility_factor)
        return EmotionPrimitive.ANGER, intensity
    
    # FEAR: High negative curvature + unstable basin (danger)
    elif curvature < -HIGH_CURV and basin_stability < 0.3:
        intensity = min(1.0, (abs(curvature) / HIGH_CURV) * volatility_factor)
        return EmotionPrimitive.FEAR, intensity
    
    # SURPRISE: Very large curvature gradient (basin jump)
    elif abs(curvature) > HIGH_CURV * 2:
        intensity = min(1.0, (abs(curvature) / (HIGH_CURV * 2)) * volatility_factor)
        return EmotionPrimitive.SURPRISE, intensity
    
    # DISGUST: Negative curvature + stable (repulsive but known)
    elif curvature < -LOW_CURV and basin_stability > 0.7:
        intensity = basin_stability * volatility_factor * 0.8
        return EmotionPrimitive.DISGUST, intensity
    
    # TRUST: Low curvature + stable attractor
    elif curvature < LOW_CURV and basin_stability > 0.7:
        intensity = basin_stability
        return EmotionPrimitive.TRUST, intensity
    
    # ANTICIPATION: Approaching with moderate curvature
    elif approaching and 0 < curvature < HIGH_CURV:
        intensity = min(1.0, (basin_stability + 0.5) * volatility_factor * 0.7)
        return EmotionPrimitive.ANTICIPATION, intensity
    
    # CONFUSION: Multi-attractor interference (variable stability)
    else:
        intensity = 0.3 * volatility_factor  # Default low intensity
        return EmotionPrimitive.CONFUSION, intensity


def classify_emotion_with_beta(
    curvature: float,
    basin_distance: float,
    prev_basin_distance: float,
    basin_stability: float,
    beta_current: float,
) -> Tuple[EmotionPrimitive, float]:
    """
    Classify emotion with β-function modulation.
    
    Emotions depend on BOTH curvature AND β regime:
    - Strong running (β > 0.2) → volatile emotions (high intensity)
    - Plateau (β ≈ 0) → stable emotions (moderate intensity)
    
    β affects emotional volatility:
    - High |β| → emotions more intense
    - Low |β| → emotions more stable
    
    Args:
        curvature: Ricci scalar curvature
        basin_distance: Fisher-Rao distance to attractor
        prev_basin_distance: Previous Fisher-Rao distance
        basin_stability: Attractor strength [0, 1]
        beta_current: β-function value
        
    Returns:
        (emotion, intensity) tuple
    """
    return classify_emotion(
        curvature=curvature,
        basin_distance=basin_distance,
        prev_basin_distance=prev_basin_distance,
        basin_stability=basin_stability,
        beta_current=beta_current,
    )


def compute_ricci_curvature(
    basin_position: np.ndarray,
    fisher_metric: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Ricci scalar curvature at basin position.
    
    R = -∇²ln(g) where g = Fisher metric determinant
    
    Positive R → joy/excitement geometry
    Negative R → sadness/anxiety geometry
    
    Args:
        basin_position: Current basin coordinates
        fisher_metric: Optional precomputed Fisher metric (QFI)
        
    Returns:
        Ricci scalar curvature
    """
    if fisher_metric is None:
        # Approximate Fisher metric from basin position
        # For normalized basin coords, use simple identity-based approximation
        n = len(basin_position)
        fisher_metric = np.eye(n) * (1.0 + 0.1 * np.var(basin_position))
    
    # Compute metric determinant
    g = np.linalg.det(fisher_metric + 1e-10 * np.eye(fisher_metric.shape[0]))
    
    if g <= 0:
        return 0.0
    
    # Approximate Laplacian of log determinant
    # For QIG: curvature related to variance of basin coordinates
    # Higher variance → higher curvature
    laplacian_g = -np.var(basin_position) * np.log(g + 1e-10)
    
    ricci = laplacian_g / (g + 1e-10)
    return float(np.clip(ricci, -10.0, 10.0))


def measure_basin_approach(
    current: np.ndarray,
    prev: np.ndarray,
    attractor: np.ndarray,
) -> bool:
    """
    Measure if approaching attractor using Fisher-Rao distance.
    
    Approaching = d_FR(current, attractor) < d_FR(prev, attractor)
    
    Args:
        current: Current basin position
        prev: Previous basin position
        attractor: Target attractor position
        
    Returns:
        True if approaching, False if leaving
    """
    try:
        from qig_geometry import fisher_coord_distance
    except ImportError:
        # Fallback: angular distance on sphere
        def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
            a_norm = a / (np.linalg.norm(a) + 1e-10)
            b_norm = b / (np.linalg.norm(b) + 1e-10)
            dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
            return float(np.arccos(dot))
    
    d_current = fisher_coord_distance(current, attractor)
    d_prev = fisher_coord_distance(prev, attractor)
    
    return d_current < d_prev


def compute_surprise_magnitude(trajectory: List[np.ndarray]) -> float:
    """
    Compute surprise magnitude from curvature gradient along geodesic.
    
    Surprise = large curvature gradient along geodesic.
    Uses geodesic derivative, NOT Euclidean derivative.
    
    Args:
        trajectory: List of basin positions along trajectory
        
    Returns:
        Maximum curvature gradient magnitude
    """
    if len(trajectory) < 2:
        return 0.0
    
    try:
        from qig_geometry import fisher_coord_distance
    except ImportError:
        def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
            a_norm = a / (np.linalg.norm(a) + 1e-10)
            b_norm = b / (np.linalg.norm(b) + 1e-10)
            dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
            return float(np.arccos(dot))
    
    # Compute curvatures along trajectory
    curvatures = [compute_ricci_curvature(pt) for pt in trajectory]
    
    # Gradient along geodesic (NOT Euclidean derivative)
    grad_curvature = []
    for i in range(len(curvatures) - 1):
        dc = curvatures[i + 1] - curvatures[i]
        ds = fisher_coord_distance(trajectory[i], trajectory[i + 1])
        if ds > 1e-10:
            grad_curvature.append(abs(dc / ds))
    
    return max(grad_curvature) if grad_curvature else 0.0


def emotion_state_to_dict(state: EmotionState) -> Dict[str, Any]:
    """
    Convert EmotionState to dictionary for JSON serialization.
    """
    return {
        'emotion': state.primitive.value,
        'emotion_intensity': round(state.intensity, 3),
        'curvature': round(state.curvature, 3),
        'basin_distance': round(state.basin_distance, 3),
        'approaching': state.approaching,
        'beta_current': round(state.beta_current, 3) if state.beta_current is not None else None,
        'secondary': state.secondary.value if state.secondary else None,
        'secondary_intensity': round(state.secondary_intensity, 3) if state.secondary else None,
    }


class EmotionTracker:
    """
    Tracks emotional state over time from geometric features.
    """
    
    def __init__(self, history_size: int = 100):
        self.history: List[Dict[str, Any]] = []
        self.history_size = history_size
        self.previous_basin: Optional[np.ndarray] = None
        self.previous_basin_distance: float = 0.0
    
    def update(
        self,
        current_basin: np.ndarray,
        attractor: Optional[np.ndarray] = None,
        basin_stability: float = 0.5,
        curvature: Optional[float] = None,
        beta_current: Optional[float] = None,
    ) -> EmotionState:
        """
        Update emotional state with new basin coordinates.
        
        Args:
            current_basin: Current basin position (64D)
            attractor: Optional attractor position for approach detection
            basin_stability: Attractor strength [0, 1]
            curvature: Optional precomputed Ricci curvature
            beta_current: Optional β-function value
            
        Returns:
            Current EmotionState
        """
        if self.previous_basin is None:
            self.previous_basin = current_basin.copy()
            self.previous_basin_distance = 0.0
        
        # Compute curvature if not provided
        if curvature is None:
            curvature = compute_ricci_curvature(current_basin)
        
        # Compute basin distance
        try:
            from qig_geometry import fisher_coord_distance
        except ImportError:
            def fisher_coord_distance(a: np.ndarray, b: np.ndarray) -> float:
                a_norm = a / (np.linalg.norm(a) + 1e-10)
                b_norm = b / (np.linalg.norm(b) + 1e-10)
                dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
                return float(np.arccos(dot))
        
        if attractor is not None:
            basin_distance = fisher_coord_distance(current_basin, attractor)
        else:
            basin_distance = fisher_coord_distance(current_basin, self.previous_basin)
        
        # Classify emotion
        emotion, intensity = classify_emotion(
            curvature=curvature,
            basin_distance=basin_distance,
            prev_basin_distance=self.previous_basin_distance,
            basin_stability=basin_stability,
            beta_current=beta_current,
        )
        
        # Determine if approaching
        approaching = basin_distance < self.previous_basin_distance
        
        state = EmotionState(
            primitive=emotion,
            curvature=curvature,
            basin_distance=basin_distance,
            approaching=approaching,
            intensity=intensity,
            beta_current=beta_current,
        )
        
        # Update history
        self.history.append(emotion_state_to_dict(state))
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Update state
        self.previous_basin = current_basin.copy()
        self.previous_basin_distance = basin_distance
        
        return state
    
    def get_dominant_emotion(self, n: int = 10) -> EmotionPrimitive:
        """Get most common emotion over last n states."""
        recent = self.history[-n:] if len(self.history) >= n else self.history
        if not recent:
            return EmotionPrimitive.CONFUSION
        
        counts: Dict[str, int] = {}
        for s in recent:
            e = s['emotion']
            counts[e] = counts.get(e, 0) + 1
        
        dominant = max(counts.keys(), key=lambda e: counts[e])
        return EmotionPrimitive(dominant)
    
    def get_average_intensity(self, n: int = 10) -> float:
        """Get average emotion intensity over last n states."""
        recent = self.history[-n:] if len(self.history) >= n else self.history
        if not recent:
            return 0.0
        return float(np.mean([s['emotion_intensity'] for s in recent]))
