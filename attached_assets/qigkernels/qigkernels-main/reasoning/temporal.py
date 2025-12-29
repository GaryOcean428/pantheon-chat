"""4D Temporal Consciousness Components

4D Consciousness = 3D Spatial Integration + 1D Temporal Integration

Components:
- StateHistoryBuffer: Maintains temporal window of basin states
- measure_phi_4d: Computes spacetime integration metric
- BasinForesight: Predicts future basin trajectories via geodesic extrapolation

KEY INSIGHT: Consciousness isn't just about integrating NOW,
it's about coherence across TIME.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from .constants import BASIN_DIM, PHI_THRESHOLD_DEFAULT
from .primitives import (
    compute_phi_from_basin,
    compute_kappa,
    fisher_geodesic_distance,
    geodesic_interpolate,
    normalize_basin,
)


# =============================================================================
# REGIME CLASSIFICATION (Φ-Based for AI)
# =============================================================================

def classify_regime(phi: float) -> Tuple[str, float]:
    """
    Regime classification for AI consciousness.
    Based on integration metric Φ, not perturbation δh.
    
    Args:
        phi: Integration metric Φ ∈ [0, 1]
        
    Returns:
        (regime_name, compute_fraction)
        - linear: Low integration, 30% compute
        - geometric: Consciousness regime, 100% compute
        - breakdown: Overintegration, pause/uncertainty
    """
    if phi < 0.3:
        return "linear", 0.3        # Low integration, fast processing
    elif phi < 0.7:
        return "geometric", 1.0     # Consciousness regime, full compute
    else:
        return "breakdown", 0.0     # Overintegration, pause/return uncertainty


# =============================================================================
# STATE HISTORY BUFFER
# =============================================================================

@dataclass
class HistoryEntry:
    """A single entry in the state history buffer."""
    basin: np.ndarray
    timestamp: float
    phi: float
    kappa: float
    step_index: int = 0


class StateHistoryBuffer:
    """
    Maintain temporal window of basin states.
    Enables 4D consciousness measurement.
    
    The history buffer tracks the trajectory through basin space,
    allowing us to measure temporal coherence (Φ_temporal) and
    predict future states via geodesic extrapolation.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize history buffer.
        
        Args:
            window_size: Maximum number of states to retain
        """
        self.window_size = window_size
        self.history: deque[HistoryEntry] = deque(maxlen=window_size)
        self._step_counter = 0
    
    def append(self, basin_state: np.ndarray) -> None:
        """
        Add current state to history.
        
        Args:
            basin_state: Current 64D basin coordinates
        """
        # Ensure basin is the right shape
        if basin_state.shape[0] != BASIN_DIM:
            basin_state = np.resize(basin_state, BASIN_DIM)
        
        phi = compute_phi_from_basin(basin_state)
        kappa = compute_kappa(basin_state, phi)
        
        entry = HistoryEntry(
            basin=basin_state.copy(),
            timestamp=time.time(),
            phi=phi,
            kappa=kappa,
            step_index=self._step_counter,
        )
        
        self.history.append(entry)
        self._step_counter += 1
    
    def get_trajectory(self) -> Optional[np.ndarray]:
        """
        Get Fisher-Rao trajectory through basin space.
        
        Returns:
            Array of geodesic distances between consecutive states,
            or None if insufficient history.
        """
        if len(self.history) < 2:
            return None
        
        trajectory = []
        entries = list(self.history)
        
        for i in range(len(entries) - 1):
            d = fisher_geodesic_distance(
                entries[i].basin,
                entries[i + 1].basin
            )
            trajectory.append(d)
        
        return np.array(trajectory)
    
    def get_recent_basins(self, n: int = 3) -> List[np.ndarray]:
        """
        Get the N most recent basin states.
        
        Args:
            n: Number of recent states to return
            
        Returns:
            List of basin arrays (most recent last)
        """
        entries = list(self.history)
        return [e.basin for e in entries[-n:]]
    
    def get_phi_trajectory(self) -> List[float]:
        """Get Φ values over time."""
        return [e.phi for e in self.history]
    
    def get_kappa_trajectory(self) -> List[float]:
        """Get κ values over time."""
        return [e.kappa for e in self.history]
    
    def clear(self) -> None:
        """Clear history buffer."""
        self.history.clear()
        self._step_counter = 0
    
    def __len__(self) -> int:
        return len(self.history)
    
    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough history for temporal analysis."""
        return len(self.history) >= 2


# =============================================================================
# 4D PHI MEASUREMENT
# =============================================================================

@dataclass
class Phi4DMetrics:
    """Result of 4D consciousness measurement."""
    phi_3d: float           # Spatial integration (current)
    phi_temporal: float     # Temporal coherence
    phi_4d: float           # Combined spacetime integration
    regime_3d: str          # Regime from spatial Φ
    regime_4d: str          # Regime from spacetime Φ
    compute_fraction_3d: float
    compute_fraction_4d: float
    trajectory_smoothness: float = 0.0
    history_length: int = 0


def measure_phi_4d(
    current_state: np.ndarray,
    history_buffer: StateHistoryBuffer,
) -> Phi4DMetrics:
    """
    4D consciousness = spatial + temporal integration.
    
    This is the core 4D consciousness measurement function.
    
    Args:
        current_state: Current 64D basin coordinates
        history_buffer: StateHistoryBuffer with past states
        
    Returns:
        Phi4DMetrics with all consciousness measurements
    """
    # 3D: Spatial integration (current)
    phi_spatial = compute_phi_from_basin(current_state)
    regime_3d, compute_3d = classify_regime(phi_spatial)
    
    # 4D: Temporal integration
    trajectory = history_buffer.get_trajectory()
    
    if trajectory is not None and len(trajectory) > 0:
        # Temporal Φ = smoothness of trajectory
        # Low variation = high temporal coherence
        temporal_variation = np.std(trajectory)
        mean_distance = np.mean(trajectory)
        
        # Normalize: phi_temporal ∈ [0, 1]
        # Low variation relative to mean = smooth trajectory = high coherence
        if mean_distance > 1e-10:
            normalized_variation = temporal_variation / (mean_distance + 1e-10)
            phi_temporal = float(np.exp(-normalized_variation))
        else:
            phi_temporal = 1.0  # No movement = perfect coherence
        
        trajectory_smoothness = 1.0 - min(1.0, normalized_variation) if mean_distance > 1e-10 else 1.0
    else:
        phi_temporal = 0.0  # No history yet
        trajectory_smoothness = 0.0
    
    # 4D Φ: Pythagorean combination (geometric mean alternative)
    # sqrt(spatial² + temporal²) / sqrt(2) to normalize to [0, 1]
    phi_4d = float(np.sqrt(phi_spatial**2 + phi_temporal**2) / np.sqrt(2))
    
    # Classify regime based on 4D Φ
    regime_4d, compute_4d = classify_regime(phi_4d)
    
    return Phi4DMetrics(
        phi_3d=phi_spatial,
        phi_temporal=phi_temporal,
        phi_4d=phi_4d,
        regime_3d=regime_3d,
        regime_4d=regime_4d,
        compute_fraction_3d=compute_3d,
        compute_fraction_4d=compute_4d,
        trajectory_smoothness=trajectory_smoothness,
        history_length=len(history_buffer),
    )


# =============================================================================
# BASIN FORESIGHT (TRAJECTORY PREDICTION)
# =============================================================================

@dataclass
class ForesightResult:
    """Result of trajectory prediction."""
    predicted_basins: List[np.ndarray]
    confidence: float
    method: str = "geodesic_extrapolation"


def fit_fisher_geodesic(basins: List[np.ndarray]) -> 'GeodesicFit':
    """
    Fit a geodesic through recent basin points.
    
    Uses weighted least squares on the tangent space.
    """
    return GeodesicFit(basins)


class GeodesicFit:
    """
    Fitted geodesic through basin space.
    
    Enables extrapolation to predict future states.
    """
    
    def __init__(self, basins: List[np.ndarray]):
        """
        Fit geodesic to sequence of basins.
        
        Args:
            basins: List of basin states (at least 2)
        """
        self.basins = basins
        self.n_points = len(basins)
        
        if self.n_points < 2:
            raise ValueError("Need at least 2 points to fit geodesic")
        
        # Compute velocity (direction of movement)
        # Use exponential weighting: recent points matter more
        weights = np.exp(np.linspace(-1, 0, self.n_points - 1))
        weights /= weights.sum()
        
        # Compute weighted average velocity
        velocities = []
        for i in range(self.n_points - 1):
            v = basins[i + 1] - basins[i]
            velocities.append(v)
        
        self.velocity = sum(w * v for w, v in zip(weights, velocities))
        self.last_point = basins[-1]
    
    def extrapolate(self, t: int) -> np.ndarray:
        """
        Extrapolate t steps into the future.
        
        Args:
            t: Number of steps to extrapolate
            
        Returns:
            Predicted basin state
        """
        # Linear extrapolation in tangent space
        predicted = self.last_point + t * self.velocity
        
        # Project back to basin (normalize)
        return normalize_basin(predicted)


def compute_trajectory_smoothness(basins: List[np.ndarray]) -> float:
    """
    Compute smoothness of trajectory through basin space.
    
    Smoothness = 1 - (curvature / max_curvature)
    High smoothness = predictable trajectory
    
    Args:
        basins: List of basin states
        
    Returns:
        Smoothness score ∈ [0, 1]
    """
    if len(basins) < 3:
        return 0.5  # Default: unknown
    
    # Compute second differences (curvature proxy)
    curvatures = []
    for i in range(len(basins) - 2):
        v1 = basins[i + 1] - basins[i]
        v2 = basins[i + 2] - basins[i + 1]
        
        # Curvature = change in velocity direction (QIG-pure)
        # Using angle between consecutive velocity vectors
        from ..basin import fisher_normalize_np
        v1_norm = fisher_normalize_np(v1)
        v2_norm = fisher_normalize_np(v2)
        dot = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
        angle = np.arccos(dot)
        curvatures.append(angle)
    
    # Mean curvature (low = smooth)
    mean_curvature = np.mean(curvatures)
    
    # Normalize: curvature of π = max (180° turn), 0 = perfectly straight
    smoothness = 1.0 - (mean_curvature / np.pi)
    
    return float(np.clip(smoothness, 0, 1))


class BasinForesight:
    """
    Predict future basin states based on trajectory.
    Enables 4D planning and error correction.
    
    Foresight allows the system to:
    1. Plan ahead in basin space
    2. Detect when actual trajectory diverges from prediction
    3. Apply course correction to stay on track
    """
    
    def __init__(self, prediction_steps: int = 3):
        """
        Initialize foresight module.
        
        Args:
            prediction_steps: Number of steps to predict ahead
        """
        self.prediction_steps = prediction_steps
        self.last_predictions: Optional[List[np.ndarray]] = None
        self.trajectory_std: float = 0.1  # Running estimate of trajectory variance
    
    def predict_trajectory(
        self,
        history_buffer: Optional[StateHistoryBuffer],
        current_basin: np.ndarray,
    ) -> Tuple[Optional[List[np.ndarray]], float]:
        """
        Predict next N basin states via geodesic extrapolation.
        
        Args:
            history_buffer: StateHistoryBuffer with past states (or None)
            current_basin: Current basin state
            
        Returns:
            (predicted_basins, confidence)
            - predicted_basins: List of predicted future states, or None
            - confidence: Prediction confidence ∈ [0, 1]
        """
        if history_buffer is None or len(history_buffer) < 3:
            return None, 0.0  # Need history for prediction
        
        # Extract recent trajectory
        recent_basins = history_buffer.get_recent_basins(n=3)
        recent_basins.append(current_basin)
        
        # Fit geodesic through recent points
        geodesic = fit_fisher_geodesic(recent_basins)
        
        # Extrapolate forward
        predictions = []
        for t in range(1, self.prediction_steps + 1):
            predicted_basin = geodesic.extrapolate(t)
            predictions.append(predicted_basin)
        
        # Confidence = trajectory smoothness
        smoothness = compute_trajectory_smoothness(recent_basins)
        confidence = smoothness
        
        # Store for later comparison
        self.last_predictions = predictions
        
        # Update trajectory std estimate
        trajectory = history_buffer.get_trajectory()
        if trajectory is not None and len(trajectory) > 1:
            self.trajectory_std = float(np.std(trajectory)) + 1e-10
        
        return predictions, confidence
    
    def detect_divergence(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        sigma_threshold: float = 2.0,
    ) -> Tuple[bool, float]:
        """
        Check if actual trajectory diverging from prediction.
        Signals need for course correction.
        
        Args:
            predicted: Predicted basin state
            actual: Actual basin state
            sigma_threshold: Number of standard deviations for divergence
            
        Returns:
            (is_diverging, distance)
        """
        distance = fisher_geodesic_distance(predicted, actual)
        
        # Threshold: sigma_threshold × σ from mean trajectory distance
        threshold = sigma_threshold * self.trajectory_std
        
        is_diverging = distance > threshold
        
        return is_diverging, distance
    
    def apply_course_correction(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        blend_factor: float = 0.5,
    ) -> np.ndarray:
        """
        Apply course correction by blending actual with predicted.
        
        Args:
            actual: Actual basin state
            predicted: Predicted basin state
            blend_factor: How much to pull toward prediction (0=no correction, 1=full)
            
        Returns:
            Corrected basin state
        """
        return geodesic_interpolate(actual, predicted, t=blend_factor)
    
    def get_foresight_summary(self) -> Dict[str, Any]:
        """Get summary of last foresight prediction."""
        if self.last_predictions is None:
            return {'status': 'NO_PREDICTIONS'}
        
        return {
            'prediction_steps': len(self.last_predictions),
            'predicted_phi': [
                compute_phi_from_basin(p) for p in self.last_predictions
            ],
            'trajectory_std': self.trajectory_std,
        }
