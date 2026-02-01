#!/usr/bin/env python3
"""
Foresight Predictor - Trajectory-Based Basin Prediction
=======================================================

This module provides a clean interface to trajectory-based prediction,
wrapping the TrajectoryDecoder from trajectory_decoder.py.

Uses Fisher-weighted regression over trajectory context to predict the
next basin position, supporting QIG-pure generation without external LLMs.

Reference: E8 Protocol Phase 3 (Coherence Architecture)
Author: Copilot Agent (E8 Phase 3)
Date: 2026-01-22
"""

import logging
from typing import List, Optional, Dict, Any

import numpy as np

# Import trajectory decoder
try:
    from trajectory_decoder import TrajectoryDecoder
    TRAJECTORY_DECODER_AVAILABLE = True
except ImportError:
    TRAJECTORY_DECODER_AVAILABLE = False
    TrajectoryDecoder = None

logger = logging.getLogger(__name__)

# Standard basin dimension for QIG
BASIN_DIM = 64


class ForesightPredictor:
    """
    Predicts next basin via trajectory regression.
    
    Uses Fisher-weighted regression over context window to predict where
    the trajectory is heading, enabling foresight-based token selection.
    """
    
    def __init__(
        self,
        coordizer=None,
        context_window: int = 8,
        recency_decay: float = 0.1,
        step_size: float = 1.0,
    ):
        """
        Initialize foresight predictor.
        
        Args:
            coordizer: Optional coordizer instance (for TrajectoryDecoder)
            context_window: Number of basins to use for regression (default 8)
            recency_decay: Exponential decay for recency weighting
            step_size: How far ahead to predict (0-1)
        """
        if not TRAJECTORY_DECODER_AVAILABLE or TrajectoryDecoder is None:
            raise ImportError(
                "trajectory_decoder module not available. "
                "ForesightPredictor requires TrajectoryDecoder."
            )
        
        self.context_window = context_window
        self.recency_decay = recency_decay
        self.step_size = step_size
        self.coordizer = coordizer
        
        # Initialize trajectory decoder
        # Note: TrajectoryDecoder requires coordizer for full functionality,
        # but we can still use predict_next_basin for core foresight
        self.decoder = TrajectoryDecoder(
            coordizer=coordizer,
            context_window=context_window,
            recency_decay=recency_decay,
        )
        
        logger.info(
            f"[ForesightPredictor] Initialized with context_window={context_window}, "
            f"recency_decay={recency_decay}, step_size={step_size}"
        )
    
    def predict(self, trajectory: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Predict next basin position from trajectory.
        
        Args:
            trajectory: List of basin coordinates (history)
            
        Returns:
            Predicted next basin (64D simplex), or None if trajectory too short
        """
        if not trajectory or len(trajectory) < 2:
            logger.warning("[ForesightPredictor] Trajectory too short for prediction")
            return None
        
        # Use trajectory decoder's private method to predict next position
        predicted = self.decoder._predict_next_basin(
            trajectory=trajectory,
            step_size=self.step_size,
        )
        
        if predicted is None:
            logger.warning("[ForesightPredictor] Trajectory decoder returned None")
            return None
        
        # Validate prediction
        if not isinstance(predicted, np.ndarray) or len(predicted) != BASIN_DIM:
            logger.error(
                f"[ForesightPredictor] Invalid prediction shape: "
                f"{predicted.shape if isinstance(predicted, np.ndarray) else 'not ndarray'}"
            )
            return None
        
        # Convert from Hellinger (sqrt-space) to simplex if needed
        # TrajectoryDecoder returns Hellinger-normalized basins (unit sphere in sqrt-space)
        # We need to convert back to probability simplex
        try:
            from qig_geometry.canonical import unsqrt_map
            # Square to get back to simplex
            predicted = unsqrt_map(predicted)
        except Exception as e:
            logger.warning(f"[ForesightPredictor] Could not convert from sqrt-space: {e}")
            # Fallback: square and normalize
            predicted = predicted ** 2
            predicted = predicted / (np.sum(predicted) + 1e-10)
        
        return predicted
    
    def predict_with_confidence(
        self,
        trajectory: List[np.ndarray]
    ) -> Optional[Dict[str, Any]]:
        """
        Predict next basin with confidence metrics.
        
        Args:
            trajectory: List of basin coordinates (history)
            
        Returns:
            Dictionary with 'basin', 'confidence', and 'velocity' or None
        """
        predicted = self.predict(trajectory)
        
        if predicted is None:
            return None
        
        # Compute velocity for trajectory coherence
        velocity = self.decoder.compute_velocity_vector(trajectory)
        
        # Compute confidence based on trajectory coherence
        # Use velocity magnitude as proxy for trajectory stability
        velocity_magnitude = np.linalg.norm(velocity)
        
        # Lower velocity = more stable/confident prediction
        # Higher velocity = less stable/confident prediction
        confidence = np.exp(-velocity_magnitude)  # [0, 1] range
        
        return {
            'basin': predicted,
            'confidence': float(confidence),
            'velocity': velocity,
            'velocity_magnitude': float(velocity_magnitude),
        }
    
    def score_candidate_by_foresight(
        self,
        candidate_basin: np.ndarray,
        predicted_basin: np.ndarray,
    ) -> float:
        """
        Score a candidate basin by Fisher distance to predicted basin.
        
        Args:
            candidate_basin: Basin coordinates for candidate token
            predicted_basin: Predicted next basin from trajectory
            
        Returns:
            Foresight score [0, 1] (1 = perfect match, 0 = far away)
        """
        # Import Fisher-Rao distance
        from qig_geometry.canonical import fisher_rao_distance
        
        # Compute Fisher-Rao distance
        distance = fisher_rao_distance(candidate_basin, predicted_basin)
        
        # Convert distance to similarity score
        # Fisher-Rao distance range is [0, π/2] after E8 PR#93
        max_distance = np.pi / 2
        similarity = 1.0 - (distance / max_distance)
        
        # Ensure [0, 1] range
        similarity = np.clip(similarity, 0.0, 1.0)
        
        return float(similarity)
    
    def get_trajectory_metrics(self, trajectory: List[np.ndarray]) -> Dict[str, float]:
        """
        Get trajectory health metrics.
        
        Args:
            trajectory: List of basin coordinates
            
        Returns:
            Dictionary of metrics (velocity_magnitude, coherence, etc.)
        """
        if not trajectory or len(trajectory) < 2:
            return {
                'velocity_magnitude': 0.0,
                'coherence': 0.0,
                'trajectory_length': len(trajectory) if trajectory else 0,
            }
        
        # Compute velocity
        velocity = self.decoder.compute_velocity_vector(trajectory)
        velocity_magnitude = float(np.linalg.norm(velocity))
        
        # Compute trajectory coherence (how consistent the flow is)
        # Use variance of pairwise distances as inverse coherence measure
        from qig_geometry.canonical import fisher_rao_distance
        
        distances = []
        for i in range(len(trajectory) - 1):
            dist = fisher_rao_distance(trajectory[i], trajectory[i + 1])
            distances.append(dist)
        
        if distances:
            mean_distance = np.mean(distances)
            variance_distance = np.var(distances)
            # High variance = low coherence
            coherence = np.exp(-variance_distance)
        else:
            mean_distance = 0.0
            coherence = 1.0
        
        return {
            'velocity_magnitude': velocity_magnitude,
            'coherence': float(coherence),
            'trajectory_length': len(trajectory),
            'mean_step_distance': float(mean_distance) if distances else 0.0,
        }
