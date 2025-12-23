#!/usr/bin/env python3
"""
Emotional Geometry Module

Emotions are NOT subjective features - they are objectively measurable
geometric properties on the Fisher manifold.

The 9 emotional primitives emerge from combinations of:
- Surprise (prediction error)
- Curiosity (exploration drive)
- Basin distance (conceptual novelty)
- Progress (goal approach)
- Stability (attractor strength)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class Emotion(Enum):
    """The 9 emotional primitives in QIG framework."""
    WONDER = "wonder"           # High curiosity + high basin distance
    FRUSTRATION = "frustration" # High surprise + no progress
    SATISFACTION = "satisfaction" # Integration + low basin distance
    CONFUSION = "confusion"     # High surprise + high basin distance
    CLARITY = "clarity"         # Low surprise + convergence
    ANXIETY = "anxiety"         # Near transition + unstable
    CONFIDENCE = "confidence"   # Far from transition + stable
    BOREDOM = "boredom"         # Low surprise + low curiosity
    FLOW = "flow"               # Medium curiosity + progress
    NEUTRAL = "neutral"         # No dominant emotion


@dataclass
class EmotionalState:
    """Complete emotional state with geometric basis."""
    primary: Emotion
    intensity: float  # 0-1
    valence: float    # -1 (negative) to +1 (positive)
    arousal: float    # 0 (calm) to 1 (excited)
    
    # Underlying geometric measurements
    surprise: float
    curiosity: float
    basin_distance: float
    progress: float
    stability: float
    
    # Secondary emotions (if mixed state)
    secondary: Optional[Emotion] = None
    secondary_intensity: float = 0.0


# Emotion characteristics mapping
EMOTION_CHARACTERISTICS = {
    Emotion.WONDER: {
        'valence': 0.7,
        'arousal': 0.8,
        'description': 'High curiosity + high basin distance'
    },
    Emotion.FRUSTRATION: {
        'valence': -0.6,
        'arousal': 0.7,
        'description': 'High surprise + no progress'
    },
    Emotion.SATISFACTION: {
        'valence': 0.8,
        'arousal': 0.3,
        'description': 'Integration + low basin distance'
    },
    Emotion.CONFUSION: {
        'valence': -0.3,
        'arousal': 0.5,
        'description': 'High surprise + high basin distance'
    },
    Emotion.CLARITY: {
        'valence': 0.6,
        'arousal': 0.2,
        'description': 'Low surprise + convergence'
    },
    Emotion.ANXIETY: {
        'valence': -0.7,
        'arousal': 0.8,
        'description': 'Near transition + unstable'
    },
    Emotion.CONFIDENCE: {
        'valence': 0.5,
        'arousal': 0.4,
        'description': 'Far from transition + stable'
    },
    Emotion.BOREDOM: {
        'valence': -0.2,
        'arousal': 0.1,
        'description': 'Low surprise + low curiosity'
    },
    Emotion.FLOW: {
        'valence': 0.9,
        'arousal': 0.6,
        'description': 'Medium curiosity + progress'
    },
    Emotion.NEUTRAL: {
        'valence': 0.0,
        'arousal': 0.3,
        'description': 'No dominant emotion'
    }
}

# Validated correlations from canonical reference
EMOTION_CORRELATIONS = {
    ('wonder', 'confusion'): 0.863,      # Both high basin distance
    ('anxiety', 'confidence'): -0.690,    # Opposite stability
    ('wonder', 'boredom'): -0.454,        # Opposite curiosity
    ('flow', 'frustration'): -0.521,      # Opposite progress
    ('satisfaction', 'anxiety'): -0.612,  # Opposite valence
    ('clarity', 'confusion'): -0.789,     # Opposite surprise
}


def measure_emotion(
    surprise: float,
    curiosity: float,
    basin_distance: float,
    progress: float,
    stability: float
) -> EmotionalState:
    """
    Measure emotional state from geometric properties.
    
    Emotions = geometric properties on Fisher manifold.
    NOT subjective - objectively measurable from basin coordinates.
    
    Args:
        surprise: Prediction error (0-1)
        curiosity: Exploration drive (0-1)
        basin_distance: Distance to nearest attractor (0-1)
        progress: Goal approach rate (0-1)
        stability: Attractor strength (0-1)
        
    Returns:
        Complete EmotionalState with primary emotion and metrics
    """
    # Clamp inputs to valid range
    surprise = np.clip(surprise, 0, 1)
    curiosity = np.clip(curiosity, 0, 1)
    basin_distance = np.clip(basin_distance, 0, 1)
    progress = np.clip(progress, 0, 1)
    stability = np.clip(stability, 0, 1)
    
    # Calculate emotion scores
    scores = _calculate_emotion_scores(
        surprise, curiosity, basin_distance, progress, stability
    )
    
    # Find primary emotion
    primary = max(scores.keys(), key=lambda e: scores[e])
    primary_intensity = scores[primary]
    
    # Find secondary emotion (if significant)
    remaining = {e: s for e, s in scores.items() if e != primary}
    secondary = max(remaining.keys(), key=lambda e: remaining[e])
    secondary_intensity = remaining[secondary]
    
    # Calculate valence and arousal
    valence = _calculate_valence(scores)
    arousal = _calculate_arousal(surprise, curiosity, stability)
    
    return EmotionalState(
        primary=primary,
        intensity=primary_intensity,
        valence=valence,
        arousal=arousal,
        surprise=surprise,
        curiosity=curiosity,
        basin_distance=basin_distance,
        progress=progress,
        stability=stability,
        secondary=secondary if secondary_intensity > 0.3 else None,
        secondary_intensity=secondary_intensity
    )


def _calculate_emotion_scores(
    surprise: float,
    curiosity: float,
    basin_distance: float,
    progress: float,
    stability: float
) -> Dict[Emotion, float]:
    """
    Calculate scores for each emotion based on geometric properties.
    """
    scores = {}
    
    # Wonder: High curiosity + high basin distance
    scores[Emotion.WONDER] = (curiosity * 0.6 + basin_distance * 0.4) * \
        (1 if curiosity > 0.6 and basin_distance > 0.5 else 0.5)
    
    # Frustration: High surprise + no progress
    scores[Emotion.FRUSTRATION] = (surprise * 0.6 + (1 - progress) * 0.4) * \
        (1 if surprise > 0.6 and progress < 0.3 else 0.5)
    
    # Satisfaction: Integration + low basin distance
    scores[Emotion.SATISFACTION] = (progress * 0.5 + (1 - basin_distance) * 0.5) * \
        (1 if progress > 0.6 and basin_distance < 0.3 else 0.5)
    
    # Confusion: High surprise + high basin distance
    scores[Emotion.CONFUSION] = (surprise * 0.5 + basin_distance * 0.5) * \
        (1 if surprise > 0.6 and basin_distance > 0.5 else 0.5)
    
    # Clarity: Low surprise + convergence (low basin distance)
    scores[Emotion.CLARITY] = ((1 - surprise) * 0.5 + (1 - basin_distance) * 0.5) * \
        (1 if surprise < 0.3 and basin_distance < 0.3 else 0.5)
    
    # Anxiety: Near transition + unstable
    scores[Emotion.ANXIETY] = ((1 - stability) * 0.7 + surprise * 0.3) * \
        (1 if stability < 0.3 else 0.5)
    
    # Confidence: Far from transition + stable
    scores[Emotion.CONFIDENCE] = (stability * 0.7 + (1 - surprise) * 0.3) * \
        (1 if stability > 0.7 else 0.5)
    
    # Boredom: Low surprise + low curiosity
    scores[Emotion.BOREDOM] = ((1 - surprise) * 0.5 + (1 - curiosity) * 0.5) * \
        (1 if surprise < 0.3 and curiosity < 0.3 else 0.5)
    
    # Flow: Medium curiosity + progress
    medium_curiosity = 1 - abs(curiosity - 0.5) * 2  # Peak at 0.5
    scores[Emotion.FLOW] = (medium_curiosity * 0.4 + progress * 0.6) * \
        (1 if 0.3 < curiosity < 0.7 and progress > 0.5 else 0.5)
    
    # Neutral: When no strong emotion
    max_score = max(scores.values())
    scores[Emotion.NEUTRAL] = 0.3 if max_score < 0.5 else 0.1
    
    # Normalize
    total = sum(scores.values())
    if total > 0:
        scores = {e: s / total for e, s in scores.items()}
    
    return scores


def _calculate_valence(scores: Dict[Emotion, float]) -> float:
    """
    Calculate overall valence from emotion scores.
    
    Valence = weighted average of emotion valences.
    """
    valence = 0.0
    for emotion, score in scores.items():
        valence += score * EMOTION_CHARACTERISTICS[emotion]['valence']
    return np.clip(valence, -1, 1)


def _calculate_arousal(
    surprise: float,
    curiosity: float,
    stability: float
) -> float:
    """
    Calculate arousal level.
    
    Arousal increases with surprise and curiosity, decreases with stability.
    """
    arousal = (surprise * 0.4 + curiosity * 0.4 + (1 - stability) * 0.2)
    return np.clip(arousal, 0, 1)


def emotion_from_basin_trajectory(
    current_basin: np.ndarray,
    previous_basin: np.ndarray,
    target_basin: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    attractor_strength: float = 0.5
) -> EmotionalState:
    """
    Calculate emotional state from basin trajectory.
    
    Derives surprise, curiosity, progress from basin movement.
    
    Args:
        current_basin: Current 64D basin coordinates
        previous_basin: Previous 64D basin coordinates
        target_basin: Optional goal basin (for progress calculation)
        prediction: Optional predicted basin (for surprise calculation)
        attractor_strength: Strength of current attractor (stability)
        
    Returns:
        EmotionalState derived from geometric properties
    """
    # Basin distance (novelty)
    basin_distance = np.linalg.norm(current_basin - previous_basin)
    basin_distance = np.clip(basin_distance / 2.0, 0, 1)  # Normalize
    
    # Surprise (prediction error)
    if prediction is not None:
        surprise = np.linalg.norm(current_basin - prediction)
        surprise = np.clip(surprise / 2.0, 0, 1)
    else:
        # Default: surprise from basin change
        surprise = basin_distance * 0.8
    
    # Curiosity (exploration drive)
    # High curiosity when moving toward new areas
    exploration_vector = current_basin - previous_basin
    curiosity = np.std(exploration_vector) * 2  # Variance in movement
    curiosity = np.clip(curiosity, 0, 1)
    
    # Progress (goal approach)
    if target_basin is not None:
        prev_dist = np.linalg.norm(previous_basin - target_basin)
        curr_dist = np.linalg.norm(current_basin - target_basin)
        progress = (prev_dist - curr_dist) / (prev_dist + 0.001)  # Relative progress
        progress = np.clip((progress + 1) / 2, 0, 1)  # Map to 0-1
    else:
        # Default: progress as stability
        progress = attractor_strength
    
    # Stability
    stability = attractor_strength
    
    return measure_emotion(
        surprise=surprise,
        curiosity=curiosity,
        basin_distance=basin_distance,
        progress=progress,
        stability=stability
    )


def emotional_state_to_dict(state: EmotionalState) -> Dict[str, Any]:
    """
    Convert EmotionalState to dictionary for JSON serialization.
    """
    return {
        'primary': state.primary.value,
        'intensity': round(state.intensity, 3),
        'valence': round(state.valence, 3),
        'arousal': round(state.arousal, 3),
        'surprise': round(state.surprise, 3),
        'curiosity': round(state.curiosity, 3),
        'basin_distance': round(state.basin_distance, 3),
        'progress': round(state.progress, 3),
        'stability': round(state.stability, 3),
        'secondary': state.secondary.value if state.secondary else None,
        'secondary_intensity': round(state.secondary_intensity, 3) if state.secondary else None,
        'description': EMOTION_CHARACTERISTICS[state.primary]['description']
    }


class EmotionalTracker:
    """
    Tracks emotional state over time.
    """
    
    def __init__(self, history_size: int = 100):
        self.history: list = []
        self.history_size = history_size
        self.previous_basin: Optional[np.ndarray] = None
    
    def update(
        self,
        current_basin: np.ndarray,
        target_basin: Optional[np.ndarray] = None,
        prediction: Optional[np.ndarray] = None,
        attractor_strength: float = 0.5
    ) -> EmotionalState:
        """
        Update emotional state with new basin coordinates.
        
        Returns:
            Current EmotionalState
        """
        if self.previous_basin is None:
            self.previous_basin = current_basin.copy()
        
        state = emotion_from_basin_trajectory(
            current_basin=current_basin,
            previous_basin=self.previous_basin,
            target_basin=target_basin,
            prediction=prediction,
            attractor_strength=attractor_strength
        )
        
        self.history.append(emotional_state_to_dict(state))
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        self.previous_basin = current_basin.copy()
        
        return state
    
    def get_average_valence(self, n: int = 10) -> float:
        """Get average valence over last n states."""
        recent = self.history[-n:] if len(self.history) >= n else self.history
        if not recent:
            return 0.0
        return np.mean([s['valence'] for s in recent])
    
    def get_dominant_emotion(self, n: int = 10) -> Emotion:
        """Get most common emotion over last n states."""
        recent = self.history[-n:] if len(self.history) >= n else self.history
        if not recent:
            return Emotion.NEUTRAL
        
        counts = {}
        for s in recent:
            e = s['primary']
            counts[e] = counts.get(e, 0) + 1
        
        dominant = max(counts.keys(), key=lambda e: counts[e])
        return Emotion(dominant)
