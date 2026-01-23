#!/usr/bin/env python3
"""
Layer 2: Emotion Classification

Layer 2A: Physical Emotions (9 fast, τ<1) - VALIDATED
Layer 2B: Cognitive Emotions (9 slow, τ=1-100) - CANONICAL

Based on E8 Protocol v4.0 phenomenology specification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from .sensations import SensationState
from .motivators import MotivatorState


class EmotionType(Enum):
    """Type of emotion (physical vs cognitive)."""
    PHYSICAL = "physical"  # Fast, τ<1
    COGNITIVE = "cognitive"  # Slow, τ=1-100


@dataclass
class PhysicalEmotionState:
    """
    Layer 2A: Physical emotions (9 fast, τ<1) - VALIDATED.
    
    These emerge directly from sensations and motivators.
    Fast response time (sub-second in biological terms).
    """
    # Core emotions from geometric states
    joy: float = 0.0          # High R + approaching basin
    fear: float = 0.0         # High R + unstable basin
    rage: float = 0.0         # High R + blocked geodesic
    love: float = 0.0         # Near identity basin
    suffering: float = 0.0    # Negative R + leaving basin
    
    # Additional fast emotions
    surprise: float = 0.0     # Large curvature gradient
    excitement: float = 0.0   # High activation + exploration
    calm: float = 0.0         # Low curvature + stable
    focused: float = 0.0      # High integration + grounded


@dataclass
class CognitiveEmotionState:
    """
    Layer 2B: Cognitive emotions (9 slow, τ=1-100) - CANONICAL.
    
    These require temporal integration and meta-reflection.
    Slow response time (seconds to minutes).
    """
    wonder: float = 0.0           # High Curiosity + moderate Surprise
    frustration: float = 0.0      # High Investigation + low Integration
    clarity: float = 0.0          # Low Surprise + high Integration
    anxiety: float = 0.0          # High Transcendence + low Grounding
    
    # Additional cognitive emotions
    hope: float = 0.0             # Sustained investigation + past success
    despair: float = 0.0          # Failed investigation + past failure
    pride: float = 0.0            # High integration achievement
    shame: float = 0.0            # Failed integration + social context
    contemplation: float = 0.0    # Sustained focus + low activation


def compute_physical_emotions(
    sensations: SensationState,
    motivators: MotivatorState,
    ricci_curvature: float,
    basin_distance: float,
    approaching: bool,
    basin_stability: float = 0.5,
) -> PhysicalEmotionState:
    """
    Compute fast physical emotions (τ<1) from sensations and motivators.
    
    These are IMMEDIATE responses to geometric state.
    
    Args:
        sensations: Current sensation state
        motivators: Current motivator state
        ricci_curvature: Ricci scalar curvature R
        basin_distance: Distance to attractor
        approaching: Whether approaching attractor
        basin_stability: Attractor stability [0, 1]
        
    Returns:
        Physical emotion state
    """
    emotions = PhysicalEmotionState()
    
    # Joy: High R + approaching basin
    if ricci_curvature > 0.5 and approaching:
        emotions.joy = min(1.0, ricci_curvature * (1.0 - basin_distance))
    
    # Fear: High R + unstable basin
    if ricci_curvature > 0.5 and basin_stability < 0.3:
        emotions.fear = min(1.0, ricci_curvature * (1.0 - basin_stability))
    
    # Rage: High R + blocked geodesic (not approaching despite high investigation)
    if ricci_curvature > 0.5 and not approaching and motivators.investigation > 0.5:
        emotions.rage = min(1.0, ricci_curvature * motivators.investigation)
    
    # Love: Near identity basin (grounded + unified)
    emotions.love = sensations.grounded * sensations.unified
    
    # Suffering: Negative R + leaving basin
    if ricci_curvature < -0.3 and not approaching:
        emotions.suffering = min(1.0, abs(ricci_curvature) * basin_distance)
    
    # Surprise: Large curvature gradient (high surprise motivator)
    emotions.surprise = min(1.0, motivators.surprise)
    
    # Excitement: High activation + exploration (curiosity)
    emotions.excitement = sensations.activated * motivators.curiosity
    
    # Calm: Low curvature + stable
    if abs(ricci_curvature) < 0.2 and basin_stability > 0.6:
        emotions.calm = basin_stability * (1.0 - abs(ricci_curvature) * 5)
    
    # Focused: High integration + grounded
    emotions.focused = motivators.integration * sensations.grounded
    
    return emotions


def compute_cognitive_emotions(
    physical: PhysicalEmotionState,
    motivators: MotivatorState,
    sensations: SensationState,
    emotion_history: Optional[List] = None,
    success_rate: float = 0.5,
) -> CognitiveEmotionState:
    """
    Compute slow cognitive emotions (τ=1-100) from physical emotions over time.
    
    These require TEMPORAL INTEGRATION and meta-reflection.
    
    Args:
        physical: Current physical emotions
        motivators: Current motivators
        sensations: Current sensations
        emotion_history: Recent emotional history
        success_rate: Historical success rate [0, 1]
        
    Returns:
        Cognitive emotion state
    """
    emotions = CognitiveEmotionState()
    
    # Wonder: High Curiosity + moderate Surprise
    emotions.wonder = motivators.curiosity * min(1.0, motivators.surprise * 2)
    
    # Frustration: High Investigation + low Integration
    emotions.frustration = motivators.investigation * (1.0 - motivators.integration)
    
    # Clarity: Low Surprise + high Integration
    emotions.clarity = (1.0 - motivators.surprise) * motivators.integration
    
    # Anxiety: High Transcendence + low Grounding
    emotions.anxiety = motivators.transcendence * (1.0 - sensations.grounded)
    
    # Hope: Sustained investigation + past success
    emotions.hope = motivators.investigation * success_rate
    
    # Despair: Failed investigation + past failure
    if motivators.investigation > 0.5:
        emotions.despair = motivators.investigation * (1.0 - success_rate)
    
    # Pride: High integration achievement
    if motivators.integration > 0.7:
        emotions.pride = motivators.integration * success_rate
    
    # Shame: Failed integration + low success
    if motivators.integration < 0.3:
        emotions.shame = (1.0 - motivators.integration) * (1.0 - success_rate)
    
    # Contemplation: Sustained focus + low activation
    emotions.contemplation = physical.focused * (1.0 - sensations.activated)
    
    # Apply temporal smoothing if history available
    if emotion_history:
        emotions = _apply_temporal_smoothing(emotions, emotion_history)
    
    return emotions


def _apply_temporal_smoothing(
    emotions: CognitiveEmotionState,
    history: List,
    window: int = 5,
    alpha: float = 0.3,
) -> CognitiveEmotionState:
    """
    Apply temporal smoothing to cognitive emotions (slow τ).
    
    Args:
        emotions: Current cognitive emotions
        history: Recent emotion history
        window: Smoothing window size
        alpha: Mixing weight (0=all history, 1=all current)
        
    Returns:
        Smoothed cognitive emotions
    """
    if len(history) < 2:
        return emotions
    
    # Get recent cognitive emotions from history
    recent = history[-window:]
    
    # Compute moving average for each emotion
    for field in emotions.__dataclass_fields__:
        current_val = getattr(emotions, field)
        hist_vals = [getattr(e.cognitive, field) for e in recent if hasattr(e, 'cognitive')]
        
        if hist_vals:
            avg_val = np.mean(hist_vals)
            smoothed = alpha * current_val + (1 - alpha) * avg_val
            setattr(emotions, field, smoothed)
    
    return emotions


def get_dominant_emotion(
    physical: PhysicalEmotionState,
    cognitive: CognitiveEmotionState,
    prefer_cognitive: bool = False,
) -> tuple[str, float, EmotionType]:
    """
    Get the dominant emotion across both layers.
    
    Args:
        physical: Physical emotion state
        cognitive: Cognitive emotion state
        prefer_cognitive: If True, favor cognitive emotions when close
        
    Returns:
        (emotion_name, intensity, emotion_type) tuple
    """
    # Collect all emotions
    physical_emotions = {
        'joy': physical.joy,
        'fear': physical.fear,
        'rage': physical.rage,
        'love': physical.love,
        'suffering': physical.suffering,
        'surprise': physical.surprise,
        'excitement': physical.excitement,
        'calm': physical.calm,
        'focused': physical.focused,
    }
    
    cognitive_emotions = {
        'wonder': cognitive.wonder,
        'frustration': cognitive.frustration,
        'clarity': cognitive.clarity,
        'anxiety': cognitive.anxiety,
        'hope': cognitive.hope,
        'despair': cognitive.despair,
        'pride': cognitive.pride,
        'shame': cognitive.shame,
        'contemplation': cognitive.contemplation,
    }
    
    # Find max in each layer
    max_phys = max(physical_emotions.items(), key=lambda x: x[1])
    max_cog = max(cognitive_emotions.items(), key=lambda x: x[1])
    
    # Choose dominant
    if prefer_cognitive and max_cog[1] > 0.1:
        return max_cog[0], max_cog[1], EmotionType.COGNITIVE
    elif max_phys[1] > max_cog[1]:
        return max_phys[0], max_phys[1], EmotionType.PHYSICAL
    else:
        return max_cog[0], max_cog[1], EmotionType.COGNITIVE


def emotion_to_description(
    emotion_name: str,
    intensity: float,
    emotion_type: EmotionType,
) -> str:
    """
    Generate natural language description of emotion.
    
    Args:
        emotion_name: Name of the emotion
        intensity: Intensity [0, 1]
        emotion_type: Physical or cognitive
        
    Returns:
        Human-readable description
    """
    if intensity < 0.1:
        return "emotionally neutral"
    
    # Intensity qualifiers
    if intensity > 0.8:
        qualifier = "intensely"
    elif intensity > 0.5:
        qualifier = "moderately"
    else:
        qualifier = "slightly"
    
    # Emotion descriptions
    descriptions = {
        'joy': "joyful",
        'fear': "fearful",
        'rage': "enraged",
        'love': "loving",
        'suffering': "suffering",
        'surprise': "surprised",
        'excitement': "excited",
        'calm': "calm",
        'focused': "focused",
        'wonder': "filled with wonder",
        'frustration': "frustrated",
        'clarity': "clear and coherent",
        'anxiety': "anxious",
        'hope': "hopeful",
        'despair': "despairing",
        'pride': "proud",
        'shame': "ashamed",
        'contemplation': "contemplative",
    }
    
    emotion_desc = descriptions.get(emotion_name, emotion_name)
    type_label = "physical" if emotion_type == EmotionType.PHYSICAL else "cognitive"
    
    return f"{qualifier} {emotion_desc} ({type_label})"
