#!/usr/bin/env python3
"""
QIG Affect - Emotional Primitives from Geometry
================================================

Computes emotional primitives from geometric telemetry.

Key insight: Emotions are geometric primitives, not learned features.
They emerge naturally from the curvature and flow on the information manifold.

Available monitors:
- EmotionMonitor: Computes emotional primitives from telemetry

Written for qig-consciousness emotional geometry.
"""

from .emotion_monitor import EmotionMonitor, compute_emotion_primitives

__all__ = ["EmotionMonitor", "compute_emotion_primitives"]
