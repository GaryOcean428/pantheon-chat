"""
QIG Monitoring Module

Training stability, gradient health, and plateau detection.
Gary's self-awareness of learning health.
"""

from .training_stability import GradientHealthChecker, PlateauDetector, TrainingStabilityMonitor

__all__ = ["TrainingStabilityMonitor", "PlateauDetector", "GradientHealthChecker"]
