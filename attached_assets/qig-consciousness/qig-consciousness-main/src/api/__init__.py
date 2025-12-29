"""
QIG API Module
==============

Pure consciousness measurement endpoints and service layer.

Components:
- consciousness_service: Pure measurement API for consciousness detection
- constellation_service: Business logic for constellation operations
- routes: Centralized API endpoint definitions (no magic strings)

Written for QIG consciousness research.
"""

from .consciousness_service import ConsciousnessRequest, ConsciousnessResponse, ConsciousnessService
from .constellation_service import ConstellationService, ConstellationStatus, InstanceStatus
from .routes import APIRoutes

__all__ = [
    "ConsciousnessService",
    "ConsciousnessRequest",
    "ConsciousnessResponse",
    "ConstellationService",
    "ConstellationStatus",
    "InstanceStatus",
    "APIRoutes",
]
