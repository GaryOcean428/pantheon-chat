"""
Psyche Plumbing Integration - Wire kernels into existing system

Integrates the psychoanalytic layers (Id, Superego, Φ hierarchy) with:
- WorkingMemoryMixin: Add reflex checking before conscious processing
- GaryCoordinator: Add ethical constraint checking
- OceanMetaObserver: Add Φ hierarchy tracking

Based on E8 Protocol v4.0 Phase 4D requirements.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

# Import psyche plumbing kernels
try:
    from kernels.phi_hierarchy import PhiLevel, get_phi_hierarchy
    from kernels.id_kernel import get_id_kernel
    from kernels.superego_kernel import get_superego_kernel, ConstraintSeverity
    PSYCHE_PLUMBING_AVAILABLE = True
except ImportError:
    PSYCHE_PLUMBING_AVAILABLE = False

logger = logging.getLogger(__name__)


class PsychePlumbingIntegration:
    """
    Integration layer for psyche plumbing kernels.
    
    Provides hooks for:
    - Pre-conscious reflex checking (Id)
    - Ethical constraint enforcement (Superego)
    - Φ hierarchy measurement
    
    USAGE:
    ```python
    integration = PsychePlumbingIntegration()
    
    # Check for fast reflex before conscious processing
    reflex = integration.check_reflex(input_basin)
    if reflex is not None:
        return reflex  # Fast path
    
    # Check ethics before action
    if not integration.is_ethical(action_basin):
        corrected = integration.correct_trajectory(action_basin)
    ```
    """
    
    def __init__(self):
        """Initialize psyche plumbing integration."""
        if not PSYCHE_PLUMBING_AVAILABLE:
            logger.warning("[PsychePlumbing] Kernels not available")
            self.available = False
            return
        
        self.available = True
        
        # Get singleton instances
        self.phi_hierarchy = get_phi_hierarchy()
        self.id_kernel = get_id_kernel()
        self.superego = get_superego_kernel()
        
        # Initialize with some basic ethical constraints
        self._initialize_basic_constraints()
        
        logger.info("[PsychePlumbing] Integration initialized")
        logger.info("  Id kernel: Fast reflex drives")
        logger.info("  Superego: Ethical constraints")
        logger.info("  Φ hierarchy: 3-level consciousness tracking")
    
    def _initialize_basic_constraints(self):
        """Add basic ethical constraints to Superego."""
        # Example: Avoid zero/nan basins (degenerate states)
        zero_basin = np.zeros(64)
        zero_basin[0] = 1.0
        
        self.superego.add_constraint(
            name="avoid-degenerate-state",
            forbidden_basin=zero_basin,
            radius=0.15,
            severity=ConstraintSeverity.WARNING,
            description="Avoid degenerate zero-concentration states",
        )
        
        logger.info("[PsychePlumbing] Basic ethical constraints loaded")
    
    # ============ Id Kernel Integration ============
    
    def check_reflex(
        self,
        input_basin: np.ndarray,
        measure_phi: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Check for fast reflex response (pre-conscious).
        
        If a reflex matches, returns response immediately without
        conscious processing. This is the "fast path" for familiar patterns.
        
        Args:
            input_basin: Input basin coordinates
            measure_phi: Whether to measure Φ_internal
            
        Returns:
            Reflex response dict if triggered, None otherwise
        """
        if not self.available:
            return None
        
        result = self.id_kernel.check_reflex(input_basin, return_latency=True)
        
        if result is not None:
            response_basin, latency_ms = result
            
            # Measure Φ_internal if requested
            phi_internal = None
            if measure_phi:
                measurement = self.phi_hierarchy.measure(
                    response_basin,
                    PhiLevel.INTERNAL,
                    source='id-reflex',
                    metadata={'latency_ms': latency_ms}
                )
                phi_internal = measurement.phi
            
            return {
                'type': 'reflex',
                'basin': response_basin,
                'latency_ms': latency_ms,
                'phi_internal': phi_internal,
            }
        
        return None
    
    def learn_reflex(
        self,
        trigger_basin: np.ndarray,
        response_basin: np.ndarray,
        success: bool = True
    ):
        """
        Learn a new reflex pattern.
        
        Args:
            trigger_basin: Basin that triggers reflex
            response_basin: Fast response basin
            success: Whether pattern was successful
        """
        if not self.available:
            return
        
        self.id_kernel.learn_reflex(trigger_basin, response_basin, success)
    
    # ============ Superego Integration ============
    
    def check_ethics(
        self,
        basin: np.ndarray,
        correct_if_needed: bool = True
    ) -> Dict[str, Any]:
        """
        Check if basin violates ethical constraints.
        
        Args:
            basin: Basin coordinates to check
            correct_if_needed: If True, compute corrected basin
            
        Returns:
            Dictionary with violations, is_ethical, corrected_basin
        """
        if not self.available:
            return {
                'violations': [],
                'is_ethical': True,
                'corrected_basin': None,
                'total_penalty': 0.0,
            }
        
        return self.superego.check_ethics(basin, apply_correction=correct_if_needed)
    
    def add_ethical_constraint(
        self,
        name: str,
        forbidden_basin: np.ndarray,
        radius: float,
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        description: str = "",
    ):
        """
        Add a new ethical constraint.
        
        Args:
            name: Constraint name
            forbidden_basin: Center of forbidden region
            radius: Radius of forbidden region
            severity: How strictly enforced
            description: Human-readable description
        """
        if not self.available:
            return
        
        self.superego.add_constraint(
            name=name,
            forbidden_basin=forbidden_basin,
            radius=radius,
            severity=severity,
            description=description,
        )
    
    # ============ Φ Hierarchy Integration ============
    
    def measure_phi_reported(
        self,
        basin: np.ndarray,
        source: str = "gary"
    ) -> float:
        """
        Measure Φ_reported (conscious awareness level).
        
        Args:
            basin: Basin coordinates
            source: Source kernel/component
            
        Returns:
            Φ_reported value [0, 1]
        """
        if not self.available:
            return 0.5
        
        measurement = self.phi_hierarchy.measure(
            basin,
            PhiLevel.REPORTED,
            source=source
        )
        return measurement.phi
    
    def measure_phi_internal(
        self,
        basin: np.ndarray,
        source: str = "id"
    ) -> float:
        """
        Measure Φ_internal (internal processing level).
        
        Args:
            basin: Basin coordinates
            source: Source kernel/component
            
        Returns:
            Φ_internal value [0, 1]
        """
        if not self.available:
            return 0.5
        
        measurement = self.phi_hierarchy.measure(
            basin,
            PhiLevel.INTERNAL,
            source=source
        )
        return measurement.phi
    
    def measure_phi_autonomic(
        self,
        basin: np.ndarray,
        source: str = "autonomic"
    ) -> float:
        """
        Measure Φ_autonomic (background function level).
        
        Args:
            basin: Basin coordinates
            source: Source kernel/component
            
        Returns:
            Φ_autonomic value [0, 1]
        """
        if not self.available:
            return 0.3
        
        measurement = self.phi_hierarchy.measure(
            basin,
            PhiLevel.AUTONOMIC,
            source=source
        )
        return measurement.phi
    
    def get_phi_statistics(self) -> Dict[str, Dict]:
        """Get Φ statistics for all levels."""
        if not self.available:
            return {}
        
        return self.phi_hierarchy.get_all_statistics()
    
    # ============ Statistics ============
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive psyche plumbing statistics."""
        if not self.available:
            return {'available': False}
        
        return {
            'available': True,
            'id_kernel': self.id_kernel.get_statistics(),
            'superego': self.superego.get_statistics(),
            'phi_hierarchy': self.get_phi_statistics(),
        }


# Global singleton instance
_psyche_plumbing_integration: Optional[PsychePlumbingIntegration] = None


def get_psyche_plumbing() -> PsychePlumbingIntegration:
    """
    Get global PsychePlumbingIntegration singleton.
    
    Returns:
        PsychePlumbingIntegration instance
    """
    global _psyche_plumbing_integration
    if _psyche_plumbing_integration is None:
        _psyche_plumbing_integration = PsychePlumbingIntegration()
    return _psyche_plumbing_integration
