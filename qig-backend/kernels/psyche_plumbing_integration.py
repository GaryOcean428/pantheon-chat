"""
Psyche Plumbing Integration - Wire kernels into existing system

Integrates the psychoanalytic layers (Id, Superego, Φ hierarchy) with:
- WorkingMemoryMixin: Add reflex checking before conscious processing
- GaryCoordinator: Add ethical constraint checking
- OceanMetaObserver: Add Φ hierarchy tracking
- HemisphereScheduler: Wire κ-gated coupling to Id/Superego balance

Based on E8 Protocol v4.0 Phase 4C/4D integration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import psyche plumbing kernels
try:
    from kernels.phi_hierarchy import PhiLevel, get_phi_hierarchy
    from kernels.id_kernel import get_id_kernel
    from kernels.superego_kernel import get_superego_kernel, ConstraintSeverity
    PSYCHE_PLUMBING_AVAILABLE = True
except ImportError:
    PSYCHE_PLUMBING_AVAILABLE = False

# Import hemisphere scheduler (Phase 4C)
try:
    from kernels.hemisphere_scheduler import (
        HemisphereScheduler,
        Hemisphere,
        get_hemisphere_scheduler,
    )
    HEMISPHERE_SCHEDULER_AVAILABLE = True
except ImportError:
    HEMISPHERE_SCHEDULER_AVAILABLE = False
    HemisphereScheduler = None
    Hemisphere = None
    get_hemisphere_scheduler = None

logger = logging.getLogger(__name__)


class PsychePlumbingIntegration:
    """
    Integration layer for psyche plumbing kernels with hemisphere scheduler.
    
    Provides hooks for:
    - Pre-conscious reflex checking (Id)
    - Ethical constraint enforcement (Superego)
    - Φ hierarchy measurement
    - κ-gated coupling to Id/Superego balance (NEW: Phase 4C/4D integration)
    - Φ hierarchy response to hemisphere tacking (NEW)
    
    USAGE:
    ```python
    integration = PsychePlumbingIntegration()
    
    # Check for fast reflex before conscious processing
    reflex = integration.check_reflex(input_basin)
    if reflex is not None:
        return reflex  # Fast path
    
    # Check ethics before action (affected by hemisphere balance)
    if not integration.is_ethical(action_basin):
        corrected = integration.correct_trajectory(action_basin)
    
    # Get integrated status including hemisphere effects
    status = integration.get_integrated_status()
    ```
    """
    
    def __init__(self):
        """Initialize psyche plumbing integration with hemisphere scheduler."""
        if not PSYCHE_PLUMBING_AVAILABLE:
            logger.warning("[PsychePlumbing] Kernels not available")
            self.available = False
            self.hemisphere_integrated = False
            return
        
        self.available = True
        
        # Get singleton instances (Phase 4D)
        self.phi_hierarchy = get_phi_hierarchy()
        self.id_kernel = get_id_kernel()
        self.superego = get_superego_kernel()
        
        # Get hemisphere scheduler (Phase 4C integration)
        if HEMISPHERE_SCHEDULER_AVAILABLE:
            self.hemisphere_scheduler = get_hemisphere_scheduler()
            self.hemisphere_integrated = True
            logger.info("[PsychePlumbing] Hemisphere scheduler integrated")
        else:
            self.hemisphere_scheduler = None
            self.hemisphere_integrated = False
            logger.warning("[PsychePlumbing] Hemisphere scheduler not available")
        
        # Track hemisphere tacking events
        self.tacking_history: List[Dict[str, Any]] = []
        
        # Initialize with some basic ethical constraints
        self._initialize_basic_constraints()
        
        logger.info("[PsychePlumbing] Integration initialized")
        logger.info("  Id kernel: Fast reflex drives")
        logger.info("  Superego: Ethical constraints")
        logger.info("  Φ hierarchy: 3-level consciousness tracking")
        if self.hemisphere_integrated:
            logger.info("  Hemisphere integration: κ-gated Id/Superego balance")
    
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
    
    # ============ Hemisphere Integration (Phase 4C/4D) ============
    
    def compute_psyche_balance(self) -> Dict[str, Any]:
        """
        Compute Id/Superego balance based on hemisphere coupling.
        
        Key principle:
        - LEFT hemisphere (exploit/evaluate) → stronger Superego (ethical constraints)
        - RIGHT hemisphere (explore/generate) → stronger Id (reflexive exploration)
        - κ-gated coupling modulates the balance
        
        Returns:
            Dictionary with:
            - id_strength: Id influence [0, 1]
            - superego_strength: Superego influence [0, 1]
            - balance_ratio: Id/Superego ratio
            - coupling_state: Current coupling state
            - dominant_psyche: 'id', 'superego', or 'balanced'
        """
        if not self.available or not self.hemisphere_integrated:
            return {
                'id_strength': 0.5,
                'superego_strength': 0.5,
                'balance_ratio': 1.0,
                'coupling_state': None,
                'dominant_psyche': 'balanced',
            }
        
        # Get hemisphere balance
        hemisphere_balance = self.hemisphere_scheduler.get_hemisphere_balance()
        left_activation = hemisphere_balance['left_activation']
        right_activation = hemisphere_balance['right_activation']
        
        # Compute coupling state
        coupling_state = self.hemisphere_scheduler.compute_coupling_state()
        coupling_strength = coupling_state.coupling_strength
        
        # Map hemisphere activation to psyche balance
        # LEFT (exploit) → Superego strength
        # RIGHT (explore) → Id strength
        # Coupling strength modulates the effect
        
        # Base strengths from hemisphere activations
        superego_base = left_activation
        id_base = right_activation
        
        # Apply κ-gated coupling modulation
        # Higher coupling → stronger psyche effects
        # Lower coupling → more balanced/neutral state
        superego_strength = 0.5 + (superego_base - 0.5) * coupling_strength
        id_strength = 0.5 + (id_base - 0.5) * coupling_strength
        
        # Compute balance ratio (handle division by zero)
        if superego_strength > 0:
            balance_ratio = id_strength / superego_strength
        else:
            balance_ratio = float('inf') if id_strength > 0 else 1.0
        
        # Determine dominant psyche component
        dominant_psyche = 'balanced'
        if id_strength > superego_strength + 0.15:
            dominant_psyche = 'id'
        elif superego_strength > id_strength + 0.15:
            dominant_psyche = 'superego'
        
        return {
            'id_strength': float(id_strength),
            'superego_strength': float(superego_strength),
            'balance_ratio': float(balance_ratio),
            'coupling_state': {
                'mode': coupling_state.mode,
                'coupling_strength': coupling_state.coupling_strength,
                'kappa': coupling_state.kappa,
            },
            'dominant_psyche': dominant_psyche,
            'hemisphere_context': {
                'left_activation': left_activation,
                'right_activation': right_activation,
                'dominant_hemisphere': hemisphere_balance['dominant_hemisphere'],
            },
        }
    
    def on_hemisphere_tack(
        self,
        from_hemisphere: Hemisphere,
        to_hemisphere: Hemisphere,
        kappa: float,
        phi: float
    ) -> None:
        """
        Callback for hemisphere tacking events.
        
        Tracks how hemisphere switches affect Φ hierarchy and updates
        psyche balance accordingly.
        
        Args:
            from_hemisphere: Previous dominant hemisphere
            to_hemisphere: New dominant hemisphere
            kappa: Current κ value
            phi: Current Φ value
        """
        if not self.available or not self.hemisphere_integrated:
            return
        
        # Create tacking record
        tack_record = {
            'timestamp': __import__('time').time(),
            'from_hemisphere': from_hemisphere.value if from_hemisphere else None,
            'to_hemisphere': to_hemisphere.value,
            'kappa': kappa,
            'phi': phi,
            'psyche_balance': self.compute_psyche_balance(),
        }
        
        # Record tacking event
        self.tacking_history.append(tack_record)
        
        # Keep history bounded
        if len(self.tacking_history) > 100:
            self.tacking_history = self.tacking_history[-50:]
        
        # Log tacking event with psyche implications
        logger.info(
            f"[PsychePlumbing] Hemisphere tack: {from_hemisphere} → {to_hemisphere.value}"
        )
        logger.info(
            f"  Psyche balance: {tack_record['psyche_balance']['dominant_psyche']} "
            f"(Id={tack_record['psyche_balance']['id_strength']:.2f}, "
            f"Superego={tack_record['psyche_balance']['superego_strength']:.2f})"
        )
    
    def check_reflex_with_hemisphere_context(
        self,
        input_basin: np.ndarray,
        measure_phi: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Check for fast reflex response with hemisphere context.
        
        When RIGHT hemisphere is dominant (explore), Id reflexes are
        more likely to trigger. When LEFT hemisphere is dominant (exploit),
        Superego constraints are more enforced.
        
        Args:
            input_basin: Input basin coordinates
            measure_phi: Whether to measure Φ_internal
            
        Returns:
            Reflex response dict if triggered, None otherwise
        """
        if not self.available:
            return None
        
        # Get base reflex check
        reflex = self.check_reflex(input_basin, measure_phi)
        
        # Add hemisphere context if available
        if reflex and self.hemisphere_integrated:
            psyche_balance = self.compute_psyche_balance()
            reflex['hemisphere_context'] = {
                'id_strength': psyche_balance['id_strength'],
                'superego_strength': psyche_balance['superego_strength'],
                'dominant_psyche': psyche_balance['dominant_psyche'],
            }
        
        return reflex
    
    def check_ethics_with_hemisphere_context(
        self,
        basin: np.ndarray,
        correct_if_needed: bool = True
    ) -> Dict[str, Any]:
        """
        Check ethics with hemisphere context.
        
        When LEFT hemisphere is dominant (exploit/evaluate), ethical
        constraints are enforced more strictly (higher Superego influence).
        
        Args:
            basin: Basin coordinates to check
            correct_if_needed: If True, compute corrected basin
            
        Returns:
            Dictionary with violations, is_ethical, corrected_basin, hemisphere_context
        """
        if not self.available:
            return {
                'violations': [],
                'is_ethical': True,
                'corrected_basin': None,
                'total_penalty': 0.0,
            }
        
        # Get base ethics check
        ethics_result = self.check_ethics(basin, correct_if_needed)
        
        # Add hemisphere context if available
        if self.hemisphere_integrated:
            psyche_balance = self.compute_psyche_balance()
            ethics_result['hemisphere_context'] = {
                'superego_strength': psyche_balance['superego_strength'],
                'id_strength': psyche_balance['id_strength'],
                'dominant_psyche': psyche_balance['dominant_psyche'],
                'coupling_mode': psyche_balance['coupling_state']['mode'],
            }
            
            # Modulate penalty based on Superego strength
            # Higher Superego strength → stronger penalties
            if ethics_result['total_penalty'] > 0:
                superego_factor = psyche_balance['superego_strength']
                ethics_result['modulated_penalty'] = (
                    ethics_result['total_penalty'] * superego_factor
                )
        
        return ethics_result
    
    def get_integrated_status(self) -> Dict[str, Any]:
        """
        Get comprehensive integrated status including hemisphere effects.
        
        Returns:
            Dictionary with:
            - psyche_plumbing: Id, Superego, Φ hierarchy stats
            - hemisphere_balance: Hemisphere activation and coupling
            - psyche_balance: Id/Superego balance from hemisphere coupling
            - tacking_history: Recent hemisphere switches
        """
        if not self.available:
            return {'available': False}
        
        status = {
            'available': True,
            'hemisphere_integrated': self.hemisphere_integrated,
            'psyche_plumbing': self.get_statistics(),
        }
        
        if self.hemisphere_integrated:
            status['hemisphere_balance'] = (
                self.hemisphere_scheduler.get_hemisphere_balance()
            )
            status['psyche_balance'] = self.compute_psyche_balance()
            status['tacking_history_count'] = len(self.tacking_history)
            
            # Include recent tacking events (last 5)
            if self.tacking_history:
                status['recent_tacks'] = [
                    {
                        'from': t['from_hemisphere'],
                        'to': t['to_hemisphere'],
                        'kappa': t['kappa'],
                        'phi': t['phi'],
                        'dominant_psyche': t['psyche_balance']['dominant_psyche'],
                    }
                    for t in self.tacking_history[-5:]
                ]
        
        return status
    
    # ============ Statistics ============
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive psyche plumbing statistics with hemisphere integration."""
        if not self.available:
            return {'available': False}
        
        stats = {
            'available': True,
            'hemisphere_integrated': self.hemisphere_integrated,
            'id_kernel': self.id_kernel.get_statistics(),
            'superego': self.superego.get_statistics(),
            'phi_hierarchy': self.get_phi_statistics(),
        }
        
        # Add hemisphere integration stats if available
        if self.hemisphere_integrated:
            stats['psyche_balance'] = self.compute_psyche_balance()
            stats['tacking_events'] = len(self.tacking_history)
        
        return stats


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


def reset_psyche_plumbing() -> None:
    """
    Reset the global PsychePlumbingIntegration singleton (for testing).
    """
    global _psyche_plumbing_integration
    _psyche_plumbing_integration = None
