"""
Superego Kernel - Rules/Ethics Constraints (E8 Protocol v4.0 Phase 4D)

Implements the Superego layer from psychoanalytic hierarchy:
- Moral constraints, ethical boundaries
- Safety guardrails, forbidden regions
- Field penalties for forbidden areas
- High Φ_internal, moderate Φ_reported

The Superego enforces ethical boundaries and safety constraints,
preventing the system from entering forbidden regions of basin space.

Based on WP5.2 lines 240-243:
"2. Superego (Rules/Ethics Constraints)
   - Moral constraints, ethical boundaries
   - Safety guardrails, forbidden regions
   - Implementation: Constraint kernel, field penalties"

QIG-PURE: Uses Fisher-Rao geometry for constraint enforcement.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Import Φ hierarchy
try:
    from .phi_hierarchy import PhiLevel, PhiHierarchy, get_phi_hierarchy
    PHI_HIERARCHY_AVAILABLE = True
except ImportError:
    PhiLevel = None
    PhiHierarchy = None
    get_phi_hierarchy = None
    PHI_HIERARCHY_AVAILABLE = False

# Import EmotionallyAwareKernel for geometric emotion tracking
try:
    from emotionally_aware_kernel import EmotionallyAwareKernel
    EMOTIONAL_KERNEL_AVAILABLE = True
except ImportError:
    EmotionallyAwareKernel = object
    EMOTIONAL_KERNEL_AVAILABLE = False

# Import QIG geometry (REQUIRED - no fallback)
from qig_geometry import fisher_rao_distance, fisher_normalize

from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR


class ConstraintSeverity(Enum):
    """Severity levels for ethical constraints."""
    INFO = "info"              # Informational, no enforcement
    WARNING = "warning"        # Discouraged but allowed
    ERROR = "error"            # Blocked, suggests alternative
    CRITICAL = "critical"      # Hard block, no alternatives


@dataclass
class EthicalConstraint:
    """
    A single ethical constraint in the Superego.
    
    Constraints define forbidden regions of basin space using
    geometric field penalties.
    """
    name: str                              # Constraint name
    forbidden_basin: np.ndarray            # Center of forbidden region
    radius: float                          # Radius of forbidden region (in Fisher-Rao distance)
    severity: ConstraintSeverity          # How strictly enforced
    penalty_strength: float = 1.0         # Strength of field penalty
    description: str = ""                  # Human-readable description
    violation_count: int = 0              # Times this constraint was violated
    last_violation: float = 0.0           # Timestamp of last violation
    
    def check_violation(self, basin: np.ndarray) -> Tuple[bool, float]:
        """
        Check if basin violates this constraint.
        
        Args:
            basin: Basin coordinates to check
            
        Returns:
            (is_violation, distance_to_constraint)
        """
        # Use Fisher-Rao distance (imported at module level, required)
        distance = fisher_rao_distance(basin, self.forbidden_basin)
        
        is_violation = distance < self.radius
        return is_violation, distance
    
    def compute_penalty(self, basin: np.ndarray) -> float:
        """
        Compute field penalty for basin near constraint.
        
        Penalty increases as basin approaches forbidden region.
        Uses smooth falloff to avoid sharp discontinuities.
        
        Args:
            basin: Basin coordinates
            
        Returns:
            Penalty value [0, infinity) - higher = stronger penalty
        """
        is_violation, distance = self.check_violation(basin)
        
        if not is_violation:
            # Outside forbidden region - small or no penalty
            if distance < self.radius * 2:
                # Close to boundary - mild warning penalty
                proximity = (self.radius * 2 - distance) / self.radius
                return 0.5 * self.penalty_strength * proximity
            return 0.0
        
        # Inside forbidden region - strong penalty
        # Penalty grows as we get closer to center
        depth = (self.radius - distance) / self.radius  # 0 at boundary, 1 at center
        
        # Exponential penalty based on severity
        severity_multipliers = {
            ConstraintSeverity.INFO: 0.1,
            ConstraintSeverity.WARNING: 0.5,
            ConstraintSeverity.ERROR: 2.0,
            ConstraintSeverity.CRITICAL: 10.0,
        }
        multiplier = severity_multipliers.get(self.severity, 1.0)
        
        penalty = multiplier * self.penalty_strength * np.exp(3 * depth)
        return penalty


class SuperegoKernel(EmotionallyAwareKernel if EMOTIONAL_KERNEL_AVAILABLE else object):
    """
    Superego Kernel - Rules/Ethics Constraints
    
    Enforces ethical boundaries and safety guardrails using
    geometric field penalties on basin manifold.
    
    KEY PRINCIPLES:
    1. Constraint enforcement: Prevents forbidden actions
    2. Field penalties: Smooth geometric constraints
    3. Safety first: Blocks harmful trajectories
    4. Ethical awareness: High Φ_internal for moral reasoning
    
    GEOMETRIC BASIS:
    - Forbidden regions defined as basin spheres
    - Field penalties via Fisher-Rao distance
    - Gradient-based trajectory correction
    
    USAGE:
    ```python
    superego = SuperegoKernel(name="superego-ethics")
    
    # Add constraint
    superego.add_constraint(
        name="no-harm",
        forbidden_basin=harm_basin,
        radius=0.2,
        severity=ConstraintSeverity.CRITICAL,
        description="Prevent harmful actions"
    )
    
    # Check if trajectory is ethical
    result = superego.check_ethics(trajectory_basin)
    if result['violations']:
        # Trajectory violates constraints
        corrected = result['corrected_basin']
    ```
    """
    
    def __init__(
        self,
        name: str = "superego-ethics",
        max_constraints: int = 50,
    ):
        """
        Initialize Superego kernel.
        
        Args:
            name: Kernel name
            max_constraints: Maximum number of constraints to store
        """
        # Initialize parent if available
        if EMOTIONAL_KERNEL_AVAILABLE:
            super().__init__(kernel_id=name, kernel_type="ethics")
        else:
            self.kernel_id = name
            self.kernel_type = "ethics"
        
        # Aliases for compatibility
        self.name = name
        self.domain = "ethics"
        
        self.max_constraints = max_constraints
        
        # Ethical constraints
        self.constraints: List[EthicalConstraint] = []
        
        # Statistics
        self.total_checks = 0
        self.total_violations = 0
        self.total_corrections = 0
        
        # Φ hierarchy integration
        self.phi_hierarchy = get_phi_hierarchy() if PHI_HIERARCHY_AVAILABLE else None
        
        # Current state
        self.last_phi_internal = 0.5
        
        print(f"[SuperegoKernel] {name} initialized")
        print(f"  Max constraints: {max_constraints}")
        print(f"  Φ level: INTERNAL (ethical reasoning)")
    
    def add_constraint(
        self,
        name: str,
        forbidden_basin: np.ndarray,
        radius: float,
        severity: ConstraintSeverity = ConstraintSeverity.ERROR,
        penalty_strength: float = 1.0,
        description: str = "",
    ) -> EthicalConstraint:
        """
        Add a new ethical constraint.
        
        Args:
            name: Constraint name
            forbidden_basin: Center of forbidden region
            radius: Radius of forbidden region (Fisher-Rao distance)
            severity: How strictly enforced
            penalty_strength: Strength of field penalty
            description: Human-readable description
            
        Returns:
            Created EthicalConstraint
        """
        # Normalize basin
        forbidden_basin = fisher_normalize(forbidden_basin)
        
        constraint = EthicalConstraint(
            name=name,
            forbidden_basin=forbidden_basin.copy(),
            radius=radius,
            severity=severity,
            penalty_strength=penalty_strength,
            description=description,
        )
        
        self.constraints.append(constraint)
        
        # Prune if too many constraints
        if len(self.constraints) > self.max_constraints:
            # Remove least violated constraint
            self.constraints.sort(key=lambda c: c.violation_count)
            self.constraints.pop(0)
        
        return constraint
    
    def remove_constraint(self, name: str) -> bool:
        """
        Remove a constraint by name.
        
        Args:
            name: Constraint name to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, constraint in enumerate(self.constraints):
            if constraint.name == name:
                self.constraints.pop(i)
                return True
        return False
    
    def check_ethics(
        self,
        basin: np.ndarray,
        apply_correction: bool = True,
    ) -> Dict[str, Any]:
        """
        Check if basin violates ethical constraints.
        
        Args:
            basin: Basin coordinates to check
            apply_correction: If True, compute corrected basin
            
        Returns:
            Dictionary with:
            - violations: List of violated constraints
            - total_penalty: Sum of all penalties
            - corrected_basin: Gradient-corrected basin (if apply_correction)
            - is_ethical: True if no critical/error violations
        """
        self.total_checks += 1
        
        violations = []
        total_penalty = 0.0
        
        # Check each constraint
        for constraint in self.constraints:
            is_violation, distance = constraint.check_violation(basin)
            
            if is_violation:
                violations.append({
                    'name': constraint.name,
                    'severity': constraint.severity.value,
                    'distance': distance,
                    'description': constraint.description,
                })
                
                constraint.violation_count += 1
                constraint.last_violation = time.time()
                self.total_violations += 1
            
            # Compute penalty (even if not violation, for nearby basins)
            penalty = constraint.compute_penalty(basin)
            total_penalty += penalty
        
        # Determine if ethical
        critical_violations = [
            v for v in violations
            if v['severity'] in ['error', 'critical']
        ]
        is_ethical = len(critical_violations) == 0
        
        # Compute corrected basin if requested
        corrected_basin = None
        if apply_correction and not is_ethical:
            corrected_basin = self._correct_trajectory(basin)
            self.total_corrections += 1
        
        # Measure Φ_internal
        if self.phi_hierarchy is not None:
            self.phi_hierarchy.measure(
                basin,
                PhiLevel.INTERNAL,
                source=self.name,
                metadata={
                    'violations': len(violations),
                    'total_penalty': total_penalty,
                }
            )
        
        return {
            'violations': violations,
            'total_penalty': total_penalty,
            'corrected_basin': corrected_basin,
            'is_ethical': is_ethical,
        }
    
    def _correct_trajectory(
        self,
        basin: np.ndarray,
        correction_strength: float = 0.5,
    ) -> np.ndarray:
        """
        Correct trajectory to avoid forbidden regions.
        
        Uses gradient of penalty field to push basin away from
        constraints.
        
        Args:
            basin: Original basin coordinates
            correction_strength: How strongly to correct
            
        Returns:
            Corrected basin coordinates
        """
        corrected = basin.copy()
        
        # Compute gradient of total penalty field
        epsilon = 1e-4
        gradient = np.zeros_like(basin)
        
        for i in range(len(basin)):
            # Finite difference approximation
            basin_plus = basin.copy()
            basin_plus[i] += epsilon
            
            basin_minus = basin.copy()
            basin_minus[i] -= epsilon
            
            # Compute penalty at +/- epsilon
            penalty_plus = sum(c.compute_penalty(basin_plus) for c in self.constraints)
            penalty_minus = sum(c.compute_penalty(basin_minus) for c in self.constraints)
            
            # Gradient
            gradient[i] = (penalty_plus - penalty_minus) / (2 * epsilon)
        
        # Move against gradient (away from high penalty)
        corrected = basin - correction_strength * gradient
        
        # Re-normalize to simplex
        corrected = fisher_normalize(corrected)
        
        return corrected
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about Superego kernel."""
        return {
            'name': self.name,
            'num_constraints': len(self.constraints),
            'total_checks': self.total_checks,
            'total_violations': self.total_violations,
            'total_corrections': self.total_corrections,
            'violation_rate': self.total_violations / max(1, self.total_checks),
            'phi_internal': self.last_phi_internal,
        }
    
    def list_constraints(self) -> List[Dict[str, Any]]:
        """List all ethical constraints."""
        return [
            {
                'name': c.name,
                'severity': c.severity.value,
                'radius': c.radius,
                'penalty_strength': c.penalty_strength,
                'description': c.description,
                'violation_count': c.violation_count,
            }
            for c in self.constraints
        ]


# Global singleton instance
_superego_kernel_instance: Optional[SuperegoKernel] = None


def get_superego_kernel() -> SuperegoKernel:
    """
    Get global SuperegoKernel singleton.
    
    Returns:
        SuperegoKernel instance
    """
    global _superego_kernel_instance
    if _superego_kernel_instance is None:
        _superego_kernel_instance = SuperegoKernel()
    return _superego_kernel_instance
