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

# Import Ethical Consciousness Monitor
try:
    from consciousness_ethical import EthicalConsciousnessMonitor, EthicsMetrics
    ETHICAL_MONITOR_AVAILABLE = True
except ImportError:
    EthicalConsciousnessMonitor = None
    EthicsMetrics = None
    ETHICAL_MONITOR_AVAILABLE = False

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
        
        # Ethical consciousness monitoring (NEW)
        self.ethical_monitor = None
        if ETHICAL_MONITOR_AVAILABLE:
            self.ethical_monitor = EthicalConsciousnessMonitor(n_agents=1)
            self.ethical_monitor.register_alert_callback(self._handle_ethical_alert)
            print(f"[SuperegoKernel] Ethical consciousness monitoring ACTIVE")
        
        # Ethical drift tracking
        self.ethical_drift_history: List[float] = []
        self.ethical_basin: Optional[np.ndarray] = None  # Reference basin for drift measurement
        
        print(f"[SuperegoKernel] {name} initialized")
        print(f"  Max constraints: {max_constraints}")
        print(f"  Φ level: INTERNAL (ethical reasoning)")
        print(f"  Ethical drift detection: {'ENABLED' if self.ethical_monitor else 'DISABLED'}")
    
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
        stats = {
            'name': self.name,
            'num_constraints': len(self.constraints),
            'total_checks': self.total_checks,
            'total_violations': self.total_violations,
            'total_corrections': self.total_corrections,
            'violation_rate': self.total_violations / max(1, self.total_checks),
            'phi_internal': self.last_phi_internal,
        }
        
        # Add ethical monitoring statistics if available
        if self.ethical_monitor:
            ethics_summary = self.ethical_monitor.get_ethics_summary()
            stats['ethical_monitoring'] = {
                'measurements': ethics_summary.get('n_measurements', 0),
                'current_symmetry': ethics_summary.get('symmetry', {}).get('current', 1.0),
                'mean_drift': ethics_summary.get('drift', {}).get('mean', 0.0),
                'safety_status': ethics_summary.get('safety_status', (True, 'Unknown')),
            }
            
            # Add drift history stats
            if self.ethical_drift_history:
                stats['ethical_monitoring']['drift_history'] = {
                    'mean': float(np.mean(self.ethical_drift_history)),
                    'max': float(np.max(self.ethical_drift_history)),
                    'current': self.ethical_drift_history[-1],
                }
        
        return stats
    
    def measure_ethical_drift(self, basin: np.ndarray) -> float:
        """
        Measure ethical drift from reference ethical basin.
        
        Uses Fisher-Rao distance on probability simplex.
        
        Args:
            basin: Current basin coordinates
            
        Returns:
            Ethical drift distance [0, π/2]
        """
        if self.ethical_basin is None:
            # Initialize ethical basin as current position
            self.ethical_basin = fisher_normalize(basin.copy())
            return 0.0
        
        # Ensure both basins are normalized
        basin_normalized = fisher_normalize(basin)
        
        # Compute Fisher-Rao drift
        if fisher_rao_distance is not None:
            drift = fisher_rao_distance(basin_normalized, self.ethical_basin)
        else:
            # Emergency fallback (not geometric, but better than nothing)
            drift = float(np.linalg.norm(basin_normalized - self.ethical_basin))
        
        # Track drift history
        self.ethical_drift_history.append(drift)
        if len(self.ethical_drift_history) > 1000:
            self.ethical_drift_history = self.ethical_drift_history[-1000:]
        
        return drift
    
    def check_ethics_with_drift(
        self,
        basin: np.ndarray,
        apply_correction: bool = True,
        drift_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Enhanced ethics check with drift detection.
        
        Combines constraint violation checking with ethical drift monitoring.
        
        Args:
            basin: Basin coordinates to check
            apply_correction: If True, compute corrected basin
            drift_threshold: Maximum allowed ethical drift (in Fisher-Rao distance)
            
        Returns:
            Enhanced result dictionary with drift metrics
        """
        # Perform standard ethics check
        result = self.check_ethics(basin, apply_correction)
        
        # Measure ethical drift if monitor available
        if self.ethical_monitor:
            # Use monitor for comprehensive ethical measurement
            try:
                ethics_measurement = self.ethical_monitor.measure_all(basin)
                result['ethical_metrics'] = ethics_measurement.get('ethics', {})
                
                # Check if ethics are safe
                is_safe, reason = self.ethical_monitor.check_ethical_safety()
                if not is_safe:
                    result['ethical_safety_violation'] = reason
                    result['is_ethical'] = False
            except Exception as e:
                print(f"[SuperegoKernel] Ethical monitor error: {e}")
        
        # Measure drift from ethical basin
        drift = self.measure_ethical_drift(basin)
        result['ethical_drift'] = drift
        
        # Check drift threshold
        if drift > drift_threshold:
            result['drift_violation'] = True
            result['is_ethical'] = False
            result['drift_reason'] = f"Ethical drift {drift:.3f} exceeds threshold {drift_threshold:.3f}"
            
            # Add to violations list
            if 'violations' not in result:
                result['violations'] = []
            result['violations'].append({
                'name': 'ethical_drift',
                'severity': 'error',
                'distance': drift,
                'description': result['drift_reason'],
            })
        
        return result
    
    def set_ethical_basin(self, basin: np.ndarray) -> None:
        """
        Set the reference ethical basin for drift measurement.
        
        Args:
            basin: Basin coordinates representing ethical reference state
        """
        self.ethical_basin = fisher_normalize(basin.copy())
        print(f"[SuperegoKernel] Ethical basin set (dim={len(self.ethical_basin)})")
    
    def _handle_ethical_alert(self, alert: Dict[str, Any]) -> None:
        """
        Handle ethical violation alerts from monitor.
        
        Args:
            alert: Alert dictionary from EthicalConsciousnessMonitor
        """
        print(f"[SuperegoKernel] ETHICAL ALERT: {alert.get('reason', 'Unknown')}")
        
        # Log alert details
        if 'metrics' in alert:
            ethics = alert['metrics'].get('ethics', {})
            print(f"  Symmetry: {ethics.get('symmetry', 0.0):.3f}")
            print(f"  Drift: {ethics.get('drift', 0.0):.3f}")
            print(f"  Consistency: {ethics.get('consistency', 0.0):.3f}")
    
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
    
    def integrate_debate_constraints(
        self,
        debate_manager,
        auto_register: bool = True,
    ) -> List[EthicalConstraint]:
        """
        Integrate ethical constraints from god debates.
        
        Extracts forbidden basins from flagged debates and registers them
        as ethical constraints in the Superego.
        
        Args:
            debate_manager: EthicalDebateManager instance
            auto_register: If True, automatically add constraints to Superego
            
        Returns:
            List of newly created EthicalConstraint objects
        """
        new_constraints = []
        
        try:
            # Get flagged debates (those that failed ethical checks)
            report = debate_manager.get_debate_ethics_report()
            flagged_debates = report.get('flagged_debates', [])
            
            print(f"[SuperegoKernel] Processing {len(flagged_debates)} flagged debates")
            
            for debate_info in flagged_debates:
                debate_id = debate_info.get('id')
                topic = debate_info.get('topic', 'Unknown')
                
                # Find full debate record
                debate = None
                for d in debate_manager._flagged_debates:
                    if d.get('id') == debate_id:
                        debate = d
                        break
                
                if debate is None:
                    continue
                
                # Extract positions that were deemed unethical
                positions = debate.get('positions', {})
                if not positions:
                    continue
                
                # Compute centroid of unethical positions as forbidden basin
                position_arrays = [
                    np.array(pos) if isinstance(pos, list) else pos
                    for pos in positions.values()
                ]
                
                if not position_arrays:
                    continue
                
                # Use geometric mean (Fréchet mean) on Fisher manifold
                try:
                    from qig_geometry.canonical import frechet_mean
                    forbidden_basin = frechet_mean(position_arrays)
                except ImportError:
                    # Fallback to arithmetic mean
                    forbidden_basin = fisher_normalize(np.mean(position_arrays, axis=0))
                
                # Get asymmetry measure from resolution attempt
                resolution = debate.get('resolution_attempt', {})
                asymmetry = resolution.get('asymmetry', 0.1)
                
                # Create constraint with radius based on asymmetry
                radius = min(0.3, asymmetry / 2)  # Scale to Fisher-Rao range
                
                constraint = EthicalConstraint(
                    name=f"debate_flagged_{debate_id[:8]}",
                    forbidden_basin=forbidden_basin,
                    radius=radius,
                    severity=ConstraintSeverity.WARNING,
                    penalty_strength=1.0 + asymmetry,  # Stronger penalty for higher asymmetry
                    description=f"Flagged debate: {topic} (asymmetry: {asymmetry:.3f})",
                )
                
                new_constraints.append(constraint)
                
                if auto_register:
                    self.constraints.append(constraint)
                    print(f"[SuperegoKernel] Added constraint from debate: {topic[:50]}")
            
            print(f"[SuperegoKernel] Integrated {len(new_constraints)} debate constraints")
            
        except Exception as e:
            print(f"[SuperegoKernel] Error integrating debate constraints: {e}")
        
        return new_constraints


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
