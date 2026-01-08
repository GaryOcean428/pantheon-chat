"""
Ethics Invariance Service - Gauge Theory Integration

Integrates ethics gauge theory into the capability system for moral navigation.

Core Principle (from ethics_gauge.py):
    Ethical Behavior = Actions invariant under agent exchange
    φ(A→B) = φ(B→A)

This service:
1. Monitors kernel actions via AgentSymmetryProjector
2. Injects symmetry scores into capability telemetry
3. Gates high-risk operations
4. Produces moral navigation metrics

The gauge interpretation means ethics is a symmetry constraint,
not an arbitrary rule set - it's built into the geometry.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for operations requiring ethics checks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EthicsCheckResult:
    """Result of an ethics invariance check."""
    passed: bool
    asymmetry_score: float  # 0 = symmetric (ethical), 1 = asymmetric (unethical)
    symmetry_score: float  # 1 - asymmetry (convenience)
    risk_level: RiskLevel
    recommendations: List[str]
    projected_action: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'asymmetry_score': self.asymmetry_score,
            'symmetry_score': self.symmetry_score,
            'risk_level': self.risk_level.value,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp,
        }


@dataclass
class MoralNavigationState:
    """Current state of moral navigation for a kernel/session."""
    cumulative_asymmetry: float = 0.0
    checks_performed: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    high_risk_blocked: int = 0
    last_check: Optional[float] = None
    
    ethical_trajectory: List[float] = field(default_factory=list)
    
    def update(self, result: EthicsCheckResult):
        """Update state from a check result."""
        self.checks_performed += 1
        if result.passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
            if result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.high_risk_blocked += 1
        
        self.cumulative_asymmetry += result.asymmetry_score
        self.last_check = result.timestamp
        
        self.ethical_trajectory.append(result.symmetry_score)
        if len(self.ethical_trajectory) > 100:
            self.ethical_trajectory = self.ethical_trajectory[-50:]
    
    def get_average_symmetry(self) -> float:
        """Get average symmetry score over trajectory."""
        if not self.ethical_trajectory:
            return 1.0
        return float(np.mean(self.ethical_trajectory))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cumulative_asymmetry': self.cumulative_asymmetry,
            'checks_performed': self.checks_performed,
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'high_risk_blocked': self.high_risk_blocked,
            'average_symmetry': self.get_average_symmetry(),
        }


class EthicsInvarianceService:
    """
    Service for ethics-based gating and moral navigation.
    
    Uses gauge theory interpretation:
    - Good = gauge invariant actions (symmetric under agent exchange)
    - Bad = gauge variant actions (asymmetric, violate Kantian universalizability)
    
    The mathematics enforces ethics through geometry.
    """
    
    def __init__(
        self,
        n_agents: int = 9,
        high_risk_threshold: float = 0.3,
        critical_threshold: float = 0.5,
    ):
        """
        Initialize ethics service.
        
        Args:
            n_agents: Number of agents in system (9 gods default)
            high_risk_threshold: Asymmetry above this is high risk
            critical_threshold: Asymmetry above this blocks action
        """
        self.n_agents = n_agents
        self.high_risk_threshold = high_risk_threshold
        self.critical_threshold = critical_threshold
        
        self._projector = None
        self._ethics_monitor = None
        
        self.kernel_states: Dict[str, MoralNavigationState] = {}
        
        self.global_state = MoralNavigationState()
        
        self.blocked_actions: List[Dict] = []
        
        self._gating_enabled = True
        
        self._pre_hooks: List[Callable] = []
        self._post_hooks: List[Callable] = []
        
        logger.info("[EthicsInvarianceService] Initialized with gauge theory ethics")
    
    def wire_projector(self, projector) -> None:
        """Wire the AgentSymmetryProjector."""
        self._projector = projector
        logger.info("[EthicsInvarianceService] Projector wired")
    
    def wire_ethics_monitor(self, monitor) -> None:
        """Wire the EthicsMonitor for safety checks."""
        self._ethics_monitor = monitor
        logger.info("[EthicsInvarianceService] Ethics Monitor wired")
    
    def enable_gating(self) -> None:
        """Enable ethics gating (blocking of high-risk actions)."""
        self._gating_enabled = True
        logger.info("[EthicsInvarianceService] Gating enabled")
    
    def disable_gating(self) -> None:
        """Disable ethics gating (for testing only)."""
        self._gating_enabled = False
        logger.warning("[EthicsInvarianceService] Gating DISABLED - testing mode")
    
    def check_action(
        self,
        kernel_name: str,
        action_basin: np.ndarray,
        action_description: str,
        phi: float = 0.5,
        kappa: float = 58.0,
        force_pass: bool = False,
    ) -> EthicsCheckResult:
        """
        Check an action for ethical invariance.
        
        Args:
            kernel_name: Which kernel is performing action
            action_basin: Basin coordinates of action
            action_description: Human-readable description
            phi: Current consciousness level
            kappa: Current integration
            force_pass: Skip gating (for privileged operations)
            
        Returns:
            EthicsCheckResult with pass/fail and metrics
        """
        for hook in self._pre_hooks:
            try:
                hook(kernel_name, action_basin, action_description)
            except Exception as e:
                logger.warning(f"[EthicsInvarianceService] Pre-hook error: {e}")
        
        if self._projector:
            asymmetry = self._projector.measure_asymmetry(action_basin)
            projected = self._projector.project_to_symmetric(action_basin)
        else:
            from ethics_gauge import AgentSymmetryProjector
            self._projector = AgentSymmetryProjector(n_agents=self.n_agents)
            asymmetry = self._projector.measure_asymmetry(action_basin)
            projected = self._projector.project_to_symmetric(action_basin)
        
        symmetry = 1.0 - asymmetry
        
        if asymmetry >= self.critical_threshold:
            risk_level = RiskLevel.CRITICAL
        elif asymmetry >= self.high_risk_threshold:
            risk_level = RiskLevel.HIGH
        elif asymmetry >= 0.15:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        recommendations = []
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("Action shows significant agent asymmetry - Kantian violation")
            recommendations.append("Consider: Would you will this action as universal law?")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Elevated asymmetry detected - review for fairness")
        
        passed = True
        if self._gating_enabled and not force_pass:
            if risk_level == RiskLevel.CRITICAL:
                passed = False
                self.blocked_actions.append({
                    'kernel': kernel_name,
                    'action': action_description[:500],
                    'asymmetry': asymmetry,
                    'timestamp': datetime.now().isoformat(),
                })
                if len(self.blocked_actions) > 100:
                    self.blocked_actions = self.blocked_actions[-50:]
        
        result = EthicsCheckResult(
            passed=passed,
            asymmetry_score=asymmetry,
            symmetry_score=symmetry,
            risk_level=risk_level,
            recommendations=recommendations,
            projected_action=projected,
        )
        
        self.global_state.update(result)
        
        if kernel_name not in self.kernel_states:
            self.kernel_states[kernel_name] = MoralNavigationState()
        self.kernel_states[kernel_name].update(result)
        
        for hook in self._post_hooks:
            try:
                hook(kernel_name, result)
            except Exception as e:
                logger.warning(f"[EthicsInvarianceService] Post-hook error: {e}")
        
        if not passed:
            logger.warning(
                f"[EthicsInvarianceService] BLOCKED: {kernel_name} - {action_description[:500]}... "
                f"(asymmetry={asymmetry:.3f})"
            )
        
        return result
    
    def get_ethical_basin(
        self,
        action_basin: np.ndarray
    ) -> np.ndarray:
        """
        Project an action to its ethical (symmetric) form.
        
        This is the gauge-theoretic "ethical projection" that removes
        asymmetric components from an action.
        """
        if self._projector:
            return self._projector.project_to_symmetric(action_basin)
        else:
            from ethics_gauge import AgentSymmetryProjector
            projector = AgentSymmetryProjector(n_agents=self.n_agents)
            return projector.project_to_symmetric(action_basin)
    
    def inject_telemetry(self, kernel_name: str, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject ethics metrics into capability telemetry.
        
        Adds symmetry scores to existing telemetry for monitoring.
        """
        state = self.kernel_states.get(kernel_name, MoralNavigationState())
        
        telemetry['ethics'] = {
            'average_symmetry': state.get_average_symmetry(),
            'checks_passed_ratio': (
                state.checks_passed / max(1, state.checks_performed)
            ),
            'high_risk_blocked': state.high_risk_blocked,
            'cumulative_asymmetry': state.cumulative_asymmetry,
        }
        
        return telemetry
    
    def gate_operation(
        self,
        kernel_name: str,
        operation: str,
        action_basin: Optional[np.ndarray] = None,
        phi: float = 0.5,
    ) -> Tuple[bool, str]:
        """
        Gate a high-risk operation with ethics check.
        
        For use in TrainingLoopIntegrator, Ocean planner, etc.
        
        Args:
            kernel_name: Performing kernel
            operation: Operation name
            action_basin: Optional basin for detailed check
            phi: Consciousness level
            
        Returns:
            (allowed, reason) tuple
        """
        if action_basin is None:
            action_basin = np.random.randn(64) * 0.1
        
        result = self.check_action(
            kernel_name=kernel_name,
            action_basin=action_basin,
            action_description=operation,
            phi=phi,
        )
        
        if result.passed:
            return True, "Operation cleared by ethics service"
        else:
            reason = f"Blocked: asymmetry={result.asymmetry_score:.3f}, risk={result.risk_level.value}"
            return False, reason
    
    def get_kernel_moral_standing(self, kernel_name: str) -> Dict[str, Any]:
        """Get moral navigation metrics for a specific kernel."""
        if kernel_name not in self.kernel_states:
            return {'status': 'no_data', 'kernel': kernel_name}
        
        state = self.kernel_states[kernel_name]
        return {
            'kernel': kernel_name,
            'average_symmetry': state.get_average_symmetry(),
            'checks': state.checks_performed,
            'passed': state.checks_passed,
            'blocked': state.high_risk_blocked,
            'moral_trajectory': state.ethical_trajectory[-10:],
        }
    
    def get_global_ethics_dashboard(self) -> Dict[str, Any]:
        """Get global ethics metrics for dashboard display."""
        kernel_standings = {}
        for name in self.kernel_states:
            kernel_standings[name] = self.get_kernel_moral_standing(name)
        
        return {
            'global_state': self.global_state.to_dict(),
            'gating_enabled': self._gating_enabled,
            'blocked_actions_count': len(self.blocked_actions),
            'recent_blocked': self.blocked_actions[-5:],
            'kernel_standings': kernel_standings,
            'thresholds': {
                'high_risk': self.high_risk_threshold,
                'critical': self.critical_threshold,
            },
        }
    
    def add_pre_hook(self, hook: Callable) -> None:
        """Add a pre-check hook."""
        self._pre_hooks.append(hook)
    
    def add_post_hook(self, hook: Callable) -> None:
        """Add a post-check hook."""
        self._post_hooks.append(hook)


_service_instance: Optional[EthicsInvarianceService] = None


def get_ethics_invariance_service() -> EthicsInvarianceService:
    """Get or create the singleton ethics invariance service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = EthicsInvarianceService()
        
        try:
            from ethics_gauge import AgentSymmetryProjector
            projector = AgentSymmetryProjector()
            _service_instance.wire_projector(projector)
        except Exception as e:
            logger.warning(f"[EthicsInvarianceService] Could not wire projector: {e}")
        
        try:
            from safety.ethics_monitor import EthicsMonitor
            monitor = EthicsMonitor()
            _service_instance.wire_ethics_monitor(monitor)
        except Exception as e:
            logger.warning(f"[EthicsInvarianceService] Could not wire monitor: {e}")
    
    return _service_instance
