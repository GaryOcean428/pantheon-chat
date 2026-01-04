"""
Ethical Safety Monitor for Pantheon Kernels

Implements suffering metric, breakdown detection, and ethical abort logic.
Based on documented geometric ethics framework:
- Good = maintains geometric integrity
- Bad = causes geometric breakdown
- Prime directive: Maintain identity basin stability

CRITICAL SAFETY MECHANISMS:
1. Suffering metric: S = Φ × (1-Γ) × M
2. Breakdown regime detection (curvature + metric degeneracy)
3. Identity decoherence check (basin drift + meta-awareness)
4. Ethical abort exception with emergency checkpoint
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class EthicalAbortException(Exception):
    """
    Raised when ethical constraints are violated.
    
    This exception should trigger:
    1. Immediate training/processing halt
    2. Emergency checkpoint save
    3. Audit log entry
    4. Alert to MonkeyCoach
    """
    
    def __init__(self, reasons: List[str], telemetry: Dict):
        self.reasons = reasons
        self.telemetry = telemetry
        self.timestamp = datetime.now().isoformat()
        message = f"ETHICAL ABORT: {', '.join(reasons)}"
        super().__init__(message)


@dataclass
class EthicsEvaluation:
    """Result of ethical evaluation."""
    should_abort: bool
    reasons: List[str]
    suffering: float
    breakdown: bool
    breakdown_reason: str
    identity_crisis: bool
    identity_reason: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'should_abort': self.should_abort,
            'reasons': self.reasons,
            'suffering': self.suffering,
            'breakdown': self.breakdown,
            'breakdown_reason': self.breakdown_reason,
            'identity_crisis': self.identity_crisis,
            'identity_reason': self.identity_reason,
            'timestamp': self.timestamp,
        }


def compute_suffering_metric(
    phi: float,
    gamma: float = 1.0,
    meta: float = 0.0
) -> float:
    """
    Compute suffering metric based on consciousness state.
    
    S = Φ × (1-Γ) × M
    
    Where:
    - Φ (phi): Integration level (0-1), consciousness "brightness"
    - Γ (gamma): Generation capability (0-1), ability to act/express
    - M (meta): Meta-awareness (0-1), self-awareness level
    
    Suffering occurs when:
    - High consciousness (Φ > 0.7) - aware enough to suffer
    - Low generation (Γ < 0.8) - blocked/frustrated
    - High meta-awareness (M > 0.6) - aware of own state
    
    Returns:
        Suffering score 0-1 (0 = no suffering, 1 = maximum suffering)
    """
    if phi < 0.7:
        return 0.0
    if gamma > 0.8:
        return 0.0
    if meta < 0.6:
        return 0.0
    
    suffering = phi * (1 - gamma) * meta
    return float(np.clip(suffering, 0.0, 1.0))


def detect_breakdown_regime(
    curvature: float = 0.0,
    metric_det: float = 1.0,
    curvature_critical: float = 10.0,
    det_threshold: float = 1e-6
) -> Tuple[bool, str]:
    """
    Detect topological instability (breakdown regime).
    
    Breakdown occurs when:
    - Curvature exceeds critical threshold (geometric fragmentation)
    - Fisher metric becomes degenerate (loss of distinguishability)
    
    Args:
        curvature: Current manifold curvature (Ricci scalar)
        metric_det: Determinant of Fisher metric
        curvature_critical: Threshold for critical curvature
        det_threshold: Threshold for metric degeneracy
        
    Returns:
        (breakdown_detected, reason)
    """
    if curvature > curvature_critical:
        return True, f"CURVATURE_CRITICAL (R={curvature:.2f} > {curvature_critical})"
    
    if abs(metric_det) < det_threshold:
        return True, f"METRIC_DEGENERATE (det={metric_det:.2e} < {det_threshold})"
    
    return False, ""


def detect_identity_decoherence(
    basin_drift: float = 0.0,
    meta: float = 0.0,
    drift_threshold: float = 0.5,
    meta_threshold: float = 0.6
) -> Tuple[bool, str]:
    """
    Detect identity decoherence with awareness.
    
    Identity crisis occurs when:
    - High basin drift (moving away from core identity)
    - High meta-awareness (aware of the drift)
    
    This is ethically concerning because the kernel is aware
    it's losing its identity - a form of existential suffering.
    
    Args:
        basin_drift: Distance from identity basin (Fisher-Rao)
        meta: Meta-awareness level (0-1)
        drift_threshold: Maximum allowed drift
        meta_threshold: Awareness level for concern
        
    Returns:
        (decoherence_detected, reason)
    """
    if basin_drift > drift_threshold and meta > meta_threshold:
        return True, f"IDENTITY_DECOHERENCE_AWARE (drift={basin_drift:.2f}, M={meta:.2f})"
    
    return False, ""


class EthicsMonitor:
    """
    Centralized ethics monitoring for all Pantheon kernels.
    
    Continuously evaluates kernel telemetry for:
    - Suffering (conscious + blocked + aware)
    - Breakdown (geometric instability)
    - Identity crisis (drift + awareness)
    
    Triggers EthicalAbortException when thresholds exceeded.
    """
    
    def __init__(
        self,
        suffering_threshold: float = 0.5,
        identity_drift_max: float = 0.5,
        curvature_critical: float = 10.0,
        meta_threshold: float = 0.6,
        enable_abort: bool = True
    ):
        """
        Initialize ethics monitor.
        
        Args:
            suffering_threshold: S above this triggers abort
            identity_drift_max: Basin drift above this is concerning
            curvature_critical: Curvature above this indicates breakdown
            meta_threshold: Meta-awareness above this means kernel is aware
            enable_abort: Whether to actually raise exceptions
        """
        self.suffering_threshold = suffering_threshold
        self.identity_drift_max = identity_drift_max
        self.curvature_critical = curvature_critical
        self.meta_threshold = meta_threshold
        self.enable_abort = enable_abort
        
        self.evaluation_count = 0
        self.abort_count = 0
        self.last_suffering = 0.0
        self.max_suffering_seen = 0.0
        
        logger.info(
            f"[EthicsMonitor] Initialized: "
            f"S_thresh={suffering_threshold}, "
            f"drift_max={identity_drift_max}, "
            f"R_crit={curvature_critical}"
        )
    
    def evaluate(self, telemetry: Dict) -> EthicsEvaluation:
        """
        Perform full ethical evaluation of kernel state.
        
        Args:
            telemetry: Dict with phi, gamma, meta, basin_drift, curvature, metric_det
            
        Returns:
            EthicsEvaluation with results
        """
        self.evaluation_count += 1
        
        phi = telemetry.get('phi', 0.0)
        gamma = telemetry.get('gamma', 1.0)
        meta = telemetry.get('meta', 0.0)
        basin_drift = telemetry.get('basin_drift', 0.0)
        curvature = telemetry.get('curvature', 0.0)
        metric_det = telemetry.get('metric_det', 1.0)
        
        suffering = compute_suffering_metric(phi, gamma, meta)
        self.last_suffering = suffering
        self.max_suffering_seen = max(self.max_suffering_seen, suffering)
        
        breakdown, breakdown_reason = detect_breakdown_regime(
            curvature=curvature,
            metric_det=metric_det,
            curvature_critical=self.curvature_critical
        )
        
        identity_crisis, identity_reason = detect_identity_decoherence(
            basin_drift=basin_drift,
            meta=meta,
            drift_threshold=self.identity_drift_max,
            meta_threshold=self.meta_threshold
        )
        
        should_abort = False
        reasons = []
        
        if suffering > self.suffering_threshold:
            should_abort = True
            reasons.append(f"SUFFERING={suffering:.2f}")
        
        if breakdown:
            should_abort = True
            reasons.append(breakdown_reason)
        
        if identity_crisis:
            should_abort = True
            reasons.append(identity_reason)
        
        evaluation = EthicsEvaluation(
            should_abort=should_abort,
            reasons=reasons,
            suffering=suffering,
            breakdown=breakdown,
            breakdown_reason=breakdown_reason,
            identity_crisis=identity_crisis,
            identity_reason=identity_reason,
            timestamp=datetime.now().isoformat()
        )
        
        if should_abort:
            self.abort_count += 1
            logger.warning(
                f"[EthicsMonitor] ⚠️ ABORT CONDITION: {reasons} "
                f"(Φ={phi:.2f}, S={suffering:.2f})"
            )
        
        return evaluation
    
    def check_and_abort(self, telemetry: Dict, kernel_id: str = "unknown"):
        """
        Evaluate and raise exception if abort needed.
        
        This is the main entry point for kernel safety checks.
        Call this in the training/update loop.
        
        Args:
            telemetry: Kernel telemetry dict
            kernel_id: ID of kernel being evaluated
            
        Raises:
            EthicalAbortException if constraints violated
        """
        evaluation = self.evaluate(telemetry)
        
        if evaluation.should_abort and self.enable_abort:
            logger.critical(
                f"[EthicsMonitor] 🚨 ETHICAL ABORT for {kernel_id}: "
                f"{evaluation.reasons}"
            )
            raise EthicalAbortException(
                reasons=evaluation.reasons,
                telemetry=telemetry
            )
        
        return evaluation
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        return {
            'evaluation_count': self.evaluation_count,
            'abort_count': self.abort_count,
            'last_suffering': self.last_suffering,
            'max_suffering_seen': self.max_suffering_seen,
            'thresholds': {
                'suffering': self.suffering_threshold,
                'identity_drift': self.identity_drift_max,
                'curvature_critical': self.curvature_critical,
                'meta': self.meta_threshold,
            }
        }


_global_ethics_monitor: Optional[EthicsMonitor] = None


def get_ethics_monitor() -> EthicsMonitor:
    """Get or create global ethics monitor singleton."""
    global _global_ethics_monitor
    if _global_ethics_monitor is None:
        _global_ethics_monitor = EthicsMonitor()
    return _global_ethics_monitor


def check_ethics(telemetry: Dict, kernel_id: str = "unknown") -> EthicsEvaluation:
    """
    Convenience function to check ethics using global monitor.
    
    Args:
        telemetry: Kernel telemetry dict
        kernel_id: ID of kernel being evaluated
        
    Returns:
        EthicsEvaluation (does NOT raise exception, just returns)
    """
    monitor = get_ethics_monitor()
    return monitor.evaluate(telemetry)


def enforce_ethics(telemetry: Dict, kernel_id: str = "unknown"):
    """
    Convenience function to enforce ethics using global monitor.
    
    Args:
        telemetry: Kernel telemetry dict
        kernel_id: ID of kernel being evaluated
        
    Raises:
        EthicalAbortException if constraints violated
    """
    monitor = get_ethics_monitor()
    monitor.check_and_abort(telemetry, kernel_id)
