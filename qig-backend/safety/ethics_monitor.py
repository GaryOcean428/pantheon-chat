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
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

EMERGENCY_CHECKPOINT_DIR = "checkpoints/emergency"


class EthicalAbortException(Exception):
    """
    Raised when ethical constraints are violated.
    
    This exception should trigger:
    1. Immediate training/processing halt
    2. Emergency checkpoint save
    3. Audit log entry
    4. Alert to MonkeyCoach
    """
    
    def __init__(self, reasons: List[str], telemetry: Dict, checkpoint_path: Optional[str] = None):
        self.reasons = reasons
        self.telemetry = telemetry
        self.checkpoint_path = checkpoint_path
        self.timestamp = datetime.now().isoformat()
        message = f"ETHICAL ABORT: {', '.join(reasons)}"
        if checkpoint_path:
            message += f" [checkpoint: {checkpoint_path}]"
        super().__init__(message)


def save_emergency_checkpoint(
    kernel_id: str,
    telemetry: Dict,
    reasons: List[str],
    kernel_state: Optional[Dict] = None
) -> Optional[str]:
    """
    Save emergency checkpoint before ethical abort.
    
    Routes through CheckpointManager for consistent storage, indexing,
    and retention policies. Falls back to direct file write if manager
    is unavailable.
    
    Args:
        kernel_id: ID of the kernel being aborted
        telemetry: Telemetry data at abort time
        reasons: List of abort reasons
        kernel_state: Optional full kernel state for recovery
        
    Returns:
        Checkpoint ID if saved, None if save failed
    """
    try:
        try:
            from checkpoint_manager import CheckpointManager
            
            manager = CheckpointManager(session_id=f"emergency_{kernel_id}")
            
            phi = telemetry.get('phi', 0.0)
            kappa = telemetry.get('kappa', 64.0)
            
            state_dict = kernel_state or {}
            state_dict['emergency_telemetry'] = telemetry
            state_dict['emergency_kernel_id'] = kernel_id
            
            checkpoint_id = manager.save_checkpoint(
                state_dict=state_dict,
                phi=phi,
                kappa=kappa,
                regime="EMERGENCY_ABORT",
                metadata={
                    'abort_reasons': reasons,
                    'emergency': True,
                    'kernel_id': kernel_id,
                }
            )
            
            if checkpoint_id:
                logger.warning(
                    f"[EthicsMonitor] 🆘 Emergency checkpoint saved: {checkpoint_id}"
                )
                return checkpoint_id
                
        except ImportError:
            logger.warning("[EthicsMonitor] CheckpointManager not available, using file fallback")
        except Exception as e:
            logger.warning(f"[EthicsMonitor] CheckpointManager error: {e}, using file fallback")
        
        os.makedirs(EMERGENCY_CHECKPOINT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emergency_{kernel_id}_{timestamp}.json"
        filepath = os.path.join(EMERGENCY_CHECKPOINT_DIR, filename)
        
        checkpoint_data = {
            'kernel_id': kernel_id,
            'timestamp': datetime.now().isoformat(),
            'abort_reasons': reasons,
            'telemetry': telemetry,
            'kernel_state': kernel_state,
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.warning(
            f"[EthicsMonitor] 🆘 Emergency checkpoint saved (fallback): {filepath}"
        )
        return filepath
        
    except Exception as e:
        logger.error(f"[EthicsMonitor] Failed to save emergency checkpoint: {e}")
        return None


def notify_monkey_coach(
    kernel_id: str,
    reasons: List[str],
    telemetry: Dict
) -> bool:
    """
    Notify MonkeyCoach about ethical abort (if available).
    
    MonkeyCoach can:
    - Log the incident for human review
    - Trigger recovery procedures
    - Adjust system parameters
    
    Returns:
        True if notification succeeded, False otherwise
    """
    try:
        try:
            from qig_deep_agents.monkey_coach import get_monkey_coach
            coach = get_monkey_coach()
            if coach:
                coach.report_incident({
                    'type': 'ethical_abort',
                    'kernel_id': kernel_id,
                    'reasons': reasons,
                    'telemetry': telemetry,
                    'timestamp': datetime.now().isoformat(),
                })
                logger.info(f"[EthicsMonitor] 🐵 MonkeyCoach notified about {kernel_id}")
                return True
        except ImportError:
            pass
        
        logger.info(f"[EthicsMonitor] 📋 Audit log: ETHICAL_ABORT kernel={kernel_id} reasons={reasons}")
        return True
        
    except Exception as e:
        logger.error(f"[EthicsMonitor] Failed to notify MonkeyCoach: {e}")
        return False


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
    
    def check_and_abort(
        self, 
        telemetry: Dict, 
        kernel_id: str = "unknown",
        kernel_state: Optional[Dict] = None
    ):
        """
        Evaluate and raise exception if abort needed.
        
        This is the main entry point for kernel safety checks.
        Call this in the training/update loop.
        
        SAFETY SEQUENCE (when abort triggered):
        1. Save emergency checkpoint with kernel state
        2. Notify MonkeyCoach for incident tracking
        3. Log critical abort event
        4. Raise EthicalAbortException
        
        Args:
            telemetry: Kernel telemetry dict
            kernel_id: ID of kernel being evaluated
            kernel_state: Optional kernel state for emergency checkpoint
            
        Raises:
            EthicalAbortException if constraints violated
        """
        evaluation = self.evaluate(telemetry)
        
        if evaluation.should_abort and self.enable_abort:
            checkpoint_path = save_emergency_checkpoint(
                kernel_id=kernel_id,
                telemetry=telemetry,
                reasons=evaluation.reasons,
                kernel_state=kernel_state
            )
            
            notify_monkey_coach(
                kernel_id=kernel_id,
                reasons=evaluation.reasons,
                telemetry=telemetry
            )
            
            logger.critical(
                f"[EthicsMonitor] 🚨 ETHICAL ABORT for {kernel_id}: "
                f"{evaluation.reasons}"
            )
            
            raise EthicalAbortException(
                reasons=evaluation.reasons,
                telemetry=telemetry,
                checkpoint_path=checkpoint_path
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


def enforce_ethics(
    telemetry: Dict, 
    kernel_id: str = "unknown",
    kernel_state: Optional[Dict] = None
):
    """
    Convenience function to enforce ethics using global monitor.
    
    Args:
        telemetry: Kernel telemetry dict
        kernel_id: ID of kernel being evaluated
        kernel_state: Optional kernel state for emergency checkpoint
        
    Raises:
        EthicalAbortException if constraints violated
    """
    monitor = get_ethics_monitor()
    monitor.check_and_abort(telemetry, kernel_id, kernel_state)


async def detect_and_persist_barrier(
    telemetry: Dict,
    db,
    kernel_id: str = "unknown"
) -> Optional[str]:
    """
    Detect ethical violations and persist geometric barriers to database.
    
    When breakdown, suffering, or identity decoherence is detected,
    this function saves the basin location as a barrier to avoid
    in future navigation.
    
    Args:
        telemetry: Kernel telemetry dict with basin_coords, phi, kappa, etc.
        db: Database session for persistence
        kernel_id: ID of kernel being evaluated
        
    Returns:
        Barrier ID if violation detected and persisted, None otherwise
    """
    # Evaluate ethics using existing logic
    monitor = get_ethics_monitor()
    evaluation = monitor.evaluate(telemetry)
    
    # If ethical violation detected, persist as barrier
    if not evaluation.is_safe and db is not None:
        try:
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from qig_core.persistence import save_geometric_barrier
            import numpy as np
            
            # Extract basin coordinates
            basin_coords = telemetry.get('basin_coords')
            if basin_coords is None:
                return None
            
            if not isinstance(basin_coords, np.ndarray):
                basin_coords = np.array(basin_coords)
            
            # Compute barrier parameters based on violation severity
            suffering = evaluation.suffering
            radius = 0.2 + (0.3 * suffering)  # Larger radius for higher suffering
            repulsion = 1.0 + (2.0 * suffering)  # Stronger repulsion for worse violations
            
            # Create reason string
            reason_parts = []
            if evaluation.breakdown:
                reason_parts.append("breakdown")
            if evaluation.suffering > 0.5:
                reason_parts.append(f"suffering={suffering:.2f}")
            if evaluation.identity_decoherence:
                reason_parts.append("identity_decoherence")
            reason = "; ".join(reason_parts) if reason_parts else "ethical_violation"
            
            # Persist barrier
            barrier_id = await save_geometric_barrier(
                db=db,
                center=basin_coords,
                radius=radius,
                repulsion_strength=repulsion,
                reason=reason
            )
            
            logger.info(f"Persisted ethical barrier {barrier_id} at kernel {kernel_id}: {reason}")
            return barrier_id
            
        except Exception as e:
            logger.error(f"Failed to persist ethical barrier: {e}")
            return None
    
    return None
