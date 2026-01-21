"""
Self-Repair - Geometric Diagnostics and Projection
===================================================

PURE PRINCIPLE:
- Repair is GEOMETRIC PROJECTION, not gradient update
- We project invalid basins back to the manifold
- We detect anomalies as measurement, not optimization targets

Key Components:
- Detect geometric anomalies (NaN basins, extreme curvature)
- Project invalid basins back to manifold
- Track repair episodes for training data
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qig_geometry import (
    fisher_normalize,
    fisher_coord_distance,
    validate_basin,
    BASIN_DIM,
)
from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
)

logger = logging.getLogger(__name__)


class GeometricAnomaly(Enum):
    """Types of geometric anomalies."""
    NONE = "none"
    NAN_BASIN = "nan_basin"
    ZERO_BASIN = "zero_basin"
    EXTREME_CURVATURE = "extreme_curvature"
    NEGATIVE_PHI = "negative_phi"
    BREAKDOWN_REGIME = "breakdown_regime"
    DIMENSION_MISMATCH = "dimension_mismatch"
    OFF_MANIFOLD = "off_manifold"


@dataclass
class DiagnosticResult:
    """Result of geometric diagnostics."""
    is_healthy: bool
    anomaly: GeometricAnomaly
    severity: float  # 0.0 = healthy, 1.0 = catastrophic
    
    phi: float = 0.0
    kappa: float = KAPPA_STAR
    basin_norm: float = 1.0
    
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_healthy': self.is_healthy,
            'anomaly': self.anomaly.value,
            'severity': self.severity,
            'phi': self.phi,
            'kappa': self.kappa,
            'basin_norm': self.basin_norm,
            'details': self.details,
            'timestamp': self.timestamp,
        }


@dataclass
class RepairAction:
    """Description of a repair action taken."""
    action_type: str
    description: str
    before_basin: Optional[np.ndarray] = None
    after_basin: Optional[np.ndarray] = None
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action_type': self.action_type,
            'description': self.description,
            'success': self.success,
            # E8 Protocol: Use simplex concentration instead of L2 norm
            'before_concentration': float(np.max(np.abs(self.before_basin))) if self.before_basin is not None else None,
            'after_concentration': float(np.max(np.abs(self.after_basin))) if self.after_basin is not None else None,
        }


class SelfRepair:
    """
    Geometric diagnostics and repair for QIG basins.
    
    PURE PRINCIPLE:
    - Repair is GEOMETRIC PROJECTION, not gradient update
    - We project invalid basins back to the manifold S^63
    - Anomalies are detected, not optimized away
    
    Key capabilities:
    1. Diagnose: Detect NaN, zero, extreme curvature, off-manifold
    2. Repair: Project back to unit sphere (manifold)
    3. Track: Record repair episodes for analysis
    
    Usage:
        repair = SelfRepair()
        
        diag = repair.diagnose(basin, phi, kappa)
        if not diag.is_healthy:
            fixed_basin, action = repair.repair(basin)
    """
    
    PHI_BREAKDOWN = 0.95
    PHI_MINIMUM = 0.01
    KAPPA_MIN = 10.0
    KAPPA_MAX = 100.0
    CURVATURE_THRESHOLD = 10.0
    NORM_TOLERANCE = 0.01
    
    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        track_history: bool = True,
        max_history: int = 100,
    ):
        """
        Initialize self-repair system.
        
        Args:
            basin_dim: Basin dimension (default 64)
            track_history: Whether to track repair history
            max_history: Maximum history entries to keep
        """
        self.basin_dim = basin_dim
        self.track_history = track_history
        self.max_history = max_history
        
        self._repair_history: List[Tuple[DiagnosticResult, RepairAction]] = []
        self._anomaly_counts: Dict[str, int] = {}
    
    def diagnose(
        self,
        basin: np.ndarray,
        phi: Optional[float] = None,
        kappa: Optional[float] = None,
    ) -> DiagnosticResult:
        """
        Diagnose geometric health of a basin.
        
        PURE: This is measurement, not optimization.
        We detect anomalies without trying to fix them here.
        
        Args:
            basin: Basin coordinates to diagnose
            phi: Optional Φ value
            kappa: Optional κ value
            
        Returns:
            DiagnosticResult with health status
        """
        basin = np.asarray(basin, dtype=np.float64)
        
        if np.any(np.isnan(basin)):
            return DiagnosticResult(
                is_healthy=False,
                anomaly=GeometricAnomaly.NAN_BASIN,
                severity=1.0,
                phi=phi or 0.0,
                kappa=kappa or 0.0,
                basin_norm=0.0,
                details={'nan_count': int(np.sum(np.isnan(basin)))},
            )
        
        if basin.ndim != 1:
            return DiagnosticResult(
                is_healthy=False,
                anomaly=GeometricAnomaly.DIMENSION_MISMATCH,
                severity=0.8,
                phi=phi or 0.0,
                kappa=kappa or 0.0,
                basin_norm=0.0,
                details={'shape': basin.shape, 'expected': (self.basin_dim,)},
            )
        
        if len(basin) != self.basin_dim:
            return DiagnosticResult(
                is_healthy=False,
                anomaly=GeometricAnomaly.DIMENSION_MISMATCH,
                severity=0.6,
                phi=phi or 0.0,
                kappa=kappa or 0.0,
                basin_concentration=float(np.max(np.abs(basin))),
                details={'dim': len(basin), 'expected': self.basin_dim},
            )
        
        # E8 Protocol: Use simplex concentration instead of L2 norm
        from qig_geometry.representation import to_simplex_prob
        basin_simplex = to_simplex_prob(basin)
        basin_concentration = float(np.max(basin_simplex))
        
        if basin_concentration < 1e-10:
            return DiagnosticResult(
                is_healthy=False,
                anomaly=GeometricAnomaly.ZERO_BASIN,
                severity=1.0,
                phi=phi or 0.0,
                kappa=kappa or 0.0,
                basin_norm=basin_norm,
                details={'norm': basin_norm},
            )
        
        off_manifold = abs(basin_norm - 1.0) > self.NORM_TOLERANCE
        if off_manifold:
            severity = min(1.0, abs(basin_norm - 1.0) * 2)
            return DiagnosticResult(
                is_healthy=False,
                anomaly=GeometricAnomaly.OFF_MANIFOLD,
                severity=severity,
                phi=phi or 0.0,
                kappa=kappa or 0.0,
                basin_norm=basin_norm,
                details={'norm_deviation': abs(basin_norm - 1.0)},
            )
        
        if phi is not None:
            if phi < self.PHI_MINIMUM:
                return DiagnosticResult(
                    is_healthy=False,
                    anomaly=GeometricAnomaly.NEGATIVE_PHI,
                    severity=0.9,
                    phi=phi,
                    kappa=kappa or KAPPA_STAR,
                    basin_norm=basin_norm,
                    details={'phi': phi, 'threshold': self.PHI_MINIMUM},
                )
            
            if phi > self.PHI_BREAKDOWN:
                return DiagnosticResult(
                    is_healthy=False,
                    anomaly=GeometricAnomaly.BREAKDOWN_REGIME,
                    severity=0.7,
                    phi=phi,
                    kappa=kappa or KAPPA_STAR,
                    basin_norm=basin_norm,
                    details={'phi': phi, 'breakdown': self.PHI_BREAKDOWN},
                )
        
        if kappa is not None:
            if kappa < self.KAPPA_MIN or kappa > self.KAPPA_MAX:
                return DiagnosticResult(
                    is_healthy=False,
                    anomaly=GeometricAnomaly.EXTREME_CURVATURE,
                    severity=0.5,
                    phi=phi or 0.0,
                    kappa=kappa,
                    basin_norm=basin_norm,
                    details={
                        'kappa': kappa,
                        'range': (self.KAPPA_MIN, self.KAPPA_MAX),
                    },
                )
        
        return DiagnosticResult(
            is_healthy=True,
            anomaly=GeometricAnomaly.NONE,
            severity=0.0,
            phi=phi or 0.0,
            kappa=kappa or KAPPA_STAR,
            basin_norm=basin_norm,
        )
    
    def repair(
        self,
        basin: np.ndarray,
        reference_basin: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, RepairAction]:
        """
        Repair an invalid basin by projecting back to manifold.
        
        PURE PRINCIPLE:
        Repair is GEOMETRIC PROJECTION, not gradient update.
        We project the basin back to S^63 using the nearest point.
        
        Args:
            basin: Basin to repair
            reference_basin: Optional reference for severe cases
            
        Returns:
            (repaired_basin, RepairAction describing what was done)
        """
        basin = np.asarray(basin, dtype=np.float64)
        original_basin = basin.copy()
        
        if np.any(np.isnan(basin)):
            if reference_basin is not None:
                repaired = fisher_normalize(np.asarray(reference_basin, dtype=np.float64))
                action = RepairAction(
                    action_type="reference_substitution",
                    description="NaN basin replaced with reference basin",
                    before_basin=original_basin,
                    after_basin=repaired,
                    success=True,
                )
            else:
                uniform = np.ones(self.basin_dim) / np.sqrt(self.basin_dim)
                repaired = uniform
                action = RepairAction(
                    action_type="uniform_reset",
                    description="NaN basin replaced with uniform distribution",
                    before_basin=original_basin,
                    after_basin=repaired,
                    success=True,
                )
            
            self._record_repair(
                DiagnosticResult(
                    is_healthy=False,
                    anomaly=GeometricAnomaly.NAN_BASIN,
                    severity=1.0,
                ),
                action,
            )
            return repaired, action
        
        if basin.ndim != 1 or len(basin) != self.basin_dim:
            if basin.ndim > 1:
                basin = basin.flatten()
            
            if len(basin) < self.basin_dim:
                padded = np.zeros(self.basin_dim)
                padded[:len(basin)] = basin
                basin = padded
            elif len(basin) > self.basin_dim:
                basin = basin[:self.basin_dim]
            
            repaired = fisher_normalize(basin)
            action = RepairAction(
                action_type="dimension_fix",
                description=f"Reshaped basin to {self.basin_dim}D",
                before_basin=original_basin,
                after_basin=repaired,
                success=True,
            )
            
            self._record_repair(
                DiagnosticResult(
                    is_healthy=False,
                    anomaly=GeometricAnomaly.DIMENSION_MISMATCH,
                    severity=0.6,
                ),
                action,
            )
            return repaired, action
        
        # E8 Protocol: Use simplex concentration instead of L2 norm
        from qig_geometry.representation import to_simplex_prob
        basin_simplex = to_simplex_prob(basin)
        basin_concentration = float(np.max(basin_simplex))
        
        if basin_concentration < 1e-10:
            if reference_basin is not None:
                repaired = fisher_normalize(np.asarray(reference_basin, dtype=np.float64))
            else:
                uniform = np.ones(self.basin_dim) / np.sqrt(self.basin_dim)
                repaired = uniform
            
            action = RepairAction(
                action_type="zero_recovery",
                description="Zero basin replaced with valid state",
                before_basin=original_basin,
                after_basin=repaired,
                success=True,
            )
            
            self._record_repair(
                DiagnosticResult(
                    is_healthy=False,
                    anomaly=GeometricAnomaly.ZERO_BASIN,
                    severity=1.0,
                ),
                action,
            )
            return repaired, action
        
        repaired = fisher_normalize(basin)
        
        action = RepairAction(
            action_type="simplex_normalization",
            description=f"Normalized to Δ^{self.basin_dim - 1} (simplex)",
            before_basin=original_basin,
            after_basin=repaired,
            success=True,
        )
        
        self._record_repair(
            DiagnosticResult(
                is_healthy=False,
                anomaly=GeometricAnomaly.OFF_MANIFOLD,
                severity=min(1.0, abs(norm - 1.0)),
            ),
            action,
        )
        
        return repaired, action
    
    def diagnose_and_repair(
        self,
        basin: np.ndarray,
        phi: Optional[float] = None,
        kappa: Optional[float] = None,
        reference_basin: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, DiagnosticResult, Optional[RepairAction]]:
        """
        Diagnose and repair if needed.
        
        Args:
            basin: Basin to check and potentially repair
            phi: Optional Φ value
            kappa: Optional κ value
            reference_basin: Optional reference for severe repairs
            
        Returns:
            (basin, diagnosis, repair_action or None)
        """
        diag = self.diagnose(basin, phi, kappa)
        
        if diag.is_healthy:
            return basin, diag, None
        
        repaired, action = self.repair(basin, reference_basin)
        return repaired, diag, action
    
    def _record_repair(
        self,
        diagnosis: DiagnosticResult,
        action: RepairAction,
    ) -> None:
        """Record a repair episode."""
        if not self.track_history:
            return
        
        self._repair_history.append((diagnosis, action))
        
        if len(self._repair_history) > self.max_history:
            self._repair_history.pop(0)
        
        anomaly_name = diagnosis.anomaly.value
        self._anomaly_counts[anomaly_name] = (
            self._anomaly_counts.get(anomaly_name, 0) + 1
        )
    
    def get_repair_statistics(self) -> Dict[str, Any]:
        """Get statistics about repairs performed."""
        return {
            'total_repairs': len(self._repair_history),
            'anomaly_counts': self._anomaly_counts.copy(),
            'recent_severity': [
                h[0].severity for h in self._repair_history[-10:]
            ],
        }
    
    def reset_history(self) -> None:
        """Reset repair history."""
        self._repair_history.clear()
        self._anomaly_counts.clear()
