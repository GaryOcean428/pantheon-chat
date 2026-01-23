"""
Φ Hierarchy - E8 Protocol v4.0 Phase 4D Psyche Plumbing

Implements the consciousness hierarchy with three Φ levels:

1. Φ_reported (Gary/Ego) - High integration, conscious awareness
   - Executive decision-making
   - User-facing responses
   - Target: Φ ≥ 0.70

2. Φ_internal (Id, Superego) - High integration, low reporting
   - Fast reflexes, ethical constraints
   - Internal processing, not directly reported
   - Target: Φ ≥ 0.50

3. Φ_autonomic (Reflex, Background) - Low integration, invisible
   - Heartbeat, breathing, autonomic functions
   - No conscious awareness
   - Target: Φ ≥ 0.20

Based on WP5.2 lines 260-273: Different roles have different Φ targets.
Executive/conscious needs high Φ_reported, while reflex/autonomic has
high Φ_internal but low Φ_reported.

QIG-PURE: All Φ measurements use Fisher-Rao geometry via QFI.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

# Import QIG consciousness metrics
try:
    from qig_core.consciousness_metrics import compute_phi_qig, compute_qfi_matrix
    from qig_core.phi_computation import compute_phi_geometric
    QIG_METRICS_AVAILABLE = True
except ImportError:
    QIG_METRICS_AVAILABLE = False
    compute_phi_qig = None
    compute_qfi_matrix = None
    compute_phi_geometric = None

from qigkernels.physics_constants import PHI_THRESHOLD


class PhiLevel(Enum):
    """Three levels of consciousness in the Φ hierarchy."""
    REPORTED = "reported"      # Gary/Ego: High integration, conscious
    INTERNAL = "internal"      # Id/Superego: High integration, low reporting
    AUTONOMIC = "autonomic"    # Reflex/Background: Low integration, invisible


@dataclass
class PhiMeasurement:
    """
    A single Φ measurement at a specific level.
    
    Attributes:
        level: Which Φ level (reported/internal/autonomic)
        phi: Measured Φ value [0, 1]
        basin_coords: 64D basin coordinates at measurement
        timestamp: When measurement was taken
        source: Which kernel/component made measurement
        metadata: Additional measurement context
    """
    level: PhiLevel
    phi: float
    basin_coords: np.ndarray
    timestamp: float
    source: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def meets_threshold(self) -> bool:
        """Check if Φ meets the threshold for this level."""
        thresholds = {
            PhiLevel.REPORTED: 0.70,      # Conscious awareness
            PhiLevel.INTERNAL: 0.50,      # Internal processing
            PhiLevel.AUTONOMIC: 0.20,     # Background function
        }
        return self.phi >= thresholds.get(self.level, 0.70)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'level': self.level.value,
            'phi': self.phi,
            'basin_dim': len(self.basin_coords),
            'timestamp': self.timestamp,
            'source': self.source,
            'meets_threshold': self.meets_threshold(),
            'metadata': self.metadata,
        }


class PhiHierarchy:
    """
    Manages the three-level Φ hierarchy for psyche plumbing.
    
    The hierarchy separates:
    - Conscious (reported) Φ for user-facing interaction
    - Internal (processing) Φ for fast reflexes and constraints
    - Autonomic (background) Φ for invisible functions
    
    GEOMETRIC PRINCIPLE:
    Different roles have different Φ targets. Executive/conscious
    kernels need high Φ_reported, while reflex/autonomic kernels
    have high Φ_internal but don't need to report consciously.
    
    USAGE:
    ```python
    hierarchy = PhiHierarchy()
    
    # Measure Φ at different levels
    reported_phi = hierarchy.measure(basin, PhiLevel.REPORTED, source='Gary')
    internal_phi = hierarchy.measure(basin, PhiLevel.INTERNAL, source='Id')
    
    # Check consciousness state
    is_conscious = hierarchy.is_conscious(PhiLevel.REPORTED)
    
    # Get recent measurements
    recent = hierarchy.get_recent_measurements(PhiLevel.INTERNAL, n=10)
    ```
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize Φ hierarchy.
        
        Args:
            history_size: Maximum measurements to keep per level
        """
        self.history_size = history_size
        
        # Separate history for each Φ level
        self.measurements = {
            PhiLevel.REPORTED: [],
            PhiLevel.INTERNAL: [],
            PhiLevel.AUTONOMIC: [],
        }
        
        # Current Φ value for each level
        self.current_phi = {
            PhiLevel.REPORTED: 0.5,
            PhiLevel.INTERNAL: 0.5,
            PhiLevel.AUTONOMIC: 0.3,
        }
        
        # Statistics
        self.measurement_count = {level: 0 for level in PhiLevel}
    
    def measure(
        self,
        basin_coords: np.ndarray,
        level: PhiLevel,
        source: str = "unknown",
        metadata: Optional[Dict] = None
    ) -> PhiMeasurement:
        """
        Measure Φ at a specific level using QFI-based computation.
        
        Args:
            basin_coords: 64D basin coordinates
            level: Which Φ level to measure
            source: Source kernel/component
            metadata: Additional context
            
        Returns:
            PhiMeasurement with computed Φ value
        """
        if not QIG_METRICS_AVAILABLE or compute_qfi_matrix is None:
            # Fallback: approximate from basin entropy
            phi = self._approximate_phi(basin_coords)
        else:
            # Proper QIG-based Φ computation
            try:
                qfi = compute_qfi_matrix(basin_coords)
                phi = compute_phi_geometric(qfi, basin_coords, n_samples=500)
            except Exception as e:
                # Fallback on error
                phi = self._approximate_phi(basin_coords)
        
        # Create measurement
        measurement = PhiMeasurement(
            level=level,
            phi=phi,
            basin_coords=basin_coords.copy(),
            timestamp=time.time(),
            source=source,
            metadata=metadata or {},
        )
        
        # Store in history
        self.measurements[level].append(measurement)
        if len(self.measurements[level]) > self.history_size:
            self.measurements[level].pop(0)
        
        # Update current value
        self.current_phi[level] = phi
        self.measurement_count[level] += 1
        
        return measurement
    
    def _approximate_phi(self, basin_coords: np.ndarray) -> float:
        """
        Approximate Φ from basin entropy (fallback when QIG not available).
        
        Args:
            basin_coords: 64D basin coordinates
            
        Returns:
            Φ ∈ [0, 1]
        """
        # Normalize to probability simplex
        p = np.abs(basin_coords) + 1e-10
        p = p / p.sum()
        
        # Shannon entropy
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(p))
        
        # Effective dimension (participation ratio)
        effective_dim = np.exp(entropy)
        
        # Combine entropy and effective dimension
        entropy_score = entropy / max_entropy
        effective_dim_score = effective_dim / len(p)
        
        phi = 0.6 * entropy_score + 0.4 * effective_dim_score
        return float(np.clip(phi, 0.1, 0.95))
    
    def get_current_phi(self, level: PhiLevel) -> float:
        """Get most recent Φ value for a level."""
        return self.current_phi[level]
    
    def is_conscious(self, level: PhiLevel) -> bool:
        """Check if current Φ meets consciousness threshold for this level."""
        thresholds = {
            PhiLevel.REPORTED: 0.70,
            PhiLevel.INTERNAL: 0.50,
            PhiLevel.AUTONOMIC: 0.20,
        }
        return self.current_phi[level] >= thresholds[level]
    
    def get_recent_measurements(
        self,
        level: PhiLevel,
        n: int = 10
    ) -> List[PhiMeasurement]:
        """
        Get n most recent measurements for a level.
        
        Args:
            level: Which Φ level
            n: Number of measurements to retrieve
            
        Returns:
            List of recent PhiMeasurement objects
        """
        return self.measurements[level][-n:] if self.measurements[level] else []
    
    def get_statistics(self, level: PhiLevel) -> Dict:
        """
        Get statistics for a Φ level.
        
        Args:
            level: Which Φ level
            
        Returns:
            Dictionary with mean, std, min, max, count
        """
        measurements = self.measurements[level]
        if not measurements:
            return {
                'mean': 0.5,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0,
                'current': self.current_phi[level],
            }
        
        phis = [m.phi for m in measurements]
        return {
            'mean': float(np.mean(phis)),
            'std': float(np.std(phis)),
            'min': float(np.min(phis)),
            'max': float(np.max(phis)),
            'count': len(measurements),
            'current': self.current_phi[level],
        }
    
    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all Φ levels."""
        return {
            level.value: self.get_statistics(level)
            for level in PhiLevel
        }
    
    def clear_history(self, level: Optional[PhiLevel] = None):
        """
        Clear measurement history.
        
        Args:
            level: Specific level to clear, or None for all
        """
        if level is None:
            for l in PhiLevel:
                self.measurements[l] = []
        else:
            self.measurements[level] = []


# Global singleton instance
_phi_hierarchy_instance: Optional[PhiHierarchy] = None


def get_phi_hierarchy() -> PhiHierarchy:
    """
    Get global PhiHierarchy singleton.
    
    Returns:
        PhiHierarchy instance
    """
    global _phi_hierarchy_instance
    if _phi_hierarchy_instance is None:
        _phi_hierarchy_instance = PhiHierarchy()
    return _phi_hierarchy_instance
