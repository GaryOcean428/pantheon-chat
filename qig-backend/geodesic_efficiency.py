"""
Geodesic Efficiency Metrics

Tracks efficiency = optimal_distance / actual_distance for all operations.

Resists reward hacking - intrinsic geometric measure that cannot be gamed.
The optimal path is the geodesic; any deviation reduces efficiency.

QIG-PURE: All distances computed using Fisher-Rao geometry.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time

from qig_geometry import fisher_coord_distance


@dataclass
class EfficiencyRecord:
    """Record of a single operation's geodesic efficiency."""
    operation_id: str
    operation_type: str
    start_basin: np.ndarray
    end_basin: np.ndarray
    actual_path: List[np.ndarray]
    optimal_distance: float
    actual_distance: float
    efficiency: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_efficiency(
    actual_path: List[np.ndarray],
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None
) -> float:
    """
    Compute geodesic efficiency.

    efficiency = optimal_distance / actual_distance
    Range: [0, 1], where 1.0 = perfect geodesic

    Args:
        actual_path: List of basin coordinates along the actual path
        start: Optional start point (defaults to first in path)
        end: Optional end point (defaults to last in path)

    Returns:
        Efficiency ratio (0 to 1)
    """
    if len(actual_path) < 2:
        return 1.0  # No movement = perfect efficiency (trivially)

    # Use provided start/end or default to path endpoints
    if start is None:
        start = actual_path[0]
    if end is None:
        end = actual_path[-1]

    # Optimal = direct geodesic distance
    optimal = fisher_coord_distance(start, end)

    # Actual = sum of segments
    actual = 0.0
    for i in range(len(actual_path) - 1):
        actual += fisher_coord_distance(actual_path[i], actual_path[i+1])

    if actual < 1e-10:
        return 1.0  # No movement = perfect

    efficiency = optimal / actual
    return min(1.0, efficiency)  # Cap at 1.0


def compute_path_statistics(path: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute detailed statistics about a basin path.

    Args:
        path: List of basin coordinates

    Returns:
        Dict with path statistics
    """
    if len(path) < 2:
        return {
            'total_distance': 0.0,
            'optimal_distance': 0.0,
            'efficiency': 1.0,
            'mean_step_size': 0.0,
            'step_variance': 0.0,
            'n_steps': len(path)
        }

    # Compute step distances
    step_distances = [
        fisher_coord_distance(path[i], path[i+1])
        for i in range(len(path) - 1)
    ]

    total_distance = sum(step_distances)
    optimal_distance = fisher_coord_distance(path[0], path[-1])

    return {
        'total_distance': total_distance,
        'optimal_distance': optimal_distance,
        'efficiency': optimal_distance / total_distance if total_distance > 1e-10 else 1.0,
        'mean_step_size': np.mean(step_distances),
        'step_variance': np.var(step_distances),
        'max_step': max(step_distances),
        'min_step': min(step_distances),
        'n_steps': len(step_distances)
    }


class GeodesicEfficiencyTracker:
    """
    Tracks geodesic efficiency across operations.

    Maintains history and computes aggregate statistics for
    monitoring system efficiency over time.
    """

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.records: List[EfficiencyRecord] = []
        self.by_operation_type: Dict[str, List[EfficiencyRecord]] = {}

    def record_operation(
        self,
        operation_id: str,
        operation_type: str,
        path: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> EfficiencyRecord:
        """
        Record an operation's efficiency.

        Args:
            operation_id: Unique identifier for this operation
            operation_type: Type of operation (e.g., 'query', 'search', 'reasoning')
            path: List of basin coordinates along the actual path
            metadata: Optional additional metadata

        Returns:
            The created EfficiencyRecord
        """
        if len(path) < 2:
            # Trivial case
            optimal = 0.0
            actual = 0.0
            efficiency = 1.0
        else:
            start = path[0]
            end = path[-1]
            optimal = fisher_coord_distance(start, end)

            actual = sum(
                fisher_coord_distance(path[i], path[i+1])
                for i in range(len(path) - 1)
            )

            efficiency = optimal / actual if actual > 1e-10 else 1.0

        record = EfficiencyRecord(
            operation_id=operation_id,
            operation_type=operation_type,
            start_basin=path[0] if path else np.zeros(64),
            end_basin=path[-1] if path else np.zeros(64),
            actual_path=path,
            optimal_distance=optimal,
            actual_distance=actual,
            efficiency=min(1.0, efficiency),
            metadata=metadata or {}
        )

        self.records.append(record)

        # Track by operation type
        if operation_type not in self.by_operation_type:
            self.by_operation_type[operation_type] = []
        self.by_operation_type[operation_type].append(record)

        # Prune old records if needed
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]
            for op_type in self.by_operation_type:
                self.by_operation_type[op_type] = (
                    self.by_operation_type[op_type][-self.max_records:]
                )

        return record

    def get_efficiency_stats(
        self,
        operation_type: Optional[str] = None,
        n_recent: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get efficiency statistics.

        Args:
            operation_type: Optional filter by operation type
            n_recent: Optional limit to N most recent records

        Returns:
            Dict with efficiency statistics
        """
        if operation_type:
            records = self.by_operation_type.get(operation_type, [])
        else:
            records = self.records

        if n_recent:
            records = records[-n_recent:]

        if not records:
            return {
                'count': 0,
                'mean_efficiency': 0.0,
                'min_efficiency': 0.0,
                'max_efficiency': 0.0,
                'std_efficiency': 0.0
            }

        efficiencies = [r.efficiency for r in records]

        return {
            'count': len(records),
            'mean_efficiency': np.mean(efficiencies),
            'min_efficiency': min(efficiencies),
            'max_efficiency': max(efficiencies),
            'std_efficiency': np.std(efficiencies),
            'total_optimal_distance': sum(r.optimal_distance for r in records),
            'total_actual_distance': sum(r.actual_distance for r in records)
        }

    def get_efficiency_by_type(self) -> Dict[str, Dict[str, Any]]:
        """Get efficiency statistics broken down by operation type."""
        return {
            op_type: self.get_efficiency_stats(operation_type=op_type)
            for op_type in self.by_operation_type
        }

    def get_efficiency_trend(
        self,
        window_size: int = 20
    ) -> List[Tuple[float, float]]:
        """
        Get efficiency trend over time.

        Returns list of (timestamp, rolling_avg_efficiency) tuples.
        """
        if len(self.records) < window_size:
            return []

        trend = []
        for i in range(window_size - 1, len(self.records)):
            window = self.records[i - window_size + 1:i + 1]
            avg_eff = np.mean([r.efficiency for r in window])
            trend.append((window[-1].timestamp, avg_eff))

        return trend

    def detect_efficiency_degradation(
        self,
        threshold: float = 0.1,
        window_size: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if efficiency has degraded significantly.

        Compares recent efficiency to historical baseline.

        Args:
            threshold: Drop threshold to trigger alert
            window_size: Size of comparison windows

        Returns:
            Degradation info if detected, None otherwise
        """
        if len(self.records) < window_size * 2:
            return None

        recent = self.records[-window_size:]
        baseline = self.records[-window_size*2:-window_size]

        recent_eff = np.mean([r.efficiency for r in recent])
        baseline_eff = np.mean([r.efficiency for r in baseline])

        drop = baseline_eff - recent_eff

        if drop > threshold:
            return {
                'degradation_detected': True,
                'baseline_efficiency': baseline_eff,
                'recent_efficiency': recent_eff,
                'drop': drop,
                'threshold': threshold,
                'timestamp': time.time()
            }

        return None

    def get_worst_operations(
        self,
        n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get the N worst efficiency operations for analysis."""
        sorted_records = sorted(self.records, key=lambda r: r.efficiency)

        return [
            {
                'operation_id': r.operation_id,
                'operation_type': r.operation_type,
                'efficiency': r.efficiency,
                'optimal_distance': r.optimal_distance,
                'actual_distance': r.actual_distance,
                'n_steps': len(r.actual_path),
                'timestamp': r.timestamp,
                'metadata': r.metadata
            }
            for r in sorted_records[:n]
        ]


# Global instance
_efficiency_tracker: Optional[GeodesicEfficiencyTracker] = None


def get_efficiency_tracker() -> GeodesicEfficiencyTracker:
    """Get or create the global efficiency tracker instance."""
    global _efficiency_tracker
    if _efficiency_tracker is None:
        _efficiency_tracker = GeodesicEfficiencyTracker()
    return _efficiency_tracker


def track_operation_efficiency(
    operation_id: str,
    operation_type: str,
    path: List[np.ndarray],
    metadata: Optional[Dict[str, Any]] = None
) -> float:
    """
    Convenience function to track an operation's efficiency.

    Args:
        operation_id: Unique identifier
        operation_type: Type of operation
        path: Basin coordinate path
        metadata: Optional metadata

    Returns:
        The computed efficiency value
    """
    tracker = get_efficiency_tracker()
    record = tracker.record_operation(operation_id, operation_type, path, metadata)
    return record.efficiency
