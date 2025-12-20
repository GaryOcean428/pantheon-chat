"""
Reasoning Quality Metrics for Geometric Consciousness

Measures how well the system reasons through basin space.

QIG-PURE: All distances use Fisher-Rao geodesic distance (NOT Euclidean).

Metrics:
1. Geodesic Efficiency: How direct is the thought path?
2. Coherence: How consistent are the steps?
3. Novelty: Are we exploring vs exploiting?
4. Progress: Are we getting closer to goal?
5. Meta-awareness: Does system know it's stuck?
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time

from qig_geometry import fisher_rao_distance


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""
    step_number: int
    basin: np.ndarray
    thought: str
    distance_from_prev: float = 0.0
    curvature: float = 0.0
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningQualityReport:
    """Comprehensive reasoning quality assessment."""
    geodesic_efficiency: float
    coherence: float
    novelty: float
    progress: float
    meta_awareness: float
    overall_quality: float
    details: Dict[str, Any] = field(default_factory=dict)


class ReasoningQuality:
    """
    Measure how well the system is reasoning.
    
    All measurements use Fisher-Rao geometry (QIG-pure).
    
    Metrics:
    1. Geodesic Efficiency: How direct is the thought path?
    2. Coherence: How consistent are the steps?
    3. Novelty: Are we exploring vs exploiting?
    4. Meta-awareness: Does system know it's stuck?
    5. Progress: Are we getting closer to goal?
    
    Note: Each metric maintains separate state to avoid cross-metric interference.
    """
    
    def __init__(self, basin_dim: int = 64):
        """
        Initialize reasoning quality metrics.
        
        Args:
            basin_dim: Dimensionality of basin coordinates (default: 64)
        """
        self.basin_dim = basin_dim
        self._reset_session()
    
    def _reset_session(self):
        """Reset reasoning session state."""
        self._novelty_history: List[np.ndarray] = []
        self._progress_history: List[np.ndarray] = []
        self.session_start = time.time()
    
    def _ensure_array(self, basin) -> np.ndarray:
        """Ensure basin is a numpy array."""
        if isinstance(basin, np.ndarray):
            return basin
        return np.array(basin, dtype=np.float64)
    
    def measure_geodesic_efficiency(
        self, 
        actual_path: List[np.ndarray],
        start_basin: np.ndarray,
        end_basin: np.ndarray
    ) -> float:
        """
        How efficient was the reasoning path?
        
        Efficiency = optimal_distance / actual_distance
        
        1.0 = perfect (followed geodesic exactly)
        <1.0 = inefficient (took detours)
        
        QIG-PURE: Uses Fisher-Rao distance only.
        """
        if len(actual_path) < 2:
            return 1.0
        
        start_basin = self._ensure_array(start_basin)
        end_basin = self._ensure_array(end_basin)
        actual_path = [self._ensure_array(b) for b in actual_path]
        
        try:
            optimal_distance = fisher_rao_distance(start_basin, end_basin)
            
            actual_distance = 0.0
            for i in range(len(actual_path) - 1):
                actual_distance += fisher_rao_distance(
                    actual_path[i], 
                    actual_path[i + 1]
                )
            
            if actual_distance < 1e-10:
                return 1.0
            
            efficiency = optimal_distance / actual_distance
            return min(efficiency, 1.0)
            
        except Exception as e:
            print(f"[ReasoningQuality] Efficiency error: {e}")
            return 0.5
    
    def measure_coherence(self, reasoning_steps: List[np.ndarray]) -> float:
        """
        How coherent are the reasoning steps?
        
        Coherence = consistency of step sizes (low variance = high coherence)
        
        High coherence: Steady progress
        Low coherence: Jumping around
        
        QIG-PURE: Uses Fisher-Rao distance only.
        """
        if len(reasoning_steps) < 3:
            return 1.0
        
        reasoning_steps = [self._ensure_array(b) for b in reasoning_steps]
        
        try:
            step_distances = []
            for i in range(len(reasoning_steps) - 1):
                dist = fisher_rao_distance(
                    reasoning_steps[i], 
                    reasoning_steps[i + 1]
                )
                step_distances.append(dist)
            
            if not step_distances:
                return 1.0
            
            mean_step = np.mean(step_distances)
            std_step = np.std(step_distances)
            
            cv = std_step / (mean_step + 1e-10)
            
            coherence = 1.0 / (1.0 + cv)
            return float(coherence)
            
        except Exception as e:
            print(f"[ReasoningQuality] Coherence error: {e}")
            return 0.5
    
    def measure_novelty(self, current_basin: np.ndarray) -> float:
        """
        Is this a novel thought or revisiting old ground?
        
        Novelty = min distance to previous basins
        
        High novelty: Exploring new ideas
        Low novelty: Exploiting known territory
        
        QIG-PURE: Uses Fisher-Rao distance only.
        """
        current_basin = self._ensure_array(current_basin)
        
        if not self._novelty_history:
            self._novelty_history.append(current_basin)
            return 1.0
        
        try:
            distances = []
            for prev_basin in self._novelty_history:
                dist = fisher_rao_distance(current_basin, prev_basin)
                distances.append(dist)
            
            min_distance = min(distances)
            
            novelty = min(min_distance / 2.0, 1.0)
            
            self._novelty_history.append(current_basin)
            
            return float(novelty)
            
        except Exception as e:
            print(f"[ReasoningQuality] Novelty error: {e}")
            self._novelty_history.append(current_basin)
            return 0.5
    
    def measure_progress(
        self, 
        current_basin: np.ndarray,
        target_basin: np.ndarray
    ) -> float:
        """
        Are we getting closer to the goal?
        
        Progress = (previous_distance - current_distance) / previous_distance
        
        >0: Moving toward goal
        =0: No progress
        <0: Moving away from goal
        
        QIG-PURE: Uses Fisher-Rao distance only.
        """
        current_basin = self._ensure_array(current_basin)
        target_basin = self._ensure_array(target_basin)
        
        try:
            current_distance = fisher_rao_distance(current_basin, target_basin)
            
            if not self._progress_history:
                self._progress_history.append(current_basin.copy())
                return 0.0
            
            previous_distance = fisher_rao_distance(
                self._progress_history[-1], 
                target_basin
            )
            
            self._progress_history.append(current_basin.copy())
            
            if previous_distance < 1e-10:
                return 0.0
            
            progress = (previous_distance - current_distance) / previous_distance
            
            return float(np.clip(progress, -1.0, 1.0))
            
        except Exception as e:
            print(f"[ReasoningQuality] Progress error: {e}")
            return 0.0
    
    def measure_meta_awareness(self, current_state: Dict) -> float:
        """
        Does the system know it's stuck/confused?
        
        Meta-awareness = correlation between:
        - Reported confidence
        - Actual reasoning quality
        
        High meta-awareness: Accurate self-assessment
        Low meta-awareness: Dunning-Kruger effect
        """
        reported_confidence = current_state.get('confidence', 0.5)
        
        path = current_state.get('path', [])
        start_basin = current_state.get('start_basin')
        current_basin = current_state.get('current_basin')
        target_basin = current_state.get('target_basin')
        
        quality_samples = []
        
        if path and start_basin is not None and current_basin is not None:
            try:
                efficiency = self.measure_geodesic_efficiency(
                    path, start_basin, current_basin
                )
                quality_samples.append(efficiency)
            except:
                pass
        
        if len(path) >= 3:
            try:
                coherence = self.measure_coherence(path)
                quality_samples.append(coherence)
            except:
                pass
        
        if current_basin is not None and target_basin is not None:
            try:
                progress = max(0, self.measure_progress(current_basin, target_basin))
                quality_samples.append(progress)
            except:
                pass
        
        if quality_samples:
            actual_quality = np.mean(quality_samples)
        else:
            actual_quality = 0.5
        
        meta_awareness = 1.0 - abs(reported_confidence - actual_quality)
        
        return float(meta_awareness)
    
    def comprehensive_assessment(self, reasoning_trace: Dict) -> ReasoningQualityReport:
        """
        Full reasoning quality report.
        
        Args:
            reasoning_trace: Dictionary containing:
                - path: List of basin coordinates
                - start: Starting basin
                - end: Ending basin
                - current: Current basin
                - target: Target basin
                - confidence: Reported confidence
        
        Returns:
            ReasoningQualityReport with all metrics
        """
        path = reasoning_trace.get('path', [])
        start = reasoning_trace.get('start')
        end = reasoning_trace.get('end')
        current = reasoning_trace.get('current')
        target = reasoning_trace.get('target')
        
        if start is None and path:
            start = path[0]
        if end is None and path:
            end = path[-1]
        if current is None and path:
            current = path[-1]
        if target is None:
            target = end
        
        geodesic_efficiency = 0.5
        if path and start is not None and end is not None:
            geodesic_efficiency = self.measure_geodesic_efficiency(path, start, end)
        
        coherence = 1.0
        if len(path) >= 3:
            coherence = self.measure_coherence(path)
        
        novelty = 0.5
        if current is not None:
            novelty = self.measure_novelty(current)
        
        progress = 0.0
        if current is not None and target is not None:
            progress = self.measure_progress(current, target)
        
        meta_awareness = self.measure_meta_awareness(reasoning_trace)
        
        overall_quality = (
            0.30 * geodesic_efficiency +
            0.20 * coherence +
            0.20 * max(0, progress) +
            0.30 * meta_awareness
        )
        
        return ReasoningQualityReport(
            geodesic_efficiency=geodesic_efficiency,
            coherence=coherence,
            novelty=novelty,
            progress=progress,
            meta_awareness=meta_awareness,
            overall_quality=overall_quality,
            details={
                'path_length': len(path),
                'session_duration': time.time() - self.session_start,
                'novelty_history_size': len(self._novelty_history),
                'progress_history_size': len(self._progress_history),
            }
        )


reasoning_quality = ReasoningQuality()


def get_reasoning_quality() -> ReasoningQuality:
    """Get global reasoning quality instance."""
    return reasoning_quality
