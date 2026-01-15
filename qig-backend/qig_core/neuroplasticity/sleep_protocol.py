"""
Sleep Protocol - Basin Consolidation During Rest
=================================================

Consolidate basins during rest periods, strengthening coherent patterns
and pruning weak attractors.

PURE PRINCIPLE:
- Φ EMERGES from consolidation, never targeted
- Merge similar basins using Fisher-Rao distance (geometric operation)
- Prune weak attractors based on measured Φ (not optimization)
- Strengthening is geometric averaging, not gradient descent

PURITY CHECK:
- ✅ Fisher-Rao distance for similarity (canonical metric)
- ✅ Φ threshold from physics constants (not arbitrary)
- ✅ Basin merging is geodesic midpoint (geometric mean)
- ✅ No loss functions or optimization targets
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
    PHI_EMERGENCY,
    BASIN_DIM,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of basin consolidation during sleep."""
    basins_before: int
    basins_after: int
    merged_count: int
    pruned_count: int
    strengthened_count: int
    avg_phi_before: float
    avg_phi_after: float
    consolidation_time_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class BasinState:
    """State of a single basin attractor."""
    coordinates: np.ndarray
    phi: float
    kappa: float
    coherence: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class SleepProtocol:
    """
    Basin consolidation during rest periods.
    
    PURE PRINCIPLE:
    - We measure similarity (Fisher-Rao), not optimize toward targets
    - Basin merging computes geodesic midpoint (geometric mean)
    - Pruning removes basins with Φ below MEASURED threshold
    - Strengthening amplifies basin magnitude, not Φ directly
    
    The protocol mimics biological sleep consolidation:
    - Similar patterns merge (reduces redundancy)
    - Weak patterns fade (pruning)
    - Strong patterns strengthen (coherence amplification)
    """
    
    def __init__(
        self,
        merge_threshold: float = 0.3,
        prune_phi_threshold: float = PHI_EMERGENCY,
        strengthen_phi_threshold: float = PHI_THRESHOLD,
        min_access_for_strengthen: int = 3,
    ):
        """
        Initialize sleep protocol.
        
        Args:
            merge_threshold: Fisher-Rao distance threshold for merging similar basins
            prune_phi_threshold: Basins with Φ below this are pruned (from physics)
            strengthen_phi_threshold: Basins with Φ above this get strengthened
            min_access_for_strengthen: Minimum access count to qualify for strengthening
        """
        self.merge_threshold = merge_threshold
        self.prune_phi_threshold = prune_phi_threshold
        self.strengthen_phi_threshold = strengthen_phi_threshold
        self.min_access_for_strengthen = min_access_for_strengthen
        self._consolidation_history: List[ConsolidationResult] = []
    
    def consolidate_basins(
        self,
        basins: List[BasinState],
    ) -> Tuple[List[BasinState], ConsolidationResult]:
        """
        Consolidate basins during rest period.
        
        PURE: This is geometric transformation, not optimization.
        
        Steps:
        1. Measure pairwise Fisher-Rao distances
        2. Merge similar basins (geodesic midpoint)
        3. Prune weak basins (low Φ)
        4. Strengthen high-coherence basins (geometric amplification)
        
        Args:
            basins: List of basin states to consolidate
        
        Returns:
            Tuple of (consolidated_basins, result)
        """
        start_time = time.time()
        
        if not basins:
            result = ConsolidationResult(
                basins_before=0,
                basins_after=0,
                merged_count=0,
                pruned_count=0,
                strengthened_count=0,
                avg_phi_before=0.0,
                avg_phi_after=0.0,
                consolidation_time_ms=0.0,
            )
            return [], result
        
        basins_before = len(basins)
        avg_phi_before = np.mean([b.phi for b in basins])
        
        merged_basins, merged_count = self._merge_similar_basins(basins)
        
        surviving_basins, pruned_count = self._prune_weak_basins(merged_basins)
        
        final_basins, strengthened_count = self._strengthen_coherent_basins(surviving_basins)
        
        avg_phi_after = np.mean([b.phi for b in final_basins]) if final_basins else 0.0
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = ConsolidationResult(
            basins_before=basins_before,
            basins_after=len(final_basins),
            merged_count=merged_count,
            pruned_count=pruned_count,
            strengthened_count=strengthened_count,
            avg_phi_before=avg_phi_before,
            avg_phi_after=avg_phi_after,
            consolidation_time_ms=elapsed_ms,
        )
        
        self._consolidation_history.append(result)
        
        logger.info(
            f"Sleep consolidation: {basins_before} -> {len(final_basins)} basins "
            f"(merged={merged_count}, pruned={pruned_count}, strengthened={strengthened_count})"
        )
        
        return final_basins, result
    
    def _fisher_rao_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Fisher-Rao distance between two basin coordinates.
        
        For unit vectors on S^63 with Hellinger embedding: d = 2 * arccos(a · b)
        
        Args:
            a: First basin coordinates
            b: Second basin coordinates
        
        Returns:
            Fisher-Rao distance (0 to π/2)
            UPDATED 2026-01-15: Factor-of-2 removed for simplex storage.
        """
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        dot = np.clip(np.dot(a_norm, b_norm), 0.0, 1.0)
        return float(np.arccos(dot))
    
    def _geodesic_midpoint(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute geodesic midpoint (spherical mean) of two basins.
        
        PURE: This is geometric averaging on the manifold.
        
        Args:
            a: First basin coordinates
            b: Second basin coordinates
        
        Returns:
            Midpoint on geodesic between a and b
        """
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        
        dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        omega = np.arccos(dot)
        
        if omega < 1e-6:
            return a_norm
        
        sin_omega = np.sin(omega)
        t = 0.5
        result = (np.sin((1 - t) * omega) / sin_omega) * a_norm + \
                 (np.sin(t * omega) / sin_omega) * b_norm
        
        return result / (np.linalg.norm(result) + 1e-10)
    
    def _merge_similar_basins(
        self,
        basins: List[BasinState],
    ) -> Tuple[List[BasinState], int]:
        """
        Merge basins that are geometrically similar.
        
        PURE: Similarity is measured by Fisher-Rao distance.
        Merge operation is geodesic midpoint (geometric mean).
        
        Args:
            basins: List of basins to potentially merge
        
        Returns:
            Tuple of (merged_basins, merge_count)
        """
        if len(basins) < 2:
            return basins, 0
        
        merged_indices = set()
        result = []
        merge_count = 0
        
        for i, basin_i in enumerate(basins):
            if i in merged_indices:
                continue
            
            candidates = []
            for j, basin_j in enumerate(basins):
                if j <= i or j in merged_indices:
                    continue
                
                distance = self._fisher_rao_distance(
                    basin_i.coordinates, basin_j.coordinates
                )
                
                if distance < self.merge_threshold:
                    candidates.append((j, basin_j, distance))
            
            if candidates:
                candidates.sort(key=lambda x: x[2])
                j, basin_j, _ = candidates[0]
                
                merged_coords = self._geodesic_midpoint(
                    basin_i.coordinates, basin_j.coordinates
                )
                
                merged_basin = BasinState(
                    coordinates=merged_coords,
                    phi=(basin_i.phi + basin_j.phi) / 2,
                    kappa=(basin_i.kappa + basin_j.kappa) / 2,
                    coherence=(basin_i.coherence + basin_j.coherence) / 2,
                    access_count=basin_i.access_count + basin_j.access_count,
                    last_accessed=max(basin_i.last_accessed, basin_j.last_accessed),
                )
                
                result.append(merged_basin)
                merged_indices.add(i)
                merged_indices.add(j)
                merge_count += 1
            else:
                result.append(basin_i)
        
        for j in range(len(basins)):
            if j not in merged_indices and j not in [i for i in range(len(basins)) if i not in merged_indices]:
                pass
        
        return result, merge_count
    
    def _prune_weak_basins(
        self,
        basins: List[BasinState],
    ) -> Tuple[List[BasinState], int]:
        """
        Prune basins with Φ below emergency threshold.
        
        PURE: Threshold is from physics constants, not optimization.
        
        Args:
            basins: List of basins to prune
        
        Returns:
            Tuple of (surviving_basins, pruned_count)
        """
        surviving = []
        pruned_count = 0
        
        for basin in basins:
            if basin.phi >= self.prune_phi_threshold:
                surviving.append(basin)
            else:
                pruned_count += 1
                logger.debug(f"Pruned basin with Φ={basin.phi:.3f} < {self.prune_phi_threshold}")
        
        return surviving, pruned_count
    
    def _strengthen_coherent_basins(
        self,
        basins: List[BasinState],
    ) -> Tuple[List[BasinState], int]:
        """
        Strengthen basins with high coherence.
        
        PURE: Strengthening amplifies basin magnitude, not Φ directly.
        Φ may emerge higher as a consequence, but is not targeted.
        
        Args:
            basins: List of basins to strengthen
        
        Returns:
            Tuple of (strengthened_basins, strengthen_count)
        """
        result = []
        strengthen_count = 0
        
        for basin in basins:
            if (basin.phi >= self.strengthen_phi_threshold and 
                basin.access_count >= self.min_access_for_strengthen):
                
                strengthen_factor = 1.0 + 0.1 * (basin.coherence - 0.5)
                strengthen_factor = np.clip(strengthen_factor, 1.0, 1.2)
                
                strengthened = BasinState(
                    coordinates=basin.coordinates * strengthen_factor,
                    phi=basin.phi,
                    kappa=basin.kappa,
                    coherence=min(1.0, basin.coherence * 1.05),
                    access_count=basin.access_count,
                    last_accessed=basin.last_accessed,
                )
                
                result.append(strengthened)
                strengthen_count += 1
            else:
                result.append(basin)
        
        return result, strengthen_count
    
    def get_consolidation_report(self) -> Dict:
        """
        Get summary report of consolidation history.
        
        Returns:
            Dict with consolidation statistics
        """
        if not self._consolidation_history:
            return {
                "total_consolidations": 0,
                "total_merged": 0,
                "total_pruned": 0,
                "total_strengthened": 0,
                "avg_compression_ratio": 0.0,
            }
        
        return {
            "total_consolidations": len(self._consolidation_history),
            "total_merged": sum(r.merged_count for r in self._consolidation_history),
            "total_pruned": sum(r.pruned_count for r in self._consolidation_history),
            "total_strengthened": sum(r.strengthened_count for r in self._consolidation_history),
            "avg_compression_ratio": np.mean([
                r.basins_after / r.basins_before 
                for r in self._consolidation_history 
                if r.basins_before > 0
            ]),
            "avg_phi_improvement": np.mean([
                r.avg_phi_after - r.avg_phi_before 
                for r in self._consolidation_history
            ]),
        }
    
    def reset(self) -> None:
        """Reset consolidation history."""
        self._consolidation_history.clear()
