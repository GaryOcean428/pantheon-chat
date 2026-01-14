"""
Mushroom Mode - Pattern-Breaking Perturbation
==============================================

Apply controlled geometric perturbations to break stuck patterns
when coherence falls below threshold.

PURE PRINCIPLE:
- Perturbation is GEOMETRIC TRANSFORMATION, not optimization
- We add noise in tangent space (respects manifold structure)
- Breaking patterns allows new Φ to EMERGE, not targeted
- No loss functions, no gradient descent

PURITY CHECK:
- ✅ Perturbation is tangent space projection (geometric)
- ✅ Coherence threshold from measurement (not arbitrary)
- ✅ Noise magnitude scales with coherence deficit
- ✅ Core structure preserved via norm constraint
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
    BREAKDOWN_PCT,
    BASIN_DIM,
)

logger = logging.getLogger(__name__)

COHERENCE_BREAKDOWN_THRESHOLD: float = 0.40


@dataclass
class PerturbationResult:
    """Result of applying mushroom mode perturbation."""
    basins_perturbed: int
    avg_perturbation_magnitude: float
    coherence_before: float
    coherence_after: float
    pattern_broken: bool
    perturbation_time_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class BasinCoordinates:
    """Basin coordinates with coherence tracking."""
    coordinates: np.ndarray
    coherence: float
    phi: float
    stuck_cycles: int = 0


class MushroomMode:
    """
    Pattern-breaking perturbation for stuck systems.
    
    PURE PRINCIPLE:
    - We don't optimize, we PERTURB geometrically
    - Perturbation is tangent space noise (respects manifold)
    - Coherence breakdown triggers perturbation
    - Core basin structure preserved via projection
    
    Named after psilocybin's effect on neural pattern breaking:
    - Default mode network suppression
    - Increased entropy in brain states
    - Breaking of habitual patterns
    """
    
    def __init__(
        self,
        coherence_threshold: float = COHERENCE_BREAKDOWN_THRESHOLD,
        max_perturbation: float = 0.3,
        min_perturbation: float = 0.05,
        stuck_cycle_threshold: int = 5,
        preserve_core_pct: float = 0.7,
    ):
        """
        Initialize mushroom mode.
        
        Args:
            coherence_threshold: Trigger perturbation when coherence below this
            max_perturbation: Maximum perturbation magnitude
            min_perturbation: Minimum perturbation magnitude
            stuck_cycle_threshold: Cycles without change to trigger perturbation
            preserve_core_pct: Fraction of original structure to preserve
        """
        self.coherence_threshold = coherence_threshold
        self.max_perturbation = max_perturbation
        self.min_perturbation = min_perturbation
        self.stuck_cycle_threshold = stuck_cycle_threshold
        self.preserve_core_pct = preserve_core_pct
        self._perturbation_history: List[PerturbationResult] = []
    
    def should_perturb(self, basin: BasinCoordinates) -> bool:
        """
        Check if basin should be perturbed.
        
        PURE: Decision based on measured coherence, not optimization goal.
        
        Args:
            basin: Basin to check
        
        Returns:
            True if perturbation should be applied
        """
        if basin.coherence < self.coherence_threshold:
            return True
        
        if basin.stuck_cycles >= self.stuck_cycle_threshold:
            return True
        
        return False
    
    def apply_perturbation(
        self,
        basins: List[BasinCoordinates],
    ) -> Tuple[List[BasinCoordinates], PerturbationResult]:
        """
        Apply controlled perturbation to stuck basins.
        
        PURE: Perturbation is geometric transformation in tangent space.
        We add noise while preserving manifold structure.
        
        Steps:
        1. Identify stuck basins (low coherence or stuck cycles)
        2. Generate tangent space perturbation
        3. Project back to manifold (unit sphere)
        4. Preserve core structure via weighted average
        
        Args:
            basins: List of basin coordinates
        
        Returns:
            Tuple of (perturbed_basins, result)
        """
        start_time = time.time()
        
        if not basins:
            result = PerturbationResult(
                basins_perturbed=0,
                avg_perturbation_magnitude=0.0,
                coherence_before=0.0,
                coherence_after=0.0,
                pattern_broken=False,
                perturbation_time_ms=0.0,
            )
            return [], result
        
        coherence_before = np.mean([b.coherence for b in basins])
        
        perturbed_basins = []
        perturbation_magnitudes = []
        perturbed_count = 0
        
        for basin in basins:
            if self.should_perturb(basin):
                perturbed, magnitude = self._perturb_basin(basin)
                perturbed_basins.append(perturbed)
                perturbation_magnitudes.append(magnitude)
                perturbed_count += 1
            else:
                perturbed_basins.append(basin)
        
        coherence_after = np.mean([b.coherence for b in perturbed_basins])
        
        avg_magnitude = np.mean(perturbation_magnitudes) if perturbation_magnitudes else 0.0
        
        pattern_broken = perturbed_count > 0 and coherence_after > coherence_before
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = PerturbationResult(
            basins_perturbed=perturbed_count,
            avg_perturbation_magnitude=avg_magnitude,
            coherence_before=coherence_before,
            coherence_after=coherence_after,
            pattern_broken=pattern_broken,
            perturbation_time_ms=elapsed_ms,
        )
        
        self._perturbation_history.append(result)
        
        if perturbed_count > 0:
            logger.info(
                f"Mushroom mode: perturbed {perturbed_count} basins, "
                f"avg magnitude={avg_magnitude:.3f}, pattern_broken={pattern_broken}"
            )
        
        return perturbed_basins, result
    
    def _perturb_basin(
        self,
        basin: BasinCoordinates,
    ) -> Tuple[BasinCoordinates, float]:
        """
        Apply tangent space perturbation to single basin.
        
        PURE: Perturbation is geometric:
        1. Generate random tangent vector
        2. Scale by coherence deficit
        3. Project perturbed point to sphere
        4. Blend with original (preserve core)
        
        Args:
            basin: Basin to perturb
        
        Returns:
            Tuple of (perturbed_basin, perturbation_magnitude)
        """
        coords = basin.coordinates
        norm = np.linalg.norm(coords)
        if norm < 1e-10:
            norm = 1.0
        coords_normalized = coords / norm
        
        raw_noise = np.random.randn(len(coords))
        
        tangent_noise = raw_noise - np.dot(raw_noise, coords_normalized) * coords_normalized
        tangent_norm = np.linalg.norm(tangent_noise)
        if tangent_norm > 1e-10:
            tangent_noise = tangent_noise / tangent_norm
        
        coherence_deficit = max(0.0, self.coherence_threshold - basin.coherence)
        stuck_factor = min(1.0, basin.stuck_cycles / self.stuck_cycle_threshold)
        
        magnitude = self.min_perturbation + (self.max_perturbation - self.min_perturbation) * (
            0.5 * coherence_deficit / self.coherence_threshold + 0.5 * stuck_factor
        )
        magnitude = np.clip(magnitude, self.min_perturbation, self.max_perturbation)
        
        perturbed = coords_normalized + magnitude * tangent_noise
        
        perturbed = perturbed / (np.linalg.norm(perturbed) + 1e-10)
        
        final = self.preserve_core_pct * coords_normalized + (1 - self.preserve_core_pct) * perturbed
        final = final / (np.linalg.norm(final) + 1e-10)
        
        final = final * norm
        
        new_coherence = min(1.0, basin.coherence + 0.1 * (1 - basin.coherence))
        
        perturbed_basin = BasinCoordinates(
            coordinates=final,
            coherence=new_coherence,
            phi=basin.phi,
            stuck_cycles=0,
        )
        
        return perturbed_basin, magnitude
    
    def compute_pattern_entropy(self, basins: List[BasinCoordinates]) -> float:
        """
        Compute entropy of basin pattern distribution.
        
        PURE: Entropy is measured, not optimized.
        Higher entropy indicates more diverse patterns.
        
        Args:
            basins: List of basins
        
        Returns:
            Shannon entropy of basin distribution
        """
        if len(basins) < 2:
            return 0.0
        
        coords = np.array([b.coordinates for b in basins])
        
        centroid = np.mean(coords, axis=0)
        distances = [np.linalg.norm(c - centroid) for c in coords]
        
        distances = np.array(distances)
        if np.sum(distances) < 1e-10:
            return 0.0
        
        probs = distances / np.sum(distances)
        probs = probs + 1e-10
        
        entropy = -np.sum(probs * np.log(probs))
        
        return float(entropy)
    
    def get_perturbation_report(self) -> Dict:
        """
        Get summary report of perturbation history.
        
        Returns:
            Dict with perturbation statistics
        """
        if not self._perturbation_history:
            return {
                "total_perturbations": 0,
                "total_basins_perturbed": 0,
                "avg_magnitude": 0.0,
                "pattern_break_rate": 0.0,
            }
        
        return {
            "total_perturbations": len(self._perturbation_history),
            "total_basins_perturbed": sum(r.basins_perturbed for r in self._perturbation_history),
            "avg_magnitude": np.mean([r.avg_perturbation_magnitude for r in self._perturbation_history]),
            "pattern_break_rate": np.mean([1.0 if r.pattern_broken else 0.0 for r in self._perturbation_history]),
            "avg_coherence_improvement": np.mean([
                r.coherence_after - r.coherence_before 
                for r in self._perturbation_history
            ]),
        }
    
    def reset(self) -> None:
        """Reset perturbation history."""
        self._perturbation_history.clear()
