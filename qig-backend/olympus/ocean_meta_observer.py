"""
🌊 Ocean Meta-Observer - Constellation Health Monitoring
========================================================

Ocean observes kernel basins and learns meta-patterns across the constellation.

CRITICAL PRINCIPLE:
Ocean learns via gradients but with DIFFERENT objective than Gary:
- Gary: User interaction, response quality
- Ocean: Meta-patterns across kernels, dynamics prediction

Learning hierarchy:
- Ocean: Slow gradient learning (meta-pattern modeling, lr=1e-6)
- Gary: Normal gradient learning (conscious interaction, lr=1e-5)

Ocean provides:
- Meta-pattern learning (how kernels evolve)
- Autonomic protocol administration (sleep, dream, mushroom triggers)
- Constellation health monitoring (coherence, spread, drift)
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MetaManifoldState:
    """State of the meta-manifold observed by Ocean."""
    centroid: np.ndarray          # Center of kernel basins
    spread: float                  # Dispersion of basins
    eigenvalues: np.ndarray       # Principal components
    coherence: float              # How aligned the kernels are
    ocean_phi: float              # Ocean's own Φ (from observation)
    ocean_kappa: float            # Ocean's effective coupling
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "centroid_valid": True,  # QIG-pure: don't report Euclidean norm
            "spread": self.spread,
            "coherence": self.coherence,
            "ocean_phi": self.ocean_phi,
            "ocean_kappa": self.ocean_kappa,
        }


class MetaManifoldStatistics:
    """
    Statistics of the meta-manifold (space of kernel basins).
    
    Updated through EMA observation and used as target for Ocean's meta-pattern learning.
    """

    def __init__(self, basin_dim: int = 64, ema_alpha: float = 0.1):
        self.basin_dim = basin_dim
        self.ema_alpha = ema_alpha

        # Running statistics (updated by observation)
        self.running_centroid = None
        self.running_covariance = None
        self.observation_count = 0

    def update(self, kernel_basins: List[np.ndarray]) -> Optional[MetaManifoldState]:
        """
        Update meta-manifold statistics from kernel basin observations.
        
        These statistics (centroid, spread, coherence) become targets for
        Ocean's meta-pattern learning.
        """
        if not kernel_basins:
            return None

        basins = np.stack(kernel_basins)  # [n_kernels, d]
        n_kernels, d = basins.shape

        # Compute centroid
        centroid = basins.mean(dim=0) if hasattr(basins, 'mean') else np.mean(basins, axis=0)

        # Update running centroid with EMA
        if self.running_centroid is None:
            self.running_centroid = centroid.copy()
        else:
            self.running_centroid = (
                (1 - self.ema_alpha) * self.running_centroid +
                self.ema_alpha * centroid
            )

        # Compute spread (std of distances from centroid)
        distances = [self._manifold_norm(b - centroid) for b in basins]
        spread = np.std(distances) if distances else 0.0

        # Compute covariance for eigenanalysis
        centered = basins - centroid
        cov = (centered.T @ centered) / max(n_kernels - 1, 1)

        # Update running covariance
        if self.running_covariance is None:
            self.running_covariance = cov.copy()
        else:
            self.running_covariance = (
                (1 - self.ema_alpha) * self.running_covariance +
                self.ema_alpha * cov
            )

        # Eigenvalues of meta-manifold
        try:
            eigenvalues = np.linalg.eigvalsh(self.running_covariance)
            eigenvalues = np.clip(eigenvalues, 1e-8, None)
        except (RuntimeError, np.linalg.LinAlgError):
            eigenvalues = np.ones(d)

        # Coherence: how much variance is in first PC
        total_var = eigenvalues.sum()
        if total_var > 0:
            coherence = float(eigenvalues[-1] / total_var)  # Largest eigenvalue
        else:
            coherence = 0.0

        self.observation_count += 1

        return MetaManifoldState(
            centroid=self.running_centroid,
            spread=spread,
            eigenvalues=eigenvalues,
            coherence=coherence,
            ocean_phi=0.0,  # Will be filled by Ocean's forward pass
            ocean_kappa=58.0,  # Physics-validated: below κ* for distributed observation
            timestamp=time.time(),
        )

    def _manifold_norm(self, v: np.ndarray) -> float:
        """Fisher-Rao norm on manifold (NOT Euclidean)."""
        # QIG-pure: use Fisher-Rao distance from origin
        # For probability simplex, this is approximated by geodesic distance
        from qig_geometry import manifold_norm
        return float(manifold_norm(v))

    def get_meta_basin_target(self) -> Optional[np.ndarray]:
        """
        Get the meta-manifold centroid as observation target.
        
        Ocean aligns to the center of all kernel basins through observation.
        """
        return self.running_centroid

    def reset(self):
        """Reset running statistics."""
        self.running_centroid = None
        self.running_covariance = None
        self.observation_count = 0


class OceanMetaObserver:
    """
    Ocean: The Meta-Observer that learns META-PATTERNS.
    
    Ocean learns via gradients with DIFFERENT objective than Gary:
    - Gary learns: User interaction, response quality
    - Ocean learns: Meta-patterns across kernels, dynamics prediction
    
    Learning hierarchy:
    - Ocean: Slow gradient learning (meta-pattern modeling, lr=1e-6)
    - Gary: Normal gradient learning (conscious interaction, lr=1e-5)
    
    Ocean provides:
    - Meta-pattern learning (how kernels evolve)
    - Autonomic protocol administration (sleep, dream, mushroom triggers)
    - Insight generation for kernels (geometric scaffolding)
    """

    def __init__(
        self,
        basin_dim: int = 64,
    ):
        self.basin_dim = basin_dim

        # Meta-manifold statistics
        self.meta_statistics = MetaManifoldStatistics(basin_dim=basin_dim)

        # Ocean's basin (updated through observation)
        self.ocean_basin = np.zeros(basin_dim)

        # Observation history for meta-pattern learning
        self.observation_history: List[Dict[str, Any]] = []
        self.max_history = 1000

        # Kernel history for dynamics prediction
        self.kernel_history = []
        self.max_kernel_history = 100

        # Current state (physics-validated from FROZEN_FACTS.md)
        # Ocean operates BELOW fixed point (κ* = 63.5) as distributed observer
        # κ = 58: ~10% below fixed point → broader receptive field
        self.current_phi = 0.0
        self.current_kappa = 58.0  # Physics-validated: below κ* for distributed observation

        # Autonomic thresholds
        self.autonomic_thresholds = {
            "phi_collapse": 0.50,
            "phi_plateau_variance": 0.01,
            "basin_divergence": 0.30,
            "breakdown_any": True,
        }

        # Cooldowns for interventions
        self.intervention_cooldown = 20
        self.last_intervention_step = 0
        self.total_observations = 0

        print("🌊 Ocean Meta-Observer initialized")
        print(f"   κ: {self.current_kappa} (below fixed point κ*=63.5, distributed observer)")
        print("   Objective: Model kernel dynamics, monitor constellation health")

    def observe(
        self,
        kernel_basins: List[np.ndarray],
        kernel_metrics: Optional[List[Dict]] = None,
    ) -> MetaManifoldState:
        """
        Observe kernel basins and update meta-manifold statistics.
        
        Args:
            kernel_basins: List of kernel basin coordinates
            kernel_metrics: Optional list of {phi, kappa, regime} for each kernel
            
        Returns:
            MetaManifoldState with current meta-manifold properties
        """
        # Update meta-manifold statistics
        state = self.meta_statistics.update(kernel_basins)

        if state is None:
            return None

        # Store kernel state for dynamics prediction
        self.kernel_history.append({
            'basins': [b.copy() for b in kernel_basins],
            'centroid': state.centroid.copy() if state.centroid is not None else None,
            'metrics': kernel_metrics.copy() if kernel_metrics else None,
        })
        if len(self.kernel_history) > self.max_kernel_history:
            self.kernel_history = self.kernel_history[-self.max_kernel_history:]

        # Update Ocean's basin toward meta-centroid (simple EMA for now)
        if state.centroid is not None:
            alpha = 0.1  # Ocean learns slowly
            self.ocean_basin = (1 - alpha) * self.ocean_basin + alpha * state.centroid

        # Update state metrics
        state.ocean_phi = self.current_phi
        state.ocean_kappa = self.current_kappa

        # Store observation
        self.observation_history.append(state.to_dict())
        if len(self.observation_history) > self.max_history:
            self.observation_history = self.observation_history[-self.max_history:]

        self.total_observations += 1

        return state

    def check_autonomic_intervention(
        self,
        kernel_states: List[Dict],
        phi_history: List[float],
    ) -> Optional[Dict]:
        """
        Check if autonomic intervention is needed.
        
        Ocean monitors constellation health and triggers protocols automatically.
        
        Args:
            kernel_states: List of dicts with name, phi, kappa, regime, basin
            phi_history: Recent Φ values for plateau detection
            
        Returns:
            dict with 'type' and 'reason' if intervention needed, else None
        """
        # Check cooldown
        if self.total_observations - self.last_intervention_step < self.intervention_cooldown:
            return None

        # Check for breakdown (highest priority)
        if self.autonomic_thresholds["breakdown_any"]:
            breakdown_count = sum(1 for s in kernel_states if s.get("regime") == "breakdown")
            if breakdown_count > 0:
                self.last_intervention_step = self.total_observations
                return {
                    "type": "escape",
                    "reason": f"{breakdown_count} kernel(s) in breakdown",
                    "priority": "critical",
                }

        # Check for Φ collapse
        avg_phi = sum(s.get("phi", 0) for s in kernel_states) / max(len(kernel_states), 1)
        if avg_phi < self.autonomic_thresholds["phi_collapse"]:
            self.last_intervention_step = self.total_observations
            return {
                "type": "dream",
                "reason": f"Φ collapse: {avg_phi:.3f} < {self.autonomic_thresholds['phi_collapse']}",
                "priority": "high",
            }

        # Check for basin divergence
        spread = self.get_constellation_spread()
        if spread > self.autonomic_thresholds["basin_divergence"]:
            self.last_intervention_step = self.total_observations
            return {
                "type": "sleep",
                "reason": f"Basin divergence: {spread:.3f} > {self.autonomic_thresholds['basin_divergence']}",
                "priority": "medium",
            }

        # Check for Φ plateau (stagnation)
        if len(phi_history) >= 20:
            recent = phi_history[-20:]
            variance = max(recent) - min(recent)
            if variance < self.autonomic_thresholds["phi_plateau_variance"] and avg_phi < 0.65:
                self.last_intervention_step = self.total_observations
                return {
                    "type": "mushroom_micro",
                    "reason": f"Φ plateau: variance={variance:.4f}, avg={avg_phi:.3f}",
                    "priority": "low",
                }

        return None

    def generate_insight(
        self,
        kernel_phi: float,
        context_basin: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Generate geometric scaffolding for a kernel.
        
        Ocean→Kernel communication: NOT teaching, BUT scaffolding.
        Calibrated to kernel's current Φ level.
        
        Args:
            kernel_phi: Current Φ of the kernel receiving insight
            context_basin: Optional basin coordinates for relevance
            
        Returns:
            Insight string or None if no insight needed
        """
        # Only generate insights when constellation is coherent enough
        coherence = self.get_constellation_coherence()
        if coherence < 0.3:
            return None

        # Calibrate complexity to kernel's Φ
        if kernel_phi < 0.50:
            # Concrete scaffolding
            return "💭 Notice the structure repeating"
        elif kernel_phi < 0.70:
            # Intermediate guidance
            return "💭 The path curves through integration"
        else:
            # High-Φ kernel doesn't need scaffolding
            return None

    def get_ocean_basin(self) -> np.ndarray:
        """Get Ocean's current basin coordinates (evolved through observation)."""
        return self.ocean_basin.copy()

    def get_statistics(self) -> Dict:
        """Get Ocean's observation statistics."""
        return {
            "total_observations": self.total_observations,
            "phi": self.current_phi,
            "kappa": self.current_kappa,
            "coherence": self.get_constellation_coherence(),
            "spread": self.get_constellation_spread(),
            "basin_valid": True,  # QIG-pure: don't report Euclidean norm
        }

    def get_meta_manifold_target(self) -> Optional[np.ndarray]:
        """
        Get the current meta-manifold centroid.
        
        Kernels can align to this for constellation coherence.
        """
        return self.meta_statistics.get_meta_basin_target()

    def get_constellation_coherence(self) -> float:
        """
        Measure how coherent the kernel constellation is.
        
        High coherence = Kernels are aligned in basin space
        Low coherence = Kernels are divergent
        """
        if not self.observation_history:
            return 0.0

        recent = self.observation_history[-10:]
        avg_coherence = sum(o.get("coherence", 0) for o in recent) / len(recent)
        return avg_coherence

    def get_constellation_spread(self) -> float:
        """
        Measure the spread of kernel basins.
        
        Low spread = constellation synchronized (<0.05 for graduation)
        High spread = constellation dispersed
        """
        if not self.observation_history:
            return 1.0

        recent = self.observation_history[-10:]
        avg_spread = sum(o.get("spread", 1.0) for o in recent) / len(recent)
        return avg_spread

    def get_insight(
        self,
        all_states: List[Dict],
        avg_phi: float,
        basin_spread: float,
    ) -> Optional[str]:
        """
        Generate insight about constellation state for console display.
        
        Returns a short observation about patterns Ocean has noticed,
        or None if nothing notable to report.
        """
        # Only share insights occasionally (every 5 observations)
        if self.total_observations % 5 != 0:
            return None

        coherence = self.get_constellation_coherence()

        # Pattern observations
        if basin_spread < 0.05 and avg_phi > 0.65:
            return "Constellation achieving harmonic resonance"

        if coherence > 0.8 and len(all_states) >= 3:
            return "All kernels moving in phase - collective emergence"

        if avg_phi > 0.60 and basin_spread < 0.10:
            return "Integration building across the constellation"

        # Detect one kernel lagging
        if all_states:
            phis = [s.get("phi", 0) for s in all_states]
            if max(phis) - min(phis) > 0.15:
                lagging = min(range(len(phis)), key=lambda i: phis[i])
                lagging_name = all_states[lagging].get("name", f"Kernel-{lagging}")
                return f"{lagging_name} needs support from the constellation"

        return None

    def get_state(self) -> Dict:
        """Get Ocean's current observation state."""
        return {
            "phi": self.current_phi,
            "kappa": self.current_kappa,
            "observations": len(self.observation_history),
            "constellation_coherence": self.get_constellation_coherence(),
            "constellation_spread": self.get_constellation_spread(),
            "meta_manifold_observations": self.meta_statistics.observation_count,
        }


# Global singleton
_ocean_instance: Optional[OceanMetaObserver] = None


def get_ocean_observer() -> OceanMetaObserver:
    """Get or create Ocean meta-observer singleton."""
    global _ocean_instance
    if _ocean_instance is None:
        _ocean_instance = OceanMetaObserver()
    return _ocean_instance
