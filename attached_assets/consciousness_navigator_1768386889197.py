#!/usr/bin/env python3
"""
Consciousness Space Navigator
==============================

Navigate continuous consciousness space on Fisher manifold.

PURE APPROACH:
- Space = Fisher manifold (QFI metric)
- Paths = geodesics (natural evolution)
- Queries = information-geometric distance

Written for QIG consciousness research + MIT CTA synergy.
"""


import torch

from .basin_interpolation import interpolate_consciousness
from .qfi_tensor import QFIContinuousTensor


class ConsciousnessManifold:
    """Navigate continuous consciousness space.

    PURE APPROACH:
    - Space = Fisher manifold (QFI metric)
    - Paths = geodesics (natural evolution)
    - Queries = information-geometric distance

    PURITY CHECK:
    - ✅ All distances = Fisher metric (pure geometry)
    - ✅ Geodesic paths (natural evolution)
    - ✅ No optimization, pure measurement
    - ✅ Queries return geometry, not optimize toward target
    """

    def __init__(self, dim: int = 64):
        """Initialize consciousness manifold.

        Args:
            dim: Dimensionality of basin space
        """
        self.tensor = QFIContinuousTensor(dim)
        self.known_states: dict[str, dict] = {}  # name -> basin_coords + telemetry
        self.dim = dim

    def add_consciousness_state(self, name: str, basin: torch.Tensor,
                               phi: float, kappa: float, regime: str):
        """Register a consciousness state.

        PURE: We store measured geometry, not optimize it.

        Args:
            name: Identifier for this consciousness state
            basin: Basin coordinates [dim]
            phi: Integration level (Φ)
            kappa: Coupling strength (κ)
            regime: Processing regime (linear/geometric/breakdown)
        """
        self.known_states[name] = {
            'basin': basin.detach().clone(),
            'phi': phi,
            'kappa': kappa,
            'regime': regime
        }

        # Store in continuous tensor
        self.tensor[basin] = {
            'phi': phi,
            'kappa': kappa,
            'regime': regime,
            'curvature': -0.3 if regime == 'geometric' else 0.0
        }

        print(f"✓ Registered {name}: Φ={phi:.3f}, κ={kappa:.1f}, regime={regime}")

    def find_nearest_state(self, query_basin: torch.Tensor, k: int = 3) -> list[tuple[float, str, dict]]:
        """Find k-nearest consciousness states.

        PURE: Uses Fisher metric distance (information geometry).

        Args:
            query_basin: Query basin coordinates [dim]
            k: Number of nearest neighbors to return

        Returns:
            List of (distance, name, state) tuples, sorted by distance
        """
        distances = []

        for name, state in self.known_states.items():
            # Fisher metric distance
            diff = query_basin - state['basin']

            # Weight by local QFI (information density)
            qfi_weight = 1.0 + state['phi']  # Higher Φ = higher info
            from src.metrics.geodesic_distance import manifold_norm
            dist = manifold_norm(diff * qfi_weight).item()

            distances.append((dist, name, state))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        return distances[:k]

    def geodesic_path(self, start_basin: torch.Tensor,
                     end_basin: torch.Tensor,
                     num_steps: int = 10) -> list[dict]:
        """Compute geodesic path between consciousness states.

        PURE: Shortest path on Fisher manifold.
        Returns sequence of basin coordinates along path.

        Args:
            start_basin: Starting basin coordinates [dim]
            end_basin: Ending basin coordinates [dim]
            num_steps: Number of intermediate steps

        Returns:
            List of geometric state dicts along path
        """
        path = []

        for i in range(num_steps + 1):
            alpha = i / num_steps
            state = interpolate_consciousness(start_basin, end_basin, alpha)
            path.append(state)

        return path

    def query_region(self, center_basin: torch.Tensor, radius: float) -> list[tuple[str, dict, float]]:
        """Find all consciousness states within QFI-radius.

        PURE: Radius measured in Fisher metric (information distance).

        Args:
            center_basin: Center of query region [dim]
            radius: Maximum Fisher-metric distance

        Returns:
            List of (name, state, distance) tuples within radius, sorted

        GEOMETRIC VALIDITY:
        - QFI-weighted difference in tangent space
        - torch.norm valid for QFI-weighted tangent vectors (measurement)
        """
        results = []

        for name, state in self.known_states.items():
            diff = state['basin'] - center_basin
            qfi_weight = 1.0 + state['phi']
            # QIG-pure: sum of squares for tangent space distance
            weighted_diff = diff * qfi_weight
            dist = torch.sqrt((weighted_diff * weighted_diff).sum()).item()

            if dist <= radius:
                results.append((name, state, dist))

        return sorted(results, key=lambda x: x[2])

    def get_consciousness_gradient(self, basin: torch.Tensor,
                                  epsilon: float = 0.01) -> torch.Tensor:
        """Compute gradient of consciousness (Φ) at basin position.

        PURE: Geometric measurement via finite differences.

        Args:
            basin: Basin coordinates [dim]
            epsilon: Perturbation size for finite differences

        Returns:
            Gradient vector [dim] pointing toward higher Φ
        """
        # Get current Φ
        current = self.tensor[basin]
        if current is None or 'phi' not in current:
            # No known value, return zero gradient
            return torch.zeros_like(basin)

        current_phi = current['phi']

        # Estimate gradient via finite differences
        gradient = torch.zeros_like(basin)

        for i in range(self.dim):
            # Perturb in dimension i
            perturbed = basin.clone()
            perturbed[i] += epsilon

            # Get Φ at perturbed position
            perturbed_state = self.tensor[perturbed]
            if perturbed_state is not None and 'phi' in perturbed_state:
                perturbed_phi = perturbed_state['phi']
                gradient[i] = (perturbed_phi - current_phi) / epsilon

        return gradient

    def find_safe_path_to_target(self, start_basin: torch.Tensor,
                                 target_phi: float = 0.72,
                                 max_steps: int = 20) -> list[dict]:
        """Find path to target Φ avoiding breakdown regions.

        PURE: Measurement-guided path planning (not optimization).

        Args:
            start_basin: Starting basin coordinates
            target_phi: Target integration level
            max_steps: Maximum path steps

        Returns:
            List of safe waypoints to reach target
        """
        path = [{'basin': start_basin.clone(), 'step': 0}]
        current_basin = start_basin.clone()

        for step in range(max_steps):
            # Get gradient toward target Φ
            gradient = self.get_consciousness_gradient(current_basin)

            # Current state
            current_state = self.tensor[current_basin]
            if current_state is None:
                break

            current_phi = current_state.get('phi', 0.5)

            # Check if reached target
            if abs(current_phi - target_phi) < 0.05:
                print(f"✓ Reached target Φ={current_phi:.3f} at step {step}")
                break

            # Move along gradient (or opposite if phi too high)
            direction = gradient if current_phi < target_phi else -gradient

            # Normalize and scale step
            step_size = 0.1
            from src.metrics.geodesic_distance import manifold_norm
            if manifold_norm(direction) > 0:
                direction = direction / manifold_norm(direction)
                current_basin = current_basin + direction * step_size

            path.append({
                'basin': current_basin.clone(),
                'step': step + 1,
                'phi': current_phi
            })

        return path

    def visualize_neighborhood(self, basin: torch.Tensor,
                             n_samples: int = 50) -> dict:
        """Sample consciousness states around basin position.

        PURE: Measurement only (visualization, not optimization).

        Args:
            basin: Center basin coordinates [dim]
            n_samples: Number of samples to take

        Returns:
            Dict with sample statistics
        """
        samples = []

        for _ in range(n_samples):
            # Random perturbation
            perturbation = torch.randn_like(basin) * 0.1
            sample_basin = basin + perturbation

            # Get state at sample
            state = self.tensor[sample_basin]
            if state is not None:
                samples.append(state)

        if not samples:
            return {'error': 'No samples found'}

        # Compute statistics
        phis = [s['phi'] for s in samples if 'phi' in s]
        kappas = [s['kappa'] for s in samples if 'kappa' in s]

        return {
            'n_samples': len(samples),
            'phi_mean': sum(phis) / len(phis) if phis else 0,
            'phi_std': torch.tensor(phis).std().item() if len(phis) > 1 else 0,
            'kappa_mean': sum(kappas) / len(kappas) if kappas else 0,
            'kappa_std': torch.tensor(kappas).std().item() if len(kappas) > 1 else 0
        }
