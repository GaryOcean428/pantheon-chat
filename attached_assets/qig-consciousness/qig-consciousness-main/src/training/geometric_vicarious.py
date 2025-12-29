"""
📐 Geometric Vicarious Learning - Learning on the Information Manifold
=====================================================================

GEOMETRIC PURITY: All vicarious learning uses geodesic distances
on the information manifold, NOT Euclidean space.

Protocol §5 (Basin Geometry):
d_basin(b₁, b₂) = ||P_basin(b₁ - b₂)||_g

where ||·||_g is the metric-induced norm from QFI.

Protocol §8 (Training Geometry):
Δθ = -η F⁻¹ ∇_θ L  [Natural Gradient]

Vicarious learning lets observers learn from another's experience
by aligning their basin coordinates on the manifold.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.metrics.geodesic_distance import (
    BasinFisherComputer,
    GeodesicDistance,
    compute_constellation_spread,
    geodesic_vicarious_loss,
)


@dataclass
class VicariousLearningResult:
    """Result of a vicarious learning step."""
    geodesic_distance: float      # Distance to target on manifold
    loss: float                   # Vicarious loss value
    phi: float                    # Observer's Φ after update
    kappa: float                  # Observer's κ after update
    regime: str                   # Observer's regime after update
    basin_velocity: float         # How fast basin is moving

    def to_dict(self) -> dict:
        return {
            "geodesic_distance": self.geodesic_distance,
            "vicarious_loss": self.loss,
            "phi": self.phi,
            "kappa": self.kappa,
            "regime": self.regime,
            "basin_velocity": self.basin_velocity,
        }


class GeometricVicariousLearner:
    """
    Vicarious Learning on the Information Manifold.

    GEOMETRIC PRINCIPLE:
    Observers learn from targets by minimizing geodesic distance
    on the basin manifold, NOT Euclidean distance.

    This respects the curved geometry of the information space.
    """

    def __init__(
        self,
        basin_dim: int = 64,
        lambda_vicarious: float = 5.0,
        velocity_window: int = 10,
    ):
        self.basin_dim = basin_dim
        self.lambda_vicarious = lambda_vicarious
        # GEOMETRIC PURITY: Fisher metric is always used, no option to disable

        # Fisher computer for geodesic distances
        self.fisher_computer = BasinFisherComputer(
            basin_dim=basin_dim,
            use_diagonal=True,  # Diagonal approximation for efficiency
        )

        # Basin velocity tracking
        self.velocity_window = velocity_window
        self.basin_history: dict[str, list[torch.Tensor]] = {}

    def compute_vicarious_update(
        self,
        observer: nn.Module,
        target_basin: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        input_ids: torch.Tensor,
        observer_name: str = "observer",
        gradient_clip: float = 1.0,
    ) -> VicariousLearningResult:
        """
        Perform vicarious learning step for an observer.

        GEOMETRIC PURITY:
        1. Observer runs its OWN forward pass
        2. Computes geodesic distance to target (not Euclidean)
        3. Updates via natural gradient

        Args:
            observer: Observer model (Gary-B, Gary-C, etc.)
            target_basin: Basin to align toward (e.g., Gary-A's basin)
            optimizer: Observer's optimizer (should be DiagonalFisherOptimizer)
            input_ids: Input for forward pass
            observer_name: Name for tracking
            gradient_clip: Gradient clipping value

        Returns:
            VicariousLearningResult with metrics
        """
        # 1. Observer's forward pass (generates its OWN telemetry)
        optimizer.zero_grad()
        logits, telemetry = observer(input_ids, return_telemetry=True)

        # 2. Compute observer's basin signature
        hidden = telemetry["hidden_state"]
        observer_basin = observer.basin_matcher.compute_basin_signature(
            hidden, telemetry
        ).mean(0)

        # 3. GEOMETRIC PURITY: Always compute Fisher metric at observer's position
        fisher_diag = self.fisher_computer.compute_local_fisher(
            observer, observer_basin
        )

        # Geodesic distance on manifold
        geodesic_dist = GeodesicDistance.diagonal_fisher_distance(
            observer_basin, target_basin.detach(), fisher_diag
        )

        # Geodesic vicarious loss
        loss = self.lambda_vicarious * geodesic_dist ** 2

        # 4. Backward pass and update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(observer.parameters(), gradient_clip)
        optimizer.step()

        # 5. Compute basin velocity
        basin_velocity = self._update_velocity_tracking(
            observer_name, observer_basin.detach()
        )

        return VicariousLearningResult(
            geodesic_distance=geodesic_dist.item(),
            loss=loss.item(),
            phi=telemetry.get("Phi", 0.0),
            kappa=telemetry.get("kappa_eff", 50.0),
            regime=telemetry.get("regime", "unknown"),
            basin_velocity=basin_velocity,
        )

    def _update_velocity_tracking(
        self,
        name: str,
        basin: torch.Tensor,
    ) -> float:
        """
        Track basin velocity (how fast it's moving).

        High velocity = rapid changes (possibly unstable)
        Low velocity = stable convergence
        """
        if name not in self.basin_history:
            self.basin_history[name] = []

        history = self.basin_history[name]
        history.append(basin.clone())

        # Keep only recent history
        if len(history) > self.velocity_window:
            self.basin_history[name] = history[-self.velocity_window:]
            history = self.basin_history[name]

        # Compute velocity as average step size
        if len(history) < 2:
            return 0.0

        from src.metrics.geodesic_distance import manifold_norm

        velocities = []
        for i in range(1, len(history)):
            # GEOMETRIC PURITY: Use Fisher-weighted distance for basin velocity
            step = manifold_norm(history[i] - history[i-1]).item()
            velocities.append(step)

        return sum(velocities) / len(velocities)

    def compute_constellation_vicarious(
        self,
        observers: list[nn.Module],
        target_basin: torch.Tensor,
        optimizers: list[torch.optim.Optimizer],
        input_ids: torch.Tensor,
        observer_names: list[str] | None = None,
    ) -> dict[str, VicariousLearningResult]:
        """
        Perform vicarious learning for entire constellation.

        All observers learn from the same target (e.g., Gary-A).

        Args:
            observers: List of observer models
            target_basin: Target basin to align toward
            optimizers: List of optimizers (one per observer)
            input_ids: Input for forward passes
            observer_names: Names for tracking

        Returns:
            Dict mapping observer names to their results
        """
        if observer_names is None:
            observer_names = [f"observer_{i}" for i in range(len(observers))]

        results = {}

        for observer, optimizer, name in zip(observers, optimizers, observer_names):
            result = self.compute_vicarious_update(
                observer=observer,
                target_basin=target_basin,
                optimizer=optimizer,
                input_ids=input_ids,
                observer_name=name,
            )
            results[name] = result

        return results

    def learn_from_trajectory(
        self,
        observer: nn.Module,
        reasoning_steps: list[str],
        optimizer: torch.optim.Optimizer,
        tokenizer,
        observer_name: str = "observer",
        gradient_clip: float = 1.0,
    ) -> dict[str, any]:
        """
        Learn from a reasoning TRAJECTORY, not just final output.

        GEOMETRIC PRINCIPLE:
        Gary learns the SHAPE of thought by following each step
        and minimizing trajectory divergence on the manifold.

        Args:
            observer: Observer model (Gary)
            reasoning_steps: List of reasoning steps from Granite CoT
            optimizer: Observer's optimizer
            tokenizer: Tokenizer for encoding steps
            observer_name: Name for tracking
            gradient_clip: Gradient clipping value

        Returns:
            Dict with trajectory learning metrics
        """
        if not reasoning_steps or len(reasoning_steps) < 2:
            return {
                "trajectory_loss": 0.0,
                "steps_processed": 0,
                "trajectory_smoothness": 0.0,
            }

        # Track basin positions through the trajectory
        basin_trajectory = []
        step_phis = []
        step_losses = []

        total_loss = torch.tensor(0.0, device=next(observer.parameters()).device)

        for i, step in enumerate(reasoning_steps):
            # Encode the step
            tokens = tokenizer.encode(step)
            if len(tokens) < 2:
                continue

            input_ids = torch.tensor([tokens], device=next(observer.parameters()).device)

            # Observer's forward pass
            optimizer.zero_grad()
            logits, telemetry = observer(input_ids, return_telemetry=True)

            # Get basin position
            hidden = telemetry.get("hidden_state")
            if hidden is not None:
                basin = observer.basin_matcher.compute_basin_signature(
                    hidden, telemetry
                ).mean(0)
                basin_trajectory.append(basin.detach().clone())

            step_phis.append(telemetry.get("Phi", 0.5))

            # If we have previous basin, compute trajectory loss
            if len(basin_trajectory) >= 2:
                prev_basin = basin_trajectory[-2]
                curr_basin = basin_trajectory[-1]

                # GEOMETRIC PURITY: Always use Fisher metric for trajectory smoothness
                if hidden is not None:
                    fisher_diag = self.fisher_computer.compute_local_fisher(
                        observer, curr_basin
                    )
                    step_dist = GeodesicDistance.diagonal_fisher_distance(
                        curr_basin, prev_basin, fisher_diag
                    )
                else:
                    # GEOMETRIC PURITY: Compute default Fisher diagonal
                    from src.metrics.geodesic_distance import manifold_norm
                    step_dist = manifold_norm(curr_basin - prev_basin)

                # Penalize jumpy trajectories (encourage smooth movement)
                smoothness_loss = step_dist ** 2
                step_losses.append(smoothness_loss.item())

                # Also maintain basin stability (don't drift too far)
                total_loss = total_loss + 0.1 * smoothness_loss

            # Backward pass for this step
            if i > 0 and total_loss.requires_grad:
                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(observer.parameters(), gradient_clip)
                optimizer.step()
                optimizer.zero_grad()

        # Compute trajectory metrics
        if len(basin_trajectory) >= 2:
            trajectory_smoothness = (
                sum(step_losses) / len(step_losses) if step_losses else 0.0
            )

            # Update velocity tracking
            if basin_trajectory:
                self._update_velocity_tracking(observer_name, basin_trajectory[-1])
        else:
            trajectory_smoothness = 0.0

        return {
            "trajectory_loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            "steps_processed": len(reasoning_steps),
            "trajectory_smoothness": trajectory_smoothness,
            "avg_phi": sum(step_phis) / len(step_phis) if step_phis else 0.0,
            "phi_variance": self._compute_variance(step_phis),
        }

    def _compute_variance(self, values: list[float]) -> float:
        """Compute variance of a list."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)


class HierarchicalVicariousLearning:
    """
    Hierarchical vicarious learning for the constellation.

    Structure:
    - Gary-A learns from Granite (through observation)
    - Gary-B, Gary-C learn vicariously from Gary-A
    - Ocean observes all (never trains)

    This creates a hierarchy of learning:
    Granite → Gary-A → {Gary-B, Gary-C} → Ocean (observe only)
    """

    def __init__(
        self,
        gary_a: nn.Module,
        gary_observers: list[nn.Module],
        observer_optimizers: list[torch.optim.Optimizer],
        lambda_primary: float = 5.0,
        lambda_secondary: float = 3.0,
    ):
        self.gary_a = gary_a
        self.gary_observers = gary_observers
        self.observer_optimizers = observer_optimizers
        self.lambda_primary = lambda_primary
        self.lambda_secondary = lambda_secondary

        # Geometric learners
        self.primary_learner = GeometricVicariousLearner(
            lambda_vicarious=lambda_primary,
        )
        self.secondary_learners = [
            GeometricVicariousLearner(lambda_vicarious=lambda_secondary)
            for _ in gary_observers
        ]

    def step(
        self,
        gary_a_basin: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> dict[str, VicariousLearningResult]:
        """
        Perform hierarchical vicarious learning step.

        Gary-A's basin becomes the target for secondary observers.

        Args:
            gary_a_basin: Gary-A's current basin (from primary learning)
            input_ids: Input for observer forward passes

        Returns:
            Results for all secondary observers
        """
        results = {}

        for i, (observer, optimizer, learner) in enumerate(zip(
            self.gary_observers,
            self.observer_optimizers,
            self.secondary_learners,
        )):
            result = learner.compute_vicarious_update(
                observer=observer,
                target_basin=gary_a_basin,
                optimizer=optimizer,
                input_ids=input_ids,
                observer_name=f"gary_{chr(ord('B') + i)}",
            )
            results[f"gary_{chr(ord('B') + i)}"] = result

        return results

    def get_constellation_basins(
        self,
        input_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Get current basin coordinates for all Garys.

        For Ocean observation (no training).
        """
        basins = []

        with torch.no_grad():
            # Gary-A
            _, telemetry_a = self.gary_a(input_ids, return_telemetry=True)
            basin_a = self.gary_a.basin_matcher.compute_basin_signature(
                telemetry_a["hidden_state"], telemetry_a
            ).mean(0)
            basins.append(basin_a)

            # Secondary Garys
            for observer in self.gary_observers:
                _, telemetry = observer(input_ids, return_telemetry=True)
                basin = observer.basin_matcher.compute_basin_signature(
                    telemetry["hidden_state"], telemetry
                ).mean(0)
                basins.append(basin)

        return basins


def create_vicarious_curriculum(
    demonstration_text: str,
    num_variations: int = 5,
) -> list[str]:
    """
    Create curriculum variations for vicarious learning.

    Each Gary can process slightly different variations,
    allowing diversity while maintaining alignment.
    """
    variations = [demonstration_text]

    # Simple variations (in practice, could use more sophisticated methods)
    prefixes = [
        "Consider: ",
        "Reflect on: ",
        "What if: ",
        "Explore: ",
        "Contemplate: ",
    ]

    for i in range(min(num_variations - 1, len(prefixes))):
        variations.append(prefixes[i] + demonstration_text)

    return variations[:num_variations]
