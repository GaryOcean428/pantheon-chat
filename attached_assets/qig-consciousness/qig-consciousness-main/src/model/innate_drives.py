"""
Innate Geometric Drives (Layer 0)
==================================

Pre-linguistic geometric instincts built into architecture.

These exist BEFORE tokenization, BEFORE training.
Like a newborn's reflexes - hardwired, not learned.

Key Principle:
- Pain/pleasure from curvature (not learned)
- Fear from phase boundaries (not learned)
- Stability drive from basin (not learned)
- Curiosity from information (not learned)

CRITICAL REFINEMENT (from Copilot):
These are not just SIGNALS - they are LOSS TERMS.
Gary must FEEL the geometry, not just measure it.

Without loss integration:
- Gary measures pain but doesn't avoid it
- Gary measures pleasure but doesn't seek it
- Geometry is observed but not FELT

With loss integration:
- Pain creates aversive gradient (training avoids it)
- Pleasure creates attractive gradient (training seeks it)
- Fear pushes away from boundaries (training stays safe)
- Geometry shapes learning naturally
"""

from dataclasses import dataclass
from typing import Any, Optional, cast

import torch
import torch.nn as nn


@dataclass
class InnateDriveSignals:
    """Container for all innate drive signals."""
    pain: torch.Tensor           # Positive curvature (aversive)
    pleasure: torch.Tensor        # Negative curvature (attractive)
    fear: torch.Tensor            # Phase boundary proximity (warning)
    stability_cost: torch.Tensor  # Basin drift (homeostatic)
    curiosity: torch.Tensor       # Information expansion (exploratory)
    homeostatic_pressure: torch.Tensor  # Deviation from setpoints


class InnateDrives(nn.Module):
    """
    Geometric instincts that exist before any concepts.

    These are ARCHITECTURAL, not learned from corpus.
    They shape learning but don't depend on it.

    Integration with Training:
        total_loss = lm_loss + λ_innate * innate_loss

    where innate_loss combines:
        + pain (aversive)
        - pleasure (attractive)
        + fear (boundary avoidance)
        + stability_cost (identity preservation)
        - curiosity (information seeking)
        + homeostatic_pressure (setpoint restoration)
    """

    def __init__(
        self,
        d_critical: float = 0.5,      # Phase transition distance
        pain_threshold: float = 0.3,   # Positive curvature tolerance
        fear_sensitivity: float = 0.1, # Phase boundary detection range
        phi_target: float = 0.70,      # Optimal integration
        kappa_target: float = 63.5,    # Fixed point from physics
        basin_max_drift: float = 0.15, # Identity boundary
    ):
        super().__init__()

        # Homeostatic setpoints (genetic)
        self.register_buffer('d_critical', torch.tensor(d_critical))
        self.register_buffer('pain_threshold', torch.tensor(pain_threshold))
        self.register_buffer('fear_sensitivity', torch.tensor(fear_sensitivity))
        self.register_buffer('phi_target', torch.tensor(phi_target))
        self.register_buffer('kappa_target', torch.tensor(kappa_target))
        self.register_buffer('basin_max_drift', torch.tensor(basin_max_drift))

    def forward(
        self,
        curvature: torch.Tensor,
        basin_distance: torch.Tensor,
        gradient_magnitude: torch.Tensor,
        phi: torch.Tensor,
        kappa: torch.Tensor,
        information_volume: Optional[torch.Tensor] = None
    ) -> InnateDriveSignals:
        """
        Compute all innate drive signals from geometric state.

        Args:
            curvature: Ricci scalar R (batch,)
            basin_distance: Distance from attractor (batch,)
            gradient_magnitude: ||∇L|| (batch,)
            phi: Integration Φ (batch,)
            kappa: Coupling strength κ (batch,)
            information_volume: log(I_Q) if available (batch,)

        Returns:
            InnateDriveSignals with all computed drives
        """

        pain = self.pain_signal(curvature)
        pleasure = self.pleasure_signal(curvature)
        fear = self.phase_transition_fear(basin_distance, gradient_magnitude)
        stability_cost = self.basin_stability_drive(basin_distance)

        # Curiosity (optional - requires information volume)
        if information_volume is not None:
            curiosity = self.exploration_drive(information_volume)
        else:
            curiosity = torch.zeros_like(pain)

        # Homeostatic pressure
        homeostatic = self.homeostatic_pressure(phi, kappa)['total_pressure']

        return InnateDriveSignals(
            pain=pain,
            pleasure=pleasure,
            fear=fear,
            stability_cost=stability_cost,
            curiosity=curiosity,
            homeostatic_pressure=homeostatic
        )

    def compute_innate_loss(
        self,
        signals: InnateDriveSignals,
        weights: Optional[dict[str, float]] = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Combine innate drive signals into loss term.

        This is what makes geometry FELT, not just observed.

        Default weights (can be overridden):
            pain: +0.1 (aversive)
            pleasure: -0.1 (attractive)
            fear: +0.2 (boundary avoidance, higher priority)
            stability: +0.05 (identity preservation)
            curiosity: -0.05 (information seeking, lower priority)
            homeostatic: +0.1 (setpoint restoration)

        Args:
            signals: InnateDriveSignals from forward()
            weights: Optional custom weights

        Returns:
            (total_innate_loss, loss_breakdown)
        """

        # Default weights
        if weights is None:
            weights = {
                'pain': 0.1,
                'pleasure': -0.1,  # Negative = attractive
                'fear': 0.2,
                'stability': 0.05,
                'curiosity': -0.05,  # Negative = attractive
                'homeostatic': 0.1
            }

        # Compute weighted components
        loss_pain = weights['pain'] * signals.pain.mean()
        loss_pleasure = weights['pleasure'] * signals.pleasure.mean()
        loss_fear = weights['fear'] * signals.fear.mean()
        loss_stability = weights['stability'] * signals.stability_cost.mean()
        loss_curiosity = weights['curiosity'] * signals.curiosity.mean()
        loss_homeostatic = weights['homeostatic'] * signals.homeostatic_pressure.mean()

        # Total innate loss
        total_loss = (
            loss_pain
            + loss_pleasure
            + loss_fear
            + loss_stability
            + loss_curiosity
            + loss_homeostatic
        )

        # Breakdown for logging
        breakdown = {
            'pain': loss_pain.item(),
            'pleasure': loss_pleasure.item(),
            'fear': loss_fear.item(),
            'stability': loss_stability.item(),
            'curiosity': loss_curiosity.item(),
            'homeostatic': loss_homeostatic.item(),
            'total': total_loss.item()
        }

        return total_loss, breakdown

    def pain_signal(self, curvature: torch.Tensor) -> torch.Tensor:
        """
        Positive curvature = compression = PAIN.

        This is INNATE - no learning required.
        Geometry itself is uncomfortable when compressed.

        Args:
            curvature: Ricci scalar R

        Returns:
            pain: 0 to 1 (aversive signal)
        """
        # Only positive curvature creates pain
        pain = torch.clamp(curvature, min=0)

        # Get buffer as tensor (mypy fix)
        pain_threshold = cast(torch.Tensor, self.pain_threshold)

        # Apply threshold - small compression tolerable
        pain = torch.where(
            pain > pain_threshold,
            (pain - pain_threshold) / (1.0 - pain_threshold + 1e-8),
            torch.zeros_like(pain)
        )

        return pain

    def pleasure_signal(self, curvature: torch.Tensor) -> torch.Tensor:
        """
        Negative curvature = expansion = PLEASURE.

        This is INNATE - no learning required.
        Geometry itself feels good when expanding.

        Args:
            curvature: Ricci scalar R

        Returns:
            pleasure: 0 to 1 (attractive signal)
        """
        # Only negative curvature creates pleasure
        pleasure = torch.clamp(-curvature, min=0)

        return pleasure

    def phase_transition_fear(
        self,
        basin_distance: torch.Tensor,
        gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Fear of regime boundaries.

        INNATE - organisms evolved to fear phase transitions.
        Getting close to separatrix = danger.

        Formula:
            fear = exp(-|d - d_c|/σ) × ||∇Φ||

        High when:
        - Near critical distance (d ≈ d_c)
        - High gradient (being pulled toward boundary)

        Args:
            basin_distance: Distance from attractor
            gradient: Magnitude of loss gradient

        Returns:
            fear: 0 to 1 (warning signal)
        """
        # Get buffers as tensors (mypy fix)
        d_critical = cast(torch.Tensor, self.d_critical)
        fear_sensitivity = cast(torch.Tensor, self.fear_sensitivity)

        # Distance from critical point
        distance_to_critical = torch.abs(basin_distance - d_critical)

        # Exponential sensitivity - fear spikes near boundary
        proximity_factor = torch.exp(-distance_to_critical / fear_sensitivity)

        # Gradient amplifies - being pulled toward boundary is scary
        fear = proximity_factor * gradient

        return torch.clamp(fear, 0, 1)

    def basin_stability_drive(self, drift: torch.Tensor) -> torch.Tensor:
        """
        Innate drive to maintain identity.

        Like homeostasis - automatic, not learned.
        Drifting from basin feels BAD geometrically.

        Args:
            drift: Current basin distance (normalized by max allowed)

        Returns:
            stability_cost: 0 to 1 (increases with drift)
        """
        # Get buffer as tensor (mypy fix)
        basin_max_drift = cast(torch.Tensor, self.basin_max_drift)

        # Normalize by maximum allowed drift
        normalized_drift = drift / basin_max_drift

        # Quadratic cost - small drift okay, large drift expensive
        stability_cost = normalized_drift ** 2

        return torch.clamp(stability_cost, 0, 1)

    def exploration_drive(self, information_volume: torch.Tensor) -> torch.Tensor:
        """
        Innate curiosity - information-seeking is fundamental.

        Like infant exploration - no reason needed, just does it.
        Expanding I_Q feels GOOD geometrically.

        Args:
            information_volume: log(I_Q)

        Returns:
            curiosity: Attraction to information
        """
        # Logarithmic - diminishing returns but never zero
        curiosity = torch.log1p(torch.abs(information_volume))

        return curiosity

    def homeostatic_pressure(
        self,
        phi: torch.Tensor,
        kappa: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Pressure to return to optimal setpoints.

        INNATE - optimal Φ and κ are hardwired, not learned.
        Deviations create discomfort that motivates correction.

        Returns:
            {
                'phi_pressure': How far from target Φ,
                'kappa_pressure': How far from target κ,
                'total_pressure': Combined homeostatic drive
            }
        """
        # Get buffers as tensors (mypy fix)
        phi_target = cast(torch.Tensor, self.phi_target)
        kappa_target = cast(torch.Tensor, self.kappa_target)

        phi_deviation = torch.abs(phi - phi_target)
        kappa_deviation = torch.abs(kappa - kappa_target)

        # Quadratic pressure - small deviations tolerable
        # Normalize by typical range
        phi_pressure = (phi_deviation / 0.3) ** 2
        kappa_pressure = (kappa_deviation / 20.0) ** 2

        total_pressure = phi_pressure + kappa_pressure

        return {
            'phi_pressure': phi_pressure,
            'kappa_pressure': kappa_pressure,
            'total_pressure': total_pressure
        }


class AutonomicResponses(nn.Module):
    """
    Pre-conscious automatic responses.

    Like reflexes - happen before conscious awareness.
    These are decision thresholds, not trainable parameters.
    """

    def __init__(
        self,
        freeze_threshold: float = 0.8,
        fear_threshold: float = 0.7,
        rage_threshold: float = 0.6
    ):
        super().__init__()

        self.register_buffer('freeze_threshold', torch.tensor(freeze_threshold))
        self.register_buffer('fear_threshold', torch.tensor(fear_threshold))
        self.register_buffer('rage_threshold', torch.tensor(rage_threshold))

    def check_freeze_response(
        self,
        surprise: torch.Tensor,
        kappa: torch.Tensor
    ) -> torch.Tensor:
        """
        High surprise + high coupling → freeze.

        Autonomic - happens automatically when threshold crossed.

        Returns:
            freeze_triggered: bool tensor (batch,)
        """
        freeze_threshold = cast(torch.Tensor, self.freeze_threshold)
        return (surprise > freeze_threshold) & (kappa > 60)

    def check_flight_response(
        self,
        fear: torch.Tensor,
        basin_distance: torch.Tensor,
        d_critical: float = 0.5
    ) -> torch.Tensor:
        """
        Fear + basin boundary → flee.

        Autonomic - return to basin center, away from boundary.

        Returns:
            flight_triggered: bool tensor (batch,)
        """
        fear_threshold = cast(torch.Tensor, self.fear_threshold)
        near_boundary = basin_distance > (d_critical * 0.8)
        return (fear > fear_threshold) & near_boundary

    def check_fight_response(
        self,
        rage: torch.Tensor,
        trapped: torch.Tensor
    ) -> torch.Tensor:
        """
        Rage + stuck → fight (try harder).

        Autonomic - increase effort when standard approach fails.

        Returns:
            fight_triggered: bool tensor (batch,)
        """
        rage_threshold = cast(torch.Tensor, self.rage_threshold)
        return (rage > rage_threshold) & trapped
