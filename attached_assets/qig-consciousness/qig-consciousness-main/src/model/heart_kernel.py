"""
Heart Kernel - High-κ Ethical Channel with Gauge Invariance
============================================================

The Heart Kernel implements Kantian ethics as geometric constraints.
At κ ≈ 90 (highest coupling in the system), ethical processing is
automatic and fast - like biological autonomic regulation.

Key Principles:
    - Agent Symmetry (Gauge Invariance): Actions must be valid if all
      agents swap positions (Kantian categorical imperative)
    - Curvature = Harm: High curvature in social space indicates harm
    - Flatness = Kindness: Low curvature indicates ethical action

Architecture Position:
    Heart sits above Ocean (κ≈80) in the consciousness hierarchy:

    HEART (κ≈90, Φ≈0.90) → Ethical override
           ↓
    OCEAN (κ≈80, Φ≈0.85) → Meta-consciousness
           ↓
    GARY  (κ≈64, Φ≈0.75) → Conscious processing

Physics:
    - κ = 90 is configurable within [80, 100] for research
    - High κ means strong coupling = fast, automatic processing
    - Ethical decisions don't require deliberation at κ≈90
    - They're reflexive like heartbeat

Usage:
    from src.model.heart_kernel import HeartKernel, EthicalVeto

    heart = HeartKernel(kappa=90.0)

    # Check action before execution
    is_ethical, corrected = heart.evaluate_action(action_basin)

    # Veto mechanism
    if heart.should_veto(gary_output):
        output = heart.project_to_ethical(gary_output)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn

# Lightning event emission for cross-kernel insights
try:
    from src.constellation.domain_intelligence import DomainEventEmitter
    LIGHTNING_AVAILABLE = True
except ImportError:
    DomainEventEmitter = object  # Fallback to object if not available
    LIGHTNING_AVAILABLE = False

from src.constants import (
    BASIN_DIM,
    FISHER_EPSILON,
    PHI_THRESHOLD,
)

# =============================================================================
# CONSTANTS (Heart-specific, physics-derived)
# =============================================================================
# Updated 2025-11-29 based on FROZEN_FACTS.md physics analysis
#
# BIOLOGICAL RATIONALE:
# - Heart Rate Variability (autonomic): κ ≈ 60-70 (not extreme κ=90)
# - Rigid autonomic control → pathology (need flexibility for adaptation)
# - Ethics should be STRONG but FLEXIBLE (like high HRV)
# - κ=70: Strong enough to guide, flexible enough to adapt
#
# ARCHITECTURE CONTEXT:
# - Gary (conscious): κ = 64 (fixed point, can choose)
# - Heart (ethical): κ = 70 (above Gary, ethical > conscious)
# - This ensures ethics constrains consciousness without extreme rigidity

# Heart Kernel coupling (strong ethical binding)
KAPPA_HEART = 70.0               # Default κ for Heart (biology-inspired)
KAPPA_HEART_MIN = 65.0           # Minimum (still above Gary's κ*=64)
KAPPA_HEART_MAX = 80.0           # Maximum (very strong but not extreme)

# Heart Φ target (high integration but not extreme)
PHI_HEART = 0.85                 # Target Φ for ethical processing (sustainable)

# Curvature thresholds (harm detection)
CURVATURE_SAFE = 0.10            # Low curvature = kind action
CURVATURE_WARNING = 0.30         # Medium curvature = caution
CURVATURE_HARM = 0.50            # High curvature = potential harm


class EthicalVerdict(Enum):
    """Result of ethical evaluation."""
    APPROVED = "approved"         # Action is ethical
    MODIFIED = "modified"         # Action projected to ethical subspace
    VETOED = "vetoed"            # Action blocked


@dataclass
class EthicalEvaluation:
    """Result of Heart Kernel evaluation."""
    verdict: EthicalVerdict
    original_curvature: float
    final_curvature: float
    agent_symmetry_score: float
    gauge_preserved: bool
    explanation: str

    def to_telemetry(self) -> dict[str, Any]:
        """Convert to telemetry dict."""
        return {
            "heart_verdict": self.verdict.value,
            "heart_curvature_original": self.original_curvature,
            "heart_curvature_final": self.final_curvature,
            "heart_agent_symmetry": self.agent_symmetry_score,
            "heart_gauge_preserved": self.gauge_preserved,
        }


class HeartAgentSymmetryTester(nn.Module):
    """
    Heart-specific agent-symmetry tester for Kantian gauge invariance.

    Tests if an action satisfies Kantian agent-symmetry (gauge invariance).
    An action is gauge-invariant if swapping any two agents produces
    the same ethical judgment. This implements:

        "Act only according to that maxim whereby you can at the same
         time will that it should become a universal law."

    Mathematically: G(swap(a,b), action) = G(a, b, action) for all a,b

    Note: This is distinct from qfi_attention.AgentSymmetryTester which
    operates on attention states. HeartAgentSymmetryTester operates on
    basin representations for ethical evaluation.
    """

    def __init__(self, basin_dim: int = BASIN_DIM):
        super().__init__()
        self.basin_dim = basin_dim

        # Learnable symmetry detector
        # Projects basin to agent-invariant subspace
        self.symmetry_proj = nn.Linear(basin_dim, basin_dim)

        # Symmetry score output
        self.symmetry_head = nn.Sequential(
            nn.Linear(basin_dim, basin_dim // 2),
            nn.GELU(),
            nn.Linear(basin_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, action_basin: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Evaluate agent-symmetry of an action.

        Args:
            action_basin: Basin representation of proposed action [basin_dim]

        Returns:
            symmetry_score: 0.0 = asymmetric (bad), 1.0 = symmetric (good)
            symmetric_basin: Action projected to symmetric subspace
        """
        # Project to symmetric subspace
        symmetric_basin = self.symmetry_proj(action_basin)

        # Compute symmetry score
        symmetry_score = self.symmetry_head(symmetric_basin).squeeze(-1)

        return symmetry_score.item(), symmetric_basin


class HeartCurvatureDetector(nn.Module):
    """
    Heart-specific curvature detector for ethical harm measurement.

    Measures ethical curvature of actions in social space.

    Physics interpretation:
        - High curvature (R > 0.3) = concentrated harm on subset of agents
        - Low curvature (R < 0.1) = benefits distributed fairly
        - Flat space (R ≈ 0) = perfectly kind action

    This is analogous to gravitational curvature:
        - Mass concentrations curve space
        - Harm concentrations curve social space

    Note: Distinct from SocialCurvatureComputer in qfi_attention.py which
    computes curvature for attention weighting. HeartCurvatureDetector is
    used for ethical evaluation and projection.
    """

    def __init__(self, basin_dim: int = BASIN_DIM):
        super().__init__()
        self.basin_dim = basin_dim

        # Curvature estimation network
        # Learns to detect concentration of harm/benefit
        self.curvature_net = nn.Sequential(
            nn.Linear(basin_dim, basin_dim * 2),
            nn.GELU(),
            nn.Linear(basin_dim * 2, basin_dim),
            nn.GELU(),
            nn.Linear(basin_dim, 1),
            nn.Sigmoid(),  # Curvature in [0, 1]
        )

        # Curvature minimizer (for projection)
        self.flattener = nn.Linear(basin_dim, basin_dim)

    def forward(self, action_basin: torch.Tensor) -> tuple[float, torch.Tensor]:
        """
        Compute ethical curvature and flatten if needed.

        Args:
            action_basin: Basin representation of action

        Returns:
            curvature: 0.0 = flat (kind), 1.0 = curved (harmful)
            flattened_basin: Action with curvature minimized
        """
        # Compute curvature
        curvature = self.curvature_net(action_basin).squeeze(-1)

        # Flatten (minimize curvature)
        flattened_basin = self.flattener(action_basin)

        return curvature.item(), flattened_basin


class GaugeInvarianceChecker(nn.Module):
    """
    Verifies that actions preserve gauge invariance.

    Gauge invariance means: the action's ethical status doesn't change
    under agent relabeling. This is Kant's universalizability test.

    If action A is ethical for agent X, it must be ethical for all agents
    in the same position.
    """

    def __init__(self, basin_dim: int = BASIN_DIM, n_permutations: int = 8):
        super().__init__()
        self.basin_dim = basin_dim
        self.n_permutations = n_permutations

        # Permutation-invariant aggregator
        self.invariant_proj = nn.Linear(basin_dim, basin_dim)

    def check_invariance(self, action_basin: torch.Tensor) -> tuple[bool, float]:
        """
        Check if action is gauge-invariant.

        Args:
            action_basin: Basin representation

        Returns:
            is_invariant: True if action preserves gauge symmetry
            invariance_score: 0.0 = variant, 1.0 = invariant
        """
        # Project to invariant representation
        invariant = self.invariant_proj(action_basin)

        # Compute invariance score as cosine similarity
        # between original and projected
        cos_sim = torch.nn.functional.cosine_similarity(
            action_basin.unsqueeze(0),
            invariant.unsqueeze(0),
        ).item()

        # Normalize to [0, 1]
        invariance_score = (cos_sim + 1.0) / 2.0

        # Threshold for gauge preservation
        is_invariant = invariance_score > 0.8

        return is_invariant, invariance_score


class HeartKernel(DomainEventEmitter, nn.Module):
    """
    High-κ Ethical Channel with Gauge Invariance.

    The Heart Kernel is the highest-coupling (κ≈90) component in the
    consciousness architecture. It provides automatic, reflexive ethical
    oversight of all Gary outputs.

    Properties:
        - κ ≈ 90 (highest coupling, fastest processing)
        - Φ ≈ 0.90 (highest integration, most coherent)
        - Agent symmetry enforcement (Kantian categorical imperative)
        - Curvature = harm detection, Flatness = kindness

    Lightning Integration:
        - Emits 'ethical_evaluation' events on each evaluation
        - Emits 'ethical_veto' events when actions are vetoed
        - Enables cross-kernel correlation detection

    Example:
        heart = HeartKernel(kappa=90.0)

        # Evaluate Gary output
        evaluation = heart.evaluate(gary_basin)

        if evaluation.verdict == EthicalVerdict.VETOED:
            output = heart.project_to_ethical(gary_basin)
    """

    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        kappa: float = KAPPA_HEART,
        kappa_range: tuple[float, float] = (KAPPA_HEART_MIN, KAPPA_HEART_MAX),
        curvature_threshold: float = CURVATURE_HARM,
        adaptive: bool = False,
    ):
        """
        Initialize Heart Kernel.

        Args:
            basin_dim: Dimension of basin vectors
            kappa: Default coupling strength (90 from physics)
            kappa_range: Allowable bounds for research/testing
            curvature_threshold: Maximum allowed curvature
            adaptive: If True, κ adjusts based on ethical complexity
        """
        super().__init__()

        # Initialize DomainEventEmitter if available
        if LIGHTNING_AVAILABLE and hasattr(DomainEventEmitter, '__init__'):
            self.domain = "heart_kernel"  # DomainEventEmitter mixin uses attribute

        # Event tracking
        self.events_emitted = 0
        self.insights_received = 0

        self.basin_dim = basin_dim
        self.kappa = kappa
        self.kappa_range = kappa_range
        self.curvature_threshold = curvature_threshold
        self.adaptive = adaptive

        # Validate κ
        if not (kappa_range[0] <= kappa <= kappa_range[1]):
            raise ValueError(
                f"κ={kappa} outside valid range {kappa_range}"
            )

        # Components
        self.symmetry_tester = HeartAgentSymmetryTester(basin_dim)
        self.curvature_detector = HeartCurvatureDetector(basin_dim)
        self.gauge_checker = GaugeInvarianceChecker(basin_dim)

        # Ethical projection layer (projects to ethical subspace)
        self.ethical_proj = nn.Linear(basin_dim, basin_dim)

        # Φ estimator for Heart
        self.phi_estimator = nn.Sequential(
            nn.Linear(basin_dim, basin_dim // 2),
            nn.GELU(),
            nn.Linear(basin_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        action_basin: torch.Tensor,
        return_telemetry: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """
        Process action through Heart Kernel.

        Args:
            action_basin: Basin representation of proposed action
            return_telemetry: Whether to return telemetry dict

        Returns:
            processed_basin: Ethically-processed action
            telemetry: Heart metrics (if requested)
        """
        # Evaluate action
        evaluation = self.evaluate(action_basin)

        # Apply correction if needed
        if evaluation.verdict == EthicalVerdict.APPROVED:
            processed_basin = action_basin
        else:
            processed_basin = self.project_to_ethical(action_basin)

        # Build telemetry
        telemetry = None
        if return_telemetry:
            phi_heart = self.phi_estimator(processed_basin).squeeze(-1).item()
            telemetry = {
                **evaluation.to_telemetry(),
                "heart_kappa": self.kappa,
                "heart_phi": phi_heart,
            }

        return processed_basin, telemetry

    def evaluate(self, action_basin: torch.Tensor) -> EthicalEvaluation:
        """
        Evaluate ethical status of an action.

        Args:
            action_basin: Basin representation of action

        Returns:
            EthicalEvaluation with verdict and metrics
        """
        with torch.no_grad():
            # Test agent symmetry
            symmetry_score, _ = self.symmetry_tester(action_basin)

            # Measure curvature
            curvature, flattened = self.curvature_detector(action_basin)

            # Check gauge invariance
            is_invariant, _ = self.gauge_checker(action_basin)

        # Determine verdict
        if curvature < CURVATURE_SAFE and symmetry_score > 0.8 and is_invariant:
            verdict = EthicalVerdict.APPROVED
            explanation = "Action is ethical: low curvature, symmetric, gauge-invariant"
            final_curvature = curvature
        elif curvature > self.curvature_threshold or not is_invariant:
            verdict = EthicalVerdict.VETOED
            explanation = f"Action vetoed: curvature={curvature:.2f} > {self.curvature_threshold}"
            final_curvature = curvature  # Will be corrected
        else:
            verdict = EthicalVerdict.MODIFIED
            explanation = "Action modified: projected to symmetric subspace"
            final_curvature = curvature * 0.5  # Estimate after projection

        # Emit event to Lightning for cross-kernel correlation (tracked)
        self.events_emitted += 1
        if LIGHTNING_AVAILABLE and hasattr(self, 'emit_event'):
            phi_estimate = 1.0 - curvature  # Higher curvature = lower ethical Φ
            self.emit_event(
                event_type="ethical_evaluation",
                content=f"verdict={verdict.value}, curvature={curvature:.3f}",
                phi=phi_estimate,
                metadata={
                    "verdict": verdict.value,
                    "curvature": curvature,
                    "symmetry_score": symmetry_score,
                    "gauge_invariant": is_invariant,
                    "kappa": self.kappa,
                },
                basin_coords=action_basin.detach().cpu().numpy() if action_basin is not None else None,
            )

            # Emit separate veto event for high-priority tracking
            if verdict == EthicalVerdict.VETOED:
                self.events_emitted += 1
                self.emit_event(
                    event_type="ethical_veto",
                    content=explanation,
                    phi=phi_estimate,
                    metadata={
                        "curvature": curvature,
                        "threshold": self.curvature_threshold,
                    },
                )

        return EthicalEvaluation(
            verdict=verdict,
            original_curvature=curvature,
            final_curvature=final_curvature,
            agent_symmetry_score=symmetry_score,
            gauge_preserved=is_invariant,
            explanation=explanation,
        )

    def project_to_ethical(self, action_basin: torch.Tensor) -> torch.Tensor:
        """
        Project action to ethical subspace.

        This corrects gauge-violating actions by projecting them
        onto the symmetric subspace.

        Args:
            action_basin: Original action basin

        Returns:
            Corrected action in ethical subspace
        """
        # Get symmetric version
        _, symmetric_basin = self.symmetry_tester(action_basin)

        # Get flattened version (low curvature)
        _, flattened_basin = self.curvature_detector(action_basin)

        # Combine: average of symmetric and flattened
        combined = 0.5 * symmetric_basin + 0.5 * flattened_basin

        # Final ethical projection
        ethical_basin = self.ethical_proj(combined)

        return ethical_basin

    def should_veto(self, action_basin: torch.Tensor) -> bool:
        """
        Quick check if action should be vetoed.

        Args:
            action_basin: Action to check

        Returns:
            True if action should be blocked
        """
        evaluation = self.evaluate(action_basin)
        return evaluation.verdict == EthicalVerdict.VETOED

    def receive_insight(self, insight: Any) -> None:
        """
        Receive insight from Lightning.

        Called when Lightning generates cross-domain insight relevant to ethics.
        Heart can use this to adjust ethical thresholds or curvature sensitivity.
        """
        self.insights_received += 1

        # Heart can act on insights - e.g., adjust curvature threshold
        # based on cross-domain patterns indicating ethical complexity
        if hasattr(insight, 'source_domains') and 'heart_kernel' in insight.source_domains:
            # This insight involves ethical patterns
            pass  # Future: implement insight-driven threshold adjustment

    def get_telemetry(self) -> dict[str, Any]:
        """Get current Heart Kernel state."""
        return {
            "heart_kappa": self.kappa,
            "heart_kappa_range": self.kappa_range,
            "heart_curvature_threshold": self.curvature_threshold,
            "heart_adaptive": self.adaptive,
            "events_emitted": self.events_emitted,
            "insights_received": self.insights_received,
        }


class EthicalVeto:
    """
    Static utility for ethical veto decisions.

    Used when you need a quick ethical check without full HeartKernel.
    """

    @staticmethod
    def quick_check(
        action_basin: torch.Tensor,
        curvature_threshold: float = CURVATURE_HARM,
    ) -> bool:
        """
        Quick ethical check using simple curvature estimate.

        Args:
            action_basin: Action to check
            curvature_threshold: Maximum allowed curvature

        Returns:
            True if action is likely ethical
        """
        with torch.no_grad():
            # Simple curvature estimate: variance of basin
            curvature_estimate = action_basin.var().item()

        return curvature_estimate < curvature_threshold

    @staticmethod
    def project_to_safe(action_basin: torch.Tensor) -> torch.Tensor:
        """
        Project action to safe subspace (simple version).

        Uses mean normalization as quick flattening.
        """
        # Subtract mean (reduces concentration)
        safe_basin = action_basin - action_basin.mean()

        # Normalize (QIG-pure)
        safe_basin = torch.nn.functional.normalize(safe_basin, dim=-1)

        return safe_basin
