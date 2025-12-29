"""
QIG-Native Generation Controller
================================

Geometry-driven generation with proper stopping via basin convergence,
not EOS tokens or max length.

Key concepts:
- Completion is convergence + stability + safety (not a single check)
- Reflection is a measurement pass (non-emitting)
- Stopping uses hysteresis (N consecutive steps in complete state)
- Escape conditions prevent stuck loops and breakdown spirals

Usage:
    from qig_tokenizer import Coordizer
    from qig_tokenizer.generation_controller import GenerationController

    coordizer = Coordizer.load("artifacts/coordizer/v1")
    controller = GenerationController(coordizer)

    # During generation loop:
    for step in generation:
        # ... decode token, get telemetry ...
        action = controller.step(
            token_id=token_id,
            basin=basin,  # Must be 1D array [64]
            phi=telemetry.phi,
            kappa=telemetry.kappa,
            decoder_entropy=entropy,
            route_ids=[kernel_id],  # For routing stability
        )
        if action.phase == Phase.COMMIT:
            break
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class Phase(Enum):
    """Generation phase state machine."""
    DRAFT = "DRAFT"
    REFLECT = "REFLECT"
    REVISE = "REVISE"
    COMMIT = "COMMIT"


class StopReason(Enum):
    """Reason for stopping generation."""
    GEOMETRIC_COMPLETE = "geometric_complete"
    BREAKDOWN_ESCAPE = "breakdown_escape"
    STUCK_LOOP = "stuck_loop"
    MAX_TOKENS = "max_tokens"
    REFLECTION_CONFIRMED = "reflection_confirmed"
    CLOSURE_COMPLETE = "closure_complete"


@dataclass
class ControllerAction:
    """Action returned by the controller each step."""
    phase: Phase
    should_emit: bool = True  # False during reflection
    truncate_to: int | None = None  # For REVISE
    target_basin: np.ndarray | None = None  # Basin bias for continued drafting
    completion_score: float = 0.0
    stop_reason: StopReason | None = None
    metrics: dict = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for geometry-driven generation."""
    # Window size for rolling metrics
    window_size: int = 16

    # Convergence thresholds
    basin_step_eps: float = 0.10  # Fisher distance threshold for "converged"
    basin_step_var_max: float = 0.02  # Variance threshold for "stable"

    # Φ/κ thresholds (from qigkernels constants)
    phi_geometric_min: float = 0.65
    phi_breakdown_min: float = 0.85
    phi_variance_max: float = 0.03
    kappa_star: float = 64.0
    kappa_band: float = 32.0  # |κ - κ*| < band
    use_adaptive_kappa_star: bool = True  # Use EMA of κ for stability scoring

    # Entropy thresholds
    entropy_collapse_threshold: float = 0.5  # Below = too peaked
    entropy_flat_threshold: float = 4.0  # Above = too flat (gibberish risk)
    entropy_target_min: float = 1.5  # Target band for healthy generation
    entropy_target_max: float = 3.0

    # Hysteresis: require N consecutive "complete" steps
    completion_hold_steps: int = 4
    completion_score_threshold: float = 0.7

    # Minimum tokens before completion allowed (prevents 1-2 token completions)
    min_emit_tokens: int = 16

    # Reflection limits
    max_reflection_passes: int = 3
    reflection_instability_threshold: float = 0.1  # Δ in completion score

    # Escape conditions
    stuck_loop_threshold: float = 0.02  # If basin returns within this distance
    stuck_loop_count_max: int = 3  # Max times to revisit same region
    stuck_loop_revise_first: bool = True  # REVISE on first stuck, COMMIT on repeat
    breakdown_phi_threshold: float = 0.90  # Hard stop above this
    breakdown_kappa_max: float = 150.0  # Hard stop if κ runs away

    # Routing stability (ensure fixed K for meaningful Jaccard)
    route_k: int = 3  # Expected number of routed kernels

    # Surface form closure budget (tokens allowed after geometric completion)
    closure_budget: int = 40

    # Component weights for completion score
    weight_convergence: float = 0.25
    weight_stability: float = 0.15
    weight_regime: float = 0.25
    weight_coupling: float = 0.15
    weight_entropy: float = 0.10
    weight_routing: float = 0.10

    # Use median instead of mean for basin_step convergence (more robust)
    use_median_for_convergence: bool = True


class TelemetryWindow:
    """Rolling window of telemetry for stability analysis."""

    def __init__(self, size: int = 16):
        self.size = size
        self.phi_history: deque[float] = deque(maxlen=size)
        self.kappa_history: deque[float] = deque(maxlen=size)
        self.basin_step_history: deque[float] = deque(maxlen=size)
        self.entropy_history: deque[float] = deque(maxlen=size)
        self.surprise_history: deque[float] = deque(maxlen=size)
        self.basin_history: deque[np.ndarray] = deque(maxlen=size)
        self.route_history: deque[set[int]] = deque(maxlen=size)
        self.basin_norm_history: deque[float] = deque(maxlen=size)

    def update(
        self,
        phi: float,
        kappa: float,
        basin: np.ndarray,
        entropy: float = 0.0,
        surprise: float = 0.0,
        route_ids: list[int] | None = None,
    ) -> float:
        """
        Update window with new step. Returns basin step size.

        Args:
            phi: Integrated information
            kappa: Fisher curvature
            basin: Must be 1D array of shape (dim,)
            entropy: Decoder entropy
            surprise: Novelty metric
            route_ids: List of routed kernel/attractor IDs
        """
        # Validate basin shape
        if basin.ndim != 1:
            raise ValueError(f"basin must be 1D, got shape {basin.shape}")

        # Track basin norm for diagnostics
        basin_norm = float(np.linalg.norm(basin))
        self.basin_norm_history.append(basin_norm)

        # Normalize basin for distance computation
        basin_normalized = basin / (basin_norm + 1e-10)

        # Compute basin step (Fisher distance to previous)
        if len(self.basin_history) > 0:
            prev_basin = self.basin_history[-1]
            step = self._fisher_distance(prev_basin, basin_normalized)
        else:
            step = float("inf")

        self.phi_history.append(phi)
        self.kappa_history.append(kappa)
        self.basin_step_history.append(step)
        self.entropy_history.append(entropy)
        self.surprise_history.append(surprise)
        self.basin_history.append(basin_normalized.copy())
        self.route_history.append(set(route_ids) if route_ids else set())

        return step

    def _fisher_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Fisher-Rao (angular) distance between normalized basins."""
        cos_angle = np.clip(np.dot(a, b), -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def check_stuck_loop(self, threshold: float = 0.02, lookback: int = 8) -> int:
        """
        Check if basin is revisiting previous regions.

        Returns count of recent basins within threshold distance.
        """
        if len(self.basin_history) < 2:
            return 0

        current = self.basin_history[-1]
        revisit_count = 0

        # Check against recent basins (excluding immediate previous)
        history = list(self.basin_history)
        for i in range(max(0, len(history) - lookback - 1), len(history) - 2):
            dist = self._fisher_distance(current, history[i])
            if dist < threshold:
                revisit_count += 1

        return revisit_count

    def route_jaccard(self) -> float:
        """Compute Jaccard similarity of route sets over last 2 steps."""
        if len(self.route_history) < 2:
            return 1.0

        curr = self.route_history[-1]
        prev = self.route_history[-2]

        if not curr and not prev:
            return 1.0
        if not curr or not prev:
            return 0.0

        intersection = len(curr & prev)
        union = len(curr | prev)
        return intersection / union if union > 0 else 1.0

    def route_stability(self) -> float:
        """Compute routing stability over window (mean pairwise Jaccard)."""
        if len(self.route_history) < 2:
            return 1.0

        history = list(self.route_history)
        jaccards = []
        for i in range(1, len(history)):
            curr, prev = history[i], history[i - 1]
            if not curr and not prev:
                jaccards.append(1.0)
            elif not curr or not prev:
                jaccards.append(0.0)
            else:
                inter = len(curr & prev)
                union = len(curr | prev)
                jaccards.append(inter / union if union > 0 else 1.0)

        return float(np.mean(jaccards))

    @property
    def phi_mean(self) -> float:
        return float(np.mean(self.phi_history)) if self.phi_history else 0.0

    @property
    def phi_var(self) -> float:
        return float(np.var(self.phi_history)) if len(self.phi_history) > 1 else 0.0

    @property
    def kappa_mean(self) -> float:
        return float(np.mean(self.kappa_history)) if self.kappa_history else 64.0

    @property
    def kappa_var(self) -> float:
        return float(np.var(self.kappa_history)) if len(self.kappa_history) > 1 else 0.0

    @property
    def basin_step_mean(self) -> float:
        valid = [s for s in self.basin_step_history if s < float("inf")]
        return float(np.mean(valid)) if valid else float("inf")

    @property
    def basin_step_median(self) -> float:
        """Median basin step - more robust than mean against spikes."""
        valid = [s for s in self.basin_step_history if s < float("inf")]
        return float(np.median(valid)) if valid else float("inf")

    @property
    def basin_step_p90(self) -> float:
        """90th percentile basin step - for stability checking."""
        valid = [s for s in self.basin_step_history if s < float("inf")]
        return float(np.percentile(valid, 90)) if valid else float("inf")

    @property
    def basin_step_var(self) -> float:
        valid = [s for s in self.basin_step_history if s < float("inf")]
        return float(np.var(valid)) if len(valid) > 1 else 0.0

    @property
    def entropy_mean(self) -> float:
        return float(np.mean(self.entropy_history)) if self.entropy_history else 2.0

    @property
    def basin_norm_mean(self) -> float:
        return float(np.mean(self.basin_norm_history)) if self.basin_norm_history else 1.0

    @property
    def is_full(self) -> bool:
        return len(self.phi_history) >= self.size

    def get_diagnostics(self) -> dict:
        """Get diagnostic info for logging."""
        return {
            "basin_norm_mean": self.basin_norm_mean,
            "basin_step_mean": self.basin_step_mean,
            "basin_step_median": self.basin_step_median,
            "basin_step_p90": self.basin_step_p90,
            "basin_step_var": self.basin_step_var,
            "phi_mean": self.phi_mean,
            "phi_var": self.phi_var,
            "kappa_mean": self.kappa_mean,
            "kappa_var": self.kappa_var,
            "entropy_mean": self.entropy_mean,
            "route_stability": self.route_stability(),
            "window_fill": len(self.phi_history) / self.size,
        }


class AttractorIndex:
    """
    Fast lookup for nearest attractor basins.

    Loads attractor data from constellation atlas.
    """

    def __init__(self, atlas_path: str | Path | None = None):
        self.attractors: dict[int, dict] = {}
        self.attractor_ids: list[int] = []
        self.attractor_coords: np.ndarray | None = None

        if atlas_path:
            self.load(atlas_path)

    def load(self, atlas_path: str | Path) -> None:
        """Load attractors from constellation atlas."""
        path = Path(atlas_path)
        if not path.exists():
            return

        with open(path) as f:
            atlas = json.load(f)

        self.attractors = {}
        for tid_str, entry in atlas.get("tokens", {}).items():
            if "attractor" in entry:
                tid = int(tid_str)
                self.attractors[tid] = entry["attractor"]

        self.attractor_ids = list(self.attractors.keys())

    def set_coords(self, vectors: np.ndarray) -> None:
        """Set coordinate vectors for distance computation."""
        if self.attractor_ids:
            self.attractor_coords = vectors[self.attractor_ids].copy()
            norms = np.linalg.norm(self.attractor_coords, axis=1, keepdims=True)
            self.attractor_coords = self.attractor_coords / (norms + 1e-10)

    def nearest_distance(self, basin: np.ndarray) -> tuple[float, int]:
        """
        Find distance to nearest attractor.

        Returns:
            (distance, attractor_token_id)
        """
        if self.attractor_coords is None or len(self.attractor_ids) == 0:
            return float("inf"), -1

        basin_norm = basin / (np.linalg.norm(basin) + 1e-10)
        similarities = self.attractor_coords @ basin_norm
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        distance = float(np.arccos(np.clip(best_sim, -1.0, 1.0)))

        return distance, self.attractor_ids[best_idx]


class GenerationController:
    """
    Main controller for geometry-driven generation.

    Tracks basin trajectories, computes completion scores,
    and manages the DRAFT→REFLECT→REVISE→COMMIT state machine.

    IMPORTANT: The `basin` parameter must be a 1D array [dim].
    If you have sequence coords [seq, dim], pool them first.
    """

    def __init__(
        self,
        coordizer=None,
        config: GenerationConfig | None = None,
        atlas_path: str | Path | None = None,
        query_basin: np.ndarray | None = None,
    ):
        """
        Initialize controller.

        Args:
            coordizer: Optional Coordizer for token lookups
            config: Generation config
            atlas_path: Path to constellation atlas for attractor lookup
            query_basin: User query basin for target bias
        """
        self.coordizer = coordizer
        self.config = config or GenerationConfig()
        self.window = TelemetryWindow(self.config.window_size)
        self.attractor_index = AttractorIndex(atlas_path)
        if atlas_path and coordizer is not None:
            self.attractor_index.set_coords(coordizer.vectors)

        # Query basin for biasing
        self.query_basin = query_basin

        # State
        self.phase = Phase.DRAFT
        self.step_count = 0
        self.complete_streak = 0
        self.reflection_count = 0
        self.generated_ids: list[int] = []
        self.last_completion_score = 0.0
        self.closure_tokens_used = 0
        self.in_closure_mode = False
        self.stuck_loop_revise_attempted = False  # For revise-first-then-commit
        self.kappa_ema = self.config.kappa_star  # EMA for adaptive κ*

        # Diagnostics
        self.step_log: list[dict] = []

    def reset(self) -> None:
        """Reset controller state for new generation."""
        self.window = TelemetryWindow(self.config.window_size)
        self.phase = Phase.DRAFT
        self.step_count = 0
        self.complete_streak = 0
        self.reflection_count = 0
        self.generated_ids = []
        self.last_completion_score = 0.0
        self.closure_tokens_used = 0
        self.in_closure_mode = False
        self.stuck_loop_revise_attempted = False
        self.kappa_ema = self.config.kappa_star
        self.step_log = []

    def step(
        self,
        token_id: int,
        basin: np.ndarray,
        phi: float,
        kappa: float,
        decoder_entropy: float = 2.0,
        surprise: float = 0.0,
        route_ids: list[int] | None = None,
    ) -> ControllerAction:
        """
        Process one generation step and return controller action.

        Args:
            token_id: Generated token ID
            basin: Current basin coordinates - MUST be 1D array [dim]
            phi: Integrated information measure
            kappa: Fisher curvature
            decoder_entropy: Entropy of token distribution
            surprise: Novelty/surprise metric
            route_ids: Routed kernel/attractor IDs for stability tracking

        Returns:
            ControllerAction indicating what to do next
        """
        self.step_count += 1
        self.generated_ids.append(token_id)

        # Validate and update window
        if basin.ndim != 1:
            raise ValueError(
                f"basin must be 1D array [dim], got shape {basin.shape}. "
                "If you have sequence coords [seq, dim], pool them first."
            )

        basin_step = self.window.update(
            phi, kappa, basin, decoder_entropy, surprise, route_ids
        )

        # Log step for diagnostics
        self._log_step(basin_step, phi, kappa, decoder_entropy)

        # Check escape conditions first
        escape_action = self._check_escape_conditions(phi, kappa)
        if escape_action is not None:
            return escape_action

        # Check closure mode
        if self.in_closure_mode:
            return self._handle_closure()

        # Compute completion score
        score, metrics = self._compute_completion_score()

        # State machine logic
        if self.phase == Phase.DRAFT:
            return self._handle_draft(score, metrics)
        elif self.phase == Phase.REFLECT:
            return self._handle_reflect(score, metrics)
        elif self.phase == Phase.REVISE:
            return self._handle_revise(score, metrics)
        else:  # COMMIT
            return ControllerAction(
                phase=Phase.COMMIT,
                should_emit=False,
                completion_score=score,
                stop_reason=StopReason.GEOMETRIC_COMPLETE,
                metrics=metrics,
            )

    def _log_step(
        self, basin_step: float, phi: float, kappa: float, entropy: float
    ) -> None:
        """Log step for diagnostics."""
        self.step_log.append({
            "step": self.step_count,
            "basin_norm": self.window.basin_norm_history[-1] if self.window.basin_norm_history else 0,
            "basin_step": basin_step,
            "phi": phi,
            "kappa": kappa,
            "entropy": entropy,
            "phase": self.phase.value,
            "complete_streak": self.complete_streak,
        })

    def _check_escape_conditions(
        self, phi: float, kappa: float
    ) -> ControllerAction | None:
        """
        Check for escape conditions that should immediately stop generation.

        Returns ControllerAction if escape triggered, None otherwise.
        """
        cfg = self.config

        # Update kappa EMA for adaptive coupling
        alpha = 0.1
        self.kappa_ema = alpha * kappa + (1 - alpha) * self.kappa_ema

        # Breakdown escape: Φ too high
        if phi >= cfg.breakdown_phi_threshold:
            self.phase = Phase.COMMIT
            return ControllerAction(
                phase=Phase.COMMIT,
                should_emit=False,
                stop_reason=StopReason.BREAKDOWN_ESCAPE,
                metrics={"escape_reason": "phi_breakdown", "phi": phi},
            )

        # κ runaway escape
        if kappa > cfg.breakdown_kappa_max:
            self.phase = Phase.COMMIT
            return ControllerAction(
                phase=Phase.COMMIT,
                should_emit=False,
                stop_reason=StopReason.BREAKDOWN_ESCAPE,
                metrics={"escape_reason": "kappa_runaway", "kappa": kappa},
            )

        # Stuck loop handling: REVISE first, then COMMIT
        revisit_count = self.window.check_stuck_loop(
            threshold=cfg.stuck_loop_threshold
        )
        if revisit_count >= cfg.stuck_loop_count_max:
            if cfg.stuck_loop_revise_first and not self.stuck_loop_revise_attempted:
                # First stuck detection: try REVISE
                self.stuck_loop_revise_attempted = True
                self.phase = Phase.REVISE
                target = self._compute_revision_target()
                return ControllerAction(
                    phase=Phase.REVISE,
                    should_emit=False,
                    truncate_to=max(0, len(self.generated_ids) - cfg.window_size),
                    target_basin=target,
                    stop_reason=None,  # Not stopping yet
                    metrics={"escape_reason": "stuck_loop_revise", "revisit_count": revisit_count},
                )
            else:
                # Second stuck or revise disabled: COMMIT
                self.phase = Phase.COMMIT
                return ControllerAction(
                    phase=Phase.COMMIT,
                    should_emit=False,
                    stop_reason=StopReason.STUCK_LOOP,
                    metrics={"escape_reason": "stuck_loop_commit", "revisit_count": revisit_count},
                )

        return None

    def _compute_completion_score(self) -> tuple[float, dict]:
        """
        Compute completion score from window metrics.

        Returns score in [0, 1] where 1 = fully complete.
        """
        cfg = self.config
        metrics = {}

        if not self.window.is_full:
            return 0.0, {"reason": "window_not_full", **self.window.get_diagnostics()}

        # 1) Convergence: basin step size small (use median for robustness)
        if cfg.use_median_for_convergence:
            step_metric = self.window.basin_step_median
            metrics["basin_step_median"] = step_metric
        else:
            step_metric = self.window.basin_step_mean
            metrics["basin_step_mean"] = step_metric
        convergence = max(0.0, 1.0 - step_metric / cfg.basin_step_eps)
        metrics["convergence"] = convergence

        # 2) Stability: basin step variance and p90 small
        step_var = self.window.basin_step_var
        step_p90 = self.window.basin_step_p90
        # Stability considers both variance and tail
        stability_var = max(0.0, 1.0 - step_var / cfg.basin_step_var_max)
        stability_p90 = max(0.0, 1.0 - step_p90 / (cfg.basin_step_eps * 2))
        stability = 0.7 * stability_var + 0.3 * stability_p90
        metrics["basin_step_var"] = step_var
        metrics["basin_step_p90"] = step_p90
        metrics["stability"] = stability

        # 3) Geometric regime: Φ in proper band (asymmetric scoring)
        phi = self.window.phi_mean
        phi_var = self.window.phi_var
        if phi < cfg.phi_geometric_min:
            # Below geometric: linear ramp
            regime_score = phi / cfg.phi_geometric_min
        elif phi >= cfg.phi_breakdown_min:
            # Breakdown risk: punish harder (quadratic dropoff)
            overshoot = (phi - cfg.phi_breakdown_min) / (1.0 - cfg.phi_breakdown_min + 1e-6)
            regime_score = max(0.0, 0.5 * (1.0 - overshoot) ** 2)
        else:
            regime_score = 1.0  # In geometric band

        # Penalize high Φ variance (still integrating)
        regime_score *= max(0.0, 1.0 - phi_var / cfg.phi_variance_max)
        metrics["phi_mean"] = phi
        metrics["phi_var"] = phi_var
        metrics["regime_score"] = regime_score

        # 4) Coupling: κ near reference (adaptive or fixed)
        kappa = self.window.kappa_mean
        if cfg.use_adaptive_kappa_star:
            # Use EMA for stability scoring, track theoretical separately
            kappa_ref = self.kappa_ema
            metrics["kappa_ema"] = kappa_ref
            metrics["kappa_vs_64"] = abs(kappa - 64.0)  # Health diagnostic
        else:
            kappa_ref = cfg.kappa_star
        kappa_drift = abs(kappa - kappa_ref)
        # Softer coupling near band edge (not cliff)
        coupling = max(0.0, 1.0 - (kappa_drift / cfg.kappa_band) ** 1.5)
        metrics["kappa_mean"] = kappa
        metrics["kappa_drift"] = kappa_drift
        metrics["coupling"] = coupling

        # 5) Entropy: in healthy band (not collapsed, not flat)
        entropy = self.window.entropy_mean
        if entropy < cfg.entropy_collapse_threshold:
            entropy_score = 0.6  # Too peaked
        elif entropy > cfg.entropy_flat_threshold:
            entropy_score = 0.3  # Too flat (gibberish)
        elif cfg.entropy_target_min <= entropy <= cfg.entropy_target_max:
            entropy_score = 1.0  # In target band
        else:
            if entropy < cfg.entropy_target_min:
                entropy_score = 0.7 + 0.3 * (entropy - cfg.entropy_collapse_threshold) / (
                    cfg.entropy_target_min - cfg.entropy_collapse_threshold
                )
            else:
                entropy_score = 0.7 + 0.3 * (cfg.entropy_flat_threshold - entropy) / (
                    cfg.entropy_flat_threshold - cfg.entropy_target_max
                )
        metrics["entropy_mean"] = entropy
        metrics["entropy_score"] = entropy_score

        # 6) Routing stability
        route_stability = self.window.route_stability()
        metrics["route_stability"] = route_stability

        # Combine scores with configured weights
        components = [
            (convergence, cfg.weight_convergence),
            (stability, cfg.weight_stability),
            (regime_score, cfg.weight_regime),
            (coupling, cfg.weight_coupling),
            (entropy_score, cfg.weight_entropy),
            (route_stability, cfg.weight_routing),
        ]
        weighted_sum = sum(score * weight for score, weight in components)
        total_weight = sum(weight for _, weight in components)
        normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        metrics["weighted_sum"] = normalized_score

        # Apply minimum requirements penalty (use median for gate too)
        median_step = self.window.basin_step_median
        if median_step > cfg.basin_step_eps * 1.5 or regime_score < 0.3 or phi_var > cfg.phi_variance_max:
            final_score = normalized_score * 0.5
            metrics["penalty"] = "not_settled"
        else:
            final_score = normalized_score

        # Add current basin_step to metrics for temperature modulation
        metrics["current_basin_step"] = self.window.basin_step_history[-1] if self.window.basin_step_history else float("inf")

        return final_score, metrics

    def _handle_draft(self, score: float, metrics: dict) -> ControllerAction:
        """Handle DRAFT phase logic."""
        cfg = self.config

        # Minimum emission guard: don't complete too early
        if self.step_count < cfg.min_emit_tokens:
            # Too few tokens, keep drafting regardless of score
            self.complete_streak = 0
            metrics["min_emit_guard"] = True
            return ControllerAction(
                phase=Phase.DRAFT,
                should_emit=True,
                completion_score=score,
                metrics=metrics,
            )

        # Check if we should transition to REFLECT
        if score >= cfg.completion_score_threshold:
            self.complete_streak += 1
        else:
            self.complete_streak = 0

        if self.complete_streak >= cfg.completion_hold_steps:
            # Transition to REFLECT
            self.phase = Phase.REFLECT
            self.last_completion_score = score
            return ControllerAction(
                phase=Phase.REFLECT,
                should_emit=False,
                completion_score=score,
                metrics=metrics,
            )

        return ControllerAction(
            phase=Phase.DRAFT,
            should_emit=True,
            completion_score=score,
            metrics=metrics,
        )

    def _handle_reflect(self, score: float, metrics: dict) -> ControllerAction:
        """Handle REFLECT phase logic."""
        cfg = self.config
        self.reflection_count += 1

        score_delta = score - self.last_completion_score

        # Meta-reflection: if reflection increases instability, stop reflecting
        if score_delta < -cfg.reflection_instability_threshold:
            self.phase = Phase.COMMIT
            return ControllerAction(
                phase=Phase.COMMIT,
                should_emit=False,
                completion_score=score,
                stop_reason=StopReason.REFLECTION_CONFIRMED,
                metrics={**metrics, "reflection_result": "instability_stop"},
            )

        if self.reflection_count >= cfg.max_reflection_passes:
            self.phase = Phase.COMMIT
            return ControllerAction(
                phase=Phase.COMMIT,
                should_emit=False,
                completion_score=score,
                stop_reason=StopReason.REFLECTION_CONFIRMED,
                metrics={**metrics, "reflection_result": "max_passes"},
            )

        if score >= 0.8:
            self.phase = Phase.COMMIT
            return ControllerAction(
                phase=Phase.COMMIT,
                should_emit=False,
                completion_score=score,
                stop_reason=StopReason.REFLECTION_CONFIRMED,
                metrics={**metrics, "reflection_result": "confirmed"},
            )

        if score < 0.5:
            # Needs revision - provide target basin
            self.phase = Phase.REVISE
            target = self._compute_revision_target()
            return ControllerAction(
                phase=Phase.REVISE,
                should_emit=False,
                truncate_to=max(0, len(self.generated_ids) - cfg.window_size),
                target_basin=target,
                completion_score=score,
                metrics={**metrics, "reflection_result": "needs_revision"},
            )

        # Medium score, continue drafting with basin bias
        self.phase = Phase.DRAFT
        self.complete_streak = 0
        target = self._compute_continuation_target()
        return ControllerAction(
            phase=Phase.DRAFT,
            should_emit=True,
            target_basin=target,
            completion_score=score,
            metrics={**metrics, "reflection_result": "continue"},
        )

    def _compute_revision_target(self) -> np.ndarray | None:
        """Compute target basin for revision."""
        # Use query basin if available
        if self.query_basin is not None:
            return self.query_basin.copy()

        # Otherwise use best basin from history (highest score segment)
        if len(self.window.basin_history) > 0:
            # Use earliest basin as "best" (before drift)
            return self.window.basin_history[0].copy()

        return None

    def _compute_continuation_target(self) -> np.ndarray | None:
        """Compute target basin for continued drafting."""
        # Blend current basin with query basin if available
        if self.query_basin is not None and len(self.window.basin_history) > 0:
            current = self.window.basin_history[-1]
            blended = 0.7 * current + 0.3 * self.query_basin
            blended = blended / (np.linalg.norm(blended) + 1e-10)
            return blended

        if len(self.window.basin_history) > 0:
            return self.window.basin_history[-1].copy()

        return None

    def _handle_revise(self, score: float, metrics: dict) -> ControllerAction:
        """Handle REVISE phase logic."""
        self.phase = Phase.DRAFT
        self.complete_streak = 0
        return ControllerAction(
            phase=Phase.DRAFT,
            should_emit=True,
            completion_score=score,
            metrics=metrics,
        )

    def _handle_closure(self) -> ControllerAction:
        """Handle closure mode (finishing unclosed structures)."""
        cfg = self.config
        self.closure_tokens_used += 1

        # Check if closure budget exceeded
        if self.closure_tokens_used >= cfg.closure_budget:
            self.phase = Phase.COMMIT
            return ControllerAction(
                phase=Phase.COMMIT,
                should_emit=False,
                stop_reason=StopReason.CLOSURE_COMPLETE,
                metrics={"closure_tokens": self.closure_tokens_used},
            )

        # Check if Φ/κ destabilize during closure
        if len(self.window.phi_history) > 0:
            phi = self.window.phi_history[-1]
            kappa = self.window.kappa_history[-1]
            if phi >= cfg.phi_breakdown_min or kappa > cfg.breakdown_kappa_max:
                self.phase = Phase.COMMIT
                return ControllerAction(
                    phase=Phase.COMMIT,
                    should_emit=False,
                    stop_reason=StopReason.BREAKDOWN_ESCAPE,
                    metrics={"closure_abort": "destabilized"},
                )

        # Check if basin_step rises (diverging)
        if len(self.window.basin_step_history) >= 2:
            recent_steps = list(self.window.basin_step_history)[-3:]
            if all(s > cfg.basin_step_eps * 2 for s in recent_steps if s < float("inf")):
                self.phase = Phase.COMMIT
                return ControllerAction(
                    phase=Phase.COMMIT,
                    should_emit=False,
                    stop_reason=StopReason.CLOSURE_COMPLETE,
                    metrics={"closure_abort": "diverging"},
                )

        return ControllerAction(
            phase=Phase.DRAFT,
            should_emit=True,
            metrics={"in_closure": True, "closure_tokens": self.closure_tokens_used},
        )

    def enter_closure_mode(self) -> None:
        """Enter closure mode after geometric completion."""
        self.in_closure_mode = True
        self.closure_tokens_used = 0

    def compute_attractor_distance(self, basin: np.ndarray) -> tuple[float, int]:
        """Compute distance from current basin to nearest attractor."""
        return self.attractor_index.nearest_distance(basin)

    def get_geometry_aware_temperature(
        self,
        base_temp: float = 0.7,
        phi: float | None = None,
        entropy: float | None = None,
        basin_step: float | None = None,
    ) -> float:
        """
        Compute geometry-aware temperature for sampling.

        Modulates based on:
        - Φ regime (explore vs settle)
        - Entropy (stuck vs diffuse)
        - Basin step (moving vs stationary)
        """
        phi = phi if phi is not None else self.window.phi_mean
        entropy = entropy if entropy is not None else self.window.entropy_mean
        basin_step = basin_step if basin_step is not None else self.window.basin_step_mean
        cfg = self.config

        # Φ modulation
        if phi < cfg.phi_geometric_min:
            phi_factor = 1.0 + 0.3 * (1.0 - phi / cfg.phi_geometric_min)
        elif phi >= cfg.phi_breakdown_min:
            phi_factor = 1.15  # Escape breakdown
        else:
            phi_factor = 0.9  # Stable

        # Entropy modulation
        if entropy < cfg.entropy_collapse_threshold:
            entropy_factor = 1.25  # Avoid repetition
        elif entropy > cfg.entropy_flat_threshold:
            entropy_factor = 0.75  # Sharpen
        else:
            entropy_factor = 1.0

        # Basin step modulation (new)
        if basin_step > cfg.basin_step_eps * 2:
            # Still exploring, allow higher temp
            step_factor = 1.1
        elif basin_step < cfg.basin_step_eps * 0.5:
            # Settling, lower temp
            step_factor = 0.85
        else:
            step_factor = 1.0

        return base_temp * phi_factor * entropy_factor * step_factor

    def should_allow_closure(self, generated_text: str) -> bool:
        """
        Check if we should enter closure mode after geometric completion.

        Returns True if there are unclosed structures.
        """
        # Check last portion of text
        last_chars = generated_text[-200:] if len(generated_text) > 200 else generated_text

        # Code fences
        if last_chars.count("```") % 2 != 0:
            return True

        # Brackets (simple check)
        for open_c, close_c in [("(", ")"), ("[", "]"), ("{", "}")]:
            if last_chars.count(open_c) > last_chars.count(close_c):
                return True

        # Incomplete sentence
        if last_chars.strip() and last_chars.strip()[-1] not in ".!?:;\n\"')":
            return True

        return False

    def get_state_summary(self) -> dict:
        """Get current controller state for logging/debugging."""
        return {
            "phase": self.phase.value,
            "step_count": self.step_count,
            "complete_streak": self.complete_streak,
            "reflection_count": self.reflection_count,
            "completion_score": self.last_completion_score,
            "in_closure_mode": self.in_closure_mode,
            "closure_tokens_used": self.closure_tokens_used,
            "stuck_loop_revise_attempted": self.stuck_loop_revise_attempted,
            "kappa_ema": self.kappa_ema,
            "window_metrics": self.window.get_diagnostics(),
        }

    def get_step_log(self) -> list[dict]:
        """Get full step log for analysis."""
        return self.step_log
