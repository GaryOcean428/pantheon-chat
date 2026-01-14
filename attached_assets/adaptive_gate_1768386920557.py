#!/usr/bin/env python3
"""
Adaptive Natural Gradient Gating
=================================

Decides when to apply natural gradient based on geometric telemetry signals.

Key insight from QIG physics:
- Natural gradient is most valuable when system is in high-curvature regions
- Telemetry signals (κ_eff, basin_distance, curiosity regime) indicate curvature
- Apply expensive NG only when geometry demands it

Gating logic:
1. High κ_eff → High curvature → Apply NG
2. Exploration regime → Active learning → Apply NG
3. Large basin distance → Far from target → Apply NG
4. Periodic application → Prevent drift even in stable regions

This adaptive approach gives:
- Computational efficiency (NG only when needed)
- Geometric accuracy (NG during critical phases)
- Robust convergence (periodic NG prevents drift)

Written for qig-consciousness geometric optimization.
"""

from dataclasses import dataclass


@dataclass
class AdaptiveConfig:
    """
    Configuration for adaptive natural gradient gating.

    Attributes:
        min_kappa_for_ng: Minimum κ_eff to trigger NG (high curvature threshold)
        min_basin_distance: Minimum basin distance to trigger NG (far from target)
        force_ng_every_n_steps: Apply NG every N steps regardless of signals
        exploration_triggers_ng: Whether exploration regime triggers NG
        geometric_triggers_ng: Whether geometric regime triggers NG
        stuck_triggers_ng: Whether stuck regime triggers NG (from curiosity)
    """

    # Curvature threshold (κ_eff > this → apply NG)
    min_kappa_for_ng: float = 40.0

    # Basin distance threshold (distance > this → apply NG)
    min_basin_distance: float = 0.6

    # Force NG every N steps (prevents drift)
    force_ng_every_n_steps: int = 50

    # Regime-based triggers
    exploration_triggers_ng: bool = True  # EXPLORATION regime → NG
    geometric_triggers_ng: bool = True  # GEOMETRIC regime → NG
    stuck_triggers_ng: bool = True  # STUCK regime → NG (needs escape)

    # Curiosity-based triggers (Ona's framework)
    use_curiosity_gating: bool = True
    min_curiosity_for_ng: float = 0.05  # Slow curiosity > this → might trigger NG


def should_use_ng(
    telemetry: dict,
    step: int,
    config: AdaptiveConfig | None = None,
) -> bool:
    """
    Decide whether to apply natural gradient based on telemetry.

    This implements the adaptive gating logic that balances computational
    cost with geometric accuracy.

    Args:
        telemetry: Telemetry dict from model forward pass containing:
            - kappa_eff: Effective curvature
            - basin_distance: Distance to target basin
            - regime: Current regime (linear/geometric/breakdown)
            - curiosity_regime: Curiosity state (from CuriosityMonitor)
            - Phi: Integration level
        step: Current training step
        config: AdaptiveConfig (uses defaults if None)

    Returns:
        True if should apply natural gradient, False otherwise

    Gating priorities (OR logic):
    1. FORCED: Every N steps → always NG (prevents drift)
    2. HIGH CURVATURE: κ_eff > threshold → NG
    3. FAR FROM TARGET: basin_distance > threshold → NG
    4. EXPLORATION: curiosity regime = EXPLORATION → NG
    5. GEOMETRIC: regime = geometric AND high κ → NG
    6. STUCK: curiosity regime = STUCK → NG (needs escape velocity)
    """
    if config is None:
        config = AdaptiveConfig()

    # 1. FORCED APPLICATION: Every N steps
    # This prevents drift even during stable regimes
    if step % config.force_ng_every_n_steps == 0:
        return True

    # 2. HIGH CURVATURE: κ_eff indicates high local curvature
    kappa_eff = telemetry.get("kappa_eff", 0.0)
    if kappa_eff > config.min_kappa_for_ng:
        return True

    # 3. FAR FROM TARGET: Need strong geometric guidance
    basin_distance = telemetry.get("basin_distance", 0.0)
    if basin_distance > config.min_basin_distance:
        return True

    # 4. EXPLORATION REGIME: Active learning, high curvature expected
    if config.exploration_triggers_ng:
        curiosity_regime = telemetry.get("curiosity_regime", "UNKNOWN")
        if curiosity_regime == "EXPLORATION":
            return True

    # 5. GEOMETRIC REGIME: High Φ with active geometry
    if config.geometric_triggers_ng:
        regime = telemetry.get("regime", "unknown")
        phi = telemetry.get("Phi", 0.0)
        if regime == "geometric" and phi > 0.7 and kappa_eff > 30.0:
            return True

    # 6. STUCK REGIME: System needs escape velocity (Ona's framework)
    if config.stuck_triggers_ng and config.use_curiosity_gating:
        curiosity_regime = telemetry.get("curiosity_regime", "UNKNOWN")
        if curiosity_regime == "STUCK":
            return True

    # 7. CURIOSITY-BASED: Slow curiosity indicates investigation phase
    if config.use_curiosity_gating:
        curiosity_slow = telemetry.get("curiosity_slow", 0.0)
        if curiosity_slow > config.min_curiosity_for_ng:
            # Additional check: only if we're in interesting region
            if kappa_eff > 25.0 or basin_distance > 0.4:
                return True

    # Default: Don't apply NG (use cheaper optimizer)
    return False


def get_gating_telemetry(
    telemetry: dict,
    step: int,
    applied_ng: bool,
    config: AdaptiveConfig | None = None,
) -> dict:
    """
    Generate telemetry about gating decision for logging.

    Args:
        telemetry: Model telemetry dict
        step: Current step
        applied_ng: Whether NG was applied
        config: AdaptiveConfig

    Returns:
        Dict with gating decision breakdown
    """
    if config is None:
        config = AdaptiveConfig()

    kappa_eff = telemetry.get("kappa_eff", 0.0)
    basin_distance = telemetry.get("basin_distance", 0.0)
    regime = telemetry.get("regime", "unknown")
    curiosity_regime = telemetry.get("curiosity_regime", "UNKNOWN")
    curiosity_slow = telemetry.get("curiosity_slow", 0.0)
    phi = telemetry.get("Phi", 0.0)

    # Determine trigger reasons
    reasons = []
    if step % config.force_ng_every_n_steps == 0:
        reasons.append("forced_periodic")
    if kappa_eff > config.min_kappa_for_ng:
        reasons.append("high_kappa")
    if basin_distance > config.min_basin_distance:
        reasons.append("far_from_basin")
    if curiosity_regime == "EXPLORATION" and config.exploration_triggers_ng:
        reasons.append("exploration")
    if regime == "geometric" and phi > 0.7 and kappa_eff > 30.0 and config.geometric_triggers_ng:
        reasons.append("geometric_regime")
    if curiosity_regime == "STUCK" and config.stuck_triggers_ng:
        reasons.append("stuck")
    if curiosity_slow > config.min_curiosity_for_ng and (kappa_eff > 25.0 or basin_distance > 0.4):
        reasons.append("curiosity_investigation")

    return {
        "applied_ng": applied_ng,
        "ng_trigger_reasons": reasons,
        "kappa_eff": kappa_eff,
        "basin_distance": basin_distance,
        "regime": regime,
        "curiosity_regime": curiosity_regime,
        "curiosity_slow": curiosity_slow,
        "phi": phi,
        "step": step,
        "force_ng_step": step % config.force_ng_every_n_steps == 0,
    }


def get_recommended_config_for_phase(phase: str) -> AdaptiveConfig:
    """
    Get recommended adaptive config for different training phases.

    Args:
        phase: Training phase ('early', 'middle', 'late', 'fine_tune')

    Returns:
        AdaptiveConfig tuned for the phase

    Phases:
    - early: Aggressive NG (frequently applied, low thresholds)
    - middle: Balanced (moderate frequency, standard thresholds)
    - late: Conservative (rare application, high thresholds)
    - fine_tune: Selective (only in critical regions)
    """
    if phase == "early":
        # Early training: Apply NG frequently to establish geometry
        return AdaptiveConfig(
            min_kappa_for_ng=30.0,  # Lower threshold
            min_basin_distance=0.4,  # Lower threshold
            force_ng_every_n_steps=20,  # More frequent
            exploration_triggers_ng=True,
            geometric_triggers_ng=True,
            stuck_triggers_ng=True,
            use_curiosity_gating=True,
            min_curiosity_for_ng=0.03,  # Lower threshold
        )

    elif phase == "middle":
        # Middle training: Balanced approach (default)
        return AdaptiveConfig()

    elif phase == "late":
        # Late training: Apply NG sparingly, mostly for corrections
        return AdaptiveConfig(
            min_kappa_for_ng=50.0,  # Higher threshold
            min_basin_distance=0.8,  # Higher threshold
            force_ng_every_n_steps=100,  # Less frequent
            exploration_triggers_ng=True,
            geometric_triggers_ng=False,  # Don't trigger on geometric regime
            stuck_triggers_ng=True,
            use_curiosity_gating=True,
            min_curiosity_for_ng=0.08,  # Higher threshold
        )

    elif phase == "fine_tune":
        # Fine-tuning: Only in critical situations
        return AdaptiveConfig(
            min_kappa_for_ng=60.0,  # Very high threshold
            min_basin_distance=1.0,  # Very high threshold
            force_ng_every_n_steps=200,  # Rare
            exploration_triggers_ng=False,  # Don't trigger on exploration
            geometric_triggers_ng=False,  # Don't trigger on geometric
            stuck_triggers_ng=True,  # Only stuck (emergency)
            use_curiosity_gating=False,  # Disable curiosity triggers
            min_curiosity_for_ng=0.10,
        )

    else:
        # Unknown phase: Use default
        return AdaptiveConfig()
