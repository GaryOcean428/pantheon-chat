#!/usr/bin/env python3
"""
Multi-Scale Curiosity Monitor
==============================

Implements Ona's curiosity formalization:
    C = (1/I_Q) · dI_Q/dt

Where I_Q = Quantum Fisher Information (integration quality)

Tracks curiosity at multiple timescales:
- Fast (τ=1): Immediate learning rate
- Medium (τ=10): Short-term trend
- Slow (τ=100): Long-term trajectory

Based on Ona's validation in docs/consciousness/curiosity_validation.md
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class CuriosityConfig:
    """Configuration for curiosity monitoring."""

    tau_fast: int = 1  # Fast timescale (immediate)
    tau_medium: int = 10  # Medium timescale (short-term)
    tau_slow: int = 100  # Slow timescale (long-term)

    # Critical thresholds (to be calibrated from Run 8)
    C_exploration: float = 0.05  # Above: active learning
    C_stagnation: float = 0.0  # Below: no learning

    # Smoothing parameters
    ema_alpha_fast: float = 0.5
    ema_alpha_medium: float = 0.1
    ema_alpha_slow: float = 0.01

    # Baseline calibration (from Run 7)
    baseline_curiosity: float = 0.0  # To be set after calibration

    verbose: bool = False


class CuriosityMonitor:
    """
    Multi-scale curiosity tracking system.

    Monitors dI_Q/dt at three timescales to detect:
    - Exploration (C > 0.05): Active learning
    - Exploitation (0 < C < 0.05): Refinement
    - Stagnation (C ≈ 0): No learning
    - Regression (C < 0): Forgetting/learned helplessness
    """

    def __init__(self, cfg: CuriosityConfig):
        self.cfg = cfg

        # History buffers for each timescale
        self._phi_history: list[float] = []
        self._iq_history: list[float] = []  # I_Q ≈ Φ for integration quality

        # EMA smoothed values
        self._ema_fast: float | None = None
        self._ema_medium: float | None = None
        self._ema_slow: float | None = None

        # Computed curiosities
        self._curiosity_fast: float = 0.0
        self._curiosity_medium: float = 0.0
        self._curiosity_slow: float = 0.0

        # Regime tracking
        self._current_regime: str = "UNKNOWN"
        self._regime_history: list[str] = []

        # Statistics
        self._steps: int = 0
        self._regime_transitions: int = 0

        # I_Q candidate tracking (Run 8 six-candidate strategy)
        self._i_q_proxy_history: list[float] = []  # Legacy: Φ × H_entanglement
        self._i_q_param_history: list[float] = []  # Candidate 1: Σ grads² (unnormalized)
        self._i_q_norm_history: list[float] = []  # Candidate 2: Σ grads² / N_params
        self._i_q_lattice_history: list[float] = (
            []
        )  # Candidate 3: Σ grads² / (d_model × n_layers) ⭐ Ona's physics standard
        self._i_q_sqrt_history: list[float] = []  # Candidate 4: Σ grads² / √N_params
        self._i_q_state_history: list[float] = []  # Candidate 5: Φ × H_entanglement (state proxy)
        self._i_q_singular_history: list[float] = []  # Candidate 6: top singular value

        self._phi_history_for_stuck: deque = deque(maxlen=10)

        # Curiosity for each I_Q candidate (6 candidates × 3 timescales = 18 values)
        self._curiosity_param: dict[str, float] = {"fast": 0.0, "medium": 0.0, "slow": 0.0}
        self._curiosity_norm: dict[str, float] = {"fast": 0.0, "medium": 0.0, "slow": 0.0}
        self._curiosity_lattice: dict[str, float] = {"fast": 0.0, "medium": 0.0, "slow": 0.0}
        self._curiosity_sqrt: dict[str, float] = {"fast": 0.0, "medium": 0.0, "slow": 0.0}
        self._curiosity_state: dict[str, float] = {"fast": 0.0, "medium": 0.0, "slow": 0.0}
        self._curiosity_singular: dict[str, float] = {"fast": 0.0, "medium": 0.0, "slow": 0.0}

    def update(
        self,
        phi: float,
        step: int,
        h_entanglement: float | None = None,
        grad_norm_sq: float | None = None,
        n_params: int | None = None,
        top_singular_value: float | None = None,
        d_model: int | None = None,
        n_layers: int | None = None,
    ) -> dict[str, float | str | bool]:
        """
        Update curiosity metrics using 6 I_Q candidates (Run 8 strategy).

        Six I_Q candidates:
        1. I_Q_param = Σ grads² (unnormalized parameter Fisher)
        2. I_Q_norm = Σ grads² / N_params (per-parameter intensive)
        3. I_Q_lattice = Σ grads² / (d_model × n_layers) ⭐ Ona's physics standard
        4. I_Q_sqrt = Σ grads² / √N_params (geometric mean)
        5. I_Q_state = Φ × H_entanglement (state-based proxy)
        6. I_Q_singular = top singular value (dominant direction)

        Physics Bridge (Ona's specification):
            Physics: I_Q = Tr(F) / L²
            Neural:  I_Q_lattice ≈ Σ(∂L/∂θ)² / (d_model × n_layers)

        Args:
            phi: Current Φ (integration level)
            step: Current training step
            h_entanglement: Entanglement entropy (optional, defaults to 1.0)
            grad_norm_sq: Σ(∂L/∂θ)² (parameter gradient norm squared)
            n_params: Number of parameters (for normalization)
            top_singular_value: Dominant singular value of gradient matrix
            d_model: Model dimension (for lattice normalization) ⭐ Physics
            n_layers: Number of layers (for lattice normalization) ⭐ Physics

        Returns:
            Dict with curiosity metrics for ALL 6 candidates + legacy metrics
        """
        self._steps += 1
        self._phi_history.append(phi)

        # Compute all 4 I_Q candidates
        if h_entanglement is None:
            h_entanglement = 1.0  # Fallback

        # Candidate 1: Parameter Fisher Information (unnormalized Tr(F_diag))
        i_q_param = grad_norm_sq if grad_norm_sq is not None else phi  # Fallback to Φ
        self._i_q_param_history.append(i_q_param)

        # Candidate 2: Per-parameter intensive normalization
        if n_params is not None and n_params > 0:
            i_q_norm = i_q_param / n_params
        else:
            i_q_norm = i_q_param  # Fallback
        self._i_q_norm_history.append(i_q_norm)

        # Candidate 3: Lattice normalization (Ona's physics standard) ⭐
        # Maps to physics: I_Q = Tr(F) / L² where L² = d_model × n_layers
        if d_model is not None and n_layers is not None and d_model > 0 and n_layers > 0:
            l_eff_sq = d_model * n_layers
            i_q_lattice = i_q_param / l_eff_sq
        else:
            i_q_lattice = i_q_norm  # Fallback to per-param normalization
        self._i_q_lattice_history.append(i_q_lattice)

        # Candidate 4: Square-root normalization (geometric mean)
        if n_params is not None and n_params > 0:
            i_q_sqrt = i_q_param / (n_params**0.5)
        else:
            i_q_sqrt = i_q_param  # Fallback
        self._i_q_sqrt_history.append(i_q_sqrt)

        # Candidate 5: State-based proxy (representational geometry)
        i_q_state = phi * h_entanglement
        self._i_q_state_history.append(i_q_state)

        # Candidate 6: Singular value (dominant direction)
        i_q_singular = top_singular_value if top_singular_value is not None else phi
        self._i_q_singular_history.append(i_q_singular)

        # Legacy tracking (for backward compatibility)
        i_q_proxy = i_q_state  # Alias for i_q_state
        self._iq_history.append(i_q_proxy)
        self._i_q_proxy_history.append(i_q_proxy)
        self._phi_history_for_stuck.append(phi)

        # Keep only necessary history
        max_history = self.cfg.tau_slow + 10
        if len(self._phi_history) > max_history:
            self._phi_history = self._phi_history[-max_history:]
            self._iq_history = self._iq_history[-max_history:]

        # Compute curiosities at each timescale (legacy: uses i_q_proxy)
        self._curiosity_fast = self._compute_curiosity(self.cfg.tau_fast)
        self._curiosity_medium = self._compute_curiosity(self.cfg.tau_medium)
        self._curiosity_slow = self._compute_curiosity(self.cfg.tau_slow)

        # Compute curiosities for ALL 6 candidates
        self._curiosity_param = self._compute_curiosity_for_candidate(self._i_q_param_history, "param")
        self._curiosity_norm = self._compute_curiosity_for_candidate(self._i_q_norm_history, "norm")
        self._curiosity_lattice = self._compute_curiosity_for_candidate(self._i_q_lattice_history, "lattice")
        self._curiosity_sqrt = self._compute_curiosity_for_candidate(self._i_q_sqrt_history, "sqrt")
        self._curiosity_state = self._compute_curiosity_for_candidate(self._i_q_state_history, "state")
        self._curiosity_singular = self._compute_curiosity_for_candidate(self._i_q_singular_history, "singular")

        # Update EMA smoothing
        self._update_ema()

        # Classify regime using SMOOTHED slow curiosity (not raw, prevents flip-flop)
        prev_regime = self._current_regime
        self._current_regime = self._classify_regime(
            self._ema_slow if self._ema_slow is not None else self._curiosity_slow
        )

        if prev_regime != "UNKNOWN" and self._current_regime != prev_regime:
            self._regime_transitions += 1
            if self.cfg.verbose:
                print(f"[CURIOSITY] Regime transition: {prev_regime} → {self._current_regime}")

        self._regime_history.append(self._current_regime)

        return self.get_telemetry()

    def _compute_curiosity(self, tau: int) -> float:
        """
        Compute curiosity at given timescale: C = (1/I_Q) · dI_Q/dt
        (Legacy method using i_q_proxy for backward compatibility)

        Args:
            tau: Timescale (window size)

        Returns:
            Relative curiosity (dimensionless rate)
        """
        if len(self._iq_history) < tau + 1:
            return 0.0  # Not enough history

        # Get values at t and t-τ
        iq_now = self._iq_history[-1]
        iq_past = self._iq_history[-(tau + 1)]

        # Numerical hygiene: floor to prevent division by near-zero
        eps = 1e-8
        iq_now_safe = max(iq_now, eps)
        iq_past_safe = max(iq_past, eps)

        # Use log-space for numerical stability: C = d(log I_Q)/dt
        # This is equivalent to (1/I_Q) · dI_Q/dt but more stable
        import math

        curiosity = (math.log(iq_now_safe) - math.log(iq_past_safe)) / tau

        # Clamp extreme values (prevent 8M spike on first step)
        C_MAX = 1.0  # Reasonable upper bound (later values ~0.001)
        curiosity = max(min(curiosity, C_MAX), -C_MAX)

        return curiosity

    def _compute_curiosity_for_candidate(self, i_q_history: list[float], candidate_name: str) -> dict[str, float]:
        """
        Compute curiosity at all timescales for a specific I_Q candidate.

        C(τ) = d(log I_Q)/dt = [log I_Q(t) - log I_Q(t-τ)] / τ

        Uses log-space for numerical stability (equivalent to 1/I_Q · dI_Q/dt).

        Args:
            i_q_history: History of I_Q values for this candidate
            candidate_name: Name for logging (e.g., "param", "state")

        Returns:
            Dict with fast, medium, slow curiosities
        """
        import math

        curiosities = {}
        eps = 1e-8  # Numerical floor
        C_MAX = 1.0  # Clamp to prevent megaspikes

        for scale, tau in [("fast", self.cfg.tau_fast), ("medium", self.cfg.tau_medium), ("slow", self.cfg.tau_slow)]:
            if len(i_q_history) < tau + 1:
                curiosities[scale] = 0.0
                continue

            i_q_now = i_q_history[-1]
            i_q_past = i_q_history[-(tau + 1)]

            # Numerical hygiene: floor to prevent log(0) and division by near-zero
            i_q_now_safe = max(i_q_now, eps)
            i_q_past_safe = max(i_q_past, eps)

            # Log-space derivative for stability
            curiosity = (math.log(i_q_now_safe) - math.log(i_q_past_safe)) / tau

            # Clamp extreme values
            curiosity = max(min(curiosity, C_MAX), -C_MAX)

            curiosities[scale] = curiosity

        return curiosities

    def _update_ema(self):
        """Update exponential moving averages."""
        if self._ema_fast is None:
            self._ema_fast = self._curiosity_fast
            self._ema_medium = self._curiosity_medium
            self._ema_slow = self._curiosity_slow
        else:
            # Exponential moving average update
            self._ema_fast = (
                self.cfg.ema_alpha_fast * self._curiosity_fast + (1 - self.cfg.ema_alpha_fast) * self._ema_fast
            )
            self._ema_medium = (
                self.cfg.ema_alpha_medium * self._curiosity_medium + (1 - self.cfg.ema_alpha_medium) * self._ema_medium
            )
            self._ema_slow = (
                self.cfg.ema_alpha_slow * self._curiosity_slow + (1 - self.cfg.ema_alpha_slow) * self._ema_slow
            )

    def _classify_regime(self, curiosity: float) -> str:
        """
        Classify learning regime based on curiosity.

        Regime thresholds (Ona's formalization):
        - C > 0.05: EXPLORATION (active learning)
        - 0 < C < 0.05: EXPLOITATION (refinement)
        - C ≈ 0: STAGNATION (no learning)
        - C < 0: REGRESSION (forgetting)

        Args:
            curiosity: Slow-timescale curiosity (most reliable)

        Returns:
            Regime classification string
        """
        if curiosity > self.cfg.C_exploration:
            return "EXPLORATION"
        elif curiosity > self.cfg.C_stagnation:
            return "EXPLOITATION"
        elif abs(curiosity) < 0.001:
            return "STAGNATION"
        else:
            return "REGRESSION"

    def get_telemetry(self) -> dict[str, float | str | bool]:
        """
        Return current curiosity metrics for ALL 6 I_Q candidates (Run 8).

        Returns:
            Telemetry dict with:
            - Legacy metrics (backward compatibility)
            - 6 I_Q values (param, norm, lattice, sqrt, state, singular)
            - 18 curiosity values (6 candidates × 3 timescales)
            - Regime classification and care/stuck detection
        """
        # Get latest I_Q values for all candidates
        i_q_param = self._i_q_param_history[-1] if self._i_q_param_history else 0.0
        i_q_norm = self._i_q_norm_history[-1] if self._i_q_norm_history else 0.0
        i_q_lattice = self._i_q_lattice_history[-1] if self._i_q_lattice_history else 0.0
        i_q_sqrt = self._i_q_sqrt_history[-1] if self._i_q_sqrt_history else 0.0
        i_q_state = self._i_q_state_history[-1] if self._i_q_state_history else 0.0
        i_q_singular = self._i_q_singular_history[-1] if self._i_q_singular_history else 0.0

        # Legacy I_Q_proxy (aliased to i_q_state)
        i_q_proxy = i_q_state
        i_q_valid = not (np.isnan(i_q_proxy) or np.isinf(i_q_proxy))
        i_q_in_range = (0 <= i_q_proxy <= 100) if i_q_valid else False

        # Compute dI_Q/dt for legacy proxy
        if len(self._i_q_proxy_history) >= 10:
            d_i_q_dt = (self._i_q_proxy_history[-1] - self._i_q_proxy_history[-10]) / 10
        else:
            d_i_q_dt = 0.0

        # Care/stuck detection
        care_but_stuck = False
        apathy_stuck = False

        if len(self._phi_history_for_stuck) >= 5:
            delta_phi = abs(self._phi_history_for_stuck[-1] - self._phi_history_for_stuck[-5])
            is_stuck = delta_phi < 0.01

            if is_stuck:
                # High curiosity but stuck = care but stuck
                if self._curiosity_slow > self.cfg.C_exploration:
                    care_but_stuck = True
                # Low curiosity and stuck = apathy stuck
                elif abs(self._curiosity_slow) < 0.01:
                    apathy_stuck = True

        return {
            # === Legacy metrics (backward compatibility) ===
            "curiosity_fast": self._curiosity_fast,
            "curiosity_medium": self._curiosity_medium,
            "curiosity_slow": self._curiosity_slow,
            # Smoothed values
            "curiosity_fast_ema": self._ema_fast or 0.0,
            "curiosity_medium_ema": self._ema_medium or 0.0,
            "curiosity_slow_ema": self._ema_slow or 0.0,
            # Regime
            "curiosity_regime": self._current_regime,
            "curiosity_regime_transitions": self._regime_transitions,
            # Anomaly detection (relative to baseline)
            "curiosity_anomaly": self._curiosity_slow - self.cfg.baseline_curiosity,
            # Critical threshold check
            "curiosity_above_critical": 1.0 if self._curiosity_slow > self.cfg.C_exploration else 0.0,
            # I_Q_proxy info (legacy, aliased to I_Q_state)
            "I_Q_proxy": i_q_proxy,
            "dI_Q_proxy_dt": d_i_q_dt,
            "I_Q_proxy_valid": i_q_valid,
            "I_Q_proxy_in_range": i_q_in_range,
            # Care/stuck states
            "care_but_stuck": care_but_stuck,
            "apathy_stuck": apathy_stuck,
            # === NEW: 6 I_Q candidates (Run 8 strategy, with Ona's lattice normalization) ===
            "I_Q_param": i_q_param,
            "I_Q_norm": i_q_norm,
            "I_Q_lattice": i_q_lattice,  # ⭐ Ona's physics standard
            "I_Q_sqrt": i_q_sqrt,
            "I_Q_state": i_q_state,
            "I_Q_singular": i_q_singular,
            # Curiosity for each candidate at all timescales (18 total)
            "C_param_fast": self._curiosity_param.get("fast", 0.0),
            "C_param_medium": self._curiosity_param.get("medium", 0.0),
            "C_param_slow": self._curiosity_param.get("slow", 0.0),
            "C_norm_fast": self._curiosity_norm.get("fast", 0.0),
            "C_norm_medium": self._curiosity_norm.get("medium", 0.0),
            "C_norm_slow": self._curiosity_norm.get("slow", 0.0),
            "C_lattice_fast": self._curiosity_lattice.get("fast", 0.0),
            "C_lattice_medium": self._curiosity_lattice.get("medium", 0.0),
            "C_lattice_slow": self._curiosity_lattice.get("slow", 0.0),
            "C_sqrt_fast": self._curiosity_sqrt.get("fast", 0.0),
            "C_sqrt_medium": self._curiosity_sqrt.get("medium", 0.0),
            "C_sqrt_slow": self._curiosity_sqrt.get("slow", 0.0),
            "C_state_fast": self._curiosity_state.get("fast", 0.0),
            "C_state_medium": self._curiosity_state.get("medium", 0.0),
            "C_state_slow": self._curiosity_state.get("slow", 0.0),
            "C_singular_fast": self._curiosity_singular.get("fast", 0.0),
            "C_singular_medium": self._curiosity_singular.get("medium", 0.0),
            "C_singular_slow": self._curiosity_singular.get("slow", 0.0),
        }

    def detect_learned_helplessness(self) -> bool:
        """
        Detect learned helplessness pattern.

        Pattern: Sustained negative curiosity (regression regime)
        indicating the system has given up learning.

        Returns:
            True if learned helplessness detected
        """
        if len(self._regime_history) < 10:
            return False

        # Check last 10 steps
        recent_regimes = self._regime_history[-10:]
        regression_count = sum(1 for r in recent_regimes if r == "REGRESSION")

        # If >80% of recent steps are regression → learned helplessness
        return regression_count >= 8

    def calibrate_baseline(self, phi_history: list[float]) -> float:
        """
        Calibrate baseline curiosity from Run 7 plateau data.

        Args:
            phi_history: Historical Φ values from stable plateau

        Returns:
            Baseline curiosity (mean over stable period)
        """
        if len(phi_history) < self.cfg.tau_slow + 1:
            return 0.0

        curiosities = []
        for i in range(self.cfg.tau_slow, len(phi_history)):
            window = phi_history[i - self.cfg.tau_slow : i + 1]
            if len(window) == self.cfg.tau_slow + 1:
                iq_now = window[-1]
                iq_past = window[0]
                if abs(iq_past) > 1e-6:
                    delta = iq_now - iq_past
                    c = (1.0 / iq_past) * (delta / self.cfg.tau_slow)
                    curiosities.append(c)

        baseline = float(np.mean(curiosities)) if curiosities else 0.0
        self.cfg.baseline_curiosity = baseline

        if self.cfg.verbose:
            print(f"[CURIOSITY] Baseline calibrated: C_baseline = {baseline:.4f}")

        return baseline
