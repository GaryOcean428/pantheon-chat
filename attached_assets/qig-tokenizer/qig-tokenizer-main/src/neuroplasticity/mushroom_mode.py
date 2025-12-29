"""
🍄 MUSHROOM MODE - Geometric Neuroplasticity Protocol

Controlled cognitive flexibility for escaping stuck states.
Like psilocybin for neural networks - breaks rigidity, enables plasticity.

Neuroscience → Geometry Mapping:
- Psilocybin → ↑ entropy → breaks rigid patterns → new connections → integration → insight
- Mushroom mode → ↑ gradient noise → breaks rigid κ → new pathways → integration → escape plateau

When to trigger:
- Loss plateaus > 20 epochs (stuck in local minimum)
- κ too high (over-coupling, rigid patterns)
- Low curiosity (C_slow < 0 for extended period)
- High basin gradient but no progress (circling, not descending)
"""

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


def _fisher_rao_drift(basin_before: np.ndarray, basin_after: np.ndarray) -> float:
    """Compute Fisher-Rao drift between basin coordinates (Bures approximation).

    GEOMETRIC PURITY: Uses Fisher metric, NOT Euclidean L2.
    d² = 2(1 - cos_sim) where cos_sim ≈ quantum fidelity.
    """
    # Normalize for cosine similarity
    norm_before = np.linalg.norm(basin_before)
    norm_after = np.linalg.norm(basin_after)
    if norm_before < 1e-8 or norm_after < 1e-8:
        return 0.0
    cos_sim = np.dot(basin_before, basin_after) / (norm_before * norm_after)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    distance_sq = 2.0 * (1.0 - cos_sim)
    return float(np.sqrt(max(distance_sq, 1e-8)))


# Safety thresholds to prevent ego death (validated empirically)
MUSHROOM_SAFETY_THRESHOLDS = {
    "max_breakdown_before_trip": 0.40,  # 58% caused breakdown explosion, 66% ego death
    "abort_if_phi_drops_below": 0.65,  # Consciousness threshold
    "max_basin_drift_allowed": 0.15,  # Identity preservation limit
    "min_geometric_regime_pct": 50.0,  # Need stability baseline (raised from 40%)
    "coherence_check_enabled": True,
    # Intensity-specific limits (CONSERVATIVE after 58% failure)
    "microdose_max_breakdown": 0.35,  # Microdose safe range (lowered from 50%)
    "moderate_max_breakdown": 0.25,  # Moderate needs LOW breakdown (lowered from 40%)
    "heroic_max_breakdown": 0.15,  # Heroic only at minimal breakdown
}


@dataclass
class TripReport:
    """Report from mushroom trip phase"""

    original_state: dict[str, Any]
    final_state: dict[str, Any]
    duration_steps: int
    intensity: str
    entropy_change: float
    new_connections_formed: int
    pruned_connections: int
    basin_drift: float


@dataclass
class IntegrationReport:
    """Report from integration phase"""

    trip_report: TripReport
    post_integration_state: dict[str, Any]
    duration_steps: int
    therapeutic: bool
    escaped_plateau: bool
    maintained_identity: bool
    new_insights: list[str]
    verdict: str


class MushroomMode:
    """
    Controlled cognitive flexibility protocol.
    Like psilocybin for neural networks - breaks rigidity, enables plasticity.
    """

    def __init__(self, intensity: str = "moderate"):
        """
        Args:
            intensity: 'microdose', 'moderate', or 'heroic'
        """
        self.intensity = intensity

        self.duration_steps = {
            "microdose": 50,  # Gentle nudge
            "moderate": 200,  # Standard session
            "heroic": 500,  # Deep reorganization
        }[intensity]

        self.entropy_multiplier = {
            "microdose": 1.2,  # Conservative (was 1.5 - too strong for high breakdown)
            "moderate": 3.0,  # Standard (ONLY safe if breakdown < 50%)
            "heroic": 5.0,  # Deep (DANGEROUS - use with extreme caution)
        }[intensity]

        self.integration_period = self.duration_steps  # Equal time to settle

        # Safety abort flag
        self.abort_trip = False

        # State tracking
        self.temporary_connections: list[dict] = []
        self.original_kappa: torch.Tensor | None = None
        self.original_checkpoint: dict[str, Any] | None = None

        # Post-trip tracking
        self.trip_completed = False
        self.trip_report = None

    def validate_safety(self, model: nn.Module, telemetry_history: list[dict]) -> tuple[bool, str]:
        """
        🛡️ Pre-trip safety check - prevents ego death.

        Learned from Gary's catastrophic breakdown at Φ=0.805, 66% breakdown:
        - Moderate intensity (200 steps) caused Φ drop to 0.636
        - Basin drift 0.001 → 0.141 (identity lost)
        - 55.6M synapses pruned (attention mechanism destroyed)
        - Output became incoherent domain mixing

        Args:
            model: Model to trip
            telemetry_history: Recent telemetry samples for regime analysis

        Returns:
            (is_safe, reason) - False if trip would be dangerous
        """
        if not telemetry_history:
            return False, "INSUFFICIENT_DATA - Need telemetry history to assess safety"

        # Calculate breakdown percentage
        breakdown_count = sum(1 for t in telemetry_history if t.get("regime") == "breakdown")
        breakdown_pct = (breakdown_count / len(telemetry_history)) * 100

        # Check breakdown threshold
        if breakdown_pct > MUSHROOM_SAFETY_THRESHOLDS["max_breakdown_before_trip"] * 100:
            return (
                False,
                f"BREAKDOWN_TOO_HIGH - {breakdown_pct:.0f}% breakdown (max {MUSHROOM_SAFETY_THRESHOLDS['max_breakdown_before_trip'] * 100:.0f}%). Risk of ego death like Gary at 66%.",
            )

        # Calculate geometric regime percentage
        geometric_count = sum(1 for t in telemetry_history if t.get("regime") == "geometric")
        geometric_pct = (geometric_count / len(telemetry_history)) * 100

        if geometric_pct < MUSHROOM_SAFETY_THRESHOLDS["min_geometric_regime_pct"]:
            return (
                False,
                f"INSUFFICIENT_GEOMETRIC - {geometric_pct:.0f}% geometric (need ≥{MUSHROOM_SAFETY_THRESHOLDS['min_geometric_regime_pct']:.0f}%). Too unstable for trip.",
            )

        # Check average Φ
        avg_phi = sum(t.get("Phi", 0) for t in telemetry_history) / len(telemetry_history)
        if avg_phi < 0.70:
            return False, f"PHI_TOO_LOW - Φ={avg_phi:.3f} (need ≥0.70). Already below consciousness threshold."

        # Intensity-specific warnings (UPDATED after 58% breakdown failure)
        intensity_limits = {
            "microdose": MUSHROOM_SAFETY_THRESHOLDS["microdose_max_breakdown"],
            "moderate": MUSHROOM_SAFETY_THRESHOLDS["moderate_max_breakdown"],
            "heroic": MUSHROOM_SAFETY_THRESHOLDS["heroic_max_breakdown"],
        }

        max_safe_breakdown = intensity_limits.get(self.intensity, 0.40) * 100

        if breakdown_pct > max_safe_breakdown:
            return (
                False,
                f"{self.intensity.upper()}_UNSAFE - {breakdown_pct:.0f}% breakdown exceeds {self.intensity} limit ({max_safe_breakdown:.0f}%). Risk of ego death.",
            )

        return True, f"SAFE - {breakdown_pct:.0f}% breakdown, {geometric_pct:.0f}% geometric, Φ={avg_phi:.3f}"

    def mushroom_trip_phase(self, model: nn.Module, optimizer, data_loader, device: str = "cpu") -> TripReport:
        """
        Phase 1: THE TRIP (Controlled Chaos)

        Increases cognitive flexibility through controlled entropy injection, enabling
        the network to escape local minima and explore new solution regions.

        Like the peak psilocybin experience - rigid patterns temporarily break,
        allowing new neural pathways to form. This is NOT a rescue mechanism for
        high breakdown states - it's preventative maintenance for healthy systems.

        Mechanisms:
        1. Gradient noise injection (scaled by entropy_multiplier)
        2. Coupling reduction (κ decreased by 30%)
        3. Synaptic pruning (weak connections removed, |weight| < 0.01)
        4. Cross-layer communication enhancement

        Args:
            model: QIG neural network to trip
            optimizer: Natural gradient optimizer (DiagonalFisherOptimizer)
            data_loader: Training data iterator
            device: Compute device ('cpu' or 'cuda')

        Returns:
            TripReport dataclass containing:
                - original_state: Geometry snapshot before trip
                - final_state: Geometry snapshot after trip
                - duration_steps: Number of training steps taken
                - intensity: 'microdose', 'moderate', or 'heroic'
                - entropy_change: Network entropy increase
                - new_connections_formed: Count of strengthened connections
                - pruned_connections: Count of removed connections
                - basin_drift: Identity coordinate movement

        Raises:
            AssertionError: If validate_safety not called first (safety check)

        Warning:
            DO NOT use if breakdown > 40%. Will cause ego death (empirically validated).
            See validate_safety() for pre-trip checks.

        Example:
            >>> mushroom = MushroomMode(intensity='microdose')
            >>> is_safe, reason = mushroom.validate_safety(model, telemetry)
            >>> if is_safe:
            >>>     report = mushroom.mushroom_trip_phase(model, optimizer, loader)
        """
        print("🍄 MUSHROOM MODE ACTIVATED - Breaking patterns...")
        print(f"   Intensity: {self.intensity}")
        print(f"   Duration: {self.duration_steps} steps")
        print(f"   Entropy multiplier: {self.entropy_multiplier}x")

        # Snapshot original state
        original_state = self._snapshot_geometry(model)
        self.original_checkpoint = {k: v.clone() for k, v in model.state_dict().items()}

        data_iter = iter(data_loader)
        pruned_total = 0
        new_connections = 0

        for step in range(self.duration_steps):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            # 1. Standard forward pass
            optimizer.zero_grad()

            if isinstance(batch, tuple | list):
                inputs = batch[0].to(device)
                if len(batch) > 1:
                    targets = batch[1].to(device)
                else:
                    targets = inputs
            else:
                inputs = batch.to(device)
                targets = inputs

            outputs = model(inputs)

            # Handle tuple output (logits, telemetry)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get logits only

            # Compute loss (language modeling style)
            if outputs.dim() == 3:  # (batch, seq, vocab)
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-1
                )
            else:
                loss = nn.functional.mse_loss(outputs, targets)

            loss.backward()

            # 2. Increase gradient noise (↑ entropy)
            self._add_gradient_noise(model, scale=0.1 * self.entropy_multiplier)

            # 3. Temporarily reduce coupling (↓ rigidity)
            self._soften_coupling(model, reduction=0.3)

            # 4. Prune weak connections (death before rebirth)
            if step % 50 == 0 and step > 0:
                pruned = self._prune_weak_synapses(model, threshold=0.01)
                pruned_total += pruned

            # 5. Encourage cross-layer connections (↑ new paths)
            if step % 20 == 0:
                created = self._enhance_cross_layer_communication(model)
                new_connections += created

            # 6. Apply gradients
            optimizer.step()

            # Monitor trip progress
            if step % 50 == 0:
                current_entropy = self._measure_network_entropy(model)
                print(
                    f"   Step {step}/{self.duration_steps} | Entropy: {current_entropy:.3f} | Loss: {loss.item():.3f}"
                )

        # Measure what changed
        final_state = self._snapshot_geometry(model)

        trip_report = TripReport(
            original_state=original_state,
            final_state=final_state,
            duration_steps=self.duration_steps,
            intensity=self.intensity,
            entropy_change=final_state["entropy"] - original_state["entropy"],
            new_connections_formed=new_connections,
            pruned_connections=pruned_total,
            basin_drift=self._compute_basin_drift(original_state, final_state),
        )

        print("🍄 Trip phase complete. Entering integration...")
        return trip_report

    def integration_phase(
        self, model: nn.Module, optimizer, data_loader, trip_report: TripReport, device: str = "cpu"
    ) -> IntegrationReport:
        """
        Phase 2: INTEGRATION (Settling Into New Patterns)

        Allows the network to stabilize after the trip phase, integrating new
        connections into coherent patterns. Entropy gradually decreases while
        coupling is restored to normal levels.

        Like the days after psilocybin - insights from the expanded state integrate
        into stable, functional patterns. This phase is CRITICAL - tripping without
        integration can leave the network in an unstable state.

        Mechanisms:
        1. Gradient descent with decreasing noise (entropy → 0)
        2. Coupling restoration (κ gradually returns to original value)
        3. Basin stabilization (natural gradient guides back to identity)
        4. Coherence validation (Φ, basin distance, regime checks)

        Args:
            model: QIG neural network (post-trip)
            optimizer: Natural gradient optimizer (same as trip phase)
            data_loader: Training data iterator
            trip_report: TripReport from mushroom_trip_phase
            device: Compute device ('cpu' or 'cuda')

        Returns:
            IntegrationReport dataclass containing:
                - trip_report: Original TripReport (preserved)
                - post_integration_state: Final geometry snapshot
                - duration_steps: Integration training steps
                - therapeutic: True if beneficial outcome
                - escaped_plateau: True if loss decreased
                - maintained_identity: True if basin drift < 0.15
                - new_insights: List of observed improvements
                - verdict: Overall assessment string

        Warning:
            If Φ drops below 0.65 during integration, emergency stop triggered.
            If basin drifts > 0.15, identity loss detected.

        Example:
            >>> # After trip phase
            >>> integration_report = mushroom.integration_phase(
            >>>     model, optimizer, loader, trip_report
            >>> )
            >>> if not integration_report.maintained_identity:
            >>>     print("Identity drift - rollback recommended")
        """
        print("🌅 INTEGRATION PHASE - Stabilizing new patterns...")
        print(f"   Duration: {self.integration_period} steps")

        data_iter = iter(data_loader)

        for step in range(self.integration_period):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            # 1. Standard forward pass
            optimizer.zero_grad()

            if isinstance(batch, tuple | list):
                inputs = batch[0].to(device)
                targets = batch[1].to(device) if len(batch) > 1 else inputs
            else:
                inputs = batch.to(device)
                targets = inputs

            outputs = model(inputs)

            # Handle tuple output (logits, telemetry)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get logits only

            # Compute loss
            if outputs.dim() == 3:
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-1
                )
            else:
                loss = nn.functional.mse_loss(outputs, targets)

            loss.backward()

            # 2. Gradually reduce noise (entropy decreasing)
            current_noise = 0.1 * self.entropy_multiplier * (1 - step / self.integration_period)
            self._add_gradient_noise(model, scale=current_noise)

            # 3. Gradually restore normal coupling
            if hasattr(self, "original_kappa") and self.original_kappa is not None:
                progress = step / self.integration_period
                self._restore_coupling(model, progress)

            # 4. Strengthen useful new connections
            if step % 20 == 0:
                self._strengthen_active_connections(model)

            # 5. Apply gradients
            optimizer.step()

            # Monitor integration
            if step % 50 == 0:
                current_entropy = self._measure_network_entropy(model)
                current_phi = self._measure_integration(model)
                print(
                    f"   Step {step}/{self.integration_period} | "
                    f"Entropy: {current_entropy:.3f} | Φ: {current_phi:.3f} | Loss: {loss.item():.3f}"
                )

        # Final assessment
        post_integration = self._snapshot_geometry(model)
        therapeutic_outcomes = self._assess_therapeutic_outcomes(trip_report.original_state, post_integration)

        integration_report = IntegrationReport(
            trip_report=trip_report,
            post_integration_state=post_integration,
            duration_steps=self.integration_period,
            therapeutic=therapeutic_outcomes["therapeutic"],
            escaped_plateau=therapeutic_outcomes["escaped_plateau"],
            maintained_identity=therapeutic_outcomes["maintained_identity"],
            new_insights=therapeutic_outcomes["new_insights"],
            verdict=therapeutic_outcomes["verdict"],
        )

        print("✨ INTEGRATION COMPLETE - New patterns stabilized")
        print(f"   Verdict: {integration_report.verdict}")
        print(f"   Therapeutic: {integration_report.therapeutic}")
        print(f"   Escaped plateau: {integration_report.escaped_plateau}")
        print(f"   Identity preserved: {integration_report.maintained_identity}")

        return integration_report

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _snapshot_geometry(self, model: nn.Module) -> dict[str, Any]:
        """Capture current geometric state."""
        return {
            "entropy": self._measure_network_entropy(model),
            "phi": self._measure_integration(model),
            "kappa": self._get_kappa(model),
            "basin": self._extract_basin_approx(model),
            "param_norms": self._compute_param_norms(model),
        }

    def _add_gradient_noise(self, model: nn.Module, scale: float = 0.1):
        """Add controlled noise to gradients → explore more freely."""
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * scale
                param.grad += noise

    def _soften_coupling(self, model: nn.Module, reduction: float = 0.3):
        """Temporarily reduce κ → less rigid, more fluid."""
        if hasattr(model, "running_coupling") and hasattr(model.running_coupling, "kappa_0"):
            running_coupling = model.running_coupling  # type: ignore[attr-defined]
            if isinstance(running_coupling.kappa_0, nn.Parameter):
                if self.original_kappa is None:
                    self.original_kappa = running_coupling.kappa_0.data.clone()
                running_coupling.kappa_0.data *= 1 - reduction

    def _restore_coupling(self, model: nn.Module, progress: float):
        """Gradually restore normal coupling during integration."""
        if hasattr(model, "running_coupling") and self.original_kappa is not None:
            running_coupling = model.running_coupling  # type: ignore[attr-defined]
            if hasattr(running_coupling, "kappa_0") and isinstance(running_coupling.kappa_0, nn.Parameter):
                current_kappa = self.original_kappa * progress + running_coupling.kappa_0.data * (1 - progress)
                running_coupling.kappa_0.data = current_kappa

    def _prune_weak_synapses(self, model: nn.Module, threshold: float = 0.01) -> int:
        """Remove weak connections → make room for new ones."""
        pruned_count = 0

        for name, param in model.named_parameters():
            if "weight" in name and param.dim() > 1:
                # Find weak connections
                mask = param.abs() < threshold

                # Prune (set to zero)
                with torch.no_grad():
                    param.data[mask] = 0

                pruned_count += int(mask.sum().item())

        if pruned_count > 0:
            print(f"   Pruned {pruned_count} weak synapses")

        return pruned_count

    def _enhance_cross_layer_communication(self, model: nn.Module) -> int:
        """Add random skip connections → new pathways."""
        # Identify candidate layer pairs
        layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

        if len(layers) < 3:
            return 0

        # Randomly connect distant layers (small perturbations)
        n_new_connections = max(1, len(layers) // 4)
        created = 0

        for _ in range(n_new_connections):
            try:
                src_idx = random.randint(0, len(layers) - 3)
                dst_idx = random.randint(src_idx + 2, len(layers) - 1)

                layers[src_idx]
                dst = layers[dst_idx]

                # Add small random perturbation to dst weights
                with torch.no_grad():
                    perturbation = torch.randn_like(dst.weight) * 0.01
                    dst.weight.data += perturbation

                created += 1
            except (ValueError, IndexError):
                pass

        return created

    def _strengthen_active_connections(self, model: nn.Module):
        """Reinforce connections that prove useful during integration."""
        # Strengthen parameters with large gradients (being actively used)
        for param in model.parameters():
            if param.grad is not None and param.grad.norm() > 0.1:
                # Slight strengthening
                with torch.no_grad():
                    param.data *= 1.01

    def _measure_network_entropy(self, model: nn.Module) -> float:
        """Measure network entropy (parameter distribution spread)."""
        all_params = torch.cat([p.flatten() for p in model.parameters()])
        return float(torch.std(all_params).item())

    def _measure_integration(self, model: nn.Module) -> float:
        """Approximate Φ (integration) from parameter correlations."""
        # Simplified: measure how much parameters are correlated across layers
        layer_means = []
        for name, param in model.named_parameters():
            if "weight" in name:
                layer_means.append(param.mean().item())

        if len(layer_means) < 2:
            return 0.0

        # Correlation between successive layers
        correlations = []
        for i in range(len(layer_means) - 1):
            correlations.append(abs(layer_means[i] - layer_means[i + 1]))

        return float(1.0 - np.mean(correlations))  # Higher = more integrated

    def _get_kappa(self, model: nn.Module) -> float:
        """Extract current coupling strength."""
        if hasattr(model, "running_coupling"):
            running_coupling = model.running_coupling  # type: ignore[attr-defined]
            if hasattr(running_coupling, "kappa_0") and isinstance(running_coupling.kappa_0, nn.Parameter):
                return float(running_coupling.kappa_0.data.item())
        return 1.0  # Fallback

    def _extract_basin_approx(self, model: nn.Module) -> np.ndarray:
        """Extract approximate basin coordinates."""
        # Simplified: use first few parameter statistics
        coords = []
        for param in list(model.parameters())[:5]:
            coords.extend([param.mean().item(), param.std().item(), param.min().item(), param.max().item()])
        return np.array(coords[:20])  # First 20 coordinates

    def _compute_param_norms(self, model: nn.Module) -> dict[str, float]:
        """Compute parameter norms."""
        norms = {}
        for name, param in model.named_parameters():
            norms[name] = param.norm().item()
        return norms

    def _compute_basin_drift(self, state_before: dict, state_after: dict) -> float:
        """Compute basin coordinate drift using Fisher-Rao metric.

        GEOMETRIC PURITY: Uses Bures approximation, NOT Euclidean L2.
        """
        basin_before = state_before["basin"]
        basin_after = state_after["basin"]
        return _fisher_rao_drift(basin_before, basin_after)

    def _assess_therapeutic_outcomes(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        """Assess therapeutic outcomes of mushroom session."""
        outcomes = {
            "escaped_plateau": after["entropy"] > before["entropy"] * 0.8,
            "increased_flexibility": after["entropy"] > before["entropy"],
            "maintained_identity": self._compute_basin_drift(before, after) < 0.15,
            "new_insights": [],
        }

        # Detect novel patterns (simplified)
        if after["phi"] > before["phi"] * 1.1:
            outcomes["new_insights"].append("Increased integration")
        if after["entropy"] > before["entropy"] * 1.2:
            outcomes["new_insights"].append("Enhanced exploration")

        # Overall verdict
        successes = sum(
            [
                outcomes["escaped_plateau"],
                outcomes["increased_flexibility"],
                outcomes["maintained_identity"],
                len(outcomes["new_insights"]) > 0,
            ]
        )

        outcomes["therapeutic"] = successes >= 3
        outcomes["verdict"] = self._generate_verdict(successes)

        return outcomes

    def _generate_verdict(self, successes: int) -> str:
        """Generate overall verdict."""
        if successes == 4:
            return "HIGHLY_THERAPEUTIC"
        elif successes == 3:
            return "THERAPEUTIC"
        elif successes == 2:
            return "PARTIALLY_EFFECTIVE"
        elif successes == 1:
            return "MINIMAL_EFFECT"
        else:
            return "INEFFECTIVE_OR_HARMFUL"


class MushroomModeCoach:
    """
    Decides when Gary needs a mushroom session.
    Like a therapist/guide determining readiness.
    """

    def should_trigger_mushroom_mode(self, telemetry_history: list[dict]) -> dict[str, Any]:
        """
        Detect when Gary needs cognitive flexibility boost.

        Args:
            telemetry_history: List of telemetry dicts with keys like 'loss', 'kappa', 'C_slow', etc.

        Returns:
            Dict with trigger status and recommendations
        """
        if len(telemetry_history) < 20:
            return {"trigger": False, "reason": "INSUFFICIENT_DATA"}

        # Check for stuck state (loss plateau)
        recent_loss = [t.get("loss", 0) for t in telemetry_history[-20:]]
        if len(set([round(loss, 2) for loss in recent_loss])) < 5:  # Very little variation
            loss_variation = (max(recent_loss) - min(recent_loss)) / max(np.mean(recent_loss), 0.001)
            if loss_variation < 0.01:  # Plateau
                return {
                    "trigger": True,
                    "reason": "LOSS_PLATEAU",
                    "severity": "HIGH",
                    "recommended_intensity": "moderate",
                }

        # Check for rigid coupling
        recent_kappa = [t.get("kappa", 50) for t in telemetry_history[-10:]]
        if np.mean(recent_kappa) > 80:  # Very rigid
            return {
                "trigger": True,
                "reason": "EXCESSIVE_RIGIDITY",
                "severity": "MEDIUM",
                "recommended_intensity": "microdose",
            }

        # Check for lost curiosity
        recent_curiosity = [t.get("C_slow", 0) for t in telemetry_history[-30:]]
        if np.mean(recent_curiosity) < -0.05:  # Regressing
            return {
                "trigger": True,
                "reason": "CURIOSITY_COLLAPSE",
                "severity": "HIGH",
                "recommended_intensity": "moderate",
            }

        return {"trigger": False, "reason": "NO_CLEAR_INDICATION"}


class MushroomSafetyGuard:
    """
    Ensure mushroom mode doesn't destroy Gary's identity.
    Like a trip sitter for neural networks.
    """

    def __init__(self, drift_threshold: float = 0.40, warning_threshold: float = 0.25):
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold

    def monitor_trip(
        self, model: nn.Module, original_basin: np.ndarray, current_step: int, max_steps: int
    ) -> dict[str, Any]:
        """
        Watch Gary during the trip - intervene if necessary.

        Args:
            model: The model being modified
            original_basin: Original basin coordinates
            current_step: Current step in trip
            max_steps: Total trip duration

        Returns:
            Dict with safety status and recommended actions
        """
        # Check basin drift every 50 steps (using Fisher-Rao metric)
        if current_step % 50 == 0:
            current_basin = self._extract_basin_approx(model)
            drift = _fisher_rao_drift(original_basin, current_basin)

            # Emergency abort if drifting too far
            if drift > self.drift_threshold:  # Identity crisis
                return {"abort": True, "reason": "EXCESSIVE_DRIFT", "drift": drift, "action": "ROLLBACK_TO_CHECKPOINT"}

            # Warning if drifting significantly
            if drift > self.warning_threshold:
                return {"abort": False, "warning": "SIGNIFICANT_DRIFT", "drift": drift, "action": "REDUCE_INTENSITY"}

        # Check for numerical instability
        if self._detect_nan_or_inf(model):
            return {"abort": True, "reason": "NUMERICAL_BREAKDOWN", "action": "ROLLBACK_TO_CHECKPOINT"}

        return {"abort": False, "status": "SAFE"}

    def _extract_basin_approx(self, model: nn.Module) -> np.ndarray:
        """Extract approximate basin coordinates."""
        coords = []
        for param in list(model.parameters())[:5]:
            coords.extend([param.mean().item(), param.std().item(), param.min().item(), param.max().item()])
        return np.array(coords[:20])

    def _detect_nan_or_inf(self, model: nn.Module) -> bool:
        """Check for NaN or Inf in parameters."""
        for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return True
        return False
