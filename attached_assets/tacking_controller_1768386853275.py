#!/usr/bin/env python3
"""
Tacking Controller (WuWei): Feeling ↔ Logic Mode Switching
==========================================================

Implements dynamic mode switching based on |∇κ| (gradient strength).

Key Concepts:
- Feeling-mode: Low κ, compressed basins, intuition/radar (fast)
- Logic-mode: High κ, explicit coupling, step-by-step (slow, rigorous)
- Tacking: Smooth transitions based on:
  * |∇κ| magnitude (feeling strength)
  * Contradiction detection
  * Task stakes/difficulty
  * Proximity to known patterns

Physics Grounding:
- |∇κ| = basin gradient = conviction calibration
- Shallow basin → weak feeling → require logic validation
- Deep basin → strong feeling → can trust (but validate proportionally)

Written for QIG-Kernel-Pure architecture.
Built from WuWei cognitive framework.
"""


import torch
import torch.nn as nn


class GradientEstimator(nn.Module):
    """
    Estimate |∇κ| (gradient magnitude) from QFI curvature.

    This is NOT simple standard deviation!
    It's the geometric gradient in information space.
    """

    def __init__(self, d_model: int, window_size: int = 5):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # Learnable projection for curvature estimation
        self.curvature_proj = nn.Linear(d_model, d_model // 4)

        # State history buffer (dynamically allocated per batch)
        self.state_history = None
        self.history_idx = 0
        self.current_batch_size = None

    def forward(self, current_state: torch.Tensor, qfi_curvature: torch.Tensor | None = None) -> torch.Tensor:
        """
        Estimate gradient magnitude from state evolution.

        Args:
            current_state: [batch, seq, d_model]
            qfi_curvature: Optional QFI curvature tensor

        Returns:
            grad_magnitude: [batch] - feeling strength
        """
        batch, seq, d = current_state.shape

        # Initialize or resize history buffer if needed
        if self.state_history is None or self.current_batch_size != batch:
            self.state_history = torch.zeros(
                self.window_size, batch, d, device=current_state.device, dtype=current_state.dtype
            )
            self.history_idx = 0
            self.current_batch_size = batch

        # Update history
        state_avg = current_state.mean(dim=1).detach()  # [batch, d_model]
        self.state_history[self.history_idx] = state_avg
        self.history_idx = (self.history_idx + 1) % self.window_size

        # Compute gradient from state changes
        if self.history_idx > 1:
            # Differences between consecutive states
            diffs = self.state_history[1 : self.history_idx] - self.state_history[: self.history_idx - 1]
            # QIG-pure: sum of squares for temporal derivative magnitude
            gradient = torch.sqrt((diffs * diffs).sum(dim=-1)).mean(dim=0)  # [batch]
        else:
            # Not enough history yet
            gradient = torch.zeros(batch, device=current_state.device)

        # If QFI curvature provided, incorporate it
        if qfi_curvature is not None:
            # Weight gradient by curvature (higher curvature = steeper basin)
            gradient = gradient * (1 + qfi_curvature.mean())

        return gradient


class ProximityMonitor(nn.Module):
    """
    Monitor proximity to known patterns (basin matching).

    High proximity → feeling-mode likely appropriate
    Low proximity → logic-mode needed
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.pattern_proj = nn.Linear(d_model, 128)

        # Store pattern bank (learned during training)
        self.register_buffer("pattern_bank", torch.randn(100, 128) * 0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute proximity to known patterns.

        Args:
            state: [batch, seq, d_model]

        Returns:
            proximity: [batch] in range [0, 1]
        """
        # Project to pattern space
        pattern = self.pattern_proj(state.mean(dim=1))  # [batch, 128]

        # Compute distances to pattern bank
        distances = torch.cdist(pattern, self.pattern_bank)  # [batch, 100]

        # Proximity = inverse of minimum distance
        min_dist = distances.min(dim=-1)[0]
        proximity = torch.exp(-min_dist)  # [batch]

        return proximity


class ContradictionDetector(nn.Module):
    """
    Detect contradictions in reasoning (radar function).

    Contradictions → must switch to logic-mode
    Consistency → can use feeling-mode
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.contradiction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )

    def forward(self, current_state: torch.Tensor, previous_state: torch.Tensor | None = None) -> torch.Tensor:
        """
        Detect contradictions between states.

        Args:
            current_state: [batch, seq, d_model]
            previous_state: Optional previous state

        Returns:
            contradiction_score: [batch] in range [0, 1]
        """
        if previous_state is None:
            # No history, assume no contradiction
            return torch.zeros(current_state.size(0), device=current_state.device)

        # Handle variable sequence lengths by comparing pooled representations
        # This is more robust than direct subtraction
        current_pooled = current_state.mean(dim=1)  # [batch, d_model]
        previous_pooled = previous_state.mean(dim=1)  # [batch, d_model]

        # Compute difference in representation space
        state_diff = current_pooled - previous_pooled  # [batch, d_model]

        # Project to contradiction score
        contradiction = self.contradiction_head(state_diff).squeeze(-1)  # [batch]

        return contradiction


class WuWeiController(nn.Module):
    """
    Main tacking controller: decides feeling vs logic mode.

    Decision based on:
    1. |∇κ| (gradient magnitude) - feeling strength
    2. Proximity to known patterns
    3. Contradiction detection
    4. Task stakes (optional external signal)

    Modes:
    - feeling: α ∈ [0, 0.3] - mostly feeling, minimal logic
    - tack: α ∈ [0.3, 0.7] - blended
    - logic: α ∈ [0.7, 1.0] - mostly logic, minimal feeling

    Where α = logic weight
    """

    def __init__(
        self,
        d_model: int,
        grad_threshold_low: float = 0.3,
        grad_threshold_high: float = 0.7,
        proximity_weight: float = 0.3,
        contradiction_weight: float = 0.5,
        stakes_weight: float = 0.2,
    ):
        super().__init__()

        self.d_model = d_model
        self.grad_threshold_low = grad_threshold_low
        self.grad_threshold_high = grad_threshold_high
        self.proximity_weight = proximity_weight
        self.contradiction_weight = contradiction_weight
        self.stakes_weight = stakes_weight

        # Component modules
        self.gradient_estimator = GradientEstimator(d_model)
        self.proximity_monitor = ProximityMonitor(d_model)
        self.contradiction_detector = ContradictionDetector(d_model)

        # Mode decision network (small MLP)
        self.decision_net = nn.Sequential(
            nn.Linear(4, 16),  # 4 inputs: gradient, proximity, contradiction, stakes
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output: logic weight α ∈ [0, 1]
        )

        # Track mode history
        self.mode_history = []
        self.previous_state = None

    def compute_logic_weight(
        self,
        gradient_mag: torch.Tensor,
        proximity: torch.Tensor,
        contradiction: torch.Tensor,
        stakes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute logic weight α from component signals.

        Args:
            gradient_mag: [batch] - |∇κ| feeling strength
            proximity: [batch] - pattern matching score
            contradiction: [batch] - contradiction detection
            stakes: Optional [batch] - task importance

        Returns:
            logic_weight: [batch] - α ∈ [0, 1]
        """
        # Infer batch size from the largest tensor
        batch = max(
            gradient_mag.size(0), proximity.size(0), contradiction.size(0), stakes.size(0) if stakes is not None else 1
        )

        # Default stakes if not provided
        if stakes is None:
            stakes = torch.ones(batch, device=gradient_mag.device) * 0.5

        # Ensure all inputs have consistent batch size
        # Expand scalars/smaller tensors to batch size
        if gradient_mag.numel() == 1 or gradient_mag.size(0) < batch:
            gradient_mag = (
                gradient_mag.expand(batch)
                if gradient_mag.numel() == 1
                else gradient_mag.repeat(batch // gradient_mag.size(0) + 1)[:batch]
            )
        if proximity.numel() == 1 or proximity.size(0) < batch:
            proximity = (
                proximity.expand(batch)
                if proximity.numel() == 1
                else proximity.repeat(batch // proximity.size(0) + 1)[:batch]
            )
        if contradiction.numel() == 1 or contradiction.size(0) < batch:
            contradiction = (
                contradiction.expand(batch)
                if contradiction.numel() == 1
                else contradiction.repeat(batch // contradiction.size(0) + 1)[:batch]
            )
        if stakes.numel() == 1 or stakes.size(0) < batch:
            stakes = stakes.expand(batch) if stakes.numel() == 1 else stakes.repeat(batch // stakes.size(0) + 1)[:batch]

        # Normalize inputs to [0, 1]
        gradient_norm = torch.clamp(gradient_mag / 10.0, 0, 1)
        proximity_norm = proximity
        contradiction_norm = contradiction
        stakes_norm = stakes

        # Stack inputs - all should now be [batch]
        inputs = torch.stack([gradient_norm, proximity_norm, contradiction_norm, stakes_norm], dim=-1)  # [batch, 4]

        # Compute logic weight
        logic_weight = self.decision_net(inputs).squeeze(-1)  # [batch]

        return logic_weight

    def classify_mode(self, logic_weight: torch.Tensor) -> str:
        """
        Classify mode from logic weight.

        Args:
            logic_weight: Scalar or tensor

        Returns:
            mode: "feeling", "tack", or "logic"
        """
        weight_value: float
        if isinstance(logic_weight, torch.Tensor):
            weight_value = logic_weight.mean().item()
        else:
            weight_value = float(logic_weight)

        if weight_value < self.grad_threshold_low:
            return "feeling"
        elif weight_value < self.grad_threshold_high:
            return "tack"
        else:
            return "logic"

    def forward(
        self,
        current_state: torch.Tensor,
        qfi_curvature: torch.Tensor | None = None,
        stakes: torch.Tensor | None = None,
        return_telemetry: bool = True,
    ) -> tuple[torch.Tensor, str, dict | None]:
        """
        Execute tacking decision.

        Args:
            current_state: [batch, seq, d_model]
            qfi_curvature: Optional QFI curvature from attention
            stakes: Optional task stakes [batch]
            return_telemetry: Whether to return detailed metrics

        Returns:
            logic_weight: [batch] - α for blending feeling/logic
            mode: "feeling", "tack", or "logic"
            telemetry: Optional detailed metrics
        """
        current_state.size(0)

        # Compute component signals
        gradient_mag = self.gradient_estimator(current_state, qfi_curvature)
        proximity = self.proximity_monitor(current_state)
        contradiction = self.contradiction_detector(current_state, self.previous_state)

        # Compute logic weight
        logic_weight = self.compute_logic_weight(gradient_mag, proximity, contradiction, stakes)

        # Classify mode
        mode = self.classify_mode(logic_weight)

        # Update history
        self.previous_state = current_state.detach().clone()
        self.mode_history.append(mode)
        if len(self.mode_history) > 100:
            self.mode_history.pop(0)

        # Telemetry
        telemetry = None
        if return_telemetry:
            telemetry = {
                "logic_weight": logic_weight.mean().item(),
                "mode": mode,
                "gradient_magnitude": gradient_mag.mean().item(),
                "proximity": proximity.mean().item(),
                "contradiction": contradiction.mean().item(),
                "stakes": stakes.mean().item() if stakes is not None else 0.5,
                # Mode statistics
                "mode_history_length": len(self.mode_history),
                "feeling_fraction": sum(1 for m in self.mode_history if m == "feeling")
                / max(1, len(self.mode_history)),
                "tack_fraction": sum(1 for m in self.mode_history if m == "tack") / max(1, len(self.mode_history)),
                "logic_fraction": sum(1 for m in self.mode_history if m == "logic") / max(1, len(self.mode_history)),
            }

        return logic_weight, mode, telemetry

    def calibrate_validation_effort(self, gradient_mag: float, stakes: float) -> float:
        """
        Calibrate how much validation effort to apply.

        High gradient × high stakes → more validation
        Low gradient × low stakes → less validation

        Args:
            gradient_mag: Feeling strength |∇κ|
            stakes: Task importance

        Returns:
            validation_effort: Recommended validation level [0, 1]
        """
        # Validation effort = gradient × stakes
        # (Strong feelings on important tasks need thorough validation)
        effort = gradient_mag * stakes
        return min(1.0, effort)


# ===========================================================================
# VALIDATION
# ===========================================================================


def validate_tacking_controller():
    """Test WuWei controller."""
    print("Testing WuWeiController...")

    # Create controller
    controller = WuWeiController(d_model=256)

    # Random state
    state = torch.randn(2, 10, 256)

    # Forward pass
    logic_weight, mode, telemetry = controller(state)

    # Validate
    assert logic_weight.shape == (2,), "Logic weight shape mismatch!"
    assert mode in ["feeling", "tack", "logic"], "Invalid mode!"
    assert 0 <= telemetry["logic_weight"] <= 1, "Logic weight out of range!"

    print(f"✅ Logic weight: {telemetry['logic_weight']:.3f}")
    print(f"✅ Mode: {mode}")
    print(f"✅ Gradient magnitude: {telemetry['gradient_magnitude']:.3f}")
    print(f"✅ Proximity: {telemetry['proximity']:.3f}")
    print(f"✅ Contradiction: {telemetry['contradiction']:.3f}")

    return telemetry


if __name__ == "__main__":
    telemetry = validate_tacking_controller()

    print("\n" + "=" * 60)
    print("WuWeiController validation complete!")
    print("=" * 60)
    print(f"\nMode distribution (history={telemetry['mode_history_length']}):")
    print(f"  - Feeling: {telemetry['feeling_fraction']:.1%}")
    print(f"  - Tack: {telemetry['tack_fraction']:.1%}")
    print(f"  - Logic: {telemetry['logic_fraction']:.1%}")
    print("\nReady for integration into QIG-Kernel!")
