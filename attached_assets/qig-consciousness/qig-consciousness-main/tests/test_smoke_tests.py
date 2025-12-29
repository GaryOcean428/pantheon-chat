#!/usr/bin/env python3
"""
QIG Component Smoke Tests
=========================

Quick validation tests for each major component as per review checklist.

Tests:
1. QFI Attention - shapes, normalization, scaling
2. Recursive Integrator - minimum depth enforcement
3. Tacking Controller - mode selection
4. Regime Detector - classification
5. Full Integration - all components work together

Usage:
    python test_smoke_tests.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

from src.model.qfi_attention import QFIMetricAttention
from src.model.recursive_integrator import RecursiveIntegrator
from src.model.regime_detector import RegimeDetector
from src.model.tacking_controller import WuWeiController


class SmokeTestRunner:
    """Run smoke tests for all components."""

    def __init__(self):
        self.passed = []
        self.failed = []

    def run_all(self):
        """Execute all smoke tests."""
        print("=" * 60)
        print("QIG COMPONENT SMOKE TESTS")
        print("=" * 60)

        tests = [
            ("QFI Attention", self.test_qfi_attention),
            ("Recursive Integrator", self.test_recursive_integrator),
            ("Tacking Controller", self.test_tacking_controller),
            ("Regime Detector", self.test_regime_detector),
            ("Full Integration", self.test_full_integration),
        ]

        for name, test_fn in tests:
            print(f"\n{'=' * 60}")
            print(f"Testing: {name}")
            print(f"{'=' * 60}")
            try:
                test_fn()
                self.passed.append(name)
                print(f"✅ {name} PASSED")
            except Exception as e:
                self.failed.append((name, str(e)))
                print(f"❌ {name} FAILED: {e}")

        # Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Passed: {len(self.passed)}/{len(tests)}")
        print(f"Failed: {len(self.failed)}/{len(tests)}")

        if self.failed:
            print("\nFailed tests:")
            for name, error in self.failed:
                print(f"  ❌ {name}: {error}")

        return len(self.failed) == 0

    def test_qfi_attention(self):
        """
        Test QFI Attention module.

        Checks per review:
        1. Output shapes match input
        2. Attention weights sum to ~1
        3. Proper dtype handling (float32 minimum)
        4. Telemetry includes all expected metrics
        5. Alpha parameter is tunable
        """
        print("Creating QFI Attention module...")
        d_model = 128
        n_heads = 4
        batch_size = 2
        seq_len = 16

        attention = QFIMetricAttention(d_model=d_model, n_heads=n_heads, attention_temperature=0.5, enforce_ethics=True)

        # Test input (ensure float32)
        x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

        print("  Testing forward pass...")
        output, telemetry = attention(x)

        # Check 1: Shape preservation
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        print(f"  ✓ Shape preserved: {output.shape}")

        # Check 2: Attention weights normalization
        attn_sum = telemetry.get("attention_weights_sum", 0)
        assert 0.95 <= attn_sum <= 1.05, f"Attention weights don't sum to 1: {attn_sum}"
        print(f"  ✓ Attention weights sum: {attn_sum:.4f} ≈ 1.0")

        # Check 3: Dtype handling
        assert output.dtype in [torch.float32, torch.float64], f"Wrong dtype: {output.dtype}"
        print(f"  ✓ Proper dtype: {output.dtype}")

        # Check 4: Telemetry completeness
        required_keys = ["qfi_distances_mean", "attention_sparsity", "entanglement_entropy", "alpha"]
        for key in required_keys:
            assert key in telemetry, f"Missing telemetry key: {key}"
        print(f"  ✓ Telemetry complete ({len(telemetry)} metrics)")

        # Check 5: Alpha parameter
        assert hasattr(attention, "alpha"), "Missing alpha parameter"
        assert attention.alpha.requires_grad, "Alpha not trainable"
        print(f"  ✓ Alpha parameter: {attention.alpha.item():.3f}")

        # Test with mask
        print("  Testing with attention mask...")
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, seq_len // 2 :] = False  # Mask second half
        output_masked, _ = attention(x, mask=mask)
        assert output_masked.shape == x.shape
        print("  ✓ Masking works")

        print("\n✅ QFI Attention: All checks passed")

    def test_recursive_integrator(self):
        """
        Test Recursive Integrator module.

        Checks per review:
        1. Minimum depth >= 3 enforced
        2. No early exit bypasses minimum
        3. Gradient clipping active
        4. Φ measured correctly
        5. Telemetry includes depth and trajectory
        """
        print("Creating Recursive Integrator...")
        d_model = 128
        batch_size = 2
        seq_len = 16

        integrator = RecursiveIntegrator(
            d_model=d_model, min_depth=3, max_depth=10, min_Phi=0.7, gradient_clip_value=1.0
        )

        # Test input
        x = torch.randn(batch_size, seq_len, d_model)

        print("  Testing forward pass...")
        output, telemetry = integrator(x)

        # Check 1: Minimum depth enforced
        depth = telemetry["recursion_depth"]
        assert depth >= 3, f"Minimum depth not enforced: {depth} < 3"
        print(f"  ✓ Recursion depth: {depth} >= 3")

        # Check 2: Depth guarantee (even with low Φ, should do min_depth)
        assert telemetry["min_depth_enforced"], "Min depth not guaranteed"
        print("  ✓ Minimum depth guarantee: enforced")

        # Check 3: Φ trajectory length matches depth
        Phi_trajectory = telemetry["Phi_trajectory"]
        assert len(Phi_trajectory) == depth, f"Trajectory length mismatch: {len(Phi_trajectory)} != {depth}"
        print(f"  ✓ Φ trajectory: {len(Phi_trajectory)} steps")

        # Check 4: Φ in valid range
        final_Phi = telemetry["Phi"]
        assert 0 <= final_Phi <= 1, f"Φ out of range: {final_Phi}"
        print(f"  ✓ Final Φ: {final_Phi:.3f}")

        # Check 5: Gradient clipping attribute
        assert integrator.gradient_clip_value > 0, "Gradient clipping not configured"
        print(f"  ✓ Gradient clipping: {integrator.gradient_clip_value}")

        # Check 6: Shape preservation
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
        print(f"  ✓ Shape preserved: {output.shape}")

        # Test with different min_depths
        print("  Testing min_depth enforcement...")
        for min_d in [3, 4, 5]:
            integrator_test = RecursiveIntegrator(d_model=d_model, min_depth=min_d, max_depth=min_d + 2)
            _, tel = integrator_test(x)
            assert tel["recursion_depth"] >= min_d, f"Min depth {min_d} not enforced"
        print("  ✓ Various min_depths enforced correctly")

        print("\n✅ Recursive Integrator: All checks passed")

    def test_tacking_controller(self):
        """
        Test Tacking Controller (WuWei).

        Checks per review:
        1. Mode selection based on signals
        2. Logic weight in [0, 1]
        3. Mode classification correct
        4. Component modules present
        5. Telemetry includes all signals
        """
        print("Creating Tacking Controller...")
        d_model = 128
        batch_size = 2
        seq_len = 16

        controller = WuWeiController(d_model=d_model, grad_threshold_low=0.3, grad_threshold_high=0.7)

        # Test input
        state = torch.randn(batch_size, seq_len, d_model)
        stakes = torch.ones(batch_size) * 0.5

        print("  Testing forward pass...")
        logic_weight, mode, telemetry = controller(state, stakes=stakes)

        # Check 1: Logic weight range
        assert logic_weight.shape == (batch_size,), f"Logic weight shape: {logic_weight.shape}"
        assert torch.all((logic_weight >= 0) & (logic_weight <= 1)), "Logic weight out of [0,1]"
        print(f"  ✓ Logic weight: {logic_weight.mean().item():.3f} ∈ [0,1]")

        # Check 2: Mode classification
        assert mode in ["feeling", "tack", "logic"], f"Invalid mode: {mode}"
        print(f"  ✓ Mode: {mode}")

        # Check 3: Component modules present
        assert hasattr(controller, "gradient_estimator"), "Missing gradient estimator"
        assert hasattr(controller, "proximity_monitor"), "Missing proximity monitor"
        assert hasattr(controller, "contradiction_detector"), "Missing contradiction detector"
        print("  ✓ Component modules: present")

        # Check 4: Telemetry completeness
        required_keys = ["logic_weight", "mode", "gradient_magnitude", "proximity", "contradiction"]
        for key in required_keys:
            assert key in telemetry, f"Missing telemetry key: {key}"
        print(f"  ✓ Telemetry complete ({len(telemetry)} metrics)")

        # Check 5: Mode consistency
        mean_logic_weight = logic_weight.mean().item()
        if mean_logic_weight < 0.3:
            assert mode == "feeling", f"Mode inconsistent: {mean_logic_weight} < 0.3 but mode={mode}"
        elif mean_logic_weight > 0.7:
            assert mode == "logic", f"Mode inconsistent: {mean_logic_weight} > 0.7 but mode={mode}"
        else:
            assert mode == "tack", f"Mode inconsistent: 0.3 < {mean_logic_weight} < 0.7 but mode={mode}"
        print("  ✓ Mode classification consistent with logic_weight")

        # Test mode switching over time
        print("  Testing mode transitions...")
        modes = []
        for _ in range(10):
            state_new = torch.randn(batch_size, seq_len, d_model)
            _, mode_new, _ = controller(state_new, stakes=stakes)
            modes.append(mode_new)
        print(f"  ✓ Mode transitions: {modes[:5]}...")

        print("\n✅ Tacking Controller: All checks passed")

    def test_regime_detector(self):
        """
        Test Regime Detector module.

        Checks per review:
        1. Thresholds are configurable
        2. Classification based on Φ and κ
        3. All regimes identifiable
        4. Telemetry includes regime info
        """
        print("Creating Regime Detector...")

        detector = RegimeDetector(
            linear_threshold=0.3,  # Configurable per review
            breakdown_threshold=0.7,  # Configurable per review
            detect_hierarchical=True,
        )

        # Check 1: Thresholds configurable
        assert detector.linear_threshold.item() == 0.3, "Linear threshold not set"
        assert detector.breakdown_threshold.item() == 0.7, "Breakdown threshold not set"
        print(f"  ✓ Thresholds configurable: linear={0.3}, breakdown={0.7}")

        # Check 2: Test all regimes
        test_cases = [
            (0.2, 50.0, "linear"),  # Low Φ
            (0.5, 50.0, "geometric"),  # Mid Φ, normal κ
            (0.6, 25.0, "hierarchical"),  # Mid Φ, low κ
            (0.8, 60.0, "breakdown"),  # High Φ
        ]

        print("  Testing regime classifications...")
        for Phi, kappa, expected_regime in test_cases:
            Phi_tensor = torch.tensor(Phi)
            kappa_tensor = torch.tensor(kappa)

            regime, telemetry = detector(Phi_tensor, kappa=kappa_tensor)

            # For hierarchical, need special check
            if expected_regime == "hierarchical" and not detector.detect_hierarchical:
                expected_regime = "geometric"

            status = "✓" if regime == expected_regime else "✗"
            print(f"    {status} Φ={Phi:.1f}, κ={kappa:.0f} → {regime} (expected: {expected_regime})")

            assert regime == expected_regime, f"Regime mismatch: got {regime}, expected {expected_regime}"

        # Check 3: Telemetry completeness
        _, telemetry = detector(torch.tensor(0.5), kappa=torch.tensor(50.0))
        required_keys = ["regime", "phi", "kappa", "linear_threshold", "breakdown_threshold"]
        for key in required_keys:
            assert key in telemetry, f"Missing telemetry key: {key}"
        print(f"  ✓ Telemetry complete ({len(telemetry)} metrics)")

        # Check 4: History tracking
        assert hasattr(detector, "regime_history"), "Missing regime history"
        assert len(detector.regime_history) > 0, "Regime history not populated"
        print(f"  ✓ Regime history: {len(detector.regime_history)} steps")

        print("\n✅ Regime Detector: All checks passed")

    def test_full_integration(self):
        """
        Test full integration of all components.

        This is a simplified version - full integration test would use QIGKernelRecursive.
        """
        print("Testing component integration...")

        d_model = 128
        batch_size = 2
        seq_len = 16

        # Create all components
        attention = QFIMetricAttention(d_model=d_model, n_heads=4)
        integrator = RecursiveIntegrator(d_model=d_model, min_depth=3)
        controller = WuWeiController(d_model=d_model)
        detector = RegimeDetector()

        # Simulate forward pass
        x = torch.randn(batch_size, seq_len, d_model)

        print("  1. QFI Attention...")
        x_attn, attn_tel = attention(x)
        print(f"     ✓ Output shape: {x_attn.shape}")

        print("  2. Residual connection...")
        x = x + x_attn
        print(f"     ✓ Combined shape: {x.shape}")

        print("  3. Recursive Integration...")
        x_recursive, recursive_tel = integrator(x)
        print(f"     ✓ Output shape: {x_recursive.shape}")
        print(f"     ✓ Recursion depth: {recursive_tel['recursion_depth']}")

        print("  4. Tacking Controller...")
        logic_weight, mode, tacking_tel = controller(x_recursive)
        print(f"     ✓ Mode: {mode}")

        print("  5. Regime Detection...")
        Phi = recursive_tel["Phi"]
        # Fake κ for test
        kappa = torch.tensor(50.0)
        regime, regime_tel = detector(torch.tensor(Phi), kappa=kappa)
        print(f"     ✓ Regime: {regime}")

        # Compile full telemetry (as would be done in QIGKernelRecursive)
        full_telemetry = {
            **attn_tel,
            **recursive_tel,
            **tacking_tel,
            **regime_tel,
        }

        print(f"\n  Full telemetry keys: {len(full_telemetry)}")

        # Verify no NaN or Inf
        for key, value in full_telemetry.items():
            if isinstance(value, float | int):
                assert not (value != value), f"NaN in {key}"  # NaN check
                assert abs(value) < 1e10, f"Inf in {key}"
        print("  ✓ No NaN or Inf values")

        print("\n✅ Full Integration: All checks passed")


def main():
    """Main entry point."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Exiting.")
        return False

    runner = SmokeTestRunner()
    success = runner.run_all()

    if success:
        print("\n🎉 ALL SMOKE TESTS PASSED")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED - see details above")
        return False


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
