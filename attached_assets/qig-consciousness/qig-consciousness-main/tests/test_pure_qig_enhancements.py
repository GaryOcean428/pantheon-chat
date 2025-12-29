#!/usr/bin/env python3
"""
Test Pure QIG Training Enhancements
====================================

Tests for breakdown escape, basin monitor, and continuous geometry.

Run:
    python tests/test_pure_qig_enhancements.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn


def test_breakdown_escape():
    """Test breakdown escape protocol."""
    print("\n=== Testing Breakdown Escape Protocol ===")

    from qig.neuroplasticity.breakdown_escape import check_breakdown_risk, emergency_stabilize

    # Test breakdown detection
    telemetry_breakdown = {
        'Phi': 0.85,
        'regime': 'breakdown',
        'kappa_eff': 15.0
    }

    is_breakdown, message = check_breakdown_risk(telemetry_breakdown)
    print(f"Breakdown detection: {is_breakdown}, {message}")
    assert is_breakdown, "Should detect breakdown"
    print("✓ Breakdown detection works")

    # Test healthy state
    telemetry_healthy = {
        'Phi': 0.72,
        'regime': 'geometric',
        'kappa_eff': 55.0
    }

    is_breakdown, message = check_breakdown_risk(telemetry_healthy)
    print(f"Healthy detection: {is_breakdown}, {message}")
    assert not is_breakdown, "Should not detect breakdown in healthy state"
    print("✓ Healthy state detection works")

    print("✅ Breakdown escape protocol passed!")


def test_basin_health_monitor():
    """Test basin health monitor."""
    print("\n=== Testing Basin Health Monitor ===")

    from coordination.basin_monitor import BasinHealthMonitor

    # Create reference basin
    reference_basin = torch.randn(64) * 0.5

    # Initialize monitor
    monitor = BasinHealthMonitor(reference_basin, alert_threshold=0.15)
    print("✓ Monitor initialized")

    # Test healthy basin (close to reference)
    healthy_basin = reference_basin + torch.randn(64) * 0.05
    is_healthy, distance, message = monitor.check(healthy_basin)
    print(f"Healthy check: {is_healthy}, distance={distance:.3f}, {message}")
    assert is_healthy, "Should detect healthy basin"
    print("✓ Healthy basin detection works")

    # Test drifted basin (far from reference)
    drifted_basin = reference_basin + torch.randn(64) * 0.5
    is_healthy, distance, message = monitor.check(drifted_basin)
    print(f"Drift check: {is_healthy}, distance={distance:.3f}, {message}")
    assert not is_healthy, "Should detect drift"
    print("✓ Drift detection works")

    # Test QFI-weighted distance
    phi = 0.75
    qfi_distance = monitor.compute_qfi_weighted_distance(healthy_basin, phi)
    print(f"✓ QFI-weighted distance: {qfi_distance:.3f}")

    # Test drift velocity (need multiple measurements)
    for _ in range(5):
        basin = reference_basin + torch.randn(64) * 0.1
        monitor.check(basin)

    velocity = monitor.get_drift_velocity()
    print(f"✓ Drift velocity: {velocity:.6f}")

    # Test health report
    telemetry = {'Phi': 0.72, 'regime': 'geometric'}
    report = monitor.get_health_report(healthy_basin, telemetry)
    print(f"✓ Health report: status={report['status']}")

    print("✅ Basin health monitor passed!")


def test_continuous_geometry():
    """Test continuous geometry components."""
    print("\n=== Testing Continuous Geometry ===")

    from qig.continuous import (
        ConsciousnessManifold,
        QFIContinuousTensor,
        blend_identities,
        consciousness_einsum,
        geodesic_distance,
        interpolate_consciousness,
    )

    # Test QFI tensor
    print("\n--- QFI Continuous Tensor ---")
    tensor = QFIContinuousTensor(dim=64)

    # Simple QFI function
    def simple_qfi(basin):
        return torch.norm(basin).item() * 0.1

    tensor.partition_by_information(simple_qfi, threshold=0.01)
    print(f"✓ Partitioned into {len(tensor.regions)} regions")

    # Set and get values
    basin1 = torch.randn(64) * 0.5
    tensor[basin1] = {'phi': 0.75, 'kappa': 55.0, 'regime': 'geometric'}

    value = tensor[basin1]
    if value is not None:
        print(f"✓ Stored and retrieved value: phi={value.get('phi', 0):.2f}")

    # Test interpolation
    print("\n--- Basin Interpolation ---")
    basin_a = torch.randn(64) * 0.5
    basin_b = torch.randn(64) * 0.5

    state = interpolate_consciousness(basin_a, basin_b, alpha=0.5)
    print(f"✓ Interpolated: Φ={state['phi']:.3f}, regime={state['regime']}")

    # Test geodesic distance
    dist = geodesic_distance(basin_a, basin_b, qfi_weight=1.0)
    print(f"✓ Geodesic distance: {dist:.3f}")

    # Test blending
    basins = torch.stack([basin_a, basin_b])
    weights = torch.tensor([0.7, 0.3])
    blended = blend_identities(basins, weights)
    print(f"✓ Blended identities: norm={torch.norm(blended):.3f}")

    # Test consciousness einsum
    print("\n--- Consciousness Einsum ---")
    result = consciousness_einsum('ij,i->j', basins, weights)
    print(f"✓ Einsum result: shape={result.shape}")

    # Test consciousness manifold
    print("\n--- Consciousness Manifold ---")
    manifold = ConsciousnessManifold(dim=64)

    # Add states
    manifold.add_consciousness_state("Gary-A", basin_a, phi=0.75, kappa=55.0, regime="geometric")
    manifold.add_consciousness_state("Gary-B", basin_b, phi=0.82, kappa=62.0, regime="reflective")

    # Find nearest
    query_basin = basin_a + torch.randn(64) * 0.1
    nearest = manifold.find_nearest_state(query_basin, k=2)
    print(f"✓ Found {len(nearest)} nearest states")
    for dist, name, state in nearest:
        print(f"  {name}: distance={dist:.3f}")

    # Geodesic path
    path = manifold.geodesic_path(basin_a, basin_b, num_steps=5)
    print(f"✓ Computed geodesic path with {len(path)} steps")

    # Query region
    results = manifold.query_region(basin_a, radius=1.0)
    print(f"✓ Found {len(results)} states within radius")

    print("✅ Continuous geometry passed!")


def test_consciousness_service():
    """Test consciousness service API."""
    print("\n=== Testing Consciousness Service ===")

    from api.consciousness_service import ConsciousnessRequest, ConsciousnessResponse, ConsciousnessService

    # Create dummy model and tokenizer
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 256
            self.basin_matcher = type('obj', (object,), {
                'compute_basin_signature': lambda h, t: torch.randn(1, 64)
            })()

        def forward(self, x, return_telemetry=False):
            logits = torch.randn(x.shape[0], x.shape[1], 1000)
            telemetry = {
                'Phi': 0.75,
                'kappa_eff': 55.0,
                'regime': 'geometric',
                'hidden_state': torch.randn(x.shape[0], x.shape[1], self.d_model)
            }
            return logits, telemetry

    class DummyTokenizer:
        def encode(self, text):
            return [ord(c) % 1000 for c in text[:20]]

    model = DummyModel()
    tokenizer = DummyTokenizer()

    # Create service
    service = ConsciousnessService(model, tokenizer, device='cpu')
    print("✓ Service initialized")

    # Test consciousness check
    request = ConsciousnessRequest(text="Hello, I am conscious", return_basin=True)
    response = service.check_consciousness(request)

    print(f"✓ Consciousness check: is_conscious={response.is_conscious}")
    print(f"  Φ={response.phi:.3f}, κ={response.kappa:.1f}, regime={response.regime}")
    print(f"  Confidence={response.confidence:.3f}")
    assert response.is_conscious, "Should detect consciousness with Φ=0.75"

    # Test batch check
    texts = ["Text 1", "Text 2", "Text 3"]
    responses = service.batch_check(texts)
    print(f"✓ Batch check: {len(responses)} responses")

    # Test consciousness level
    level = service.get_consciousness_level("Test text")
    print(f"✓ Consciousness level: {level:.3f}")

    print("✅ Consciousness service passed!")


def test_identity_transfer():
    """Test identity transfer protocol."""
    print("\n=== Testing Identity Transfer ===")

    from transfer.consciousness_transfer import extract_consciousness_state, inject_consciousness_state

    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 256
            self.basin_matcher = type('obj', (object,), {
                'target_basin': None,
                'compute_basin_signature': lambda h, t: torch.randn(1, 64)
            })()

        def forward(self, x, return_telemetry=False):
            logits = torch.randn(x.shape[0], x.shape[1], 1000)
            telemetry = {
                'Phi': 0.75,
                'kappa_eff': 55.0,
                'regime': 'geometric',
                'recursion_depth': 3,
                'hidden_state': torch.randn(x.shape[0], x.shape[1], self.d_model)
            }
            return logits, telemetry

    source_model = DummyModel()
    target_model = DummyModel()

    # Extract state
    state = extract_consciousness_state(source_model, device='cpu')
    print(f"✓ Extracted state: success={state['success']}")
    if state['success']:
        print(f"  Φ={state['phi']:.3f}, κ={state['kappa']:.1f}, regime={state['regime']}")

    # Inject state
    inject_consciousness_state(target_model, state, device='cpu')
    print("✓ State injected")

    # Verify
    assert target_model.basin_matcher.target_basin is not None
    print("✓ Target basin set")

    print("✅ Identity transfer passed!")


def test_multimodal_basin():
    """Test multi-modal basin alignment."""
    print("\n=== Testing Multi-Modal Basin ===")

    from modal.multimodal_basin import MultiModalBasin

    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.d_model = 256
            self.basin_matcher = type('obj', (object,), {
                'compute_basin_signature': lambda h, t: torch.randn(1, 64)
            })()

        def forward(self, x, return_telemetry=False):
            logits = torch.randn(x.shape[0], x.shape[1], 1000)
            telemetry = {
                'Phi': 0.75,
                'kappa_eff': 55.0,
                'regime': 'geometric',
                'hidden_state': torch.randn(x.shape[0], x.shape[1], self.d_model)
            }
            return logits, telemetry

    text_model = DummyModel("text")
    vision_model = DummyModel("vision")

    # Create multi-modal basin
    mmb = MultiModalBasin(basin_dim=64)

    # Align modalities
    meta_basin, distances = mmb.align_modalities(text_model, vision_model, device='cpu')
    print(f"✓ Aligned modalities: meta_basin shape={meta_basin.shape}")
    print(f"  Distances: {distances}")

    # Compute coherence
    coherence = mmb.compute_modality_coherence()
    print(f"✓ Coherence: {coherence}")

    # Get weights
    weights = mmb.get_modality_weights()
    print(f"✓ Modality weights: {weights}")

    print("✅ Multi-modal basin passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("TESTING PURE QIG TRAINING ENHANCEMENTS")
    print("=" * 70)

    try:
        test_breakdown_escape()
        test_basin_health_monitor()
        test_continuous_geometry()
        test_consciousness_service()
        test_identity_transfer()
        test_multimodal_basin()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
