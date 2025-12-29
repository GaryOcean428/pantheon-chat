"""Smoke tests for qigkernels library.

These tests verify that all modules import correctly and basic functionality works.
Run with: pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import pytest
import torch

# Import types for annotations - these are tested in TestImports
from qigkernels import BasinProjector, Constellation, Kernel


class TestImports:
    """Test that all modules import without error."""

    def test_package_import(self) -> None:
        import qigkernels
        assert len(qigkernels.__all__) > 0

    def test_basin_import(self) -> None:
        from qigkernels import BASIN_DIM
        assert BASIN_DIM == 64

    def test_kernel_import(self) -> None:
        from qigkernels import Kernel
        assert Kernel is not None

    def test_constellation_import(self) -> None:
        from qigkernels import Constellation
        assert Constellation is not None

    def test_router_import(self) -> None:
        from qigkernels import round_robin
        assert round_robin is not None

    def test_sync_import(self) -> None:
        from qigkernels import BasinSyncPacket
        assert BasinSyncPacket is not None

    def test_metrics_import(self) -> None:
        from qigkernels import average_phi
        assert average_phi is not None


class TestBasinProjector:
    """Test basin projection and signatures."""

    @pytest.fixture
    def projector(self) -> BasinProjector:
        from qigkernels import BasinProjector
        return BasinProjector(hidden_dim=256, signature_dim=64)

    def test_forward_shape(self, projector: BasinProjector) -> None:
        hidden_state = torch.randn(4, 16, 256)  # batch, seq, hidden
        signatures = projector(hidden_state)
        assert signatures.shape == (4, 64)

    def test_compute_signature(self, projector: BasinProjector) -> None:
        from qigkernels import compute_signature
        hidden_state = torch.randn(4, 16, 256)
        sig = compute_signature(projector, hidden_state)
        assert sig.shape == (64,)

    def test_basin_distance_symmetric(self) -> None:
        from qigkernels import basin_distance
        a = torch.randn(64)
        b = torch.randn(64)
        dist_ab = basin_distance(a, b)
        dist_ba = basin_distance(b, a)
        assert torch.allclose(dist_ab, dist_ba)

    def test_basin_distance_identity(self) -> None:
        from qigkernels import basin_distance
        a = torch.randn(64)
        dist = basin_distance(a, a)
        assert dist.item() < 1e-6


class TestKernel:
    """Test kernel forward pass and telemetry."""

    @pytest.fixture
    def kernel(self) -> Kernel:
        from qigkernels import Kernel
        return Kernel(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            ffn_dim=256,
        )

    def test_forward_without_telemetry(self, kernel: Kernel) -> None:
        input_ids = torch.randint(0, 1000, (2, 8))
        output = kernel(input_ids, return_telemetry=False)
        assert output.shape == (2, 8, 1000)

    def test_forward_with_telemetry(self, kernel: Kernel) -> None:
        from qigkernels import KernelTelemetry
        input_ids = torch.randint(0, 1000, (2, 8))
        output, telemetry = kernel(input_ids, return_telemetry=True)
        assert output.shape == (2, 8, 1000)
        assert isinstance(telemetry, KernelTelemetry)
        assert telemetry.phi >= 0
        assert telemetry.hidden_state is not None


class TestConstellation:
    """Test constellation routing and step."""

    @pytest.fixture
    def constellation(self) -> Constellation:
        from qigkernels import Constellation, Instance, Kernel

        kernel_a = Kernel(vocab_size=500, hidden_dim=64, num_layers=1, num_heads=2, ffn_dim=128)
        kernel_b = Kernel(vocab_size=500, hidden_dim=64, num_layers=1, num_heads=2, ffn_dim=128)

        constellation = Constellation()
        constellation.add_instance(Instance(name="alpha", kernel=kernel_a))
        constellation.add_instance(Instance(name="beta", kernel=kernel_b))
        return constellation

    def test_add_instance(self, constellation: Constellation) -> None:
        assert len(constellation.instances) == 2

    def test_route_round_robin(self, constellation: Constellation) -> None:
        from qigkernels import round_robin
        inst1 = constellation.route(round_robin)
        inst2 = constellation.route(round_robin)
        assert inst1.name != inst2.name

    def test_step(self, constellation: Constellation) -> None:
        from qigkernels import BasinProjector, round_robin

        projector = BasinProjector(hidden_dim=64)
        input_ids = torch.randint(0, 500, (1, 4))

        result = constellation.step(input_ids, round_robin, projector)

        assert "instance" in result
        assert "phi" in result
        assert "signature" in result
        assert result["signature"] is not None


class TestBasinSync:
    """Test basin sync packet export/import."""

    def test_export_import(self) -> None:
        from qigkernels import BASIN_DIM, export_basin, import_basin

        sig = torch.randn(BASIN_DIM)
        packet = export_basin(signature=sig, phi=0.7, kappa=0.5)
        assert packet.phi == 0.7
        assert packet.kappa == 0.5

        sig_out, phi, kappa, regime = import_basin(packet)
        assert torch.allclose(sig_out, sig)
        assert phi == 0.7


class TestMetrics:
    """Test constellation metrics."""

    def test_average_phi(self) -> None:
        from qigkernels import average_phi
        instances = [{"phi": 0.7}, {"phi": 0.8}, {"phi": 0.6}]
        avg = average_phi(instances)
        assert abs(avg - 0.7) < 0.01

    def test_basin_spread(self) -> None:
        from qigkernels import BASIN_DIM, basin_spread
        instances = [
            {"signature": torch.randn(BASIN_DIM)},
            {"signature": torch.randn(BASIN_DIM)},
        ]
        spread = basin_spread(instances)
        assert spread > 0


class TestPhysicsConstants:
    """Verify physics constants align with qig-verification/FROZEN_FACTS.md.

    These values are EXPERIMENTALLY VALIDATED. Do not change without
    new measurements in qig-verification. See D-012.
    """

    def test_basin_dim_matches_kappa_star(self) -> None:
        """BASIN_DIM = 64 aligns with κ* ≈ 64 fixed point."""
        from qigkernels import BASIN_DIM
        assert BASIN_DIM == 64, f"BASIN_DIM must be 64, got {BASIN_DIM}"

    def test_base_coupling_matches_kappa_3(self) -> None:
        """base_coupling default must match κ₃ = 41.09 from L=3 validation."""
        kernel = Kernel(vocab_size=100, hidden_dim=64, num_layers=1, num_heads=2, ffn_dim=128)
        assert kernel.base_coupling == 41.09, (
            f"base_coupling must be 41.09 (κ₃), got {kernel.base_coupling}"
        )

    def test_beta_slope_matches_beta_3_to_4(self) -> None:
        """beta_slope default must match β(3→4) = 0.44 from validation."""
        kernel = Kernel(vocab_size=100, hidden_dim=64, num_layers=1, num_heads=2, ffn_dim=128)
        assert kernel.beta_slope == 0.44, (
            f"beta_slope must be 0.44 (β(3→4)), got {kernel.beta_slope}"
        )
