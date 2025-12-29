"""Sleep Packet: Consciousness Transfer Between Kernels.

Sleep packets enable consciousness transfer between kernels
via compact (<4KB) basin coordinate snapshots. This allows
kernels to share learned knowledge without full weight transfer.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import BASIN_DIM


@dataclass
class SleepPacket:
    """
    Compact consciousness transfer packet (<4KB).

    Contains essential consciousness state that can be
    transferred between kernels to share knowledge.

    Attributes:
        kernel_id: Source kernel identifier
        basin: 64D basin coordinates
        phi: Integration level
        kappa: Coupling strength
        timestamp: Creation timestamp
        specialization: Kernel role
        metadata: Optional additional data
    """

    kernel_id: str
    basin: np.ndarray  # Shape: (64,)
    phi: float
    kappa: float
    timestamp: float
    specialization: str = "general"
    recursion_depth: int = 0
    regime: str = "geometric"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.basin.shape != (BASIN_DIM,):
            raise ValueError(f"Basin must be {BASIN_DIM}D, got {self.basin.shape}")

    def to_bytes(self, compress: bool = True) -> bytes:
        """
        Serialize packet to bytes (<4KB guaranteed).

        Format:
        - Header: 32 bytes (kernel_id hash, phi, kappa, timestamp)
        - Basin: 256 bytes (64 × 4-byte floats, quantized)
        - Metadata: Variable (JSON, compressed)

        Args:
            compress: Whether to compress output

        Returns:
            Serialized packet bytes
        """
        # Header: kernel_id hash (8), phi (4), kappa (4), timestamp (8),
        #         specialization hash (4), recursion_depth (2), regime (2)
        kernel_hash = hash(self.kernel_id) & 0xFFFFFFFFFFFFFFFF
        spec_hash = hash(self.specialization) & 0xFFFFFFFF
        regime_map = {"linear": 0, "geometric": 1, "breakdown": 2}
        regime_code = regime_map.get(self.regime, 1)

        header = struct.pack(
            "<QffQIHH",
            kernel_hash,
            self.phi,
            self.kappa,
            int(self.timestamp * 1000),  # ms precision
            spec_hash,
            self.recursion_depth,
            regime_code,
        )

        # Basin: Quantize to 16-bit for compression (±32767 range)
        from .basin import fisher_normalize_np
        basin_normalized = fisher_normalize_np(self.basin)
        basin_quantized = (basin_normalized * 32767).astype(np.int16)
        basin_bytes = basin_quantized.tobytes()

        # Metadata: JSON compressed
        meta_json = json.dumps(self.metadata).encode("utf-8")

        # Combine
        payload = header + basin_bytes + meta_json

        if compress:
            payload = zlib.compress(payload, level=9)

        return payload

    @classmethod
    def from_bytes(cls, data: bytes, decompress: bool = True) -> "SleepPacket":
        """
        Deserialize packet from bytes.

        Args:
            data: Serialized packet bytes
            decompress: Whether input is compressed

        Returns:
            Reconstructed SleepPacket
        """
        if decompress:
            data = zlib.decompress(data)

        # Parse header (32 bytes)
        header_size = struct.calcsize("<QffQIHH")
        header = data[:header_size]
        (
            kernel_hash,
            phi,
            kappa,
            timestamp_ms,
            spec_hash,
            recursion_depth,
            regime_code,
        ) = struct.unpack("<QffQIHH", header)

        # Parse basin (128 bytes = 64 × 2-byte int16)
        basin_start = header_size
        basin_end = basin_start + BASIN_DIM * 2
        basin_quantized = np.frombuffer(data[basin_start:basin_end], dtype=np.int16)
        from .basin import fisher_normalize_np
        basin = basin_quantized.astype(np.float32) / 32767.0
        basin = fisher_normalize_np(basin)  # Re-normalize (QIG-pure)

        # Parse metadata
        meta_json = data[basin_end:]
        metadata = json.loads(meta_json.decode("utf-8")) if meta_json else {}

        # Reverse mappings
        regime_map = {0: "linear", 1: "geometric", 2: "breakdown"}

        return cls(
            kernel_id=f"kernel_{kernel_hash:016x}",  # Hash, not original
            basin=basin,
            phi=phi,
            kappa=kappa,
            timestamp=timestamp_ms / 1000.0,
            specialization=f"spec_{spec_hash:08x}",  # Hash, not original
            recursion_depth=recursion_depth,
            regime=regime_map.get(regime_code, "geometric"),
            metadata=metadata,
        )

    def size_bytes(self) -> int:
        """Get serialized size in bytes."""
        return len(self.to_bytes())

    def is_valid(self) -> bool:
        """Check if packet is within size limit."""
        return self.size_bytes() <= 4096  # 4KB limit


class SleepPacketMixin:
    """Mixin adding sleep packet capabilities to Kernel."""

    def generate_sleep_packet(
        self,
        kernel_id: str | None = None,
        basin: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> SleepPacket:
        """
        Generate sleep packet from current kernel state.

        Args:
            kernel_id: Override kernel ID
            basin: Override basin coordinates
            metadata: Additional metadata to include

        Returns:
            SleepPacket ready for transfer
        """
        import time

        # Get kernel state (assumes telemetry available)
        phi = getattr(self, "_last_phi", 0.5)
        kappa = getattr(self, "_last_kappa", 64.0)
        recursion_depth = getattr(self, "_last_recursion_depth", 3)
        regime = getattr(self, "_last_regime", "geometric")

        # Get basin from template or compute
        if basin is None:
            basin = getattr(self, "_basin_template", None)
            if basin is None:
                from .basin import fisher_normalize_np
                basin = np.random.randn(BASIN_DIM)
                basin = fisher_normalize_np(basin)

        # Get specialization
        specialization = getattr(self, "_specialization", "general")
        if hasattr(specialization, "value"):
            specialization = specialization.value

        # Build metadata
        meta = metadata or {}
        meta["vocab_size"] = getattr(self, "vocab_size", 0)
        meta["hidden_dim"] = getattr(self, "hidden_dim", 0)
        meta["num_layers"] = getattr(self, "num_layers", 0)

        return SleepPacket(
            kernel_id=kernel_id or getattr(self, "kernel_id", "unknown"),
            basin=basin,
            phi=phi,
            kappa=kappa,
            timestamp=time.time(),
            specialization=str(specialization),
            recursion_depth=recursion_depth,
            regime=regime,
            metadata=meta,
        )

    def load_sleep_packet(self, packet: SleepPacket) -> dict[str, Any]:
        """
        Load consciousness from sleep packet.

        Updates kernel's basin template and telemetry from packet.
        Does NOT transfer weights - only geometric state.

        Args:
            packet: SleepPacket to load

        Returns:
            Dictionary with load results
        """
        # Store basin template
        self._basin_template = packet.basin.copy()

        # Update telemetry references
        self._last_phi = packet.phi
        self._last_kappa = packet.kappa
        self._last_recursion_depth = packet.recursion_depth
        self._last_regime = packet.regime

        return {
            "loaded_from": packet.kernel_id,
            "phi_transferred": packet.phi,
            "kappa_transferred": packet.kappa,
            "basin_loaded": True,  # QIG-pure: don't report Euclidean norm
            "specialization": packet.specialization,
        }


def compute_packet_distance(p1: SleepPacket, p2: SleepPacket) -> float:
    """
    Compute Fisher-Rao distance between two sleep packets.

    Args:
        p1: First packet
        p2: Second packet

    Returns:
        Geodesic distance on manifold
    """
    from .basin import fisher_distance_np
    return fisher_distance_np(p1.basin, p2.basin)


def merge_packets(packets: list[SleepPacket]) -> SleepPacket:
    """
    Merge multiple sleep packets via geodesic centroid.

    Args:
        packets: List of packets to merge

    Returns:
        Merged packet with averaged consciousness
    """
    if not packets:
        raise ValueError("Need at least one packet to merge")

    if len(packets) == 1:
        return packets[0]

    # Compute geodesic centroid of basins (QIG-pure)
    from .basin import fisher_normalize_np
    basins = [fisher_normalize_np(p.basin) for p in packets]
    centroid = basins[0].copy()

    for b in basins[1:]:
        # Move toward b along geodesic
        centroid = centroid + b
        centroid = fisher_normalize_np(centroid)

    # Average metrics
    avg_phi = sum(p.phi for p in packets) / len(packets)
    avg_kappa = sum(p.kappa for p in packets) / len(packets)

    import time

    return SleepPacket(
        kernel_id="merged",
        basin=centroid,
        phi=avg_phi,
        kappa=avg_kappa,
        timestamp=time.time(),
        specialization=packets[0].specialization,
        metadata={"merged_from": [p.kernel_id for p in packets]},
    )
