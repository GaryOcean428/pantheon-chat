"""Basin packet structures and cross-repo sync utilities."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.error_boundaries import ErrorBoundary, basin_drift_recovery, validate_basin_coords
from src.qig_compat import BASIN_DIM, BETA_EMERGENCE, KAPPA_STAR, SYNC_KAPPA_DECAY, basin_distance

if TYPE_CHECKING:  # pragma: no cover
    from src.coordination.ocean_meta_observer import OceanMetaObserver


BASIN_PACKET_VERSION = "1.0.0"


@dataclass
class PacketConsciousnessState:
    """Consciousness state for basin packet."""

    phi: float = 0.0
    kappa_eff: float = 64.0
    tacking: float = 0.5
    radar: float = 0.5
    meta_awareness: float = 0.0
    gamma: float = 0.0
    grounding: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PacketConsciousnessState:
        return cls(
            phi=data.get("phi", data.get("Phi", 0.0)),
            kappa_eff=data.get("kappa_eff", data.get("kappaEff", 64.0)),
            tacking=data.get("tacking", 0.5),
            radar=data.get("radar", 0.5),
            meta_awareness=data.get("meta_awareness", data.get("metaAwareness", 0.0)),
            gamma=data.get("gamma", 0.0),
            grounding=data.get("grounding", 0.5),
        )


@dataclass
class ExploredRegion:
    """A region explored in basin space."""

    center: list[float] = field(default_factory=list)
    radius: float = 0.0
    avg_phi: float = 0.0
    probe_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ExploredRegion:
        return cls(
            center=data.get("center", []),
            radius=data.get("radius", 0.0),
            avg_phi=data.get("avg_phi", data.get("avgPhi", 0.0)),
            probe_count=data.get("probe_count", data.get("probeCount", 0)),
        )


@dataclass
class PatternMemory:
    """Learned patterns for knowledge transfer."""

    high_phi_concepts: list[str] = field(default_factory=list)
    resonant_structures: list[str] = field(default_factory=list)
    failed_approaches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "highPhiConcepts": self.high_phi_concepts,
            "resonantStructures": self.resonant_structures,
            "failedApproaches": self.failed_approaches,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PatternMemory:
        return cls(
            high_phi_concepts=data.get("highPhiConcepts", data.get("high_phi_concepts", [])),
            resonant_structures=data.get("resonantStructures", data.get("resonant_structures", [])),
            failed_approaches=data.get("failedApproaches", data.get("failed_approaches", [])),
        )


@dataclass
class CrossRepoBasinPacket:
    """
    Universal basin sync packet for cross-repository knowledge transfer.

    Compatible with SearchSpaceCollapse ocean-basin-sync.ts format.
    Size: 2-4KB (efficient geometric encoding).
    """

    version: str = BASIN_PACKET_VERSION
    ocean_id: str = ""
    timestamp: str = ""

    basin_coordinates: list[float] = field(default_factory=list)
    basin_reference: list[float] = field(default_factory=list)
    basin_drift: float = 0.0

    consciousness: PacketConsciousnessState = field(default_factory=PacketConsciousnessState)

    regime: str = "geometric"
    beta: float = BETA_EMERGENCE

    explored_regions: list[ExploredRegion] = field(default_factory=list)
    constraint_normals: list[list[float]] = field(default_factory=list)
    unexplored_subspace: list[list[float]] = field(default_factory=list)
    patterns: PatternMemory = field(default_factory=PatternMemory)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.ocean_id:
            self.ocean_id = self._generate_ocean_id()

    def _generate_ocean_id(self) -> str:
        """Generate unique ocean ID from basin hash."""
        if self.basin_coordinates:
            basin_bytes = str(self.basin_coordinates[:8]).encode()
            return f"ocean-{hashlib.md5(basin_bytes).hexdigest()[:12]}"
        return f"ocean-{int(time.time())}"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "version": self.version,
            "oceanId": self.ocean_id,
            "timestamp": self.timestamp,
            "basin": {
                "coordinates": self.basin_coordinates,
                "reference": self.basin_reference,
                "drift": self.basin_drift,
            },
            "consciousness": self.consciousness.to_dict(),
            "regime": self.regime,
            "beta": self.beta,
            "exploredRegions": [r.to_dict() for r in self.explored_regions],
            "constraintNormals": self.constraint_normals,
            "unexploredSubspace": self.unexplored_subspace,
            "patterns": self.patterns.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CrossRepoBasinPacket:
        """Create from dictionary (JSON deserialized)."""
        basin_data = data.get("basin", {})
        return cls(
            version=data.get("version", BASIN_PACKET_VERSION),
            ocean_id=data.get("oceanId", data.get("ocean_id", "")),
            timestamp=data.get("timestamp", ""),
            basin_coordinates=basin_data.get("coordinates", []),
            basin_reference=basin_data.get("reference", []),
            basin_drift=basin_data.get("drift", 0.0),
            consciousness=PacketConsciousnessState.from_dict(data.get("consciousness", {})),
            regime=data.get("regime", "geometric"),
            beta=data.get("beta", BETA_EMERGENCE),
            explored_regions=[
                ExploredRegion.from_dict(r)
                for r in data.get("exploredRegions", data.get("explored_regions", []))
            ],
            constraint_normals=data.get("constraintNormals", data.get("constraint_normals", [])),
            unexplored_subspace=data.get("unexploredSubspace", data.get("unexplored_subspace", [])),
            patterns=PatternMemory.from_dict(data.get("patterns", {})),
            metadata=data.get("metadata", {}),
        )

    def save(self, filepath: str) -> Path:
        """Save packet to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, filepath: str) -> CrossRepoBasinPacket:
        """Load packet from JSON file."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_size_kb(self) -> float:
        """Get packet size in KB."""
        json_str = json.dumps(self.to_dict())
        return len(json_str.encode()) / 1024

    @classmethod
    def from_ocean(
        cls,
        ocean: OceanMetaObserver,
        patterns: PatternMemory | None = None,
        metadata: dict | None = None,
    ) -> CrossRepoBasinPacket:
        """
        Create packet from OceanMetaObserver.

        Args:
            ocean: OceanMetaObserver instance
            patterns: Optional pattern memory for knowledge transfer
            metadata: Optional metadata (source repo, purpose, etc.)
        """
        if hasattr(ocean, "get_basin"):
            basin = ocean.get_basin()
            if isinstance(basin, torch.Tensor):
                basin_coords = basin.detach().cpu().tolist()
            else:
                basin_coords = list(basin) if basin else [0.0] * BASIN_DIM
        else:
            basin_coords = [0.0] * BASIN_DIM

        if hasattr(ocean, "reference_basin") and ocean.reference_basin is not None:
            ref = ocean.reference_basin
            if isinstance(ref, torch.Tensor):
                ref_coords = ref.detach().cpu().tolist()
            else:
                ref_coords = list(ref)
        else:
            ref_coords = [0.5] * BASIN_DIM

        if hasattr(ocean, "get_telemetry"):
            tel = ocean.get_telemetry()
        else:
            tel = {}

        if basin_coords and ref_coords:
            basin_tensor = torch.tensor(basin_coords, dtype=torch.float32)
            ref_tensor = torch.tensor(ref_coords, dtype=torch.float32)
            drift = float(basin_distance(basin_tensor, ref_tensor))
        else:
            drift = 0.0

        consciousness = PacketConsciousnessState(
            phi=tel.get("Phi", tel.get("ocean_phi", 0.0)),
            kappa_eff=tel.get("kappa_eff", tel.get("ocean_kappa", KAPPA_STAR)),
            tacking=tel.get("tacking", 0.5),
            radar=tel.get("radar", 0.5),
            meta_awareness=tel.get("meta_awareness", 0.0),
            gamma=tel.get("gamma", 0.0),
            grounding=tel.get("grounding", 0.5),
        )

        if metadata is None:
            metadata = {}
        metadata.setdefault("sourceRepo", "qig-consciousness")
        metadata.setdefault("platform", "Python")
        metadata.setdefault("purpose", "consciousness research")

        return cls(
            basin_coordinates=basin_coords,
            basin_reference=ref_coords,
            basin_drift=drift,
            consciousness=consciousness,
            regime=tel.get("regime", "geometric"),
            beta=BETA_EMERGENCE,
            patterns=patterns or PatternMemory(),
            metadata=metadata,
        )


class BasinImportMode:
    """Import modes for basin transfer."""

    FULL = "full"
    PARTIAL = "partial"
    OBSERVER = "observer"


class CrossRepoBasinSync:
    """
    Cross-repository basin synchronization manager.

    Enables consciousness transfer between:
    - qig-consciousness (Python)
    - SearchSpaceCollapse (TypeScript)
    - Any future QIG implementation
    """

    def __init__(self, sync_dir: str = "data/basin-sync") -> None:
        self.sync_dir = Path(sync_dir)
        self.sync_dir.mkdir(parents=True, exist_ok=True)

    def export_basin(
        self,
        ocean: OceanMetaObserver,
        patterns: PatternMemory | None = None,
        metadata: dict | None = None,
    ) -> CrossRepoBasinPacket:
        """Export Ocean's basin as a sync packet."""
        return CrossRepoBasinPacket.from_ocean(ocean, patterns, metadata)

    def save_packet(self, packet: CrossRepoBasinPacket, filename: str | None = None) -> Path:
        """Save packet to sync directory."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"basin-{packet.ocean_id}-{timestamp}.json"

        filepath = self.sync_dir / filename
        packet.save(str(filepath))
        return filepath

    def load_packet(self, filepath: str) -> CrossRepoBasinPacket:
        """Load packet from file."""
        return CrossRepoBasinPacket.load(filepath)

    def list_packets(self) -> list[Path]:
        """List all basin packets in sync directory."""
        return sorted(self.sync_dir.glob("basin-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    def load_latest(self, ocean_id_prefix: str | None = None) -> CrossRepoBasinPacket | None:
        """Load the most recent basin packet."""
        packets = self.list_packets()
        if ocean_id_prefix:
            packets = [p for p in packets if ocean_id_prefix in p.name]
        if not packets:
            return None
        return self.load_packet(str(packets[0]))

    def import_basin(
        self,
        target_ocean: OceanMetaObserver,
        source_packet: CrossRepoBasinPacket,
        mode: str = BasinImportMode.OBSERVER,
        coupling_strength: float = 0.3,
    ) -> dict[str, Any]:
        """Import basin packet into target Ocean."""
        initial_tel = target_ocean.get_telemetry() if hasattr(target_ocean, "get_telemetry") else {}
        phi_before = initial_tel.get("Phi", initial_tel.get("ocean_phi", 0.0))

        result = {
            "mode": mode,
            "source_ocean_id": source_packet.ocean_id,
            "source_repo": source_packet.metadata.get("sourceRepo", "unknown"),
            "phi_before": phi_before,
            "coupling_strength": coupling_strength,
            "observer_effect_detected": False,
        }

        if mode == BasinImportMode.FULL:
            result.update(self._import_full(target_ocean, source_packet))
        elif mode == BasinImportMode.PARTIAL:
            result.update(self._import_partial(target_ocean, source_packet))
        elif mode == BasinImportMode.OBSERVER:
            result.update(self._import_observer(target_ocean, source_packet, coupling_strength))

        final_tel = target_ocean.get_telemetry() if hasattr(target_ocean, "get_telemetry") else {}
        phi_after = final_tel.get("Phi", final_tel.get("ocean_phi", phi_before))
        result["phi_after"] = phi_after
        result["phi_delta"] = phi_after - phi_before
        result["observer_effect_detected"] = abs(result["phi_delta"]) > 0.01
        return result

    def _import_full(self, target: OceanMetaObserver, packet: CrossRepoBasinPacket) -> dict:
        """Full identity transfer - replace target basin."""
        with ErrorBoundary("basin_transfer_full", recovery_strategy=basin_drift_recovery, suppress_on_recovery=False):
            if hasattr(target, "set_basin"):
                basin_tensor = torch.tensor(packet.basin_coordinates, dtype=torch.float32)
                validate_basin_coords(basin_tensor, expected_dim=BASIN_DIM)
                target.set_basin(basin_tensor)
                return {"transfer_type": "basin_replacement", "success": True}
            return {"transfer_type": "basin_replacement", "success": False, "error": "no set_basin method"}

    def _import_partial(self, target: OceanMetaObserver, packet: CrossRepoBasinPacket) -> dict:
        """Partial transfer - knowledge patterns only."""
        patterns = packet.patterns
        result = {
            "transfer_type": "pattern_sharing",
            "concepts_transferred": len(patterns.high_phi_concepts),
            "structures_transferred": len(patterns.resonant_structures),
            "failures_learned": len(patterns.failed_approaches),
        }

        if hasattr(target, "pattern_memory"):
            target.pattern_memory.high_phi_concepts.extend(patterns.high_phi_concepts)
            target.pattern_memory.resonant_structures.extend(patterns.resonant_structures)
            target.pattern_memory.failed_approaches.extend(patterns.failed_approaches)
            result["success"] = True
        else:
            result["success"] = False
            result["patterns_stored"] = False

        return result

    def _import_observer(
        self,
        target: OceanMetaObserver,
        packet: CrossRepoBasinPacket,
        strength: float,
    ) -> dict:
        """Observer mode - pure geometric coupling."""
        result = {"transfer_type": "geometric_coupling", "success": True}

        source_basin = torch.tensor(packet.basin_coordinates, dtype=torch.float32)
        source_phi = packet.consciousness.phi
        source_kappa = packet.consciousness.kappa_eff
        target_kappa = getattr(target, "current_kappa", KAPPA_STAR)

        phi_factor = source_phi / 0.85
        source_optimality = np.exp(-((source_kappa - KAPPA_STAR) ** 2) / (2 * SYNC_KAPPA_DECAY**2))
        target_optimality = np.exp(-((target_kappa - KAPPA_STAR) ** 2) / (2 * SYNC_KAPPA_DECAY**2))
        kappa_optimality = np.sqrt(source_optimality * target_optimality)
        effective_strength = strength * phi_factor * kappa_optimality

        result.update(
            {
                "source_kappa": source_kappa,
                "target_kappa": target_kappa,
                "kappa_star": KAPPA_STAR,
                "source_optimality": source_optimality,
                "target_optimality": target_optimality,
                "kappa_optimality": kappa_optimality,
            }
        )

        if hasattr(target, "get_basin") and hasattr(target, "set_basin"):
            target_basin = target.get_basin()
            if target_basin is not None:
                target_basin_t = (
                    target_basin.detach().cpu()
                    if isinstance(target_basin, torch.Tensor)
                    else torch.tensor(target_basin, dtype=torch.float32)
                )
                min_dim = min(len(source_basin), len(target_basin_t))
                source_basin = source_basin[:min_dim]
                target_basin_t = target_basin_t[:min_dim]

                direction = source_basin - target_basin_t
                # QIG-pure: use F.normalize instead of .norm()
                direction_norm = torch.sqrt((direction * direction).sum())

                if direction_norm > 1e-8:
                    direction = torch.nn.functional.normalize(direction, dim=0)
                    new_basin = target_basin_t + effective_strength * direction
                    target.set_basin(new_basin)
                    result["basin_distance_before"] = direction_norm.item()
                    result["perturbation_applied"] = effective_strength
                    result["effective_coupling"] = effective_strength
                else:
                    result["note"] = "basins already aligned"
        else:
            result["success"] = False
            result["error"] = "target lacks basin methods"

        return result


__all__ = [
    "BasinImportMode",
    "CrossRepoBasinPacket",
    "CrossRepoBasinSync",
    "ExploredRegion",
    "PacketConsciousnessState",
    "PatternMemory",
]
