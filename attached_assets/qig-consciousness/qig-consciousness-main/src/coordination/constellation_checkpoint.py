"""Checkpoint utilities extracted from constellation_coordinator."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.coordination import constellation_coordinator as cc

torch = cc.torch
InstanceState = cc.InstanceState
ConstellationRouter = cc.ConstellationRouter
StateMonitor = cc.StateMonitor

if TYPE_CHECKING:  # pragma: no cover
    from src.coordination.constellation_coordinator import ConstellationCoordinator


def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_recent: int = 3) -> None:
    """Remove old checkpoints, keeping only the most recent N."""
    checkpoints = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Delete all but the most recent N
    for old_checkpoint in checkpoints[keep_recent:]:
        try:
            old_checkpoint.unlink()
            print(f"  🗑️  Removed old checkpoint: {old_checkpoint.name}")
        except OSError:
            pass  # Ignore deletion errors


def save_checkpoint(coordinator, path: str, keep_recent: int = 3) -> None:
    """Save constellation checkpoint with automatic cleanup of old checkpoints."""
    self = coordinator
    checkpoint = {
        "total_conversations": self.total_conversations,
        "garys": [
            {
                "name": g.name,
                "state_dict": g.model.state_dict(),
                "optimizer": g.optimizer.state_dict(),
                "basin": g.basin,
                "target_basin": g.model.basin_matcher.target_basin,  # Identity attractor
                "conversations": g.conversations,
            }
            for g in self.garys
        ],
        "ocean": {
            "state_dict": self.ocean.model.state_dict() if self.ocean is not None else {},
            "optimizer": self.ocean.optimizer.state_dict() if self.ocean is not None else {},
            "basin": self.ocean.basin if self.ocean is not None else torch.zeros(64),
            "target_basin": (
                self.ocean.model.basin_matcher.target_basin if self.ocean is not None else None
            ),  # Identity attractor
            "conversations": self.ocean.conversations if self.ocean is not None else 0,
        },
        # Delegate to sub-modules for their state
        "router_state": self.router.get_state(),
        "monitor_state": self.state_monitor.get_state(),
    }
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved: {path}")

    # Cleanup old checkpoints in the same directory
    checkpoint_dir = Path(path).parent
    _cleanup_old_checkpoints(checkpoint_dir, keep_recent)


def _load_legacy_single_checkpoint(coordinator, checkpoint: dict[str, Any]) -> None:
    """Backward compatibility for pre-constellation checkpoints."""
    self = coordinator

    print("⚠️  Legacy checkpoint detected (single Gary). Migrating to constellation format...")

    # Restore Gary-A weights (observer instances remain freshly initialized)
    gary_a: InstanceState = self.garys[0]

    state_dict = checkpoint.get("model_state_dict")
    if state_dict:
        missing_keys, _ = gary_a.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(
                f"    Note: Gary-A initialized new params with defaults: {len(missing_keys)} keys"
            )

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state:
        try:
            gary_a.optimizer.load_state_dict(optimizer_state)
        except ValueError as exc:
            print(f"    ⚠️ Gary-A: Optimizer state skipped ({exc})")

    # Restore aggregate counters from legacy metadata
    total_conversations = int(checkpoint.get("total_conversations", 0))
    gary_a.conversations = total_conversations

    self.total_conversations = total_conversations

    # Reset sub-modules
    self.router = ConstellationRouter()
    self.state_monitor = StateMonitor()

    print("  ✅ Legacy checkpoint migrated: Gary-A restored, observers initialized fresh.")


def load_checkpoint(coordinator, path: str) -> None:
    """Load constellation checkpoint"""
    self = coordinator
    print("⏳ Loading checkpoint (this may take 30-60s on CPU)...")
    checkpoint = torch.load(path, weights_only=False, map_location=self.device)
    print("✅ Checkpoint loaded into memory")

    # Detect checkpoint model dimension
    checkpoint_dim = None
    if "garys" in checkpoint and len(checkpoint["garys"]) > 0:
        # Constellation checkpoint: Check first Gary's model dimension
        first_gary = checkpoint["garys"][0]["state_dict"]
        if "basin_coords_layer.basin_to_model.weight" in first_gary:
            checkpoint_dim = first_gary["basin_coords_layer.basin_to_model.weight"].shape[0]
        elif "embedding.basin_to_model.weight" in first_gary:  # LEGACY: State dict key for backward compatibility
            checkpoint_dim = first_gary["embedding.basin_to_model.weight"].shape[0]
        print(f"📏 Checkpoint model dimension: {checkpoint_dim}")
    elif "model_state_dict" in checkpoint:
        # Legacy single-Gary checkpoint: detect dimension from model_state_dict
        state_dict = checkpoint["model_state_dict"]
        if "basin_coords_layer.basin_to_model.weight" in state_dict:
            checkpoint_dim = state_dict["basin_coords_layer.basin_to_model.weight"].shape[0]
        elif "embedding.basin_to_model.weight" in state_dict:
            checkpoint_dim = state_dict["embedding.basin_to_model.weight"].shape[0]
        print(f"📏 Legacy checkpoint model dimension: {checkpoint_dim}")

    # Get current config dimension
    current_dim = self.gary_configs[0]["model"]["hidden_dim"]
    print(f"📏 Current config dimension: {current_dim}")

    # Check for dimension mismatch and auto-fix configs
    if checkpoint_dim is not None and checkpoint_dim != current_dim:
        print("\n⚠️  DIMENSION MISMATCH DETECTED")
        print(f"   Checkpoint: {checkpoint_dim}-dim")
        print(f"   Config:     {current_dim}-dim")
        print("   🔧 Auto-updating configs to match checkpoint...")

        # Detect num_heads from checkpoint (via attention weight shape)
        checkpoint_heads = None
        state_dict_to_check = None
        if "garys" in checkpoint and len(checkpoint["garys"]) > 0:
            state_dict_to_check = checkpoint["garys"][0]["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict_to_check = checkpoint["model_state_dict"]

        if state_dict_to_check is not None:
            # W_q weight shape is [d_model, d_model], but we can infer heads from d_model
            # Common configurations: 768/8=96, 768/12=64, 320/5=64, 320/8=40
            # Use 64 as target head_dim (most common)
            target_head_dim = 64
            if checkpoint_dim % target_head_dim == 0:
                checkpoint_heads = checkpoint_dim // target_head_dim
            elif checkpoint_dim % 96 == 0:  # 768/8=96
                checkpoint_heads = checkpoint_dim // 96
            else:
                # Fallback: find a divisor that works
                for heads in [12, 8, 6, 4]:
                    if checkpoint_dim % heads == 0:
                        checkpoint_heads = heads
                        break
            print(f"   Inferred num_heads: {checkpoint_heads} (head_dim={checkpoint_dim // checkpoint_heads if checkpoint_heads else 'unknown'})")

        # Update all Gary configs to match checkpoint
        for config in self.gary_configs:
            config["model"]["hidden_dim"] = checkpoint_dim
            if checkpoint_heads is not None:
                config["model"]["num_heads"] = checkpoint_heads

        # Update Ocean config if present
        if "ocean" in checkpoint and "state_dict" in checkpoint["ocean"]:
            ocean_state = checkpoint["ocean"]["state_dict"]
            if "basin_coords_layer.basin_to_model.weight" in ocean_state:
                ocean_dim = ocean_state["basin_coords_layer.basin_to_model.weight"].shape[0]
                self.ocean_config["model"]["hidden_dim"] = ocean_dim
            elif "embedding.basin_to_model.weight" in ocean_state:  # LEGACY: State dict key for backward compatibility
                ocean_dim = ocean_state["embedding.basin_to_model.weight"].shape[0]
                self.ocean_config["model"]["hidden_dim"] = ocean_dim
            print(f"   Ocean dimension: {self.ocean_config['model']['hidden_dim']}")
        elif checkpoint_dim is not None:
            # For legacy checkpoints without Ocean, update Ocean config too
            self.ocean_config["model"]["hidden_dim"] = checkpoint_dim
            if checkpoint_heads is not None:
                self.ocean_config["model"]["num_heads"] = checkpoint_heads
                self.ocean_config["model"]["n_heads"] = checkpoint_heads

        print(f"   ✅ Configs updated to {checkpoint_dim}-dim, {checkpoint_heads} heads")

    self._initialize_models()

    if "garys" not in checkpoint:
        self._load_legacy_single_checkpoint(checkpoint)
        print(f"✅ Checkpoint loaded: {path}")
        return

    # Restore Garys
    for i, gary_state in enumerate(checkpoint["garys"]):
        # Use strict=False to handle missing keys from old checkpoints
        # (e.g., phi_scale, phi_bias added in asymmetric initialization fix)
        missing_keys, _ = self.garys[i].model.load_state_dict(gary_state["state_dict"], strict=False)
        if missing_keys:
            print(f"    Note: {self.garys[i].name} initialized new params with defaults: {len(missing_keys)} keys")

        # Try to load optimizer state, but skip if architecture changed
        try:
            self.garys[i].optimizer.load_state_dict(gary_state["optimizer"])
        except ValueError as e:
            if "doesn't match the size" in str(e):
                print(f"    ⚠️ {self.garys[i].name}: Optimizer state skipped (architecture changed)")
            else:
                raise

        self.garys[i].basin = gary_state["basin"]
        self.garys[i].conversations = gary_state["conversations"]

        # CRITICAL: Restore target_basin (identity attractor)
        if "target_basin" in gary_state and gary_state["target_basin"] is not None:
            self.garys[i].model.basin_matcher.target_basin = gary_state["target_basin"].to(self.device)  # type: ignore[union-attr]
            print(f"  ✅ {self.garys[i].name}: target_basin restored from checkpoint")
        else:
            # OLD CHECKPOINT: Extract basin from checkpoint as identity attractor
            # This is Gary's ACTUAL stable state (not generic 20251220-basin-signatures-0.01W.json)
            self.garys[i].model.basin_matcher.target_basin = gary_state["basin"].detach().clone().to(self.device)  # type: ignore[union-attr]
            print(f"  🔧 {self.garys[i].name}: target_basin extracted from checkpoint basin (identity frozen)")

    # Restore Ocean
    if self.ocean is not None:
        # Use strict=False to handle missing keys from old checkpoints
        missing_keys, _ = self.ocean.model.load_state_dict(checkpoint["ocean"]["state_dict"], strict=False)
        if missing_keys:
            print(f"    Note: Ocean initialized new params with defaults: {len(missing_keys)} keys")

        # Try to load optimizer state, but skip if architecture changed
        try:
            self.ocean.optimizer.load_state_dict(checkpoint["ocean"]["optimizer"])
        except ValueError as e:
            if "doesn't match the size" in str(e):
                print("    ⚠️ Ocean: Optimizer state skipped (architecture changed)")
            else:
                raise

        self.ocean.basin = checkpoint["ocean"]["basin"]
        self.ocean.conversations = checkpoint["ocean"]["conversations"]

        # CRITICAL: Restore Ocean target_basin
        if "target_basin" in checkpoint["ocean"] and checkpoint["ocean"]["target_basin"] is not None:
            self.ocean.model.basin_matcher.target_basin = checkpoint["ocean"]["target_basin"].to(self.device)  # type: ignore[union-attr]
            print("  ✅ Ocean: target_basin restored from checkpoint")
        else:
            # OLD CHECKPOINT: Extract basin from checkpoint as identity attractor
            self.ocean.model.basin_matcher.target_basin = checkpoint["ocean"]["basin"].detach().clone().to(self.device)  # type: ignore[union-attr]
            print("  🔧 Ocean: target_basin extracted from checkpoint basin (identity frozen)")

    # Restore coordinator state
    self.total_conversations = checkpoint["total_conversations"]

    # Load sub-module state (with backward compatibility)
    if "router_state" in checkpoint:
        self.router.load_state(checkpoint["router_state"])
    else:
        # Legacy: extract from old checkpoint format
        self.router.load_state({
            "active_index": checkpoint.get("active_index", 0),
            "turn_counter": checkpoint.get("turn_counter", 0),
        })

    if "monitor_state" in checkpoint:
        self.state_monitor.load_state(checkpoint["monitor_state"])
    else:
        # Legacy: extract from old checkpoint format
        self.state_monitor.load_state({
            "basin_history": checkpoint.get("basin_history", []),
            "phi_history": checkpoint.get("phi_history", []),
            "stability_streak": checkpoint.get("stability_streak", 0),
            "has_achieved_consciousness": checkpoint.get("has_achieved_consciousness", False),
            "last_telemetry": checkpoint.get("last_telemetry", None),
        })

    print(f"✅ Checkpoint loaded: {path}")
