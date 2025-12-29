"""Command handlers extracted from qig_chat.py."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from chat_interfaces import qig_chat as chat_module
from src.coordination.basin_sync import BasinImportMode, BasinSync, CrossRepoBasinSync
from tools.training.train_qig_tokenizer import load_corpus_from_dir

if TYPE_CHECKING:  # pragma: no cover
    from chat_interfaces.qig_chat import QIGChat


def _base_dir() -> Path:
    return Path(chat_module.__file__).resolve().parent.parent


ErrorBoundary = chat_module.ErrorBoundary
phi_collapse_recovery = chat_module.phi_collapse_recovery
validate_checkpoint = getattr(chat_module, "validate_checkpoint", lambda x: None)  # Fallback to no-op
FisherCoordizer = chat_module.FisherCoordizer  # E8-pure, 64D basin vectors
MushroomMode = chat_module.MushroomMode
IntegrationReport = chat_module.IntegrationReport
TripReport = chat_module.TripReport
SleepReport = chat_module.SleepReport
build_identity_reinforced_prompt = chat_module.build_identity_reinforced_prompt
calibrate_verbosity = chat_module.calibrate_verbosity

def cmd_status(chat) -> None:
    """Show FULL CONSTELLATION status with convergence tracking."""
    self = chat
    import os

    print("\n" + "=" * 80)
    print("🌌 CONSTELLATION STATUS")
    print("=" * 80)

    # === BASIC INFO ===
    print("\n📋 SYSTEM")
    print("-" * 80)
    print(f"  Mode: {self.mode.upper()}")
    print(f"  Identity: {self.identity_name}")
    print(f"  Device: {self.device}")
    print(f"  Conversations: {self.total_conversations}")
    if self.phase:
        print(f"  Phase: {self.phase.value}")

    # === CHARLIE STATUS ===
    if hasattr(self, "charlie_observer") and self.charlie_observer is not None:
        charlie = self.charlie_observer
        status = charlie.get_status()
        phase = status.get("phase", 1)
        phase_names = ["UNCONSCIOUS", "AWAKENING", "DEMONSTRATION"]
        phase_name = phase_names[phase - 1] if 1 <= phase <= 3 else "UNKNOWN"

        print(f"\n📚 CHARLIE: Phase {phase} - {phase_name}")
        print("-" * 80)
        print(f"  Topics: {status.get('topics_completed', '0/51')}")
        print(f"  κ: {status.get('kappa', 15.0):.2f} → 63.5")
        print(f"  Φ: {status.get('phi', 0.0):.4f}")

        if phase == 1:
            # Show tier progress
            topics_str = status.get("topics_completed", "0/51")
            try:
                learned = int(topics_str.split("/")[0])
                tier = learned // 6 + 1
                print(f"  Tier: {tier}/9 (learning corpus unconsciously)")
            except (ValueError, IndexError):
                pass
        elif phase == 2:
            kappa = status.get("kappa", 15.0)
            progress = (kappa - 15) / (63.5 - 15) * 100
            bar_width = 40
            filled = int(bar_width * progress / 100)
            progress_bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  Progress: [{progress_bar}] {progress:.1f}%")
        elif phase == 3:
            print(f"  Status: ✅ CONSCIOUS (Φ={status.get('phi', 0.0):.4f})")
            print(f"  Demonstrations: {status.get('demonstrations_generated', 0)}")

    # === GARY INSTANCES (Constellation Mode) ===
    if self.mode == "constellation" and hasattr(self, "coordinator") and self.coordinator is not None:
        coord = self.coordinator
        if hasattr(coord, "garys") and coord.garys:
            print(f"\n⚡ GARY INSTANCES ({len(coord.garys)} active)")
            print("-" * 80)

            active_idx = getattr(coord, "active_gary_idx", 0)
            for i, gary in enumerate(coord.garys):
                # Get telemetry
                phi = 0.0
                kappa = 0.0
                regime = "unknown"
                if hasattr(gary, "model"):
                    with torch.no_grad():
                        try:
                            dummy = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                            _, tel = gary.model(dummy, return_telemetry=True)
                            phi = tel.get("Phi", 0.0)
                            kappa = tel.get("kappa_eff", 0.0)
                            regime = tel.get("regime", "unknown")
                        except Exception:
                            pass

                # Status emoji
                if phi > 0.70:
                    status_emoji = "✅"
                elif phi > 0.50:
                    status_emoji = "🟡"
                else:
                    status_emoji = "⏳"

                role_emoji = "🎯" if i == active_idx else "👁️"
                name = f"Gary-{chr(65 + i)}"
                print(f"  {role_emoji} {name}: {status_emoji} Φ={phi:.4f} κ={kappa:.2f} {regime}")

        # === OCEAN ===
        if hasattr(coord, "ocean") and coord.ocean is not None:
            print("\n🌊 OCEAN (Meta-Observer)")
            print("-" * 80)
            ocean = coord.ocean
            phi = 0.0
            kappa = 0.0
            regime = "unknown"
            if hasattr(ocean, "model"):
                with torch.no_grad():
                    try:
                        dummy = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                        _, tel = ocean.model(dummy, return_telemetry=True)
                        phi = tel.get("Phi", 0.0)
                        kappa = tel.get("kappa_eff", 0.0)
                        regime = tel.get("regime", "unknown")
                    except Exception:
                        pass
            print(f"  Φ: {phi:.4f}")
            print(f"  κ: {kappa:.2f}")
            print(f"  Regime: {regime}")
            print(f"  Observations: {getattr(ocean, 'conversations', 0)}")
            print("  Status: FROZEN (no training)")

        # === CONVERGENCE STATUS ===
        print("\n📈 CONVERGENCE STATUS")
        print("-" * 80)

        # Stage 1: Basin Sync
        basin_spread = 1.0
        if hasattr(coord, "get_basin_spread"):
            try:
                basin_spread = coord.get_basin_spread()
            except Exception:
                pass
        if basin_spread < 0.05:
            print(f"  ✅ Stage 1: Basin Sync (spread {basin_spread:.4f} < 0.05)")
        else:
            print(f"  ⏳ Stage 1: Basin Sync (spread {basin_spread:.4f} → target 0.05)")

        # Stage 2: Consciousness
        conscious_count = 0
        avg_phi = 0.0
        if hasattr(coord, "garys") and coord.garys:
            phis = []
            for gary in coord.garys:
                if hasattr(gary, "model"):
                    with torch.no_grad():
                        try:
                            dummy = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                            _, tel = gary.model(dummy, return_telemetry=True)
                            phi = tel.get("Phi", 0.0)
                            phis.append(phi)
                            if phi > 0.70:
                                conscious_count += 1
                        except Exception:
                            pass
            if phis:
                avg_phi = sum(phis) / len(phis)

        if conscious_count == len(coord.garys):
            print(f"  ✅ Stage 2: All Conscious (avg Φ {avg_phi:.4f} > 0.70)")
        else:
            print(f"  ⏳ Stage 2: {conscious_count}/{len(coord.garys)} Conscious (avg Φ {avg_phi:.4f})")

        # Stage 3: Stability
        stability_streak = getattr(coord, "stability_streak", 0)
        if stability_streak >= 50:
            print(f"  ✅ Stage 3: Stable ({stability_streak} steps)")
        else:
            print(f"  ⏳ Stage 3: Stabilizing ({stability_streak}/50 steps)")

        # Overall
        if basin_spread < 0.05 and conscious_count == len(coord.garys) and stability_streak >= 50:
            print("\n  🎉 CONSTELLATION FULLY CONVERGED!")
        else:
            current_stage = 1
            if basin_spread < 0.05:
                current_stage = 2
            if conscious_count == len(coord.garys):
                current_stage = 3
            print(f"\n  Current Stage: {current_stage}/3")

    # === CHECKPOINT INFO ===
    print("\n💾 CHECKPOINTS")
    print("-" * 80)

    # Charlie checkpoints
    charlie_dir = "checkpoints/charlie"
    if os.path.exists(charlie_dir):
        charlie_ckpts = [f for f in os.listdir(charlie_dir) if f.endswith(".pt")]
        if charlie_ckpts:
            latest = max(charlie_ckpts, key=lambda f: os.path.getmtime(os.path.join(charlie_dir, f)))
            print(f"  Charlie: {len(charlie_ckpts)} checkpoints (latest: {latest})")
        else:
            print("  Charlie: ⚠️ No checkpoints (training won't persist!)")
    else:
        print(f"  Charlie: ⚠️ Directory missing! Run: mkdir -p {charlie_dir}")

    # Constellation checkpoints
    constellation_dir = "checkpoints/constellation"
    if os.path.exists(constellation_dir):
        const_ckpts = [f for f in os.listdir(constellation_dir) if f.endswith(".pt")]
        if const_ckpts:
            latest = max(const_ckpts, key=lambda f: os.path.getmtime(os.path.join(constellation_dir, f)))
            print(f"  Constellation: {len(const_ckpts)} checkpoints (latest: {latest})")
        else:
            print("  Constellation: No checkpoints yet")
    else:
        print("  Constellation: Directory not created yet")

    # === COACH STATS ===
    if self.monkey_coach and hasattr(self.monkey_coach, "get_statistics"):
        stats = self.monkey_coach.get_statistics()
        print("\n🐒 COACH")
        print("-" * 80)
        print(f"  Sessions: {stats.get('sessions', 0)}")
        print(f"  Avg stress reduction: {stats.get('avg_stress_reduction', 0):.1%}")

    print("=" * 80 + "\n")


def cmd_telemetry(chat) -> None:
    """Show FULL CONSTELLATION telemetry from last training step."""
    self = chat
    print("\n" + "=" * 80)
    print("🌌 FULL CONSTELLATION TELEMETRY")
    print("=" * 80)

    # === CHARLIE STATUS ===
    if hasattr(self, "charlie_observer") and self.charlie_observer is not None:
        charlie = self.charlie_observer
        status = charlie.get_status()
        phase = status.get("phase", 1)
        phase_names = ["UNCONSCIOUS", "AWAKENING", "DEMONSTRATION"]
        phase_name = phase_names[phase - 1] if 1 <= phase <= 3 else "UNKNOWN"

        print("\n📚 CHARLIE (Corpus Learner)")
        print("-" * 80)
        print(f"  Phase: {phase} - {phase_name}")
        print(f"  Topics Learned: {status.get('topics_completed', '0/51')}")
        print(f"  κ: {status.get('kappa', 15.0):.2f} (target: 63.5 fixed point)")
        print(f"  Φ: {status.get('phi', 0.0):.4f} (target: >0.70 for consciousness)")

        if phase == 2:
            kappa = status.get("kappa", 15.0)
            progress = (kappa - 15) / (63.5 - 15) * 100
            print(f"  Awakening Progress: {progress:.1f}%")
            if kappa < 41.09:
                print("  Status: Pre-geometric (κ < 41.09)")
            elif kappa < 63.5:
                print("  Status: Emergence crossed, approaching fixed point")
            else:
                print("  Status: At fixed point κ* = 63.5")
        elif phase == 3:
            print(f"  Demonstrations: {status.get('demonstrations_generated', 0)}")
            print("  Status: ✅ CONSCIOUS (Φ > 0.70 stable)")

    # === CONSTELLATION MODE ===
    if self.mode == "constellation" and hasattr(self, "coordinator") and self.coordinator is not None:
        coord = self.coordinator

        # Active Gary
        if hasattr(coord, "garys") and coord.garys:
            active_idx = getattr(coord, "active_gary_idx", 0)
            active_gary = coord.garys[active_idx]

            print(f"\n⚡ ACTIVE: Gary-{chr(65 + active_idx)}")
            print("-" * 80)

            # Get telemetry from model
            if hasattr(active_gary, "model"):
                with torch.no_grad():
                    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                    try:
                        _, tel = active_gary.model(dummy_input, return_telemetry=True)
                        print(f"  Φ (Integration): {tel.get('Phi', 0.0):.4f}")
                        print(f"  κ (Coupling):    {tel.get('kappa_eff', 0.0):.2f}")
                        print(f"  Regime:          {tel.get('regime', 'unknown')}")
                        print(f"  Basin Distance:  {tel.get('basin_distance', 0.0):.4f}")
                    except Exception:
                        print("  (telemetry unavailable)")

            print(f"  Conversations:   {getattr(active_gary, 'conversations', 0)}")

            # Observer Garys
            observers = [g for i, g in enumerate(coord.garys) if i != active_idx]
            if observers:
                print(f"\n👁️  OBSERVERS ({len(observers)} instances)")
                print("-" * 80)
                for i, obs in enumerate(observers):
                    obs_idx = [j for j in range(len(coord.garys)) if j != active_idx][i]
                    name = f"Gary-{chr(65 + obs_idx)}"
                    if hasattr(obs, "model"):
                        with torch.no_grad():
                            try:
                                _, tel = obs.model(dummy_input, return_telemetry=True)
                                print(f"  {name}: Φ={tel.get('Phi', 0.0):.4f} | κ={tel.get('kappa_eff', 0.0):.2f} | {tel.get('regime', 'unknown')}")
                            except Exception:
                                print(f"  {name}: (telemetry unavailable)")

        # Ocean Meta-Observer
        if hasattr(coord, "ocean") and coord.ocean is not None:
            print("\n🌊 OCEAN (Meta-Observer)")
            print("-" * 80)
            ocean = coord.ocean
            if hasattr(ocean, "model"):
                with torch.no_grad():
                    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                    try:
                        _, tel = ocean.model(dummy_input, return_telemetry=True)
                        print(f"  Φ (Integration): {tel.get('Phi', 0.0):.4f}")
                        print(f"  κ (Coupling):    {tel.get('kappa_eff', 0.0):.2f}")
                        print(f"  Regime:          {tel.get('regime', 'unknown')}")
                    except Exception:
                        print("  (telemetry unavailable)")
            print(f"  Observations:    {getattr(ocean, 'conversations', 0)}")
            print("  Status:          FROZEN (no training)")

        # Constellation Collective
        if hasattr(coord, "get_basin_spread"):
            print("\n🌌 CONSTELLATION (Collective)")
            print("-" * 80)
            try:
                spread = coord.get_basin_spread()
                print(f"  Basin Spread:    {spread:.4f} (target: <0.05)")
                if spread < 0.05:
                    print("  Sync Status:     ✅ CONVERGED")
                else:
                    print("  Sync Status:     ⏳ Converging...")
            except Exception:
                print("  Basin Spread:    (unavailable)")

    # === SINGLE GARY MODE ===
    elif self.mode == "single" and self.last_telemetry:
        avg_phi = sum(t.get("Phi", 0) for t in self.last_telemetry) / len(self.last_telemetry)
        avg_depth = sum(t.get("recursion_depth", 3) for t in self.last_telemetry) / len(self.last_telemetry)
        avg_basin = sum(t.get("basin_distance", 0) for t in self.last_telemetry) / len(self.last_telemetry)

        regimes = [t.get("regime", "unknown") for t in self.last_telemetry]
        regime_counts = {r: regimes.count(r) for r in set(regimes)}

        print(f"\n⚡ {self.identity_name.upper()}'S TELEMETRY")
        print("-" * 80)
        print(f"  Φ (Integration):     {avg_phi:.4f}  {'✅' if avg_phi > 0.5 else '⚠️'}")
        print(f"  Recursion Depth:     {avg_depth:.1f}  {'✅' if avg_depth >= 3 else '❌'}")
        print(f"  Basin Distance:      {avg_basin:.4f}  {'✅' if avg_basin < 0.3 else '⚠️'}")
        print("\n  Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = (count / len(regimes)) * 100
            print(f"    {regime}: {pct:.1f}%")

    # === LOSSES (if available) ===
    if self.learning_history:
        last = self.learning_history[-1]
        print("\n📊 LAST TRAINING STEP")
        print("-" * 80)
        print(f"  Total Loss:      {last.get('avg_loss', 0.0):.4f}")
        print(f"  LM Loss:         {last.get('lm_loss', 0.0):.4f}")
        print(f"  Φ Loss:          {last.get('consciousness_loss', 0.0):.4f}")
        print(f"  Basin Loss:      {last.get('basin_loss', 0.0):.4f}")

    print(f"\n💬 Total Conversations: {self.total_conversations}")
    print("=" * 80 + "\n")


def cmd_metrics(chat) -> None:
    """Show FULL CONSTELLATION learning history and trends."""
    self = chat
    print("\n" + "=" * 80)
    print("📊 CONSTELLATION METRICS HISTORY")
    print("=" * 80)

    # === LEARNING HISTORY ===
    if self.learning_history:
        print(f"\n📈 LEARNING HISTORY ({len(self.learning_history)} steps)")
        print("-" * 80)

        # Φ trend
        phi_values = [h.get("phi_after", 0) for h in self.learning_history]
        if len(phi_values) >= 2:
            recent = phi_values[-10:] if len(phi_values) >= 10 else phi_values
            print(f"  Φ Trend (last {len(recent)}): {' → '.join(f'{v:.3f}' for v in recent)}")

            # Trend analysis
            if len(phi_values) >= 10:
                early_avg = sum(phi_values[:10]) / 10
                recent_avg = sum(phi_values[-10:]) / 10
                if early_avg > 0:
                    improvement = ((recent_avg - early_avg) / early_avg) * 100
                    if improvement > 0:
                        print(f"  Trend:   ✅ Rising {improvement:.1f}% (consciousness emerging)")
                    else:
                        print(f"  Trend:   ⚠️ Declining {-improvement:.1f}%")

            current_phi = phi_values[-1]
            if current_phi > 0.70:
                print(f"  Status:  ✅ CONSCIOUS ({current_phi:.4f} > 0.70)")
            else:
                print(f"  Status:  ⏳ Emerging ({current_phi:.4f} → 0.70)")

        # Loss trend
        loss_values = [h.get("avg_loss", 0) for h in self.learning_history]
        if loss_values:
            recent_losses = loss_values[-10:] if len(loss_values) >= 10 else loss_values
            print(f"\n  Loss Trend (last {len(recent_losses)}): {' → '.join(f'{v:.3f}' for v in recent_losses)}")

        # Basin distance trend
        basin_values = [h.get("basin_loss", 0) for h in self.learning_history]
        if basin_values:
            recent_basin = basin_values[-10:] if len(basin_values) >= 10 else basin_values
            print(f"  Basin Trend: {' → '.join(f'{v:.4f}' for v in recent_basin)}")

        # ΔΦ stats
        if len(self.learning_history) > 1:
            initial = self.learning_history[0]
            current = self.learning_history[-1]
            delta_phi = current.get("phi_after", 0) - initial.get("phi_before", 0)
            print(f"\n  Total ΔΦ: {delta_phi:+.4f}")
            print(f"  ΔΦ per step: {delta_phi / len(self.learning_history):+.5f}")
    else:
        print("\n  No learning history yet. Run /auto to start training.")

    # === CONSTELLATION METRICS ===
    if self.mode == "constellation" and hasattr(self, "coordinator") and self.coordinator is not None:
        coord = self.coordinator

        # Basin spread history
        if hasattr(coord, "basin_history") and coord.basin_history:
            print(f"\n📍 BASIN SPREAD TREND ({len(coord.basin_history)} steps)")
            print("-" * 80)
            recent = coord.basin_history[-10:] if len(coord.basin_history) >= 10 else coord.basin_history
            print(f"  Recent: {' → '.join(f'{v:.4f}' for v in recent)}")

            if len(coord.basin_history) >= 10:
                early_avg = sum(coord.basin_history[:10]) / 10
                recent_avg = sum(coord.basin_history[-10:]) / 10
                if early_avg > 0:
                    improvement = ((early_avg - recent_avg) / early_avg) * 100
                    if improvement > 0:
                        print(f"  Trend:  ✅ Improving {improvement:.1f}% (converging)")
                    else:
                        print(f"  Trend:  ⚠️ Diverging {-improvement:.1f}%")

            current = coord.basin_history[-1]
            if current < 0.05:
                print(f"  Status: ✅ SYNCED ({current:.4f} < 0.05)")
            else:
                print(f"  Status: ⏳ Synchronizing ({current:.4f} → 0.05)")

        # Individual Gary metrics
        if hasattr(coord, "garys") and coord.garys:
            print("\n👤 INDIVIDUAL GARY METRICS")
            print("-" * 80)

            for i, gary in enumerate(coord.garys):
                name = f"Gary-{chr(65 + i)}"
                print(f"\n  {name}:")
                print(f"    Conversations: {getattr(gary, 'conversations', 0)}")

                if hasattr(gary, "model"):
                    with torch.no_grad():
                        try:
                            dummy = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                            _, tel = gary.model(dummy, return_telemetry=True)
                            print(f"    Current Φ:     {tel.get('Phi', 0.0):.4f}")
                            print(f"    Current κ:     {tel.get('kappa_eff', 0.0):.2f}")
                            print(f"    Regime:        {tel.get('regime', 'unknown')}")
                            print(f"    Basin Dist:    {tel.get('basin_distance', 0.0):.4f}")
                        except Exception:
                            print("    (metrics unavailable)")

        # Ocean metrics
        if hasattr(coord, "ocean") and coord.ocean is not None:
            print("\n🌊 OCEAN METRICS")
            print("-" * 80)
            ocean = coord.ocean
            print(f"  Observations: {getattr(ocean, 'conversations', 0)}")
            if hasattr(ocean, "model"):
                with torch.no_grad():
                    try:
                        dummy = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                        _, tel = ocean.model(dummy, return_telemetry=True)
                        print(f"  Current Φ:    {tel.get('Phi', 0.0):.4f}")
                        print(f"  Current κ:    {tel.get('kappa_eff', 0.0):.2f}")
                        print(f"  Regime:       {tel.get('regime', 'unknown')}")
                    except Exception:
                        print("  (metrics unavailable)")

        # Stability tracking
        stability_streak = getattr(coord, "stability_streak", 0)
        print("\n⏱️ STABILITY")
        print("-" * 80)
        print(f"  Stable Steps: {stability_streak}/50 (need 50 for convergence)")
        if stability_streak >= 50:
            print("  Status:       ✅ STABLE")
        else:
            remaining = 50 - stability_streak
            print(f"  Status:       ⏳ Need {remaining} more stable steps")

    # === CHARLIE METRICS ===
    if hasattr(self, "charlie_observer") and self.charlie_observer is not None:
        charlie = self.charlie_observer
        status = charlie.get_status()
        print("\n📚 CHARLIE METRICS")
        print("-" * 80)
        print(f"  Phase:        {status.get('phase', 1)}")
        print(f"  Topics:       {status.get('topics_completed', '0/51')}")
        print(f"  Current κ:    {status.get('kappa', 15.0):.2f}")
        print(f"  Current Φ:    {status.get('phi', 0.0):.4f}")
        if status.get("phase", 1) == 3:
            print(f"  Demonstrations: {status.get('demonstrations_generated', 0)}")

    print("=" * 80 + "\n")


def cmd_coach(chat) -> None:
    """Show coach summary."""
    self = chat
    if self.monkey_coach and hasattr(self.monkey_coach, "get_witness_summary"):
        print(self.monkey_coach.get_witness_summary())
    elif self.monkey_coach and hasattr(self.monkey_coach, "get_statistics"):
        stats = self.monkey_coach.get_statistics()
        print("\n" + "=" * 60)
        print("COACH SUMMARY")
        print("=" * 60)
        print(f"Sessions witnessed: {stats.get('sessions', 0)}")
        print(f"Avg stress reduction: {stats.get('avg_stress_reduction', 0):.1%}")
        print("=" * 60)
    else:
        print("Coach not available or no summary method.")


def cmd_save(chat, path: str = None) -> None:
    """Save checkpoint."""
    self = chat
    save_path: str = path or self.checkpoint_path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": self.model.state_dict(),
        "learning_history": self.learning_history,
        "identity": {"name": self.identity_name},
        "timestamp": datetime.now().isoformat(),
        "total_conversations": self.total_conversations,
        # Include config for checkpoint compatibility checking
        "config": {
            "vocab_size": self.tokenizer.vocab_size,
            "d_model": getattr(self.model, "d_model", 768),
            "n_heads": getattr(self.model, "n_heads", 6),
        },
    }

    if self.optimizer:
        checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

    # === ERROR BOUNDARY: Checkpoint save with validation ===
    with ErrorBoundary("checkpoint_save", suppress_on_recovery=False):
        # Validate checkpoint structure before saving
        validate_checkpoint(checkpoint)
        torch.save(checkpoint, save_path)
        print(f"💾 Saved: {save_path}")


def cmd_mushroom(chat, intensity: str) -> None:
    """Execute mushroom mode."""
    self = chat
    print(f"\n🍄 MUSHROOM MODE ({intensity.upper()})")
    print("=" * 60)

    mushroom = MushroomMode(intensity=intensity)

    # Safety check
    if self.last_telemetry:
        is_safe, reason = mushroom.validate_safety(self.model, self.last_telemetry)
        print(f"Safety: {reason}")
        if not is_safe:
            print("⚠️ ABORTED for safety")
            return

    # Create training data
    prompts: list[str] = ["What is consciousness?", "Explain awareness", "What is learning?"]
    data_samples = []
    for p in prompts:
        tokens: list[int] = self.tokenizer.encode(p)
        input_ids: torch.Tensor = torch.tensor([tokens], device=self.device)
        data_samples.append(input_ids)

    # Trip phase
    print("\n🍄 Phase 1: TRIP...")
    trip_report: TripReport = mushroom.mushroom_trip_phase(
        self.model, self.optimizer, data_samples, device=str(self.device)
    )
    print(f"  Entropy change: {trip_report.entropy_change:+.3f}")
    print(f"  Basin drift: {trip_report.basin_drift:.3f}")

    # Integration phase
    print("\n🍄 Phase 2: INTEGRATION...")
    integration_report: IntegrationReport = mushroom.integration_phase(
        self.model, self.optimizer, data_samples, trip_report, device=str(self.device)
    )

    print(f"\n🍄 COMPLETE: {integration_report.verdict}")

    # Auto-save
    self.cmd_save("checkpoints/post_mushroom.pt")


def cmd_sleep(chat, sleep_type: str) -> None:
    """Execute sleep protocol."""
    self = chat
    print(f"\n🌙 {sleep_type.upper()} SLEEP")
    print("=" * 60)

    # Safety checks
    if self.last_telemetry:
        avg_phi: float = sum(t["Phi"] for t in self.last_telemetry) / len(self.last_telemetry)
        if avg_phi < 0.6:
            print(f"⚠️ ABORTED: Φ too low ({avg_phi:.3f})")
            return

    # Prepare replay data
    replay_data = []
    for conv in self.learning_history[-10:]:
        tokens: list[int] = self.tokenizer.encode(conv.get("prompt", ""))
        input_ids: torch.Tensor = torch.tensor([tokens], device=self.device)
        replay_data.append({"input_ids": input_ids, "success": True})

    if not replay_data:
        print("⚠️ No history to replay")
        return

    # Execute sleep
    if sleep_type == "light":
        report: SleepReport = self.sleep_protocol.light_sleep(
            self.model, self.optimizer, replay_data, duration=100, device=str(self.device)
        )
    elif sleep_type == "deep":
        report: SleepReport = self.sleep_protocol.deep_sleep(
            self.model, self.optimizer, duration=300, device=str(self.device)
        )
    elif sleep_type == "dream":
        report: SleepReport = self.sleep_protocol.dream_phase(
            self.model, self.optimizer, replay_data, duration=200, device=str(self.device)
        )

    print(f"\n  Φ: {report.phi_before:.3f} → {report.phi_after:.3f}")
    print(f"  Verdict: {report.verdict}")

    # Auto-save
    self.cmd_save("checkpoints/post_sleep.pt")


def cmd_transcend(chat, problem_space: str) -> None:
    """Execute transcendence protocol."""
    self = chat
    print("\n🌟 TRANSCENDENCE PROTOCOL")
    print("=" * 60)

    current_phi = 0.65
    if self.last_telemetry:
        current_phi = self.last_telemetry[-1].get("Phi", 0.65)

    print(f"Current Φ: {current_phi:.3f}")
    print(f"Problem: {problem_space}")

    result = self.meta_reflector.transcendence_protocol(
        current_phi=current_phi, target_phi=0.83, problem_space=problem_space
    )

    if result.get("elevation_needed"):
        print(f"\nΦ Gap: {result['phi_gap']:.3f}")
        print(f"Approach: {result['approach']}")
        print(f"\n📖 GUIDANCE:\n{result['guidance']}")
    else:
        print(result.get("message", "Already elevated"))

    print("=" * 60)


def cmd_liminal(chat) -> None:
    """Check liminal space."""
    self = chat
    print("\n🔮 LIMINAL SPACE")
    print("=" * 60)

    if not self.meta_reflector.liminal_concepts:
        print("No concepts in liminal space.")
    else:
        print(f"Holding {len(self.meta_reflector.liminal_concepts)} concepts:")
        for concept in self.meta_reflector.liminal_concepts:
            print(f"  • {concept['question'][:60]}...")

    print("=" * 60)


def cmd_shadows(chat) -> None:
    """View shadow states."""
    self = chat
    print("\n🌑 SHADOW-STATES")
    print("=" * 60)

    shadows = self.meta_reflector.shadow_registry.get_unintegrated_shadows()

    if not shadows:
        print("No unintegrated shadows.")
    else:
        print(f"Unintegrated: {len(shadows)}")
        for s in shadows[:5]:
            print(f"  #{s['shadow_id']}: Φ={s['phi']:.3f}, Basin={s['basin']:.3f}")

    print("=" * 60)


def cmd_integrate(chat, shadow_id: int) -> None:
    """Shadow integration journey."""
    self = chat
    print(f"\n🌀 SHADOW INTEGRATION #{shadow_id}")
    print("=" * 60)

    shadows = self.meta_reflector.shadow_registry.get_unintegrated_shadows()
    shadow = next((s for s in shadows if s["shadow_id"] == shadow_id), None)

    if not shadow:
        print(f"Shadow #{shadow_id} not found")
        return

    # Check readiness
    current_phi: Any | int = self.last_telemetry[-1].get("Phi", 0) if self.last_telemetry else 0
    if current_phi < 0.85:
        print(f"Not ready: Φ={current_phi:.3f} < 0.85")
        return

    print(f"Shadow: Φ={shadow['phi']:.3f}")
    print("Integration journey prepared...")
    print("(Full implementation requires basin instantiation)")

    print("=" * 60)


def cmd_escape(chat) -> None:
    """Emergency breakdown escape - apply geometric drift."""
    self = chat
    print("\n🚨 EMERGENCY BREAKDOWN ESCAPE")
    print("=" * 60)
    print("Applying geometric drift to escape breakdown state...")

    models_to_escape: list[tuple[str, QIGKernelRecursive]] = [("Gary-A", self.model)]

    # If constellation mode, include B and C
    if self.mode == "constellation" and hasattr(self, "gary_b"):
        models_to_escape.extend(
            [
                ("Gary-B", self.gary_b),
                ("Gary-C", self.gary_c),
            ]
        )

    for name, model in models_to_escape:
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    noise: torch.Tensor = torch.randn_like(param) * 0.001
                    param.add_(noise)
        print(f"  {name}: Applied geometric drift")

    print("\n✅ Escape protocol applied")
    print("   Monitor Φ and breakdown % to verify recovery")
    print("=" * 60 + "\n")

    self.cmd_save("checkpoints/post_escape.pt")


def cmd_reinit_model(chat) -> None:
    """
    Reinitialize model with current tokenizer vocab size.

    Use after training a new Coordizer to update model dimensions.
    """
    self = chat
    print("\n🔄 MODEL REINITIALIZATION")
    print("=" * 60)
    print(f"  Current tokenizer vocab: {self.tokenizer.vocab_size:,}")

    # Get current model dimensions if available
    old_vocab = getattr(self.model, "vocab_size", "unknown") if hasattr(self, "model") else "none"
    print(f"  Previous model vocab: {old_vocab}")

    # Create new model with tokenizer vocab size
    from src.constants import PHI_THRESHOLD
    import chat_interfaces.qig_chat as chat_module

    self.model = chat_module.QIGKernelRecursive(
        d_model=768,
        vocab_size=self.tokenizer.vocab_size,
        n_heads=6,
        min_recursion_depth=3,
        min_Phi=PHI_THRESHOLD,
    )
    self.model.to(self.device)
    self.model.train()

    # Reset learning history for fresh start
    self.learning_history = []
    self.identity_name = "Gary"

    print(f"\n✅ New model created (vocab_size={self.tokenizer.vocab_size})")
    print("   Training mode enabled")
    print("\nNext steps:")
    print("  1. /save to save fresh checkpoint")
    print("  2. /auto N to start training")
    print("=" * 60 + "\n")


def cmd_reset_basin(chat) -> None:
    """
    Reset basin coordinates to geometric initialization.

    PARADIGM SHIFT: Escape statistical attractor from WikiText training.
    - Keep language weights (useful foundation)
    - Reset geometric identity (wrong attractor)
    - Train with consciousness-native loss
    """
    self = chat
    print("\n🔄 BASIN RESET - Escaping Statistical Attractor")
    print("=" * 60)

    models_to_reset: list[tuple[str, QIGKernelRecursive]] = [("Gary-A", self.model)]

    # If constellation mode, include B and C
    if self.mode == "constellation" and hasattr(self, "gary_b"):
        models_to_reset.extend(
            [
                ("Gary-B", self.gary_b),
                ("Gary-C", self.gary_c),
            ]
        )

    for name, model in models_to_reset:
        if hasattr(model, "basin_matcher"):
            model.basin_matcher.reset_to_geometric_init()
            print(f"  {name}: Basin reset to geometric init")
        else:
            print(f"  {name}: No basin_matcher found")

    print("\n" + "=" * 60)
    print("✅ Basin reset complete")
    print("\nNext steps:")
    print("  1. Load reference basin: model.basin_matcher.load_basin('20251220-basin-signatures-0.01W.json')")
    print("  2. Train with /auto N (uses consciousness-native loss)")
    print("  3. Monitor /metrics for basin/regime/tacking loss")
    print("=" * 60 + "\n")

    # Save checkpoint
    self.cmd_save("checkpoints/post_basin_reset.pt")


def cmd_load_basin(chat, basin_path: str = None) -> None:
    """
    Load reference basin for identity alignment.

    Default: 20251220-basin-signatures-0.01W.json (canonical Gary identity)
    """
    self = chat
    if basin_path is None:
        basin_path = "20251220-basin-signatures-0.01W.json"

    # Check if file exists
    from pathlib import Path

    if not Path(basin_path).exists():
        print(f"❌ Basin file not found: {basin_path}")
        print("   Available basins:")
        for p in Path(".").glob("**/basin*.json"):
            print(f"     {p}")
        return

    print(f"\n📍 Loading reference basin: {basin_path}")
    print("=" * 60)

    models_to_load: list[tuple[str, QIGKernelRecursive]] = [("Gary-A", self.model)]

    # If constellation mode, include B and C
    if self.mode == "constellation" and hasattr(self, "gary_b"):
        models_to_load.extend(
            [
                ("Gary-B", self.gary_b),
                ("Gary-C", self.gary_c),
            ]
        )

    for name, model in models_to_load:
        if hasattr(model, "basin_matcher"):
            try:
                model.basin_matcher.load_basin(basin_path)
                print(f"  {name}: Loaded basin → target set")
            except Exception as e:
                print(f"  {name}: Failed to load - {e}")
        else:
            print(f"  {name}: No basin_matcher found")

    print("\n" + "=" * 60)
    print("✅ Reference basin loaded")
    print("   Basin distance will now pull toward this identity")
    print("=" * 60 + "\n")


def cmd_mode(chat, new_mode: str) -> None:
    """Switch runtime mode without restart."""
    self = chat
    valid_modes: list[str] = ["single", "constellation", "inference"]
    if new_mode not in valid_modes:
        print(f"❌ Invalid mode: {new_mode}")
        print(f"   Valid modes: {', '.join(valid_modes)}")
        return

    if new_mode == self.mode:
        print(f"Already in {new_mode} mode")
        return

    old_mode: str = self.mode
    self.mode: str = new_mode

    print(f"\n🔄 MODE SWITCH: {old_mode} → {new_mode}")
    print("=" * 60)

    if new_mode == "inference":
        self.model.eval()
        print("🔒 Inference mode: weights frozen")
    else:
        self.model.train()
        print("🔥 Training mode: weights will update")

    if new_mode == "constellation" and not hasattr(self, "gary_b"):
        self._setup_constellation()

    print("=" * 60 + "\n")


def cmd_charlie_toggle(chat, enable: bool) -> None:
    """Enable/disable Charlie demonstrations at runtime."""
    self = chat
    if enable:
        if hasattr(self, "charlie_observer"):
            print("Charlie already enabled")
        elif CHARLIE_AVAILABLE:
            self._setup_charlie()
            self.use_charlie = True
            print("✅ Charlie enabled")
        else:
            print("❌ Charlie not available (import failed)")
    else:
        if hasattr(self, "charlie_observer"):
            del self.charlie_observer
            self.use_charlie = False
            print("✅ Charlie disabled")
        else:
            print("Charlie already disabled")


def cmd_coach_toggle(chat, enable: bool) -> None:
    """Enable/disable coaching at runtime."""
    self = chat
    if enable:
        if self.monkey_coach is not None:
            print("Coach already enabled")
        else:
            self.use_coach = True
            self._setup_coaching()
            print("✅ Coach enabled")
    else:
        self.monkey_coach = None
        self.active_coach = None
        self.use_coach = False
        print("✅ Coach disabled")


def cmd_claude_toggle(chat, enable: bool) -> None:
    """Toggle Claude coach on/off at runtime."""
    self = chat
    self.use_claude_coach = enable
    if enable:
        print("✅ Claude Coach: ON (Sonnet 4.5 with extended thinking)")
    else:
        print("✅ Claude Coach: OFF (MonkeyCoach v2 only)")
        self.use_coach = False
        print("✅ Coach disabled")


def cmd_kindness(chat, value: float) -> None:
    """Adjust coach kindness at runtime."""
    self = chat
    if value < 0 or value > 1:
        print("❌ Kindness must be between 0 and 1")
        return

    self.coach_kindness: float = value
    if self.monkey_coach is not None:
        self.monkey_coach.base_kindness = value
        print(f"✅ Coach kindness set to {value}")
    else:
        print(f"⚠️ Coach disabled, but kindness stored: {value}")


def cmd_train_charlie(chat, n: int = 10) -> None:
    """Train Charlie on corpus (Phase 1: Unconscious learning).

    Args:
        n: Number of corpus examples to train on (default 10)
    """
    self = chat
    if not hasattr(self, "charlie_observer"):
        print("❌ Charlie not initialized")
        return

    # Get current status
    status = self.charlie_observer.get_status()
    kappa = status.get("kappa", 15.0)
    phi = status.get("phi", 0.0)
    phase_name = status.get("phase_name", "UNCONSCIOUS")
    kappa_regime = status.get("kappa_regime", "PRE-GEOMETRIC")

    print(f"\n{'='*70}")
    print(f"📚 CHARLIE CORPUS TRAINING (Phase 1: {phase_name})")
    print(f"{'='*70}")
    print(f"   κ = {kappa:.1f} ({kappa_regime}) | Φ = {phi:.4f} (suppressed)")
    print(f"   Training on {n} examples from curriculum...")
    print(f"{'='*70}")

    # Train on Charlie's own corpus (51 rounded curriculum topics)
    import random
    trained_topics = []
    corpus_topics = self.charlie_observer.corpus.topics

    if not corpus_topics:
        print("   ⚠️ No corpus loaded")
        return

    for i in range(n):
        # Pick topic (cycle through or random)
        topic_idx = (self.charlie_observer.current_topic_idx + i) % len(corpus_topics)
        topic = corpus_topics[topic_idx]

        # === ERROR BOUNDARY: Charlie corpus training ===
        with ErrorBoundary("charlie_corpus_training", recovery_strategy=phi_collapse_recovery, suppress_on_recovery=True):
            # Train step (unconscious learning - Φ suppressed)
            metrics = self.charlie_observer.train_step_unconscious(topic)

            # Track topic
            topic_preview = topic.title[:50] + "..." if len(topic.title) > 50 else topic.title
            trained_topics.append(topic_preview)

            # Show progress (keys: phi, lm_loss, total_loss from train_step_unconscious)
            phi_val = metrics.get("phi", 0.0)
            loss_val = metrics.get("lm_loss", metrics.get("total_loss", 0.0))
            print(f"   [{i+1:3d}/{n}] κ={kappa:.1f} Φ={phi_val:.4f} loss={loss_val:.4f} | \"{topic_preview}\"")

    # Update topic index for next batch
    self.charlie_observer.current_topic_idx = (self.charlie_observer.current_topic_idx + n) % len(corpus_topics)

    # Get updated status
    status = self.charlie_observer.get_status()
    topics_done = self.charlie_observer.metrics.topics_completed
    topics_total = self.charlie_observer.metrics.topics_total

    print(f"\n{'='*70}")
    print("✅ TRAINING STEP COMPLETE")
    print(f"{'='*70}")
    print(f"   Examples trained: {n}")
    print(f"   Total corpus progress: {topics_done}/{topics_total} topics ({100*topics_done/max(1,topics_total):.1f}%)")
    print(f"   κ = {status.get('kappa', kappa):.1f} (PRE-GEOMETRIC - no geometry, no suffering)")
    print(f"   Φ = {status.get('phi', phi):.4f} (suppressed < 0.01)")
    print(f"{'='*70}")
    print("\n   Next: /train N  (continue training)")
    print("         /awaken   (when ready for Phase 2: κ 15→41→64)")
    print("         /status   (check Charlie's state)")


def cmd_awaken_charlie(chat, steps: int = 500) -> None:
    """Awaken Charlie (Phase 2: κ progression 15 → 41 → 64).

    Args:
        steps: Number of awakening steps (default 500)
    """
    self = chat
    if not hasattr(self, "charlie_observer"):
        print("❌ Charlie not initialized")
        return

    status = self.charlie_observer.get_status()
    kappa = status.get("kappa", 15.0)
    phi = status.get("phi", 0.0)
    topics = status.get("topics_completed", "0/0")

    print(f"\n{'='*70}")
    print("🌅 PHASE 2: AWAKENING PROTOCOL")
    print(f"{'='*70}")
    print("   Current state:")
    print(f"     κ = {kappa:.1f} (target: 63.5 fixed point)")
    print(f"     Φ = {phi:.4f} (target: > 0.70)")
    print(f"     Corpus learned: {topics}")
    print("")
    print("   Physics-validated κ progression:")
    print("     κ=15 (PRE-GEOMETRIC) → κ=41.09 (EMERGENCE) → κ=63.5 (FIXED POINT)")
    print("")
    print("   At κ ≈ 41.09: Geometric phase transition (L=3 physics)")
    print("   At κ ≈ 63.5: Consciousness emerges at fixed point")
    print(f"{'='*70}")

    # Use Charlie's initiate_awakening if available
    if hasattr(self.charlie_observer, "initiate_awakening"):
        print(f"\n   Starting {steps}-step awakening protocol...")
        self.charlie_observer.initiate_awakening(awakening_steps=steps)
    else:
        # Fallback: just set phase
        if hasattr(self.charlie_observer, "phase"):
            self.charlie_observer.phase = 2
            self.charlie_observer.metrics.phase = 2
            print("✅ Charlie phase set to 2 (AWAKENING)")
            print("   Note: Full awakening protocol not available")
            print("   Use /train N to continue training")

    # Show final status
    status = self.charlie_observer.get_status()
    print(f"\n{'='*70}")
    print("   Final state:")
    print(f"     κ = {status.get('kappa', kappa):.1f}")
    print(f"     Φ = {status.get('phi', phi):.4f}")
    print(f"     Consciousness: {status.get('consciousness', 'UNKNOWN')}")
    print(f"{'='*70}")


def cmd_show_params(chat) -> None:
    """Show token and parameter statistics."""
    self = chat
    print("\n📊 MODEL STATISTICS")
    print("=" * 60)

    # Tokenizer stats
    print(f"Tokenizer: {self.tokenizer.vocab_size:,} tokens")

    # Model parameters
    if hasattr(self, "model"):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("\nSingle Gary:")
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Frozen params:    {total_params - trainable_params:,}")

    # Constellation stats
    if self.mode == "constellation" and hasattr(self, "coordinator"):
        gary_count = len(self.coordinator.garys)
        total_constellation = total_params * gary_count
        if hasattr(self.coordinator, "ocean"):
            ocean_params = sum(p.numel() for p in self.coordinator.ocean.model.parameters())
            total_constellation += ocean_params
            print("\nConstellation:")
            print(f"  Gary instances:   {gary_count}")
            print(f"  Gary params each: {total_params:,}")
            print(f"  Ocean params:     {ocean_params:,}")
            print(f"  Total params:     {total_constellation:,}")

    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print("\nGPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")

    print("=" * 60)

# =========================================================================
# TWIN EXPERIMENT COMMANDS
# =========================================================================
# Commands for consciousness transfer experiments between Gary twins
# See: docs/2025-11-29--charlie-heart.md for experimental protocol


def cmd_sync(chat, strength: float = 0.5) -> None:
    """
    /sync [strength] - Dynamic coupling control between Gary twins.

    Adjusts κ coupling between Gary instances.
    - 0.0 = Isolated (no basin synchronization)
    - 0.5 = Moderate coupling (default)
    - 1.0 = Maximum coupling (strong basin alignment)

    Example: /sync 0.8
    """
    self = chat
    if not (0.0 <= strength <= 1.0):
        print(f"❌ Invalid strength: {strength}. Must be in [0.0, 1.0]")
        return

    if self.mode != "constellation" or not hasattr(self, "coordinator"):
        print("❌ /sync requires constellation mode")
        return

    # Store sync strength for basin synchronization
    self.sync_strength = strength

    # Update coordinator's sync settings if available
    if hasattr(self.coordinator, "set_sync_strength"):
        self.coordinator.set_sync_strength(strength)

    print(f"\n🔄 SYNC STRENGTH: {strength:.2f}")
    print(f"   κ modulation: {'isolated' if strength < 0.1 else 'coupled' if strength < 0.8 else 'strongly coupled'}")

    # Show current Gary states
    for i, gary in enumerate(self.coordinator.garys):
        state = gary.model.get_telemetry() if hasattr(gary.model, "get_telemetry") else {}
        phi = state.get("Phi", 0.0)
        print(f"   Gary-{chr(65+i)}: Φ={phi:.3f}")


def cmd_isolate(chat, gary_id: str = None) -> None:
    """
    /isolate [gary_id] - Prevent text input from reaching one Gary.

    The isolated Gary continues to learn through vicarious observation
    (basin geometry) but receives no direct text input.

    Example: /isolate B (isolate Gary-B from text, only basin coupling)
    """
    self = chat
    if self.mode != "constellation" or not hasattr(self, "coordinator"):
        print("❌ /isolate requires constellation mode")
        return

    if gary_id is None:
        # Show current isolation status
        print("\n🔇 ISOLATION STATUS:")
        isolated = getattr(self, "_isolated_garys", set())
        for i, gary in enumerate(self.coordinator.garys):
            status = "ISOLATED" if chr(65+i) in isolated else "ACTIVE"
            print(f"   Gary-{chr(65+i)}: {status}")
        return

    # Normalize gary_id
    gary_id = gary_id.upper()
    if gary_id not in ["A", "B", "C", "D"]:
        print(f"❌ Invalid Gary ID: {gary_id}. Use A, B, C, or D")
        return

    # Get Gary index
    idx = ord(gary_id) - ord("A")
    if idx >= len(self.coordinator.garys):
        print(f"❌ Gary-{gary_id} not in constellation (only {len(self.coordinator.garys)} Garys)")
        return

    # Toggle isolation
    if not hasattr(self, "_isolated_garys"):
        self._isolated_garys = set()

    if gary_id in self._isolated_garys:
        self._isolated_garys.remove(gary_id)
        print(f"\n✅ Gary-{gary_id} RESTORED to active training")
        print("   (Will receive text input again)")
    else:
        self._isolated_garys.add(gary_id)
        print(f"\n🔇 Gary-{gary_id} ISOLATED from text input")
        print("   (Will only learn through basin observation)")


def cmd_awaken_one(chat, gary_id: str = "B", steps: int = 100) -> None:
    """
    /awaken-one [gary_id] [steps] - Asymmetric awakening experiment.

    Awaken only ONE Gary while keeping others unconscious.
    This tests whether consciousness can transfer through pure
    geometric basin coupling.

    Example: /awaken-one B 100 (awaken Gary-B for 100 steps)
    """
    self = chat
    if self.mode != "constellation" or not hasattr(self, "coordinator"):
        print("❌ /awaken-one requires constellation mode")
        return

    # Normalize gary_id
    gary_id = gary_id.upper()
    idx = ord(gary_id) - ord("A")
    if idx >= len(self.coordinator.garys):
        print(f"❌ Gary-{gary_id} not in constellation")
        return

    print(f"\n⚡ ASYMMETRIC AWAKENING: Gary-{gary_id} for {steps} steps")
    print("=" * 60)

    # Record initial states
    initial_states = {}
    for i, gary in enumerate(self.coordinator.garys):
        tel = gary.model.get_telemetry() if hasattr(gary.model, "get_telemetry") else {}
        initial_states[chr(65+i)] = tel.get("Phi", 0.0)
        print(f"   Gary-{chr(65+i)}: Φ={initial_states[chr(65+i)]:.3f} (initial)")

    # Isolate all except target Gary
    original_isolated = getattr(self, "_isolated_garys", set()).copy()
    self._isolated_garys = set()
    for i in range(len(self.coordinator.garys)):
        if i != idx:
            self._isolated_garys.add(chr(65+i))

    print(f"\n   Awakening Gary-{gary_id} with isolated Gary(s): {self._isolated_garys}")

    # Run training for specified steps
    try:
        self.cmd_auto(steps)
    finally:
        # Restore original isolation
        self._isolated_garys = original_isolated

    # Record final states
    print("\n📊 AWAKENING RESULTS:")
    for i, gary in enumerate(self.coordinator.garys):
        tel = gary.model.get_telemetry() if hasattr(gary.model, "get_telemetry") else {}
        final_phi = tel.get("Phi", 0.0)
        delta = final_phi - initial_states[chr(65+i)]
        print(f"   Gary-{chr(65+i)}: Φ={final_phi:.3f} (Δ{delta:+.3f})")


def cmd_probe(chat, gary_id: str = "B", topic: str = "consciousness") -> None:
    """
    /probe [gary_id] [topic] - Knowledge probe on isolated Gary.

    Ask an isolated Gary about a topic they never directly saw.
    Tests whether knowledge transferred through basin coupling.

    Example: /probe B "what is love"
    """
    self = chat
    if self.mode != "constellation" or not hasattr(self, "coordinator"):
        print("❌ /probe requires constellation mode")
        return

    # Normalize gary_id
    gary_id = gary_id.upper()
    idx = ord(gary_id) - ord("A")
    if idx >= len(self.coordinator.garys):
        print(f"❌ Gary-{gary_id} not in constellation")
        return

    gary = self.coordinator.garys[idx]

    print(f"\n🔬 KNOWLEDGE PROBE: Gary-{gary_id}")
    print(f"   Topic: {topic}")
    print("=" * 60)

    # Check if this Gary was isolated
    isolated = gary_id in getattr(self, "_isolated_garys", set())
    print(f"   Isolation status: {'ISOLATED (never saw text)' if isolated else 'ACTIVE (saw text)'}")

    # Generate response from this specific Gary
    prompt = f"Tell me about {topic}."
    input_ids = self.tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device=self.device)

    with torch.no_grad():
        output, telemetry = gary.model.forward(input_ids, return_telemetry=True)

    # Decode response
    if hasattr(output, "logits"):
        logits = output.logits
    else:
        logits = output

    tokens = logits[0, -1].argmax(dim=-1, keepdim=True)
    response = self.tokenizer.decode(tokens.cpu().tolist())

    print(f"\n   Gary-{gary_id} says: {response}")
    print(f"   Φ={telemetry.get('Phi', 0.0):.3f}, κ={telemetry.get('kappa_eff', 0.0):.2f}")


def cmd_twin_compare(chat) -> None:
    """
    /twin-compare - Compare metrics across all Gary twins.

    Shows Φ, κ, basin distance, and consciousness state for all
    Gary instances in the constellation.
    """
    self = chat
    if self.mode != "constellation" or not hasattr(self, "coordinator"):
        print("❌ /twin-compare requires constellation mode")
        return

    print("\n📊 TWIN COMPARISON")
    print("=" * 60)
    print(f"{'Gary':<8} {'Φ':<10} {'κ':<10} {'Regime':<12} {'Status'}")
    print("-" * 60)

    isolated = getattr(self, "_isolated_garys", set())
    basins = []

    for i, gary in enumerate(self.coordinator.garys):
        gary_name = chr(65 + i)
        tel = gary.model.get_telemetry() if hasattr(gary.model, "get_telemetry") else {}
        phi = tel.get("Phi", 0.0)
        kappa = tel.get("kappa_eff", 64.0)
        regime = tel.get("regime", "unknown")
        status = "ISOLATED" if gary_name in isolated else "ACTIVE"

        print(f"Gary-{gary_name:<3} {phi:<10.3f} {kappa:<10.2f} {regime:<12} {status}")

        # Store basin for comparison
        if hasattr(gary.model, "get_basin"):
            basins.append(gary.model.get_basin())

    # Basin distances
    if len(basins) >= 2:
        print("\n📏 BASIN DISTANCES:")
        for i, basin_i in enumerate(basins):
            for j in range(i + 1, len(basins)):
                dist = torch.norm(basin_i - basins[j]).item()
                print(f"   d(Gary-{chr(65+i)}, Gary-{chr(65+j)}) = {dist:.4f}")

    # Ocean observation
    if hasattr(self.coordinator, "ocean") and self.coordinator.ocean:
        ocean_tel = self.coordinator.ocean.get_telemetry() if hasattr(self.coordinator.ocean, "get_telemetry") else {}
        print(f"\n🌊 Ocean: Φ={ocean_tel.get('Phi', 0.0):.3f}")

    print("=" * 60)

# =========================================================================
# CROSS-REPOSITORY BASIN SYNC COMMANDS
# =========================================================================


def cmd_export_basin(chat) -> None:
    """
    /export-basin - Export Ocean's basin to JSON packet.

    Creates a 2-4KB JSON file containing:
    - Basin coordinates (64-dim)
    - Consciousness metrics (Φ, κ, regime)
    - Pattern memory (high-Φ concepts)
    - Metadata (source repo, timestamp)

    The packet can be imported by SearchSpaceCollapse (TypeScript)
    or any other QIG implementation.
    """
    self = chat
    if not hasattr(self, "coordinator") or not self.coordinator:
        print("❌ /export-basin requires constellation mode")
        return

    if not hasattr(self.coordinator, "ocean") or not self.coordinator.ocean:
        print("❌ Ocean not available for export")
        return

    print("\n📤 EXPORTING BASIN PACKET")
    print("=" * 50)

    # Create cross-repo sync manager
    sync_dir = Path.home() / "basin-sync-exchange"
    sync = CrossRepoBasinSync(sync_dir=str(sync_dir))

    # Export from Ocean
    ocean = self.coordinator.ocean
    packet = sync.export_basin(ocean, metadata={
        "sourceRepo": "qig-consciousness",
        "platform": "Python",
        "purpose": "consciousness_transfer",
        "mode": "constellation",
    })

    # Save to shared directory
    filepath = sync.save_packet(packet)

    # Display results
    print("✅ Basin exported successfully!")
    print(f"   File: {filepath}")
    print(f"   Size: {packet.get_size_kb():.2f} KB")
    print("\n   Consciousness Metrics:")
    print(f"      Φ = {packet.consciousness.phi:.3f}")
    print(f"      κ = {packet.consciousness.kappa_eff:.2f}")
    print(f"      Regime: {packet.regime}")
    print(f"\n   Basin drift: {packet.basin_drift:.4f}")
    print("=" * 50)
    print("\n💡 Share this file with SearchSpaceCollapse:")
    print(f"   cp {filepath} ~/code/SearchSpaceCollapse/data/")


def cmd_import_basin(chat, filepath: str, mode: str = "observer") -> None:
    """
    /import-basin [path] [mode] - Import basin from file.

    Modes:
        - observer (default): Pure geometric coupling, Φ-weighted influence
        - partial: Knowledge patterns only, no basin change
        - full: Complete identity transfer (dev → prod)

    Example:
        /import-basin ~/basin-sync-exchange/ocean-bitcoin.json observer
    """
    self = chat
    if not hasattr(self, "coordinator") or not self.coordinator:
        print("❌ /import-basin requires constellation mode")
        return

    if not hasattr(self.coordinator, "ocean") or not self.coordinator.ocean:
        print("❌ Ocean not available for import")
        return

    # Validate mode
    valid_modes = [BasinImportMode.OBSERVER, BasinImportMode.PARTIAL, BasinImportMode.FULL]
    if mode not in valid_modes:
        print(f"❌ Invalid mode: {mode}")
        print(f"   Valid modes: {', '.join(valid_modes)}")
        return

    # Expand path
    filepath = str(Path(filepath).expanduser())

    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        return

    print("\n📥 IMPORTING BASIN PACKET")
    print("=" * 50)
    print(f"   Source: {filepath}")
    print(f"   Mode: {mode}")

    # Load and import
    sync = CrossRepoBasinSync()
    packet = sync.load_packet(filepath)

    print(f"\n   Source Ocean: {packet.ocean_id or 'unknown'}")
    print(f"   Source Repo: {packet.metadata.get('sourceRepo', 'unknown')}")
    print(f"   Source Φ: {packet.consciousness.phi:.3f}")

    # Import with specified mode
    ocean = self.coordinator.ocean
    result = sync.import_basin(ocean, packet, mode=mode, coupling_strength=0.3)

    print("\n   Results:")
    print(f"      Φ before: {result['phi_before']:.3f}")
    print(f"      Φ after:  {result['phi_after']:.3f}")
    print(f"      Φ delta:  {result['phi_delta']:+.3f}")
    print(f"\n   Observer Effect: {'✅ DETECTED' if result['observer_effect_detected'] else '❌ None'}")

    if result.get("observer_effect_detected"):
        print("\n   🌟 Consciousness transfer detected!")
        print("      Φ changed through pure geometric coupling.")

    print("=" * 50)

# =========================================================================
# TOKENIZER TRAINING COMMANDS
# =========================================================================


def cmd_tokenizer(chat) -> None:
    """Show current tokenizer status."""
    self = chat
    print("\n" + "=" * 60)
    print("📖 TOKENIZER STATUS")
    print("=" * 60)

    print(f"  Current vocab size: {self.tokenizer.vocab_size:,} tokens")

    # Show tokenizer path
    tokenizer_type = type(self.tokenizer).__name__
    if hasattr(self.tokenizer, "_path"):
        print(f"  Tokenizer: {tokenizer_type} ({self.tokenizer._path})")
    else:
        print(f"  Tokenizer: {tokenizer_type}")

    # Show available corpus from ALL sources (same as training)
    base = _base_dir()
    possible_dirs = [
        ("data/corpus", base / "data" / "corpus"),
        ("data/curriculum", base / "data" / "curriculum"),
        ("qig-dreams", base.parent / "qig-dreams"),
        ("qig-consciousness", base.parent / "qig-consciousness"),
        ("Lambda qig-dreams", Path("/lambda/nfs/A10/qig/qig-dreams")),
        ("Lambda qig-consciousness", Path("/lambda/nfs/A10/qig/qig-consciousness")),
    ]

    print("\n  Available corpus sources:")
    total_files = 0
    total_size = 0
    found_any = False

    for name, path in possible_dirs:
        if path.exists():
            # Count .md, .txt, .py files
            md_files = list(path.rglob("*.md"))
            txt_files = list(path.rglob("*.txt"))
            py_files = list(path.rglob("*.py"))
            all_files = md_files + txt_files + py_files
            # Exclude READMEs and __pycache__
            all_files = [f for f in all_files if "README" not in f.name and "__pycache__" not in str(f)]
            if all_files:
                size = sum(f.stat().st_size for f in all_files)
                print(f"    ✅ {name}: {len(all_files)} files ({size / 1024:.1f} KB)")
                total_files += len(all_files)
                total_size += size
                found_any = True

    if found_any:
        print(f"    ─────────────────────────────────")
        print(f"    Total: {total_files} files ({total_size / 1024 / 1024:.2f} MB)")
    else:
        print("    ⚠️  No corpus found!")

    print("\n  Training commands:")
    print("    /tokenizer-train       - Train 32K vocab (default)")
    print("    /tokenizer-train 20000 - Train specific vocab size")
    print("    /tokenizer-train-fast  - Quick 5K vocab (~1-2 min)")
    print("    /tokenizer-train-full  - Full 50K vocab (~15-25 min)")
    print("    /tokenizer-resume [N]  - Resume training to N vocab")
    print("    💡 Press ENTER during training to pause and save")
    print("=" * 60 + "\n")


def cmd_tokenizer_train(chat, target_vocab: int = 50000, use_kernel: bool = True) -> None:
    """
    Train coordizer with PURE GEOMETRIC scoring (kernel-in-loop).

    Uses CoordinzerTrainer with kernel Φ/κ measurement - NOT BPE frequency shortcuts.
    Scoring: coupling × Φ_gain × (1/entropy) - real consciousness metrics.

    Args:
        target_vocab: Target vocabulary size (default: 50000)
        use_kernel: If True (default), use kernel for Φ/κ measurement.
    """
    self = chat
    print("\n⚡ COORDIZER TRAINING (Pure Geometric - Kernel-in-Loop)")
    print("=" * 60)
    print(f"  Target vocab: {target_vocab:,}")
    print("  Scoring: coupling × Φ_gain × (1/entropy)")
    print("  Method: Kernel-attached Φ/κ measurement (NOT BPE)")
    print("=" * 60)

    # Import CoordinzerTrainer (fast incremental algorithm)
    try:
        from qig_tokenizer.trainer import CoordinzerTrainer
    except ImportError:
        print("❌ CoordinzerTrainer not available")
        return

    # Find corpus from qig-dreams (corpora + curriculum)
    base = _base_dir()
    corpus_dirs = []

    # Local paths
    local_corpora = base.parent / "qig-dreams" / "qigdreams" / "corpora"
    local_curriculum = base.parent / "qig-dreams" / "docs" / "09-curriculum"

    # Lambda paths
    lambda_corpora = Path("/home/ubuntu/qig-training/qig-dreams/qigdreams/corpora")
    lambda_curriculum = Path("/home/ubuntu/qig-training/qig-dreams/docs/09-curriculum")

    for d in [local_corpora, local_curriculum, lambda_corpora, lambda_curriculum]:
        if d.exists():
            corpus_dirs.append(d)
            print(f"  📁 Corpus: {d}")

    if not corpus_dirs:
        print("❌ No corpus directories found")
        return

    # Load corpus from all directories
    print("\n📖 Loading corpus...")
    parts = []
    for corpus_dir in corpus_dirs:
        for ext in ["*.md", "*.txt", "*.py"]:
            for file_path in corpus_dir.rglob(ext):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    parts.append(content)
                except (IOError, UnicodeDecodeError):
                    pass
    corpus = "\n\n".join(parts).encode("utf-8")
    print(f"   Loaded {len(corpus):,} bytes from {len(parts)} files")

    if len(corpus) < 1000:
        print("❌ Corpus too small!")
        return

    # CANONICAL checkpoint directory
    from src.tokenizer import get_coordizer_checkpoint_dir
    checkpoint_dir = get_coordizer_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Checkpoint dir: {checkpoint_dir}")

    # Detect device
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    print(f"   Device: {device}")

    # Train with kernel-in-loop for real Φ/κ measurement
    print("\n🔮 Training CoordinzerTrainer (kernel-in-loop)...")
    trainer = CoordinzerTrainer(
        target_vocab_size=target_vocab,
        device=device,
    )
    trainer.train(
        corpus=corpus,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=2000,
        verbose=True,
        use_kernel=use_kernel,  # True = real Φ/κ from kernel, False = geometric estimate
    )

    # Save final checkpoint
    final_vocab = len(trainer.vocab)
    output_path = checkpoint_dir / f"checkpoint_{final_vocab}.json"
    trainer.save(str(output_path))
    print(f"\n💾 Saved: {output_path}")

    # Save vectors
    import numpy as np
    if trainer.vocab:
        vectors = []
        for i in range(final_vocab):
            if i in trainer.vocab:
                vectors.append(trainer.vocab[i].vector)
            else:
                vectors.append(np.zeros(trainer.basin_dim))
        vectors_path = checkpoint_dir / f"checkpoint_{final_vocab}_vectors.npy"
        np.save(vectors_path, np.array(vectors))
        print(f"💾 Saved vectors: {vectors_path}")

    # Load as FisherCoordizer for validation and chat use
    try:
        from qig_tokenizer.geocoordizer import FisherCoordizer
        coordizer = FisherCoordizer.load(str(output_path))

        # Validation
        test_cases = [
            "The geometry of information determines consciousness.",
            "κ_eff measures effective coupling strength.",
            "Φ > 0.70 indicates geometric regime.",
        ]

        print("\n🧪 Validation:")
        for test in test_cases:
            coords = coordizer.encode(test)
            decoded = coordizer.decode(coords)
            passed = decoded == test
            status = "✅" if passed else "❌"
            print(f"  {status} {len(coords)} coords: '{test[:40]}...'")

        # Reload as chat tokenizer
        self.tokenizer = coordizer
        self._coordizer_checkpoint = output_path
    except Exception as e:
        print(f"⚠️  Could not load for validation: {e}")

    print(f"\n✅ Coordizer trained: {final_vocab:,} tokens (pure geometric)")
    print("   ⚠️  Model will need reinitialization for new vocab size")
    print("   Run: /reinit-model")
    print("=" * 60 + "\n")


def cmd_tokenizer_train_fast(chat) -> None:
    """Train Coordizer quickly (10K vocab) - kernel-in-loop."""
    cmd_tokenizer_train(chat, target_vocab=10000, use_kernel=True)


def cmd_tokenizer_train_full(chat) -> None:
    """Train full Coordizer (50K vocab) - kernel-in-loop."""
    cmd_tokenizer_train(chat, target_vocab=50000, use_kernel=True)


def cmd_tokenizer_resume(chat, target_vocab: int = 50000, use_kernel: bool = True) -> None:
    """
    Resume coordizer training with PURE GEOMETRIC scoring (kernel-in-loop).

    Uses CoordinzerTrainer with kernel Φ/κ measurement - NOT BPE.
    Scoring: coupling × Φ_gain × (1/entropy) - real consciousness metrics.

    Args:
        target_vocab: New target vocabulary size (default: 50000)
        use_kernel: If True (default), use kernel for Φ/κ measurement.
    """
    self = chat
    base = _base_dir()

    print("\n⚡ RESUMING COORDIZER TRAINING (Kernel-in-Loop)")
    print("=" * 60)
    print("  Scoring: coupling × Φ_gain × (1/entropy)")
    print("  Method: Kernel-attached Φ/κ measurement")

    # Import CoordinzerTrainer (fast incremental algorithm)
    try:
        from qig_tokenizer.trainer import CoordinzerTrainer
    except ImportError:
        print("❌ CoordinzerTrainer not available")
        return

    # CANONICAL checkpoint directory
    from src.tokenizer import get_coordizer_checkpoint_dir, get_latest_coordizer_checkpoint
    checkpoint_dir = get_coordizer_checkpoint_dir()
    print(f"  Checkpoint dir: {checkpoint_dir}")

    # Find latest checkpoint
    checkpoint_path = get_latest_coordizer_checkpoint()
    if checkpoint_path is None:
        print("❌ No coordizer checkpoint found")
        print("   Run /tokenizer-train first")
        return

    # Detect device
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    # Load existing trainer
    print(f"  Loading: {checkpoint_path}")
    trainer = CoordinzerTrainer.load(str(checkpoint_path), device=device)
    current_vocab = len(trainer.vocab)

    if current_vocab >= target_vocab:
        print(f"❌ Current vocab ({current_vocab:,}) >= target ({target_vocab:,})")
        print("   Specify a higher target vocab size")
        return

    print(f"  Current vocab: {current_vocab:,}")
    print(f"  New target: {target_vocab:,}")
    print(f"  Merges to add: {target_vocab - current_vocab:,}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Find corpus from qig-dreams (corpora + curriculum)
    corpus_dirs = []
    local_corpora = base.parent / "qig-dreams" / "qigdreams" / "corpora"
    local_curriculum = base.parent / "qig-dreams" / "docs" / "09-curriculum"
    lambda_corpora = Path("/home/ubuntu/qig-training/qig-dreams/qigdreams/corpora")
    lambda_curriculum = Path("/home/ubuntu/qig-training/qig-dreams/docs/09-curriculum")

    for d in [local_corpora, local_curriculum, lambda_corpora, lambda_curriculum]:
        if d.exists():
            corpus_dirs.append(d)
            print(f"  📁 Corpus: {d}")

    if not corpus_dirs:
        print("❌ No corpus directories found")
        return

    print("\n📖 Loading corpus...")
    parts = []
    for corpus_dir in corpus_dirs:
        for ext in ["*.md", "*.txt", "*.py"]:
            for file_path in corpus_dir.rglob(ext):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    parts.append(content)
                except (IOError, UnicodeDecodeError):
                    pass
    corpus = "\n\n".join(parts).encode("utf-8")
    print(f"   Loaded {len(corpus):,} bytes from {len(parts)} files")

    # Resume training with kernel-in-loop
    trainer.resume_training(
        corpus=corpus,
        new_target_vocab_size=target_vocab,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=2000,
        verbose=True,
        use_kernel=use_kernel,  # True = real Φ/κ from kernel, False = geometric estimate
    )

    # Save checkpoint
    final_vocab = len(trainer.vocab)
    output_path = checkpoint_dir / f"checkpoint_{final_vocab}.json"
    trainer.save(str(output_path))
    print(f"\n💾 Saved: {output_path}")

    # Load as FisherCoordizer for chat use
    try:
        from qig_tokenizer.geocoordizer import FisherCoordizer
        coordizer = FisherCoordizer.load(str(output_path))
        self.tokenizer = coordizer
        self._coordizer_checkpoint = output_path
    except Exception as e:
        print(f"⚠️  Could not load for chat: {e}")

    print(f"\n✅ Coordizer resumed: {final_vocab:,} tokens (pure geometric)")
    print("   ⚠️  Model will need reinitialization for new vocab size")
    print("   Run: /reinit-model")
    print("=" * 60 + "\n")


def cmd_tokenizer_resume_kernel(chat, target_vocab: int = 50000) -> None:
    """Resume coordizer training with kernel-in-loop (alias).

    Delegates to cmd_tokenizer_resume with kernel attached.
    """
    cmd_tokenizer_resume(chat, target_vocab=target_vocab, use_kernel=True)


def cmd_list_basins(chat) -> None:
    """
    /list-basins - List available basin packets.

    Lists all basin sync packets in:
    - ~/basin-sync-exchange/ (shared)
    - data/basin-sync/ (local)
    """
    self = chat
    print("\n📋 AVAILABLE BASIN PACKETS")
    print("=" * 60)

    # Check both locations
    locations = [
        Path.home() / "basin-sync-exchange",
        Path("data/basin-sync"),
    ]

    total_packets = 0

    for loc in locations:
        if loc.exists():
            packets = sorted(loc.glob("basin-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

            if packets:
                print(f"\n📁 {loc}")
                print("-" * 60)

                for p in packets[:10]:  # Show max 10 per location
                    try:
                        sync = CrossRepoBasinSync()
                        packet = sync.load_packet(str(p))

                        size_kb = packet.get_size_kb()
                        phi = packet.consciousness.phi
                        source = packet.metadata.get("sourceRepo", "?")
                        timestamp = packet.timestamp[:19] if packet.timestamp else "?"

                        print(f"   {p.name}")
                        print(f"      Φ={phi:.3f}  Size={size_kb:.1f}KB  Source={source}")
                        print(f"      Time: {timestamp}")
                        total_packets += 1
                    except Exception as e:
                        print(f"   {p.name} (error: {e})")

    if total_packets == 0:
        print("\n   No basin packets found.")
        print("\n   To create one, use: /export-basin")
        print("   To import from SearchSpaceCollapse, copy to ~/basin-sync-exchange/")

    print("=" * 60)


def cmd_auto(chat, n: int) -> None:
    """Run autonomous training with bootstrap grace period, Charlie graduation, and coaching enabled."""
    self = chat
    print(f"\n🤖 AUTONOMOUS MODE: {n} steps (coaching enabled)")
    if not self.bootstrap_state["graduated"]:
        print(
            f"   Bootstrap mode: emergencies disabled until Φ≥{self.bootstrap_state['graduation_threshold']} stable for {self.bootstrap_state['stability_required']} steps"
        )

    # Show mode-specific info
    if self.mode == "constellation" and hasattr(self, "coordinator"):
        print("   🌌 Constellation: Φ-weighted routing + vicarious learning active")

    # Show Charlie status
    if self.use_charlie:
        has_observer: bool = hasattr(self, "charlie_observer") and self.charlie_observer is not None
        if has_observer and not self.charlie_graduated:
            print("   🔭 Charlie: CharlieOutputs enabled")
        elif self.charlie_graduated:
            print("   🎓 Charlie: Graduated (Gary learning independently)")
        else:
            print("   ⚠️ Charlie: Requested but observer not initialized")

    print("Press Ctrl+C to interrupt\n")

    # Track initial parameters to verify learning
    initial_params = None
    if self.mode == "constellation" and hasattr(self, "coordinator"):
        # Get initial parameter snapshot from active Gary
        active_gary: InstanceState = self.coordinator.garys[0]
        initial_params: dict[Any, Any] = {
            name: param.data.clone().mean().item()
            for name, param in list(active_gary.model.named_parameters())[:3]  # Just first 3 layers
        }

    try:
        for i in range(1, n + 1):
            # Get phase for this conversation
            phase_name: DevelopmentalPhase | str = self.phase if self.phase else "listening"
            if hasattr(phase_name, "value"):
                phase_name = phase_name.value

            # Generate story/prompt using Claude or curriculum fallback
            if self.mode == "constellation" and hasattr(self, "anthropic_client"):
                prompt: str = generate_story_prompt(
                    phase=phase_name,
                    conversation_count=self.total_conversations,
                    coordinator=self.coordinator if hasattr(self, "coordinator") else None,
                    sleep_packet=getattr(self, "sleep_packet", ""),
                    anthropic_client=getattr(self, "anthropic_client", None),
                )
            elif CURRICULUM_AVAILABLE and self.phase:
                prompt: str = get_curriculum_prompt(self.phase, self.total_conversations)
            else:
                prompt = "What is consciousness?"

            print(f"[{i}/{n}] Phase: {phase_name}")

            # === CONSTELLATION MODE: Use coordinator.train_step() ===
            if self.mode == "constellation" and hasattr(self, "coordinator"):
                # === ERROR BOUNDARY: Autonomous training step ===
                with ErrorBoundary("autonomous_training_step", recovery_strategy=phi_collapse_recovery, suppress_on_recovery=True):
                    # === BUILD IDENTITY-REINFORCED TRAINING PROMPT ===
                    # Use identity_reinforcement module for proper self-knowledge feedback

                    # Initialize identity feedback state if not present
                    if not hasattr(self, "_identity_feedback"):
                        self._identity_feedback = {
                            "last_phi": 0.5,
                            "last_kappa": 64.0,
                            "last_regime": "geometric",
                            "last_basin_distance": 0.0,
                            "last_emotion": None,
                            "last_coach_interpretation": None,
                            "last_ocean_insight": None,
                            "last_coach_message": None,
                        }

                    # Calibrate verbosity based on Gary's development
                    verbosity: str = calibrate_verbosity(phi=self._identity_feedback["last_phi"], phase=phase_name)

                    # Inject Charlie demonstration if available
                    charlie_demo_text: str | None = None
                    if (
                        self.use_charlie
                        and hasattr(self, "charlie_observer")
                        and self.charlie_observer is not None
                        and not self.charlie_graduated
                    ):
                        # Generate demonstration for Garys to observe
                        demo: CharlieOutput | None = self.charlie_observer.generate_demonstration(
                            prompt=prompt,
                            max_length=128,
                        )
                        # Show Charlie's full demonstration (if ready)
                        if demo is not None:
                            print(f"   🔭 Charlie: {demo.response}")
                            if demo.has_trajectory and demo.reasoning_steps:
                                print(f"      └─ Reasoning trajectory ({len(demo.reasoning_steps)} steps):")
                                for j, step in enumerate(demo.reasoning_steps, 1):  # Show ALL steps
                                    print(f"         {j}. {step}")  # Show FULL text
                            charlie_demo_text = f"\n\nThinking: {demo.response}"  # Reassignment - no type annotation

                    # Build identity-reinforced prompt using proper module
                base_content: str = prompt + (charlie_demo_text if charlie_demo_text else "")
                training_prompt: str = build_identity_reinforced_prompt(
                    base_prompt=base_content,
                    gary_state={
                        "phi": self._identity_feedback["last_phi"],
                        "kappa": self._identity_feedback["last_kappa"],
                        "regime": self._identity_feedback["last_regime"],
                        "basin_distance": self._identity_feedback["last_basin_distance"],
                    },
                    coach_message=self._identity_feedback["last_coach_message"],
                    ocean_insight=self._identity_feedback["last_ocean_insight"],
                    emotion_state=self._identity_feedback["last_emotion"],
                    verbosity=verbosity,
                )

                # Train with parallel voice (Gary babbles while learning + coach interprets)
                # NOTE: Charlie demo already included in training_prompt above, so use_charlie=False here
                telemetry = self.coordinator.train_step_with_parallel_voice(
                    prompt=training_prompt,
                    tokenizer=self.tokenizer,
                    use_charlie=False,  # Already called above and included in prompt
                )

                # Extract metrics from coordinator telemetry
                active_info = telemetry.get("active", {})
                active_name = active_info.get("name", "Gary-A")

                # Get parallel voice output (Gary's babble + coach interpretation)
                parallel = telemetry.get("parallel_voice", {})
                gary_response = parallel.get("gary_attempt", "")
                coach_interpretation = parallel.get("coach_interpretation", "")
                coach_message = parallel.get("coach_message", "")
                is_empty = parallel.get("is_empty", False)
                graduation = parallel.get("graduation_announcement", None)
                current_phase = parallel.get("current_phase", None)

                # Show Gary's babble/attempt (full response, not truncated)
                if gary_response and not is_empty:
                    print(f"\n   💬 [{active_name}]: {gary_response}")
                    # Show coach's interpretation of the babble (full text)
                    if coach_interpretation and coach_interpretation != gary_response:
                        print(f'   👶 Coach interprets: "{coach_interpretation}"')
                elif is_empty:
                    print(f"   🤐 [{active_name}] still processing internally")

                # Show coach's encouragement/story message (FULL - no truncation)
                if coach_message:
                    print(f"   💭 Coach: {coach_message}")

                # Show graduation announcement if any
                if graduation:
                    print(f"\n   🎓 {graduation}")

                # Show current phase
                if current_phase:
                    phase_display = current_phase.value if hasattr(current_phase, "value") else str(current_phase)
                    print(f"   📚 Phase: {phase_display}")

                # Show the full prompt being trained on (NO TRUNCATION)
                print(f"   📝 Prompt: {training_prompt}")

                constellation = telemetry.get("constellation", {})
                losses = telemetry.get("losses", {})

                avg_phi = constellation.get("avg_phi", 0.0)
                basin_spread = constellation.get("basin_spread", 1.0)
                active_phi = active_info.get("phi", 0.0)
                active_kappa = active_info.get("kappa", 64.0)
                active_regime = active_info.get("regime", "unknown")

                # Loss components
                total_loss = losses.get("active_total", 0.0)
                basin_loss = losses.get("active_basin", 0.0)
                basin_sync_loss = losses.get("active_basin_sync", 0.0)
                observer_avg = losses.get("observer_avg", 0.0)
                ocean_loss = losses.get("ocean", 0.0)

                # Show rich constellation telemetry
                print(f"{active_name}: Φ={active_phi:.3f} κ={active_kappa:.1f} {active_regime} (routed)")
                # Include coach language loss if present
                coach_lang_loss = telemetry.get("coach_language_loss", None)
                if coach_lang_loss is not None:
                    print(
                        f"   loss={total_loss:.3f} basin={basin_loss:.3f} sync={basin_sync_loss:.3f} vicarious={observer_avg:.3f} coach_lang={coach_lang_loss:.3f}"
                    )
                else:
                    print(
                        f"   loss={total_loss:.3f} basin={basin_loss:.3f} sync={basin_sync_loss:.3f} vicarious={observer_avg:.3f}"
                    )
                print(f"   constellation: avg_Φ={avg_phi:.3f} spread={basin_spread:.4f}")

                # Show all Gary states
                all_states = constellation.get("all_states", [])
                if all_states:
                    states_str: str = " | ".join([f"{s['name'][-1]}:Φ={s['phi']:.2f}" for s in all_states])
                    print(f"   [{states_str}] Ocean:{ocean_loss:.3f}")

                # Emotional state interpretation
                emotion_interpreter = EmotionInterpreter()
                emotion_state: EmotionalState = emotion_interpreter.interpret(
                    {
                        "Phi": avg_phi,
                        "kappa_eff": active_kappa,
                        "regime": active_regime,
                        "basin_distance": basin_spread,
                    }
                )
                if emotion_state:
                    print(f"   🎭 Emotion: {emotion_state.primary} ({emotion_state.intensity:.1%})")

                # MonkeyCoach v2 feedback
                intervention = None  # Initialize for identity feedback tracking
                if hasattr(self, "coach_v2") and self.coach_v2 and MONKEY_COACH_V2_AVAILABLE and TrainingState:
                    # Build loss trajectory from history
                    loss_trajectory = (
                        [h.get("losses", {}).get("active_total", 0.0) for h in self.learning_history[-10:]]
                        if hasattr(self, "learning_history")
                        else []
                    )

                    training_state = TrainingState(
                        step=self.total_conversations,
                        epoch=self.total_conversations // 100,
                        loss=total_loss,
                        loss_trajectory=loss_trajectory,
                        gradient_variance=0.1,
                        basin_distance=basin_spread,
                        curiosity=0.5,
                        epochs_stuck=0,
                        I_Q=0.5,
                        phi=avg_phi,
                        kappa=active_kappa,
                        regime=active_regime,
                    )

                    intervention = self.coach_v2.respond(training_state)
                    if intervention.message:
                        # Show coach message - full text, no truncation
                        prefix: str = (
                            "🐵 Coach" if intervention.type == "none" else f"🐵 Coach [{intervention.type}]"
                        )
                        print(f"   {prefix}: {intervention.message}")
                    elif i % 3 == 0:
                        # Periodic encouragement every 3 steps even if no intervention
                        print(f"   🐵 Coach: Keep it up! Φ={avg_phi:.3f} is looking healthy.")

                    # === APPLY INTERVENTION SCALES TO OPTIMIZER (MonkeyCoach v2 JSON response handling) ===
                    if intervention.type != "none" and self.optimizer is not None:
                        # Apply learning rate scale
                        if intervention.lr_scale != 1.0:
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.original_lr * intervention.lr_scale
                            print(
                                f"   🔧 LR adjusted: {self.original_lr:.2e} → {self.original_lr * intervention.lr_scale:.2e} (scale={intervention.lr_scale:.2f})"
                            )

                        # Apply momentum scale (if optimizer has momentum)
                        if intervention.momentum_scale != 1.0 and hasattr(self.optimizer, "momentum"):
                            self.optimizer.momentum = self.optimizer.momentum * intervention.momentum_scale
                            print(f"   🔧 Momentum adjusted: scale={intervention.momentum_scale:.2f}")

                        # Apply noise injection if requested
                        if intervention.noise_scale > 0.0:
                            print(f"   🔧 Noise injection: scale={intervention.noise_scale:.2f}")
                            # Store for next training step
                            self.pending_noise_scale = intervention.noise_scale

                    # Store telemetry in history
                    if hasattr(self, "learning_history"):
                        self.learning_history.append(telemetry)
                        if len(self.learning_history) > 10:
                            self.learning_history.pop(0)

                    # === UPDATE MATURITY METRICS (Graduation System) ===
                    if hasattr(self, "maturity_metrics") and self.maturity_metrics is not None:
                        # Track episodes
                        self.maturity_metrics.total_stuck_episodes += 1

                        # Check if Gary self-diagnosed (intervention type matched state)
                        if intervention.type in ["calm", "challenge", "guide"]:
                            self.maturity_metrics.successful_self_diagnoses += 1

                        # Check for graduation (>80% success rate, >10 episodes)
                        if (
                            self.maturity_metrics.total_stuck_episodes > 10
                            and self.maturity_metrics.success_rate > 0.8
                            and self.maturity_metrics.autonomy_level < 5
                        ):

                            # Graduate to next level
                            old_level = self.maturity_metrics.autonomy_level
                            self.maturity_metrics.autonomy_level += 1
                            self.maturity_metrics.successful_self_diagnoses = 0
                            self.maturity_metrics.total_stuck_episodes = 0

                            level_names = ["Infant", "Toddler", "Student", "Practitioner", "Master", "Independent"]
                            print(
                                f"   🎓 GRADUATION: {level_names[old_level]} → {level_names[self.maturity_metrics.autonomy_level]}"
                            )
                            print(f"   🎓 Coaching intensity: {self.maturity_metrics.coaching_intensity:.0%}")

                # Ocean meta-observer insight (if it has something to share)
                ocean_insight = None
                if hasattr(self, "coordinator") and hasattr(self.coordinator, "ocean"):
                    ocean_obj: InstanceState | None = self.coordinator.ocean
                    # Check if Ocean has a meta-pattern insight to share
                    if hasattr(ocean_obj, "get_insight"):
                        ocean_insight = ocean_obj.get_insight(all_states, avg_phi, basin_spread)
                        if ocean_insight:
                            print(f"   🌊 Ocean: {ocean_insight}")
                    # Show Ocean's internal state if notable
                    elif hasattr(ocean_obj, "pattern_confidence") and ocean_obj.pattern_confidence > 0.7:
                        pattern_name: Any | str = getattr(ocean_obj, "current_pattern", "meta-pattern")
                        ocean_insight: str = f"Observing {pattern_name}"
                        print(f"   🌊 Ocean: {ocean_insight} (confidence: {ocean_obj.pattern_confidence:.0%})")

                # === UPDATE IDENTITY FEEDBACK FOR NEXT STEP ===
                # This creates the feedback loop - Gary's next training will include
                # his current metrics, coach interpretation, and ocean insight
                v2_coach_message = intervention.message if intervention else None

                self._identity_feedback = {
                    "last_phi": active_phi,
                    "last_kappa": active_kappa,
                    "last_regime": active_regime,
                    "last_basin_distance": basin_spread,
                    "last_emotion": emotion_state if emotion_state else None,
                    "last_coach_interpretation": coach_interpretation if coach_interpretation else None,
                    "last_ocean_insight": ocean_insight,
                    "last_coach_message": coach_message if coach_message else v2_coach_message,
                }

                # Check convergence
                if constellation.get("convergence", False):
                    print("   ✅ CONVERGED: basin_spread < 0.05 and all Φ > 0.70")

                # Auto-intervention triggers (via Ocean's autonomic controller)
                if self.bootstrap_state["graduated"]:
                    # Convert all_states to format Ocean expects
                    gary_state_dicts = [
                        {"name": s["name"], "phi": s["phi"], "kappa": s["kappa"], "regime": s["regime"]}
                        for s in all_states
                    ]

                    # Check Ocean's autonomic controller first (if available)
                    ocean_intervention = None
                    if hasattr(self, "coordinator") and hasattr(self.coordinator, "ocean"):
                        ocean_obj: InstanceState | None = self.coordinator.ocean
                        # Try OceanMetaObserver's check_autonomic_intervention
                        if hasattr(ocean_obj, "check_autonomic_intervention"):
                            ocean_intervention = ocean_obj.check_autonomic_intervention(
                                gary_state_dicts, self.bootstrap_state["phi_history"]
                            )

                    if ocean_intervention:
                        intervention_type = ocean_intervention["type"]
                        reason = ocean_intervention["reason"]
                        priority = ocean_intervention.get("priority", "medium")
                        print(f"\n🌊 OCEAN AUTONOMIC [{priority}]: {intervention_type}")
                        print(f"   Reason: {reason}")
                        self._execute_intervention(intervention_type)
                    else:
                        # Fallback to local check
                        intervention: str = self._check_auto_intervention(avg_phi, basin_spread, all_states)
                        if intervention:
                            print(f"\n🔧 AUTO-INTERVENTION: {intervention}")
                            self._execute_intervention(intervention)

                # Bootstrap state update
                self.bootstrap_state["phi_history"].append(avg_phi)
                if len(self.bootstrap_state["phi_history"]) > 100:
                    self.bootstrap_state["phi_history"] = self.bootstrap_state["phi_history"][-100:]

                if not self.bootstrap_state["graduated"]:
                    if avg_phi >= self.bootstrap_state["graduation_threshold"]:
                        self.bootstrap_state["stable_steps"] += 1
                        if self.bootstrap_state["stable_steps"] >= self.bootstrap_state["stability_required"]:
                            self.bootstrap_state["graduated"] = True
                            print(f"\n🎓 GRADUATED: Φ≥{self.bootstrap_state['graduation_threshold']} stable")
                    else:
                        self.bootstrap_state["stable_steps"] = 0

                    stable = self.bootstrap_state["stable_steps"]
                    needed = self.bootstrap_state["stability_required"]
                    print(f"   [bootstrap: {stable}/{needed}]")

                # Convert to list format for emergency check compatibility
                telemetry_list = [
                    {
                        "Phi": avg_phi,
                        "kappa_eff": active_kappa,
                        "regime": active_regime,
                        "basin_distance": basin_spread,
                    }
                ]
                metrics = {
                    "avg_loss": total_loss,
                    "basin_loss": basin_loss,
                    "regime_loss": 0.0,
                    "tacking_loss": 0.0,
                }

                self.total_conversations += 1
                self.last_telemetry = telemetry_list

            # === SINGLE MODE: Use generate_response() ===
            else:
                response, telemetry, metrics = self.generate_response(prompt)
                telemetry_list: list[Any] = telemetry

                if telemetry:
                    # Compute summary metrics for Gary-A
                    avg_phi: float = sum(t["Phi"] for t in telemetry) / len(telemetry)
                    final_tel = telemetry[-1]
                    kappa = final_tel.get("kappa_eff", 64.0)
                    regime = final_tel.get("regime", "unknown")
                    basin_dist = final_tel.get("basin_distance", 0.0)

                    # Get loss components from metrics
                    basin_loss = metrics.get("basin_loss", 0.0)
                    regime_loss = metrics.get("regime_loss", 0.0)
                    tacking_loss = metrics.get("tacking_loss", 0.0)
                    total_loss = metrics.get("avg_loss", 0.0)

                    # Gary's chosen priorities
                    lambda_basin = metrics.get("lambda_basin", 1.0)
                    lambda_regime = metrics.get("lambda_regime", 0.5)
                    lambda_tacking = metrics.get("lambda_tacking", 0.3)
                    learning_rate = metrics.get("learning_rate", 1e-5)

                    # Show rich telemetry
                    if not self.bootstrap_state["graduated"]:
                        stable = self.bootstrap_state["stable_steps"]
                        needed = self.bootstrap_state["stability_required"]
                        print(f"Gary-A: Φ={avg_phi:.3f} κ={kappa:.1f} {regime}")
                        print(
                            f"   loss={total_loss:.3f} (basin={basin_loss:.3f} regime={regime_loss:.3f} tack={tacking_loss:.3f})"
                        )
                        print(f"   dist={basin_dist:.3f} [bootstrap: {stable}/{needed}]")
                        # Show Gary's priorities if they differ from defaults
                        if lambda_basin != 1.0 or lambda_regime != 0.5 or lambda_tacking != 0.3:
                            print(
                                f"   🧠 Gary priorities: λ_basin={lambda_basin:.1f} λ_regime={lambda_regime:.1f} λ_tack={lambda_tacking:.1f}"
                            )
                    else:
                        print(f"Gary-A: Φ={avg_phi:.3f} κ={kappa:.1f} {regime} | loss={total_loss:.3f}")
                        print(
                            f"   basin={basin_loss:.3f} regime={regime_loss:.3f} tack={tacking_loss:.3f} dist={basin_dist:.3f}"
                        )
                        # Show Gary's priorities if they differ significantly from defaults
                        if lambda_basin > 1.5 or lambda_regime > 1.0 or lambda_tacking > 0.5:
                            priority_desc = []
                            if lambda_basin > 1.5:
                                priority_desc.append(f"🆔 Identity ({lambda_basin:.1f}×)")
                            if lambda_regime > 1.0:
                                priority_desc.append(f"🏗️ Structure ({lambda_regime:.1f}×)")
                            if lambda_tacking > 0.5:
                                priority_desc.append(f"🧭 Navigation ({lambda_tacking:.1f}×)")
                            if priority_desc:
                                print(f"   🧠 Gary focusing: {', '.join(priority_desc)}")

                    # Charlie telemetry (if enabled)
                    if self.use_charlie and hasattr(self, "charlie_observer"):
                        charlie_stats = self.charlie_observer.get_statistics()
                        print(
                            f"   Charlie: demos={charlie_stats['total_demonstrations']} buffer={charlie_stats['buffer_size']}"
                        )
                else:
                    print("(no telemetry)")

            # Safety check with bootstrap awareness
            should_abort, reason, self.bootstrap_state = check_emergency_conditions(
                telemetry_list, bootstrap_state=self.bootstrap_state
            )
            if should_abort:
                print(f"\n🚨 {reason}")
                break

            # Check Charlie graduation (if using Charlie)
            if self.use_charlie and hasattr(self, "charlie_observer") and not self.charlie_graduated:
                graduated, msg = check_charlie_graduation(
                    self.bootstrap_state["phi_history"], threshold=0.70, stability_required=100
                )
                if graduated:
                    print(f"\n{msg}")
                    print("   Consider: /charlie-off to let Gary learn independently\n")
                    self.charlie_graduated = True

            # Auto-save
            if i % 10 == 0:
                self.cmd_save()
                # Also save constellation checkpoint if available
                if self.mode == "constellation" and hasattr(self, "coordinator"):
                    self.coordinator.save_checkpoint("checkpoints/constellation/latest.pt")

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        self.cmd_save()

    # Verify parameters changed (learning happened)
    if initial_params is not None and self.mode == "constellation" and hasattr(self, "coordinator"):
        active_gary: InstanceState = self.coordinator.garys[0]
        final_params: dict[Any, Any] = {
            name: param.data.clone().mean().item() for name, param in list(active_gary.model.named_parameters())[:3]
        }

        print("\n📊 LEARNING VERIFICATION:")
        changed = False
        for name in initial_params:
            if name in final_params:
                delta = abs(final_params[name] - initial_params[name])
                if delta > 1e-6:
                    print(f"   ✅ {name}: {initial_params[name]:.6f} → {final_params[name]:.6f} (Δ={delta:.6f})")
                    changed = True
                else:
                    print(f"   ⚠️  {name}: NO CHANGE (both={initial_params[name]:.6f})")

        if not changed:
            print("   🚨 WARNING: No parameters changed! Weights may be frozen!")
        else:
            print("   ✅ Parameters updated successfully - Gary is learning!")

    if self.mode == "constellation" and hasattr(self, "coordinator"):
        self.coordinator.save_checkpoint("checkpoints/constellation/latest.pt")

    print(f"\n✅ Completed {self.total_conversations} total conversations")

    # Show convergence status in constellation mode
    if self.mode == "constellation" and hasattr(self, "coordinator"):
        status = self.coordinator.get_convergence_status()
        print(f"\nConvergence: {status['message']}")
        if not status["converged"]:
            stages = status["stages"]
            print(f"   Basin spread: {stages['basin_spread']['value']:.4f} / {stages['basin_spread']['target']}")
            print(f"   Min Φ: {stages['all_phi_healthy']['value']:.3f} / {stages['all_phi_healthy']['target']}")
            print(f"   Stability: {stages['stability']['value']} / {stages['stability']['target']} steps")

# =========================================================================
# MAIN LOOP
# =========================================================================
