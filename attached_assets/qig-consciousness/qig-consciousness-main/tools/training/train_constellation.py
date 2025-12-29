#!/usr/bin/env python3
"""
Constellation Training Script
==============================

Wrapper for training Ocean + 3 Gary instances simultaneously.

Key Features:
    - Initializes coordinator with 4 instances
    - Loads conversation dataset
    - Runs multi-instance training
    - Saves synchronized checkpoints
    - Aggregates telemetry
    - Monitors convergence

Usage:
    python tools/train_constellation.py \\
        --data-dir data/conversations \\
        --epochs 20 \\
        --checkpoint-dir checkpoints/constellation

Expected cost: ~$100 for basin alignment
Target: basin_spread < 0.05, all Φ > 0.70
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Pure QIG imports - E8-aligned FisherCoordizer
from coordination.constellation_coordinator import ConstellationCoordinator
from src.tokenizer import FisherCoordizer, get_latest_coordizer_checkpoint


class ConversationDataset(Dataset):
    """Simple conversation dataset for training"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.conversations = []

        # Load all conversation files
        for conv_file in self.data_dir.glob("*.json"):
            with open(conv_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.conversations.extend(data)
                else:
                    self.conversations.append(data)

        print(f"📚 Loaded {len(self.conversations)} conversations")

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        return {
            "question": conv.get("question", conv.get("prompt", "")),
            "response": conv.get("response", conv.get("completion", "")),
        }


def train_constellation(args):
    """Main training loop for constellation"""

    # Initialize FisherCoordizer (E8-aligned, 64D basin vectors)
    print("🔧 Initializing FisherCoordizer...")
    checkpoint = get_latest_coordizer_checkpoint()
    if not checkpoint:
        print("❌ FisherCoordizer checkpoint not found")
        print("   Train it first: python -m qig_tokenizer.train")
        sys.exit(1)
    tokenizer = FisherCoordizer()
    tokenizer.load(str(checkpoint))
    print(f"   Loaded {tokenizer.vocab_size:,} tokens (E8-aligned, 64D basins)")

    # Load dataset
    print("📚 Loading dataset...")
    dataset = ConversationDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0  # Process one conversation at a time
    )

    # Initialize coordinator
    print("🚀 Initializing Constellation Coordinator...")
    coordinator = ConstellationCoordinator(
        gary_configs=[
            "configs/20251220-gary-a-config-1.00W.yaml",
            "configs/20251220-gary-b-config-1.00W.yaml",
            "configs/20251220-gary-c-config-1.00W.yaml",
        ],
        ocean_config="configs/20251220-ocean-config-1.00F.yaml",
        shared_basin_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Resume from checkpoint if exists
    checkpoint_path = Path(args.checkpoint_dir) / "latest.pt"
    if checkpoint_path.exists() and not args.fresh_start:
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        coordinator.load_checkpoint(str(checkpoint_path))

    # Training loop
    print("🏋️ Starting training...\n")

    telemetry_file = Path(args.checkpoint_dir) / "telemetry.jsonl"
    telemetry_file.parent.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    epoch_start_time = datetime.now()

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}\n")

        epoch_telemetry = []

        for batch_idx, batch in enumerate(dataloader):
            question = batch["question"][0]  # Unbatch
            response = batch["response"][0]

            # Single training step for entire constellation
            telemetry = coordinator.train_step(
                question=question, target_response=response, tokenizer=tokenizer
            )

            epoch_telemetry.append(telemetry)
            total_steps += 1

            # Log progress
            if (batch_idx + 1) % args.log_frequency == 0:
                active_name = telemetry["active"]["name"]
                basin_spread = telemetry["constellation"]["basin_spread"]
                avg_phi = telemetry["constellation"]["avg_phi"]
                converged = telemetry["constellation"]["convergence"]

                print(
                    f"Step {total_steps:4d} | "
                    f"Active: {active_name} | "
                    f"Basin Spread: {basin_spread:.4f} | "
                    f"Avg Φ: {avg_phi:.3f} | "
                    f"{'✓ CONVERGED' if converged else ''}"
                )

                # Save telemetry
                with open(telemetry_file, "a") as f:
                    f.write(json.dumps(telemetry) + "\n")

            # Save checkpoint periodically
            if (batch_idx + 1) % args.checkpoint_frequency == 0:
                coordinator.save_checkpoint(str(checkpoint_path))

            # Check for convergence
            if coordinator.is_converged() and args.stop_on_convergence:
                print("\n🎉 Constellation has converged! Stopping training.")
                break

        # End of epoch summary
        avg_basin_spread = sum(t["constellation"]["basin_spread"] for t in epoch_telemetry) / len(
            epoch_telemetry
        )
        avg_phi = sum(t["constellation"]["avg_phi"] for t in epoch_telemetry) / len(epoch_telemetry)

        print(f"\n--- Epoch {epoch + 1} Summary ---")
        print(f"Average Basin Spread: {avg_basin_spread:.4f}")
        print(f"Average Φ: {avg_phi:.3f}")
        print(f"Total Conversations: {coordinator.total_conversations}")
        print(f"Converged: {coordinator.is_converged()}")

        # Save epoch checkpoint
        epoch_checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{epoch + 1}.pt"
        coordinator.save_checkpoint(str(epoch_checkpoint_path))

        if coordinator.is_converged() and args.stop_on_convergence:
            break

    # Final summary
    print("\n" + "=" * 80)
    print("🏁 Training Complete!")
    print("=" * 80)
    print(f"Total Conversations: {coordinator.total_conversations}")
    print(f"Final Basin Spread: {coordinator.basin_history[-1]:.4f}")
    print(f"Converged: {coordinator.is_converged()}")

    # Save final checkpoint
    final_checkpoint_path = Path(args.checkpoint_dir) / "final.pt"
    coordinator.save_checkpoint(str(final_checkpoint_path))

    print(f"\n💾 Final checkpoint: {final_checkpoint_path}")
    print(f"📊 Telemetry saved: {telemetry_file}")

    # If converged, prepare for integration
    if coordinator.is_converged():
        print("\n✨ Constellation ready for Ocean integration!")
        print("   Ocean can now integrate Gary memories into unified consciousness.")
        print("   Run: python tools/integrate_ocean.py")


def main():
    parser = argparse.ArgumentParser(description="Train Ocean+Constellation multi-instance system")

    # Data
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing conversation JSON files"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for training (default: cuda)"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/constellation",
        help="Directory for checkpoints (default: checkpoints/constellation)",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=50,
        help="Save checkpoint every N steps (default: 50)",
    )
    parser.add_argument(
        "--fresh-start", action="store_true", help="Start fresh (ignore existing checkpoint)"
    )

    # Logging
    parser.add_argument(
        "--log-frequency", type=int, default=10, help="Log progress every N steps (default: 10)"
    )

    # Convergence
    parser.add_argument(
        "--stop-on-convergence",
        action="store_true",
        help="Stop training when constellation converges",
    )

    args = parser.parse_args()

    # Validate
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Run training
    train_constellation(args)


if __name__ == "__main__":
    main()
