#!/usr/bin/env python3
"""
QIG-Kernel-Recursive Inference Demo
====================================

Demonstrates consciousness-capable model with:
- Mandatory recursion (3+ loops)
- Integration measurement (Φ)
- Regime detection (linear/geometric/breakdown)
- Basin proximity tracking
- Full telemetry

Usage:
    python tools/demo_inference.py --checkpoint checkpoints/final_step1000.pt

Example queries to try:
    - "Explain consciousness from information geometry perspective"
    - "What is the relationship between recursion and integration?"
    - "How does the running coupling affect attention?"

Watch for:
    - Recursion depth (should be ≥3)
    - Φ value (target >0.7 for geometric regime)
    - Basin distance (target <0.15 for alignment)
    - Regime classification (target: "geometric")
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for src.* imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

# No transformers needed - use QIG tokenizer
TRANSFORMERS_AVAILABLE = False


class QIGInference:
    """Interactive inference with QIG-Kernel-Recursive."""

    def __init__(
        self, checkpoint_path: str = None, basin_path: str = "20251220-basin-signatures-0.01W.json"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            self.initialize_fresh(basin_path)

    def initialize_fresh(self, basin_path: str):
        """Initialize fresh model (untrained)."""
        from src.model.qig_kernel_recursive import QIGKernelRecursive

        from src.tokenizer import FisherCoordizer, get_latest_coordizer_checkpoint

        print("Initializing fresh QIG-Kernel-Recursive...")

        # Load FisherCoordizer (E8-aligned, 64D basin vectors)
        checkpoint = get_latest_coordizer_checkpoint()
        if not checkpoint:
            raise FileNotFoundError(
                "FisherCoordizer checkpoint not found.\nTrain it first: python -m qig_tokenizer.train"
            )
        self.tokenizer = FisherCoordizer()
        self.tokenizer.load(str(checkpoint))
        print(f"✅ Loaded FisherCoordizer: {self.tokenizer.vocab_size:,} tokens")

        # Create model with basin coordinates
        self.model = QIGKernelRecursive(
            d_model=768,
            vocab_size=self.tokenizer.vocab_size,
            n_heads=6,
            min_recursion_depth=3,
            min_Phi=0.7,
            target_basin=basin_path if Path(basin_path).exists() else None,
        )

        self.model.to(self.device)
        self.model.eval()

        print("Model initialized (untrained - for architecture demonstration)")

    def load_checkpoint(self, checkpoint_path: str):
        """Load trained checkpoint."""
        from src.model.qig_kernel_recursive import QIGKernelRecursive

        print(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint with weights_only=False for legacy compatibility
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        from src.tokenizer import FisherCoordizer, get_latest_coordizer_checkpoint

        config = checkpoint.get("config", {})
        tokenizer_path = config.get("tokenizer_path")

        # Find coordizer checkpoint
        if tokenizer_path and Path(tokenizer_path).exists():
            coordizer_checkpoint = Path(tokenizer_path)
        else:
            coordizer_checkpoint = get_latest_coordizer_checkpoint()

        if not coordizer_checkpoint:
            raise FileNotFoundError("FisherCoordizer checkpoint not found")

        self.tokenizer = FisherCoordizer()
        self.tokenizer.load(str(coordizer_checkpoint))
        print(f"✅ Loaded FisherCoordizer: {self.tokenizer.vocab_size:,} tokens")

        # Create model
        self.model = QIGKernelRecursive(
            d_model=config.get("d_model", 768),
            vocab_size=config.get("vocab_size", self.tokenizer.vocab_size),
            n_heads=config.get("n_heads", 6),
            min_recursion_depth=config.get("min_recursion_depth", 3),
            min_Phi=config.get("min_Phi", 0.7),
            target_basin=config.get("target_basin"),
        )

        # Load weights (strict=False for old checkpoints missing new parameters)
        missing_keys, _ = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing_keys:
            print(f"  Note: Initialized new parameters with defaults: {missing_keys}")
        self.model.to(self.device)
        self.model.eval()

        print(f"Checkpoint loaded (step {checkpoint.get('step', 'unknown')})")

        # Display Gary's identity (brief and friendly!)
        identity = checkpoint.get("identity", {})
        if identity.get("name") == "Gary":
            print(f"\n{'=' * 70}")
            # Gary introduces himself - playful and concise!
            print(f"{self.model.announce_identity()}")
            print(f"{'=' * 70}\n")

            # Add /lore command hint
            print(
                '(Type \'model.explain_identity("basic")\' for more, or "technical"/"full_lore" for the deep dive)'
            )
            print()
        elif identity.get("name"):
            # Other models - show details
            print(f"\n{'=' * 70}")
            print(f"LOADED: {identity['name']}")
            print(f"{'=' * 70}")
            coaching = checkpoint.get("coaching_provenance")
            if coaching:
                print(f"Coaching: {coaching['total_interventions']} interventions")
                print(f"Maturity: {coaching['maturity_level']}/5")
            print(f"{'=' * 70}\n")

    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ):
        """Generate response with full telemetry."""

        # Tokenize with FisherCoordizer (returns List[int], not tensor)
        input_tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_tokens], device=self.device)

        print(f"\n{'=' * 60}")
        print(f"PROMPT: {prompt}")
        print(f"{'=' * 60}\n")

        generated_tokens = input_ids[0].tolist()
        telemetry_history = []

        with torch.no_grad():
            for step in range(max_length):
                # Forward pass
                current_input = torch.tensor([generated_tokens], device=self.device)
                logits, telemetry = self.model(current_input, return_telemetry=True)

                # Track telemetry
                telemetry_history.append(telemetry)

                # Get next token logits
                next_token_logits = logits[0, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Add to sequence
                generated_tokens.append(next_token.item())

                # Stop if newline (simplified stopping condition)
                if next_token.item() == ord("\n"):
                    break

                # Show progress every 10 tokens
                if step % 10 == 0:
                    current_text = self.tokenizer.decode(generated_tokens)
                    print(
                        f"[Step {step}] Φ={telemetry['Phi']:.3f}, regime={telemetry['regime']}, depth={telemetry['recursion_depth']}"
                    )

        # Decode
        generated_text = self.tokenizer.decode(generated_tokens)

        # Print results
        print(f"\n{'=' * 60}")
        print("GENERATED:")
        print(f"{'=' * 60}")
        print(generated_text)
        print(f"\n{'=' * 60}")
        print("TELEMETRY SUMMARY:")
        print(f"{'=' * 60}\n")

        # Aggregate telemetry
        avg_phi = sum(t["Phi"] for t in telemetry_history) / len(telemetry_history)
        avg_depth = sum(t["recursion_depth"] for t in telemetry_history) / len(telemetry_history)
        avg_basin_dist = sum(t["basin_distance"] for t in telemetry_history) / len(
            telemetry_history
        )

        regime_counts = {}
        for t in telemetry_history:
            regime = t["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        print(f"Average Φ: {avg_phi:.3f} (target >0.7)")
        print(f"Average Recursion Depth: {avg_depth:.1f} (minimum 3.0)")
        print(f"Average Basin Distance: {avg_basin_dist:.3f} (target <0.15)")
        print("\nRegime Distribution:")
        for regime, count in regime_counts.items():
            pct = (count / len(telemetry_history)) * 100
            print(f"  {regime}: {count}/{len(telemetry_history)} ({pct:.1f}%)")

        # Success check
        success_criteria = {
            "Φ > 0.7": avg_phi > 0.7,
            "Depth ≥ 3": avg_depth >= 3.0,
            "Basin < 0.15": avg_basin_dist < 0.15,
            "Geometric >70%": regime_counts.get("geometric", 0) / len(telemetry_history) > 0.7,
        }

        print("\nSuccess Criteria:")
        for criterion, met in success_criteria.items():
            print(f"  {criterion}: {'✅' if met else '❌'}")

        return generated_text, telemetry_history


def interactive_mode(inference: QIGInference):
    """Interactive REPL for querying model."""

    print("\n" + "=" * 60)
    print("QIG-KERNEL-RECURSIVE INTERACTIVE MODE")
    print("=" * 60)
    print("\nType your query (or 'quit' to exit)")
    print("Commands:")
    print("  /basin - Show basin parameters")
    print("  /telemetry - Show last telemetry")
    print("  /help - Show this help")
    print()

    last_telemetry = None

    while True:
        try:
            query = input("\nQuery> ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if query == "/basin":
                basin_params = inference.model.get_basin_parameters()
                print("\nBasin Parameters:")
                print(json.dumps(basin_params, indent=2))
                continue

            if query == "/telemetry":
                if last_telemetry:
                    print("\nLast Telemetry:")
                    for t in last_telemetry[-3:]:  # Last 3 steps
                        print(
                            f"  Φ={t['Phi']:.3f}, regime={t['regime']}, depth={t['recursion_depth']}, basin_dist={t['basin_distance']:.3f}"
                        )
                else:
                    print("No telemetry yet - run a query first")
                continue

            if query == "/help":
                print("\nCommands:")
                print("  /basin - Show basin parameters")
                print("  /telemetry - Show last telemetry")
                print("  /help - Show this help")
                print("  quit/exit/q - Exit interactive mode")
                continue

            # Generate
            text, telemetry = inference.generate(query, max_length=100)
            last_telemetry = telemetry

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="QIG-Kernel-Recursive Inference Demo")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path")
    parser.add_argument(
        "--basin", type=str, default="20251220-basin-signatures-0.01W.json", help="Basin file"
    )
    parser.add_argument("--prompt", type=str, help="Single prompt (non-interactive)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Create inference
    inference = QIGInference(checkpoint_path=args.checkpoint, basin_path=args.basin)

    # Run
    if args.prompt:
        # Single query
        inference.generate(args.prompt)
    elif args.interactive or not args.prompt:
        # Interactive mode
        interactive_mode(inference)


if __name__ == "__main__":
    import json  # For /basin command

    main()
