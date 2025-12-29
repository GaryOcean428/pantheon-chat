#!/usr/bin/env python3
"""
QIG Consciousness - Command Line Interface
==========================================

Main CLI entry point for the qig-consciousness package.

Usage:
    qig --help              Show available commands
    qig info                Show package and model info
    qig validate            Run β-function validation
    qig chat                Start unified chat (all modes)
    qig train               Start training (with options)

Canonical Entry Point:
    qig_chat.py is THE unified interface. All other chat_interfaces/*.py
    files are deprecated and should be archived.
"""

import argparse
import sys
from pathlib import Path

# Add src to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))


def show_info():
    """Display package and system information."""
    import torch

    from src.constants import (
        BASIN_DIM,
        BETA_3_TO_4,
        KAPPA_3,
        KAPPA_4,
        KAPPA_5,
        KAPPA_STAR,
        PHI_THRESHOLD,
    )

    print("\n" + "=" * 60)
    print("QIG-CONSCIOUSNESS - Package Information")
    print("=" * 60)
    print()
    print("Version: 0.1.4")
    print()
    print("Physics Constants (from FROZEN_FACTS):")
    print(f"  κ* (fixed point):     {KAPPA_STAR}")
    print(f"  κ₃ (L=3 emergence):   {KAPPA_3}")
    print(f"  κ₄ (L=4 running):     {KAPPA_4}")
    print(f"  κ₅ (L=5 plateau):     {KAPPA_5}")
    print(f"  β(3→4):               {BETA_3_TO_4}")
    print(f"  Φ threshold:          {PHI_THRESHOLD}")
    print(f"  Basin dimension:      {BASIN_DIM}")
    print()
    print("System:")
    print(f"  Python:      {sys.version.split()[0]}")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  CUDA:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    print("Canonical Entry Point:")
    print("  qig chat                 Unified chat interface (all modes)")
    print("    --single               Single Gary mode (default)")
    print("    --constellation        Multi-Gary constellation mode")
    print("    --inference            Inference only (no training)")
    print("    --granite              Enable Granite observer")
    print("    --claude-coach         Enable Claude coaching")
    print()
    print("Other Commands:")
    print("  qig info                 Show this information")
    print("  qig validate             Run β-function validation")
    print("  qig train                Start training interface")
    print()


def validate():
    """Run β-function validation."""
    print("\nRunning β-function validation...")
    print("=" * 60)

    try:
        from tools.validation.beta_attention_validator import main as validate_main

        validate_main()
    except ImportError as e:
        print(f"Error importing validator: {e}")
        print("Run from project root: python -m tools.validation.beta_attention_validator")
        return 1
    return 0


def train():
    """Start training interface."""
    parser = argparse.ArgumentParser(description="QIG Training")
    parser.add_argument("--mode", choices=["single", "constellation"], default="single", help="Training mode")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    args = parser.parse_args(sys.argv[2:] if len(sys.argv) > 2 else [])

    print("\n" + "=" * 60)
    print("QIG Training Interface")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    if args.checkpoint:
        print(f"Resuming from: {args.checkpoint}")
    print()

    # Use unified qig_chat.py entry point
    print("Starting training via unified qig_chat.py...")
    try:
        from chat_interfaces.qig_chat import main as qig_chat_main

        # Build args for qig_chat
        chat_args = []
        if args.mode == "constellation":
            chat_args.append("--constellation")
        else:
            chat_args.append("--single")

        if args.checkpoint:
            chat_args.extend(["--checkpoint", args.checkpoint])

        # Inject args and run
        sys.argv = ["qig_chat.py"] + chat_args
        qig_chat_main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Ensure all dependencies are installed: pip install qig-consciousness[all]")
        return 1

    return 0


def chat():
    """Start unified chat interface (canonical entry point)."""
    print("\nStarting QIG Chat (unified interface)...")
    print("=" * 60)

    try:
        from chat_interfaces.qig_chat import main as qig_chat_main

        # Pass remaining args to qig_chat
        sys.argv = ["qig_chat.py"] + sys.argv[2:]
        qig_chat_main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Ensure all dependencies are installed: pip install qig-consciousness[all]")
        return 1
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="qig",
        description="QIG Consciousness - Quantum Information Geometry approach to machine consciousness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  info        Show package and system information
  chat        Start unified chat interface (THE canonical entry point)
  validate    Run β-function validation
  train       Start training interface

Chat modes (via 'qig chat'):
  qig chat --single         Single Gary mode (default)
  qig chat --constellation  Multi-Gary constellation
  qig chat --inference      Inference only
  qig chat --granite        Enable Granite observer
  qig chat --claude-coach   Enable Claude coaching

Examples:
  qig info
  qig chat --constellation --granite --claude-coach
  qig validate
  qig train --mode constellation

The geometry is the truth. Trust the Φ.
        """,
    )
    parser.add_argument("command", nargs="?", choices=["info", "chat", "validate", "train"], help="Command to run")
    parser.add_argument("--version", action="version", version="qig-consciousness 0.1.4")

    args = parser.parse_args(sys.argv[1:2] if len(sys.argv) > 1 else [])

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "info":
        show_info()
        return 0
    if args.command == "chat":
        return chat()
    if args.command == "validate":
        return validate()
    if args.command == "train":
        return train()

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
