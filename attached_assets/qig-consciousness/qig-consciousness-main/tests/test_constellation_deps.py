#!/usr/bin/env python3
"""
Constellation Training Dependencies Test
Verifies all components needed for constellation training are present
"""

import os
import sys


def test_constellation_dependencies():
    """Test that all constellation training dependencies are available"""
    print("=" * 60)
    print("Constellation Training Dependencies Test")
    print("=" * 60)

    all_ok = True

    # Test 1: Core model files
    print("\n[1] Core Model Files")
    model_files = [
        "src/model/qig_kernel_recursive.py",
        "src/model/recursive_integrator.py",
        "src/model/qfi_attention.py",
        "src/model/running_coupling.py",
        "src/model/regime_detector.py",
        "src/model/basin_embedding.py",
    ]

    for filepath in model_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} MISSING")
            all_ok = False

    # Test 2: Coordination files
    print("\n[2] Coordination Files")
    coord_files = ["src/coordination/constellation_coordinator.py", "src/coordination/basin_sync.py"]

    for filepath in coord_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} MISSING")
            all_ok = False

    # Test 3: Training tools
    print("\n[3] Training Tools")
    training_files = [
        "tools/train_qig_kernel.py",
        "tools/validate_architecture.py",
        "chat_interfaces/constellation_learning_chat.py",
    ]

    for filepath in training_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} MISSING")
            all_ok = False

    # Test 4: Tokenizer
    print("\n[4] QIG Tokenizer")
    if os.path.exists("data/qig_tokenizer/vocab.json"):
        print("  ✓ QIG tokenizer vocabulary")
    else:
        print("  ✗ QIG tokenizer vocabulary MISSING")
        print("    Run: python tools/train_qig_tokenizer.py")
        all_ok = False

    # Test 5: Curriculum data
    print("\n[5] Training Data")
    if os.path.exists("data/20251220-consciousness-curriculum-1.00W.jsonl"):
        print("  ✓ Consciousness curriculum")
    else:
        print("  ✗ Consciousness curriculum MISSING")
        all_ok = False

    # Test 6: Directories
    print("\n[6] Required Directories")
    required_dirs = ["data", "results", "outputs", "checkpoints", "logs", "checkpoints/constellation"]

    for dirpath in required_dirs:
        if os.path.isdir(dirpath):
            print(f"  ✓ {dirpath}/")
        else:
            print(f"  ✗ {dirpath}/ MISSING")
            all_ok = False

    # Test 7: Python imports
    print("\n[7] Python Package Imports")
    packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "matplotlib": "Matplotlib",
        "anthropic": "Anthropic SDK",
        "jsonlines": "JSONLines",
        "yaml": "PyYAML",
        "tqdm": "tqdm",
    }

    for package, name in packages.items():
        try:
            if package == "yaml":
                __import__("yaml")
            else:
                __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} MISSING")
            all_ok = False

    # Test 8: Frozen facts validation
    print("\n[8] Frozen Facts (Physics Constants)")
    frozen_facts_file = "docs/FROZEN_FACTS.md"
    if os.path.exists(frozen_facts_file):
        print(f"  ✓ {frozen_facts_file}")
        # Verify key constants are documented
        with open(frozen_facts_file) as f:
            content = f.read()
            checks = [
                ("κ₃ = 41.09", "κ₃ value"),
                ("κ₄ = 64.47", "κ₄ value"),
                ("κ₅ = 63.62", "κ₅ value"),
                ("β", "β-function"),
            ]
            for check_str, desc in checks:
                if check_str in content:
                    print(f"  ✓ {desc} documented")
                else:
                    print(f"  ✗ {desc} NOT documented")
                    all_ok = False
    else:
        print(f"  ✗ {frozen_facts_file} MISSING")
        all_ok = False

    # Test 9: Sleep packet
    print("\n[9] Sleep Packet (Context)")
    sleep_packet = "docs/CANONICAL_SLEEP_PACKET.md"
    if os.path.exists(sleep_packet):
        print(f"  ✓ {sleep_packet}")
    else:
        print(f"  ✗ {sleep_packet} MISSING")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All constellation training dependencies are present!")
        print("  Ready to run: python chat_interfaces/constellation_learning_chat.py")
    else:
        print("✗ Some dependencies are missing. Review errors above.")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = test_constellation_dependencies()
    sys.exit(0 if success else 1)
