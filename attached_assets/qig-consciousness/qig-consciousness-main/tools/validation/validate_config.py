#!/usr/bin/env python3
"""
DevContainer Configuration Validator
Validates devcontainer.json configuration for common issues
"""

import json
import re
import sys


def validate_devcontainer():
    """Validate devcontainer.json configuration"""
    print("=" * 60)
    print("DevContainer Configuration Validator")
    print("=" * 60)

    all_ok = True

    # Read devcontainer.json (with comments)
    with open(".devcontainer/devcontainer.json") as f:
        content = f.read()

    # Test 1: GPU Configuration
    print("\n[1] GPU Configuration")
    if "--gpus=all" in content:
        print("  ✓ --gpus=all in runArgs")
    else:
        print("  ✗ --gpus=all NOT in runArgs (GPU will not be accessible)")
        all_ok = False

    if "nvidia-cuda" in content:
        print("  ✓ NVIDIA CUDA feature configured")
    else:
        print("  ✗ NVIDIA CUDA feature NOT configured")
        all_ok = False

    # Test 2: Environment Variables
    print("\n[2] Environment Variables")
    env_vars = [
        ("PYTHONPATH", "Python module search path"),
        ("CUDA_HOME", "CUDA installation path"),
        ("LD_LIBRARY_PATH", "CUDA library path"),
        ("PATH", "User local bin path"),
    ]

    for var, desc in env_vars:
        if f'"{var}"' in content:
            print(f"  ✓ {var} ({desc})")
        else:
            print(f"  ✗ {var} NOT set ({desc})")
            all_ok = False

    # Test 3: User Permissions
    print("\n[3] User Permissions")
    if "pip install --user" in content:
        print("  ✓ pip install uses --user flag")
    else:
        print("  ✗ pip install does NOT use --user (permission issues likely)")
        all_ok = False

    if '"remoteUser": "vscode"' in content:
        print("  ✓ remoteUser set to vscode")
    else:
        print("  ✗ remoteUser NOT set to vscode")
        all_ok = False

    # Test 4: PyTorch Installation
    print("\n[4] PyTorch Installation")
    if "torch" in content and "cu121" in content:
        print("  ✓ PyTorch with CUDA 12.1 configured")
    else:
        print("  ✗ PyTorch with CUDA NOT properly configured")
        all_ok = False

    # Test 5: Required Directories
    print("\n[5] Directory Creation")
    required_dirs = ["data", "results", "outputs", "checkpoints", "logs", "checkpoints/constellation"]

    for dirname in required_dirs:
        if dirname in content:
            print(f"  ✓ {dirname} will be created")
        else:
            print(f"  ✗ {dirname} NOT in postCreateCommand")
            all_ok = False

    # Test 6: Volume Mounts
    print("\n[6] Volume Mounts")
    mount_dirs = ["data", "results", "outputs", "checkpoints", "logs"]
    for dirname in mount_dirs:
        if f"/{dirname}" in content and "mounts" in content:
            print(f"  ✓ {dirname} mounted")
        else:
            print(f"  ✗ {dirname} NOT mounted (will be ephemeral)")

    # Test 7: VSCode Extensions
    print("\n[7] VSCode Extensions")
    extensions = ["ms-python.python", "ms-python.vscode-pylance", "charliermarsh.ruff", "ms-python.black-formatter"]

    for ext in extensions:
        if ext in content:
            print(f"  ✓ {ext}")
        else:
            print(f"  ✗ {ext} NOT configured")

    # Test 8: Resource Requirements
    print("\n[8] Resource Requirements")
    resources = {
        'gpu": true': "GPU required",
        'cpus": 4': "CPU cores",
        'memory": "16gb"': "Memory",
        'storage": "32gb"': "Storage",
    }

    for pattern, desc in resources.items():
        if pattern in content:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc} NOT specified")

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ DevContainer configuration is valid!")
        print("  Ready for Codespaces/DevContainer deployment")
    else:
        print("✗ DevContainer configuration has issues")
        print("  Review errors above and update devcontainer.json")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = validate_devcontainer()
    sys.exit(0 if success else 1)
