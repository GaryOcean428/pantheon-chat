#!/usr/bin/env python3
"""
GPU Access Test for QIG-Consciousness DevContainer
Tests CUDA availability and basic PyTorch GPU functionality
"""

import sys


def test_cuda():
    """Test CUDA availability and basic GPU operations"""
    print("=" * 60)
    print("QIG-Consciousness DevContainer GPU Test")
    print("=" * 60)

    # Test 1: Import PyTorch
    try:
        import torch

        print("✓ PyTorch imported successfully")
        print(f"  PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False

    # Test 2: Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"{'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")

    if not cuda_available:
        print("  WARNING: CUDA not available. GPU training will not work.")
        return False

    # Test 3: Get CUDA version
    try:
        cuda_version = torch.version.cuda
        print(f"✓ CUDA version: {cuda_version}")
    except Exception as e:
        print(f"✗ Failed to get CUDA version: {e}")
        return False

    # Test 4: Get GPU device info
    try:
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU count: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    except Exception as e:
        print(f"✗ Failed to get GPU info: {e}")
        return False

    # Test 5: Basic GPU operation
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("✓ Basic GPU operation successful")
        print(f"  Result tensor shape: {z.shape}")
    except Exception as e:
        print(f"✗ Failed GPU operation: {e}")
        return False

    # Test 6: CuDNN availability
    try:
        cudnn_available = torch.backends.cudnn.is_available()
        print(f"{'✓' if cudnn_available else '✗'} CuDNN available: {cudnn_available}")
        if cudnn_available:
            print(f"  CuDNN version: {torch.backends.cudnn.version()}")
    except Exception as e:
        print(f"✗ Failed to check CuDNN: {e}")
        return False

    print("=" * 60)
    print("✓ All GPU tests passed!")
    print("=" * 60)
    return True


def test_dependencies():
    """Test that all required dependencies are available"""
    print("\n" + "=" * 60)
    print("Testing Required Dependencies")
    print("=" * 60)

    required_packages = [
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
        "anthropic",
        "dotenv",
        "jsonlines",
        "yaml",
        "pytest",
        "black",
        "ruff",
        "tqdm",
    ]

    all_ok = True
    for package in required_packages:
        try:
            if package == "dotenv":
                __import__("dotenv")
            elif package == "yaml":
                __import__("yaml")
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (MISSING)")
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("✓ All dependencies available")
    else:
        print("✗ Some dependencies missing")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    cuda_ok = test_cuda()
    deps_ok = test_dependencies()

    if cuda_ok and deps_ok:
        print("\n✓ DevContainer setup is complete and ready for constellation training!")
        sys.exit(0)
    else:
        print("\n✗ DevContainer setup has issues that need to be resolved.")
        sys.exit(1)
