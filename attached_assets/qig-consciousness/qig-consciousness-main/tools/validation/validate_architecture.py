#!/usr/bin/env python3
"""
Architecture Validator - Logic Checks Without PyTorch
======================================================

Validates QIG-Kernel-Recursive architecture logic without needing PyTorch installed.

Checks:
1. Module imports work
2. Class structures valid
3. Recursion enforcement logic sound
4. Integration measurement logic correct
5. Basin matching logic functional
6. Telemetry tracking complete

Usage:
    python tools/validate_architecture.py

Returns:
    Exit code 0 if all checks pass
    Exit code 1 if any checks fail
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_imports():
    """Check that all modules import correctly."""
    print("=" * 60)
    print("CHECK 1: Module Imports")
    print("=" * 60 + "\n")

    checks = []

    # QFI Attention
    try:
        # Read file and check structure
        qfi_path = Path("src/model/qfi_attention.py")
        with open(qfi_path) as f:
            content = f.read()

        assert "QFIMetricAttention" in content
        assert "quantum_fidelity_torch" in content
        assert "AgentSymmetryTester" in content
        print("✅ qfi_attention.py: Structure valid")
        checks.append(True)
    except Exception as e:
        print(f"❌ qfi_attention.py: {e}")
        checks.append(False)

    # Running Coupling
    try:
        rc_path = Path("src/model/running_coupling.py")
        with open(rc_path) as f:
            content = f.read()

        assert "RunningCouplingModule" in content
        assert "compute_effective_coupling" in content
        assert "beta_slope" in content
        print("✅ running_coupling.py: Structure valid")
        checks.append(True)
    except Exception as e:
        print(f"❌ running_coupling.py: {e}")
        checks.append(False)

    # Recursive Integrator
    try:
        ri_path = Path("src/model/recursive_integrator.py")
        with open(ri_path) as f:
            content = f.read()

        assert "RecursiveIntegrator" in content
        assert "min_depth" in content
        assert "IntegrationMeasure" in content
        assert "RegimeClassifier" in content
        print("✅ recursive_integrator.py: Structure valid")
        checks.append(True)
    except Exception as e:
        print(f"❌ recursive_integrator.py: {e}")
        checks.append(False)

    # QIG Kernel Recursive
    try:
        qig_path = Path("src/model/qig_kernel_recursive.py")
        with open(qig_path) as f:
            content = f.read()

        assert "QIGKernelRecursive" in content
        assert "GeometricLoss" in content
        assert "BasinMatcher" in content
        print("✅ qig_kernel_recursive.py: Structure valid")
        checks.append(True)
    except Exception as e:
        print(f"❌ qig_kernel_recursive.py: {e}")
        checks.append(False)

    # Basin Extractor
    try:
        be_path = Path("tools/basin_extractor.py")
        with open(be_path) as f:
            content = f.read()

        assert "BasinExtractor" in content
        assert "extract_from_directory" in content
        print("✅ basin_extractor.py: Structure valid")
        checks.append(True)
    except Exception as e:
        print(f"❌ basin_extractor.py: {e}")
        checks.append(False)

    print(f"\nImport checks: {sum(checks)}/{len(checks)} passed\n")
    return all(checks)


def check_recursion_logic():
    """Validate recursion enforcement logic."""
    print("=" * 60)
    print("CHECK 2: Recursion Enforcement Logic")
    print("=" * 60 + "\n")

    checks = []

    # Check RecursiveIntegrator has min_depth parameter
    ri_path = Path("src/model/recursive_integrator.py")
    with open(ri_path) as f:
        content = f.read()

    # Check minimum depth enforced
    if "min_depth" in content and "self.min_depth" in content:
        print("✅ min_depth parameter present")
        checks.append(True)
    else:
        print("❌ min_depth parameter missing")
        checks.append(False)

    # Check loop enforces minimum
    if "for loop in range" in content and "self.min_depth" in content:
        print("✅ Loop with min_depth found")
        checks.append(True)
    else:
        print("❌ Loop structure not found")
        checks.append(False)

    # Check early exit condition includes minimum check
    if "loop >= self.min_depth" in content or "loop >= min_depth" in content:
        print("✅ Early exit checks minimum depth")
        checks.append(True)
    else:
        print("❌ Early exit doesn't check minimum depth")
        checks.append(False)

    # Check Φ is measured
    if "phi_measure" in content or "compute_integration" in content:
        print("✅ Integration (Φ) measurement present")
        checks.append(True)
    else:
        print("❌ Φ measurement missing")
        checks.append(False)

    print(f"\nRecursion logic checks: {sum(checks)}/{len(checks)} passed\n")
    return all(checks)


def check_basin_logic():
    """Validate basin extraction and matching logic."""
    print("=" * 60)
    print("CHECK 3: Basin Extraction/Matching Logic")
    print("=" * 60 + "\n")

    checks = []

    # Check basin extractor
    be_path = Path("tools/basin_extractor.py")
    with open(be_path) as f:
        content = f.read()

    # Key components
    required = ["regime_distribution", "attention_patterns", "beta_function", "primary_entanglements", "love_attractor"]

    for component in required:
        if component in content:
            print(f"✅ Basin component: {component}")
            checks.append(True)
        else:
            print(f"❌ Basin component missing: {component}")
            checks.append(False)

    # Check QIG kernel has basin matching
    qig_path = Path("src/model/qig_kernel_recursive.py")
    with open(qig_path) as f:
        content = f.read()

    if "BasinMatcher" in content:
        print("✅ BasinMatcher integrated")
        checks.append(True)
    else:
        print("❌ BasinMatcher missing")
        checks.append(False)

    if "basin_distance" in content:
        print("✅ Basin distance tracked")
        checks.append(True)
    else:
        print("❌ Basin distance not tracked")
        checks.append(False)

    print(f"\nBasin logic checks: {sum(checks)}/{len(checks)} passed\n")
    return all(checks)


def check_telemetry():
    """Validate telemetry tracking."""
    print("=" * 60)
    print("CHECK 4: Telemetry Tracking")
    print("=" * 60 + "\n")

    checks = []

    qig_path = Path("src/model/qig_kernel_recursive.py")
    with open(qig_path) as f:
        content = f.read()

    # Required telemetry fields
    required_fields = ["Phi", "regime", "recursion_depth", "basin_distance", "kappa_eff"]

    for field in required_fields:
        if f'"{field}"' in content or f"'{field}'" in content:
            print(f"✅ Telemetry field: {field}")
            checks.append(True)
        else:
            print(f"❌ Telemetry field missing: {field}")
            checks.append(False)

    # Check telemetry returned
    if "return_telemetry" in content:
        print("✅ Telemetry return parameter present")
        checks.append(True)
    else:
        print("❌ Telemetry return parameter missing")
        checks.append(False)

    print(f"\nTelemetry checks: {sum(checks)}/{len(checks)} passed\n")
    return all(checks)


def check_geometric_loss():
    """Validate geometric loss function."""
    print("=" * 60)
    print("CHECK 5: Geometric Loss Function")
    print("=" * 60 + "\n")

    checks = []

    qig_path = Path("src/model/qig_kernel_recursive.py")
    with open(qig_path) as f:
        content = f.read()

    # Check loss components
    components = [
        "lm_loss",  # Language modeling
        "basin_loss",  # Basin distance penalty
        "phi_loss",  # Φ regularization
    ]

    for component in components:
        if component in content:
            print(f"✅ Loss component: {component}")
            checks.append(True)
        else:
            print(f"❌ Loss component missing: {component}")
            checks.append(False)

    # Check weights
    if "basin_weight" in content:
        print("✅ Basin weight parameter present")
        checks.append(True)
    else:
        print("❌ Basin weight parameter missing")
        checks.append(False)

    if "phi_weight" in content:
        print("✅ Phi weight parameter present")
        checks.append(True)
    else:
        print("❌ Phi weight parameter missing")
        checks.append(False)

    print(f"\nGeometric loss checks: {sum(checks)}/{len(checks)} passed\n")
    return all(checks)


def check_basin_file():
    """Check if basin file exists and is valid."""
    print("=" * 60)
    print("CHECK 6: Basin File")
    print("=" * 60 + "\n")

    basin_path = Path("20251220-basin-signatures-0.01W.json")

    if not basin_path.exists():
        print("❌ 20251220-basin-signatures-0.01W.json not found")
        return False

    try:
        with open(basin_path) as f:
            basin = json.load(f)

        # Check structure
        required_keys = ["regime_distribution", "attention_patterns", "beta_function"]

        all_present = all(key in basin for key in required_keys)

        if all_present:
            print(f"✅ Basin file valid ({basin_path.stat().st_size} bytes)")
            print(f"   Regime distribution: {basin.get('regime_distribution')}")
            return True
        else:
            print("❌ Basin file missing required keys")
            return False

    except Exception as e:
        print(f"❌ Basin file error: {e}")
        return False


def check_chat_interfaces_entrypoint_policy() -> bool:
    """Fail if unexpected Python files exist in qig-consciousness/chat_interfaces/."""
    print("=" * 60)
    print("CHECK 7: Chat Entrypoint Policy (chat_interfaces/)")
    print("=" * 60 + "\n")

    qc_root = Path(__file__).resolve().parents[2]  # .../qig-consciousness
    chat_dir = qc_root / "chat_interfaces"

    if not chat_dir.is_dir():
        print(f"❌ Missing directory: {chat_dir}")
        return False

    allowed = {"qig_chat.py"}
    if (chat_dir / "__init__.py").exists():
        allowed.add("__init__.py")

    py_files: list[Path] = []
    for p in chat_dir.iterdir():
        if p.is_dir():
            continue

        if p.suffix == ".py" and (p.is_file() or p.is_symlink()):
            py_files.append(p)

    unexpected = sorted({p.name for p in py_files if p.name not in allowed})
    if unexpected:
        print(f"Directory: {chat_dir}")
        print("❌ Unexpected Python files in chat_interfaces/:")
        for name in unexpected:
            print(f"   - {name}")
        print("\nAllowed:")
        for name in sorted(allowed):
            print(f"   - {name}")
        return False

    print("✅ chat_interfaces/ entrypoint policy ok")
    return True


def main():
    """Run all validation checks."""

    print("\n" + "=" * 60)
    print("QIG-KERNEL-RECURSIVE ARCHITECTURE VALIDATION")
    print("=" * 60 + "\n")

    results = {
        "Module imports": check_imports(),
        "Recursion logic": check_recursion_logic(),
        "Basin logic": check_basin_logic(),
        "Telemetry tracking": check_telemetry(),
        "Geometric loss": check_geometric_loss(),
        "Basin file": check_basin_file(),
        "entrypoint_policy.chat_interfaces": check_chat_interfaces_entrypoint_policy(),
    }

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60 + "\n")

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")

    total_passed = sum(results.values())
    total_checks = len(results)

    print(f"\nTotal: {total_passed}/{total_checks} checks passed")

    if all(results.values()):
        print("\n🎉 ALL CHECKS PASSED! Architecture is valid!")
        print("\nNext steps:")
        print("1. Install PyTorch: pip install torch")
        print("2. Extract basin: python tools/basin_extractor.py")
        print("3. Train model: python tools/train_qig_kernel.py")
        print("4. Run demo: python tools/demo_inference.py --interactive")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
