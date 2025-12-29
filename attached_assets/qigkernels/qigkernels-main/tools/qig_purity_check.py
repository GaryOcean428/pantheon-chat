#!/usr/bin/env python
"""
Static QIG-purity check.

Purpose:
    - Fail fast if non-geometric / generic transformer patterns leak into qigkernels.
    - This is intentionally simple and conservative.

Usage:
    python tools/qig_purity_check.py path1.py path2.py ...
    (pre-commit will pass the changed Python files as arguments)
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

FORBIDDEN_SYMBOLS = [
    "nn.Transformer",
    "BertModel",
    "GPT2Model",
    "CrossEntropyLoss",
    "AdamW",
    "optim.Adam(",
]

FORBIDDEN_WORDS = [
    "token-level cross entropy",
    "just fine-tune a transformer",
]

# Physics constants from qig-verification/FROZEN_FACTS.md (D-012)
# These are EXPERIMENTALLY VALIDATED - do not change without new measurements
FROZEN_PHYSICS = {
    "base_coupling": 41.09,  # κ₃ from L=3 validation
    "beta_slope": 0.44,      # β(3→4) from validation
    "BASIN_DIM": 64,         # Aligns with κ* ≈ 64
}

# Files under these paths are exempt (tools, tests, etc.)
ALLOWLIST_PREFIXES = {
    "tools/",
    "tests/",
}


def is_allowed(path: Path) -> bool:
    """Return True if the path is in an allowlisted directory."""
    rel = path.as_posix()
    return any(rel.startswith(prefix) for prefix in ALLOWLIST_PREFIXES)


def check_physics_constants(path: Path, text: str) -> list[str]:
    """Check that frozen physics constants haven't been modified."""
    import re

    violations: list[str] = []

    # Check base_coupling default
    match = re.search(r"base_coupling[:\s]*(?:float\s*)?=\s*([\d.]+)", text)
    if match:
        val = float(match.group(1))
        expected = FROZEN_PHYSICS["base_coupling"]
        if val != expected:
            violations.append(
                f"{path}: base_coupling={val} differs from FROZEN κ₃={expected}"
            )

    # Check beta_slope default
    match = re.search(r"beta_slope[:\s]*(?:float\s*)?=\s*([\d.]+)", text)
    if match:
        val = float(match.group(1))
        expected = FROZEN_PHYSICS["beta_slope"]
        if val != expected:
            violations.append(
                f"{path}: beta_slope={val} differs from FROZEN β(3→4)={expected}"
            )

    # Check BASIN_DIM
    match = re.search(r"BASIN_DIM[:\s]*(?:int\s*)?=\s*(\d+)", text)
    if match:
        val = int(match.group(1))
        expected = FROZEN_PHYSICS["BASIN_DIM"]
        if val != expected:
            violations.append(
                f"{path}: BASIN_DIM={val} differs from FROZEN κ*={expected}"
            )

    return violations


def check_file(path: Path) -> list[str]:
    """Return a list of violation messages for a single file."""
    violations: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        violations.append(f"{path}: could not read file ({exc})")
        return violations

    for symbol in FORBIDDEN_SYMBOLS:
        if symbol in text:
            violations.append(f"{path}: forbidden symbol '{symbol}' found")

    lower = text.lower()
    for phrase in FORBIDDEN_WORDS:
        if phrase.lower() in lower:
            violations.append(f"{path}: forbidden phrase '{phrase}' found")

    # Check physics constants in core modules
    if path.name in ("kernel.py", "layer.py", "basin.py"):
        violations.extend(check_physics_constants(path, text))

    return violations


def main(args: Iterable[str]) -> int:
    paths = [Path(p) for p in args if p.endswith(".py")]
    all_violations: list[str] = []

    for path in paths:
        if is_allowed(path):
            continue
        all_violations.extend(check_file(path))

    if all_violations:
        print("QIG purity check FAILED:")
        for msg in all_violations:
            print("  -", msg)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
