#!/usr/bin/env python3
"""Validate QIG geometric purity gates.

Runs the canonical geometry validation scripts used in CI.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run_command(command: Iterable[str], label: str) -> None:
    print(f"\n▶ {label}")
    result = subprocess.run(list(command), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def ensure_command_available(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(f"Required command not found in PATH: {command}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    ensure_command_available("bash")
    ensure_command_available("rg")

    scripts = {
        "Purity pattern scan": repo_root / "scripts" / "validate-purity-patterns.sh",
        "QFI canonical path validation": repo_root / "scripts" / "validate-qfi-canonical-path.sh",
    }

    for label, script_path in scripts.items():
        if not script_path.exists():
            raise RuntimeError(f"Missing required script: {script_path}")
        run_command(["bash", str(script_path)], label)

    print("\n✅ Geometric purity validation complete.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)
