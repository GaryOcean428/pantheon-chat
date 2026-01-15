#!/usr/bin/env python3
"""
Validate banned geometry patterns and direct SQL writes outside canonical paths.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    REPO_ROOT / "server",
    REPO_ROOT / "shared",
]

SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "docs",
}

SCAN_EXTENSIONS = {".py", ".ts", ".tsx", ".js"}

BANNED_PATTERNS = {
    "COSINE_SIMILARITY": re.compile(r"cosine_similarity\s*\(", re.IGNORECASE),
    "F_NORMALIZE": re.compile(r"\bF\.normalize\s*\(", re.IGNORECASE),
    "NP_LINALG_NORM": re.compile(r"np\.linalg\.norm\s*\(", re.IGNORECASE),
    "UNIT_SPHERE_LANGUAGE": re.compile(r"unit sphere", re.IGNORECASE),
}

DIRECT_SQL_PATTERN = re.compile(
    r"(INSERT INTO|UPDATE\s+\w+\s+SET|DELETE FROM)\s+coordizer_vocabulary",
    re.IGNORECASE,
)

ALLOWED_SQL_PATHS = {
    Path("server/persistence/coordizer-vocabulary.ts"),
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def scan_file(path: Path) -> list[str]:
    violations: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return violations

    for name, pattern in BANNED_PATTERNS.items():
        if pattern.search(text):
            violations.append(f"{path}: {name}")

    if DIRECT_SQL_PATTERN.search(text):
        relative = path.relative_to(REPO_ROOT)
        if relative not in ALLOWED_SQL_PATHS:
            violations.append(f"{path}: DIRECT_SQL_WRITE")

    return violations


def main() -> int:
    violations: list[str] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if should_skip(path):
                continue
            if path.suffix not in SCAN_EXTENSIONS:
                continue
            violations.extend(scan_file(path))

    if violations:
        print("❌ Purity validation failed:")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print("✅ Purity validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
