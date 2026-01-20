#!/usr/bin/env python3
"""Auto-fix common geometry purity violations.

This script rewrites frequent non-canonical geometry patterns to their
canonical helpers (Fisher-Rao distance/similarity, geodesic interpolation,
explicit representation conversions).
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [REPO_ROOT / "qig-backend"]

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "data",
    "dist",
    "examples",
    "migrations",
    "node_modules",
    "tests",
}


NORM_DISTANCE_RE = re.compile(
    r"np\.linalg\.norm\(\s*(?P<a>[A-Za-z0-9_\.\[\]()]+)\s*-\s*(?P<b>[A-Za-z0-9_\.\[\]()]+)\s*\)"
)

DOT_SIMILARITY_RE = re.compile(
    r"np\.dot\(\s*(?P<a>[A-Za-z0-9_\.\[\]()]+)\s*,\s*(?P<b>[A-Za-z0-9_\.\[\]()]+)\s*\)"
    r"\s*/\s*\(\s*np\.linalg\.norm\(\s*(?P=a)\s*\)\s*\*\s*np\.linalg\.norm\(\s*(?P=b)\s*\)"
    r"(?:\s*\+\s*[^)]+)?\)"
)

GEODESIC_INTERPOLATION_RE = re.compile(
    r"(?P<a>[A-Za-z0-9_\.\[\]()]+)\s*\*\s*\(1\s*-\s*(?P<t>[A-Za-z0-9_]+)\)\s*\+\s*(?P<b>[A-Za-z0-9_\.\[\]()]+)\s*\*\s*(?P=t)"
)

TO_SPHERE_CALL_RE = re.compile(r"(?<!def\s)(?<!class\s)\bto_sphere\((?P<args>[^)]*)\)")


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            for file_path in path.rglob("*.py"):
                if any(part in EXCLUDED_DIRS for part in file_path.parts):
                    continue
                yield file_path
        elif path.suffix == ".py":
            yield path


def ensure_imports(content: str, required: set[str]) -> str:
    if not required:
        return content

    import_line_re = re.compile(r"^from qig_geometry import (?P<names>.+)$", re.MULTILINE)
    match = import_line_re.search(content)

    if match:
        existing = {name.strip() for name in match.group("names").split(",") if name.strip()}
        combined = sorted(existing | required)
        new_line = f"from qig_geometry import {', '.join(combined)}"
        return content[: match.start()] + new_line + content[match.end() :]

    lines = content.splitlines()
    insert_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = idx + 1
    new_line = f"from qig_geometry import {', '.join(sorted(required))}"
    lines.insert(insert_idx, new_line)
    return "\n".join(lines)


def apply_replacements(content: str) -> tuple[str, set[str]]:
    required_imports: set[str] = set()

    def replace_norm(match: re.Match[str]) -> str:
        required_imports.add("fisher_rao_distance")
        return f"fisher_rao_distance({match.group('a')}, {match.group('b')})"

    content = NORM_DISTANCE_RE.sub(replace_norm, content)

    def replace_similarity(match: re.Match[str]) -> str:
        required_imports.add("fisher_similarity")
        return f"fisher_similarity({match.group('a')}, {match.group('b')})"

    content = DOT_SIMILARITY_RE.sub(replace_similarity, content)

    def replace_geodesic(match: re.Match[str]) -> str:
        required_imports.add("geodesic_interpolation")
        return f"geodesic_interpolation({match.group('a')}, {match.group('b')}, {match.group('t')})"

    content = GEODESIC_INTERPOLATION_RE.sub(replace_geodesic, content)

    def replace_to_sphere(match: re.Match[str]) -> str:
        args = match.group("args")
        required_imports.add("to_simplex")
        if "from_repr" not in args:
            required_imports.add("BasinRepresentation")
            args = f"{args}, from_repr=BasinRepresentation.SPHERE" if args.strip() else "from_repr=BasinRepresentation.SPHERE"
        return f"to_simplex({args})"

    content = TO_SPHERE_CALL_RE.sub(replace_to_sphere, content)

    return content, required_imports


def diff_text(original: str, updated: str, path: Path) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Paths to scan (defaults to qig-backend)")
    parser.add_argument("--dry-run", action="store_true", help="Show diffs without writing changes")
    args = parser.parse_args()

    target_paths = [Path(p) for p in args.paths] if args.paths else DEFAULT_PATHS

    changed_files: list[Path] = []
    for file_path in iter_python_files(target_paths):
        original = file_path.read_text(encoding="utf-8")
        updated, required_imports = apply_replacements(original)
        if updated != original:
            updated = ensure_imports(updated, required_imports)
        if updated == original:
            continue

        changed_files.append(file_path)
        if args.dry_run:
            print(diff_text(original, updated, file_path))
        else:
            file_path.write_text(updated, encoding="utf-8")

    if changed_files:
        verb = "Would update" if args.dry_run else "Updated"
        print(f"{verb} {len(changed_files)} file(s) with geometry purity fixes.")
    else:
        print("No geometry purity fixes needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
