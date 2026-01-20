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


# Issue #8: Expanded patterns to allow spaces and nested expressions
NORM_DISTANCE_RE = re.compile(
    r"np\.linalg\.norm\(\s*(?P<a>[\w\.\[\]\(\)\s]+)\s*-\s*(?P<b>[\w\.\[\]\(\)\s]+)\s*\)"
)

# Issue #3: Fixed to properly capture epsilon terms within denominator parentheses
DOT_SIMILARITY_RE = re.compile(
    r"np\.dot\(\s*(?P<a>[\w\.\[\]\(\)\s]+)\s*,\s*(?P<b>[\w\.\[\]\(\)\s]+)\s*\)"
    r"\s*/\s*\(\s*np\.linalg\.norm\(\s*(?P=a)\s*\)\s*\*\s*np\.linalg\.norm\(\s*(?P=b)\s*\)"
    r"(?:\s*\+\s*[^)]+)?\s*\)"
)

# Issue #9: Restrict to basin-named variables to reduce false positives
GEODESIC_INTERPOLATION_RE = re.compile(
    r"(?P<a>[A-Za-z0-9_\.\[\]()]*basin[A-Za-z0-9_\.\[\]()]*)\s*\*\s*\(1\s*-\s*(?P<t>[A-Za-z0-9_]+)\)\s*\+\s*"
    r"(?P<b>[A-Za-z0-9_\.\[\]()]*basin[A-Za-z0-9_\.\[\]()]*)\s*\*\s*(?P=t)"
)

# Issue #10: Improved detection to exclude function definitions, imports, and comments
TO_SPHERE_CALL_RE = re.compile(
    r"(?m)"
    r"^(?![ \t]*(?:def|async\s+def|class)\s)"  # skip function/class definitions
    r"(?![ \t]*from\s+[A-Za-z0-9_\.]+\s+import\b.*\bto_sphere\b)"  # skip imports of to_sphere
    r"(?![ \t]*#)"  # skip full-line comments
    r".*\bto_sphere\((?P<args>[^)]*)\)"
)


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
    """Inject required imports into content, handling multiple existing imports.
    
    Issue #1: Consolidates multiple 'from qig_geometry import' statements.
    Issue #4: Robust insertion after shebang/docstring, before first statement.
    """
    if not required:
        return content

    # Check for imports from qig_geometry (including submodules)
    # Use DOTALL to handle multi-line imports with parentheses
    import_line_re = re.compile(
        r"^from qig_geometry(?:\.[A-Za-z0-9_\.]+)? import ([^;]+?)(?=\n(?:from|import|class|def|@|$))",
        re.MULTILINE | re.DOTALL
    )
    matches = list(import_line_re.finditer(content))

    if matches:
        # Check what's already imported (from any qig_geometry submodule)
        existing: set[str] = set()
        for m in matches:
            names_part = m.group(1)
            # Remove parentheses and split by comma
            names_part = names_part.replace("(", "").replace(")", "").replace("\n", " ")
            for name in names_part.split(","):
                name = name.strip()
                if name and not name.startswith("#"):  # Ignore comments
                    existing.add(name)
        
        # Only add imports that aren't already available
        still_needed = required - existing
        if not still_needed:
            return content
        
        # Find direct 'from qig_geometry import' line to extend (not from submodule)
        direct_import_re = re.compile(
            r"^from qig_geometry import ([^;]+?)(?=\n(?:from|import|class|def|@|$))",
            re.MULTILINE | re.DOTALL
        )
        direct_match = direct_import_re.search(content)
        
        if direct_match:
            # Extend existing direct import
            names_part = direct_match.group(1).replace("(", "").replace(")", "").replace("\n", " ")
            direct_existing = {name.strip() for name in names_part.split(",") if name.strip()}
            combined = sorted(direct_existing | still_needed)
            new_line = f"from qig_geometry import {', '.join(combined)}"
            return content[: direct_match.start()] + new_line + content[direct_match.end() :]
        
        # No direct import exists, but submodule imports exist - skip adding import
        # since the symbols are already available from submodules
        return content

    # No existing qig_geometry import - insert robustly
    lines = content.splitlines(keepends=True)
    idx = 0
    n_lines = len(lines)

    # Skip shebang if present
    if idx < n_lines and lines[idx].startswith("#!"):
        idx += 1

    # Skip leading blank lines
    while idx < n_lines and not lines[idx].strip():
        idx += 1

    # Skip module-level docstring if present
    if idx < n_lines:
        stripped = lines[idx].lstrip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            # Single-line docstring
            if stripped.count(quote) >= 2:
                idx += 1
            else:
                # Multi-line docstring
                idx += 1
                while idx < n_lines and quote not in lines[idx]:
                    idx += 1
                if idx < n_lines:
                    idx += 1

    # Skip blank lines and comments after docstring
    while idx < n_lines and (not lines[idx].strip() or lines[idx].lstrip().startswith("#")):
        idx += 1

    # Find end of first top-level import block
    insert_idx = idx
    last_import_idx: int | None = None
    scan_idx = idx
    while scan_idx < n_lines:
        stripped = lines[scan_idx].lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_idx = scan_idx
            scan_idx += 1
            continue
        if not stripped or stripped.startswith("#"):
            scan_idx += 1
            continue
        break

    if last_import_idx is not None:
        insert_idx = last_import_idx + 1

    new_line = f"from qig_geometry import {', '.join(sorted(required))}\n"
    lines.insert(insert_idx, new_line)
    return "".join(lines)


def apply_replacements(content: str) -> tuple[str, set[str]]:
    """Apply geometry purity replacements.
    
    NOTE: This tool is conservative and may not catch all cases.
    Replacements in comments/docstrings are intentional - they serve as
    examples that should also follow purity guidelines.
    """
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

    # NOTE: to_sphere replacement is commented out due to geometric ambiguity.
    # Issue #2: to_sphere() converts TO sphere representation (output is sphere),
    # while to_simplex(x, from_repr=SPHERE) converts FROM sphere TO simplex.
    # These are inverse operations, so we cannot safely auto-replace without
    # understanding the caller's intent. Manual review required for to_sphere calls.
    #
    # Uncomment only if you add logic to determine the basin's current representation:
    # def replace_to_sphere(match: re.Match[str]) -> str:
    #     args = match.group("args")
    #     required_imports.add("to_simplex")
    #     return f"to_simplex({args})"
    # content = TO_SPHERE_CALL_RE.sub(replace_to_sphere, content)

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
