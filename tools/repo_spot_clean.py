#!/usr/bin/env python3
"""
Move nonconforming documentation files outside docs/ into docs/99-quarantine/.
Generates an index of moved files for review.
"""

from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
QUARANTINE_DIR = DOCS_DIR / "99-quarantine"
INDEX_PATH = QUARANTINE_DIR / "INDEX.md"

ALLOWED_ROOT_FILES = {
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
}

SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "docs",
}


def is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith("qig-") for part in path.parts)


def quarantine_docs() -> list[tuple[Path, Path]]:
    moved: list[tuple[Path, Path]] = []
    for path in REPO_ROOT.rglob("*.md"):
        if is_skipped(path):
            continue
        if path.parent == REPO_ROOT and path.name in ALLOWED_ROOT_FILES:
            continue
        if path.is_file() and not str(path).startswith(str(DOCS_DIR)):
            relative = path.relative_to(REPO_ROOT)
            prefix = "root" if relative.parent == Path(".") else str(relative.parent).replace("/", "__")
            dest_name = f"{prefix}__{path.name}"
            dest_path = QUARANTINE_DIR / dest_name
            QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), dest_path)
            moved.append((path, dest_path))
    return moved


def write_index(moved: list[tuple[Path, Path]]) -> None:
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quarantined Documentation Files",
        "",
        "The following files were moved because they were documentation files outside `docs/`.",
        "",
    ]
    if not moved:
        lines.append("No files moved.")
    else:
        for src, dest in moved:
            lines.append(f"- `{src}` → `{dest}`")
    INDEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    moved = quarantine_docs()
    write_index(moved)
    print(f"Moved {len(moved)} file(s) to {QUARANTINE_DIR}")


if __name__ == "__main__":
    main()
