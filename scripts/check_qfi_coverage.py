#!/usr/bin/env python3
"""Check QFI coverage enforcement requirements.

This script performs lightweight static checks to ensure the
QFI pipeline is wired into schema + persistence layers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing required file: {path}") from exc


def require_patterns(label: str, content: str, patterns: Iterable[tuple[str, str]]) -> None:
    for description, pattern in patterns:
        if not re.search(pattern, content, re.MULTILINE | re.DOTALL):
            raise RuntimeError(f"{label}: missing {description} (pattern: {pattern})")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    schema_path = repo_root / "shared" / "schema.ts"
    schema_content = read_text(schema_path)
    require_patterns(
        "shared/schema.ts",
        schema_content,
        [
            ("qfi_score column", r"qfiScore\s*:\s*doublePrecision\(\"qfi_score\"\)"),
        ],
    )

    # Validate both vocabulary persistence files exist and have qfiScore writes
    # Both server/vocabulary-persistence.ts and server/persistence/coordizer-vocabulary.ts
    # are canonical locations for qfiScore writes per validate-qfi-canonical-path.sh (line 71)
    # and are excluded from purity checks in validate-purity-patterns.sh (line 66-67)
    
    # Older persistence file - just check for qfiScore writes
    old_persistence_path = repo_root / "server" / "vocabulary-persistence.ts"
    old_persistence_content = read_text(old_persistence_path)
    require_patterns(
        str(old_persistence_path.relative_to(repo_root)),
        old_persistence_content,
        [
            ("qfiScore writes", r"qfiScore"),
        ],
    )
    
    # Newer persistence file - check for qfiScore writes and activeVocabularyFilter
    new_persistence_path = repo_root / "server" / "persistence" / "coordizer-vocabulary.ts"
    new_persistence_content = read_text(new_persistence_path)
    require_patterns(
        str(new_persistence_path.relative_to(repo_root)),
        new_persistence_content,
        [
            ("qfiScore writes", r"qfiScore"),
            ("activeVocabularyFilter", r"activeVocabularyFilter"),
            (
                "activeVocabularyFilter enforces qfiScore",
                r"activeVocabularyFilter[\s\S]{0,500}qfiScore[\s\S]{0,500}IS NOT NULL",
            ),
        ],
    )

    print("✅ QFI coverage checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
