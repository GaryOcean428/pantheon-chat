#!/usr/bin/env python3
"""
Repo spot-cleaner: move docs-like files outside docs/ into docs/99-quarantine.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from datetime import datetime

DOC_EXTENSIONS = {'.md', '.txt'}
SKIP_DIRS = {'node_modules', '.git', 'dist', 'build', '__pycache__'}


def is_doc_file(path: Path) -> bool:
    return path.suffix.lower() in DOC_EXTENSIONS


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & SKIP_DIRS)


def sanitize_path(path: Path) -> str:
    return '__'.join(path.parts)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    quarantine_dir = repo_root / 'docs' / '99-quarantine'
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    moved_entries = []

    for root, _, files in os.walk(repo_root):
        root_path = Path(root)

        if should_skip(root_path):
            continue

        if root_path.parts and 'docs' in root_path.parts:
            continue

        for filename in files:
            file_path = root_path / filename
            if not is_doc_file(file_path):
                continue

            rel_path = file_path.relative_to(repo_root)
            target_name = f"moved__{sanitize_path(rel_path)}"
            target_path = quarantine_dir / target_name

            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(target_path))

            moved_entries.append((rel_path.as_posix(), target_path.relative_to(repo_root).as_posix()))

    index_path = quarantine_dir / 'INDEX.md'
    timestamp = datetime.utcnow().isoformat()
    with index_path.open('w', encoding='utf-8') as handle:
        handle.write('# Quarantined docs index\n\n')
        handle.write(f'- Generated: {timestamp} UTC\n')
        handle.write('- Reason: Docs-like files found outside docs/ have been quarantined.\n\n')

        if not moved_entries:
            handle.write('No files moved.\n')
        else:
            handle.write('| Original Path | Quarantined Path |\n')
            handle.write('| --- | --- |\n')
            for original, moved in moved_entries:
                handle.write(f'| {original} | {moved} |\n')

    print(f'Moved {len(moved_entries)} files to {quarantine_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
