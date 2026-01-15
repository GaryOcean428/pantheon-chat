#!/usr/bin/env python3
import os
import shutil
from datetime import datetime


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DOCS_DIR = os.path.join(ROOT_DIR, 'docs')
QUARANTINE_DIR = os.path.join(DOCS_DIR, '99-quarantine')
INDEX_PATH = os.path.join(QUARANTINE_DIR, 'INDEX.md')

DOC_EXTENSIONS = {'.md', '.markdown', '.mdx', '.txt', '.rst', '.adoc'}
ALLOWLIST = {'README.md', 'LICENSE', 'AGENTS.md', 'CLAUDE.md'}
SKIP_DIRS = {'node_modules', 'dist', '.git', '.cache', 'docs'}


def ensure_dirs() -> None:
    os.makedirs(QUARANTINE_DIR, exist_ok=True)


def is_doc_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in DOC_EXTENSIONS


def find_docs_outside_docs() -> list[tuple[str, str]]:
    moved = []
    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if root.startswith(DOCS_DIR):
            continue
        for filename in files:
            if filename in ALLOWLIST:
                continue
            if not is_doc_file(filename):
                continue
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, ROOT_DIR)
            moved.append((full_path, rel_path))
    return moved


def quarantine_files(files: list[tuple[str, str]]) -> list[tuple[str, str]]:
    moved_records = []
    for full_path, rel_path in files:
        safe_name = rel_path.replace(os.sep, '__')
        destination = os.path.join(QUARANTINE_DIR, f'{safe_name}')
        shutil.move(full_path, destination)
        moved_records.append((rel_path, os.path.relpath(destination, ROOT_DIR)))
    return moved_records


def write_index(moved_records: list[tuple[str, str]]) -> None:
    timestamp = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    lines = [
        '# Quarantined Docs Index',
        '',
        f'Generated: {timestamp}',
        '',
        'Files moved here because they were documentation artifacts located outside the docs/ tree.',
        '',
        '| Original Path | Quarantined Path |',
        '| --- | --- |',
    ]
    for original, quarantined in moved_records:
        lines.append(f'| `{original}` | `{quarantined}` |')
    lines.append('')
    with open(INDEX_PATH, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def main() -> None:
    ensure_dirs()
    docs_to_move = find_docs_outside_docs()
    if not docs_to_move:
        print('No stray docs found.')
        return

    moved_records = quarantine_files(docs_to_move)
    write_index(moved_records)
    print(f'Moved {len(moved_records)} files to {QUARANTINE_DIR}.')


if __name__ == '__main__':
    main()
