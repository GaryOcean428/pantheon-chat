#!/usr/bin/env python3
"""
Validate all markdown links in the repository.

This script checks all markdown files for broken internal links.
External links are allowed (not checked) as they may be temporarily unavailable.

Exit codes:
  0 - All links valid
  1 - Broken links found
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set


def extract_markdown_links(content: str, filepath: str) -> List[Tuple[str, str, int]]:
    """
    Extract all markdown links from content.
    Returns: [(link_text, link_path, line_number), ...]
    """
    links = []
    # Match [text](path) format
    link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    
    for line_num, line in enumerate(content.split('\n'), 1):
        for match in re.finditer(link_pattern, line):
            text = match.group(1)
            path = match.group(2)
            links.append((text, path, line_num))
    
    return links


def is_external_link(link_path: str) -> bool:
    """Check if a link is external (http/https)."""
    return link_path.startswith('http://') or link_path.startswith('https://')


def resolve_link_path(link_path: str, source_file: Path, repo_root: Path) -> Path:
    """
    Resolve a markdown link to an absolute path.
    
    Args:
        link_path: The link path from markdown
        source_file: The markdown file containing the link
        repo_root: Repository root directory
    
    Returns:
        Resolved absolute path
    """
    # Remove anchors (#section)
    link_path = link_path.split('#')[0]
    
    # Remove query strings
    link_path = link_path.split('?')[0]
    
    if not link_path:
        return None
    
    # Handle relative paths
    if link_path.startswith('./'):
        link_path = link_path[2:]
    
    # If it starts with /, it's relative to repo root
    if link_path.startswith('/'):
        return repo_root / link_path[1:]
    
    # Otherwise, relative to source file directory
    return (source_file.parent / link_path).resolve()


def validate_file_links(filepath: Path, repo_root: Path) -> Tuple[List[str], List[str]]:
    """
    Validate all links in a markdown file.
    
    Returns:
        (valid_links, broken_links)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return [], []
    
    links = extract_markdown_links(content, str(filepath))
    valid_links = []
    broken_links = []
    
    for text, link_path, line_num in links:
        # Skip external links
        if is_external_link(link_path):
            continue
        
        # Resolve the link
        resolved_path = resolve_link_path(link_path, filepath, repo_root)
        
        if resolved_path is None:
            continue
        
        # Check if file exists
        if resolved_path.exists():
            valid_links.append(link_path)
        else:
            rel_source = filepath.relative_to(repo_root)
            broken_links.append(
                f"  ❌ {rel_source}:{line_num}\n"
                f"     Link: [{text}]({link_path})\n"
                f"     Expected: {resolved_path}"
            )
    
    return valid_links, broken_links


def find_markdown_files(repo_root: Path, exclude_dirs: Set[str]) -> List[Path]:
    """Find all markdown files in the repository."""
    markdown_files = []
    
    for root, dirs, files in os.walk(repo_root):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(Path(root) / file)
    
    return markdown_files


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate markdown links')
    parser.add_argument('--warn-only', action='store_true',
                      help='Only warn about broken links, do not fail')
    parser.add_argument('--file', type=str,
                      help='Check only specific file(s) (comma-separated)')
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.resolve()
    
    # Directories to exclude from checking
    exclude_dirs = {
        '.git', 'node_modules', 'dist', 'build', '__pycache__',
        '.pytest_cache', '.mypy_cache', 'venv', '.venv'
    }
    
    print("=" * 80)
    print("Markdown Link Validation")
    print("=" * 80)
    print(f"Repository: {repo_root}")
    print()
    
    # Find all markdown files or use specified files
    if args.file:
        file_paths = args.file.split(',')
        markdown_files = [repo_root / f.strip() for f in file_paths]
        print(f"Checking {len(markdown_files)} specified file(s)\n")
    else:
        markdown_files = find_markdown_files(repo_root, exclude_dirs)
        print(f"Found {len(markdown_files)} markdown files\n")
    
    # Validate each file
    all_broken_links = []
    total_valid = 0
    
    for filepath in sorted(markdown_files):
        valid, broken = validate_file_links(filepath, repo_root)
        total_valid += len(valid)
        all_broken_links.extend(broken)
    
    # Report results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"✅ Valid internal links: {total_valid}")
    print(f"❌ Broken internal links: {len(all_broken_links)}")
    print()
    
    if all_broken_links:
        print("BROKEN LINKS:")
        print("-" * 80)
        for broken in all_broken_links[:50]:  # Show first 50 to avoid overwhelming output
            print(broken)
            print()
        
        if len(all_broken_links) > 50:
            print(f"... and {len(all_broken_links) - 50} more broken links")
            print()
        
        if args.warn_only:
            print("\n⚠️  Warning: Broken links detected but not failing (--warn-only mode)")
            return 0
        else:
            print("\n⚠️  Fix these broken links before merging!")
            return 1
    else:
        print("✅ All internal links are valid!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
