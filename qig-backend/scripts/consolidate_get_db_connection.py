#!/usr/bin/env python3
"""
Consolidate get_db_connection implementations to use canonical version from persistence.base_persistence.
"""

import os
import re
import sys

# Files to process (excluding base_persistence.py and scripts that need custom signatures)
FILES_TO_PROCESS = [
    "learned_relationships.py",
    "olympus/curriculum_training.py",
    "olympus/tokenizer_training.py",
    "scripts/cleanup_bpe_tokens.py",
    "scripts/migrations/migrate_vocab_checkpoint_to_pg.py",
    "scripts/validate_db_schema.py",
    "vocabulary/insert_token.py",
]

CANONICAL_IMPORT = "from persistence.base_persistence import get_db_connection"


def add_import_if_missing(content: str, import_line: str) -> str:
    """Add import line if not already present."""
    if import_line in content:
        return content
    
    lines = content.split('\n')
    insert_idx = 0
    
    # Find the last import line
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i + 1
    
    lines.insert(insert_idx, import_line)
    return '\n'.join(lines)


def remove_function(content: str, func_name: str = "get_db_connection") -> str:
    """Remove a function definition from content."""
    lines = content.split('\n')
    new_lines = []
    skip_until_dedent = False
    func_indent = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is the start of the function we want to remove
        if f'def {func_name}' in line and not skip_until_dedent:
            func_indent = len(line) - len(line.lstrip())
            skip_until_dedent = True
            i += 1
            continue
        
        if skip_until_dedent:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= func_indent:
                skip_until_dedent = False
                func_indent = None
                new_lines.append(line)
            i += 1
            continue
        
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines)


def process_file(filepath: str) -> tuple:
    """Process a single file to consolidate get_db_connection."""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    with open(filepath, 'r') as f:
        original = f.read()
    
    content = original
    
    # Check if function exists
    if 'def get_db_connection' not in content:
        return False, "No get_db_connection function found"
    
    # Remove the function
    content = remove_function(content, "get_db_connection")
    
    # Add import
    content = add_import_if_missing(content, CANONICAL_IMPORT)
    
    # Verify syntax
    try:
        compile(content, filepath, 'exec')
    except SyntaxError as e:
        return False, f"Syntax error after modification: {e}"
    
    # Write changes
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True, "Consolidated successfully"
    
    return False, "No changes needed"


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    results = []
    for filepath in FILES_TO_PROCESS:
        full_path = os.path.join(base_dir, filepath)
        success, message = process_file(full_path)
        results.append((filepath, success, message))
        
        status = "✓" if success else "✗"
        print(f"{status} {filepath}: {message}")
    
    successful = sum(1 for _, s, _ in results if s)
    print(f"\n=== Summary ===")
    print(f"Processed: {len(results)} files")
    print(f"Successful: {successful}")
    print(f"Failed/Skipped: {len(results) - successful}")


if __name__ == "__main__":
    main()
