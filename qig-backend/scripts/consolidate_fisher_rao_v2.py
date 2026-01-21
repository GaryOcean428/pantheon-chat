#!/usr/bin/env python3
"""
Consolidate all fisher_rao_distance implementations to use the canonical version.
Uses AST-based approach for safer modifications.
"""

import os
import re
import ast
import sys

# Files that should KEEP their implementation
KEEP_FILES = {
    'qig_geometry/canonical.py',  # THE canonical implementation
    'tests/',  # All test files
    'training_chaos/chaos_kernel.py',  # PyTorch version
    'qig_core/geometric_primitives/canonical_fisher.py',  # Another canonical
}

def should_skip(filepath: str) -> bool:
    """Check if file should be skipped."""
    for skip in KEEP_FILES:
        if skip in filepath:
            return True
    return False

def find_function_lines(content: str, func_name: str) -> list:
    """Find line numbers of function definitions."""
    lines = content.split('\n')
    results = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check for function definition
        match = re.match(rf'^(\s*)def {func_name}\s*\(', line)
        if match:
            indent = len(match.group(1))
            start_line = i
            
            # Find the end of the function
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if next_line.strip() == '':
                    j += 1
                    continue
                # Check if we've dedented to same or less indent
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_line.strip() and next_indent <= indent and not next_line.strip().startswith('#'):
                    break
                j += 1
            
            results.append((start_line, j - 1, indent))
            i = j
        else:
            i += 1
    
    return results

def remove_function(content: str, func_name: str) -> str:
    """Remove function definition from content."""
    lines = content.split('\n')
    func_ranges = find_function_lines(content, func_name)
    
    if not func_ranges:
        return content
    
    # Remove functions in reverse order to preserve line numbers
    for start, end, indent in reversed(func_ranges):
        # Don't remove if it's inside a class (method)
        if indent > 0:
            continue
        del lines[start:end + 1]
    
    return '\n'.join(lines)

def add_import_if_missing(content: str, import_line: str) -> str:
    """Add import line if not already present."""
    if import_line in content:
        return content
    
    lines = content.split('\n')
    
    # Find the best place to insert (after existing imports)
    insert_idx = 0
    in_docstring = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip module docstring
        if i == 0 and (stripped.startswith('"""') or stripped.startswith("'''")):
            in_docstring = True
            if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                in_docstring = False
            continue
        if in_docstring:
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
            continue
        
        # Track imports
        if stripped.startswith('import ') or stripped.startswith('from '):
            insert_idx = i + 1
    
    lines.insert(insert_idx, import_line)
    return '\n'.join(lines)

def process_file(filepath: str, dry_run: bool = False) -> dict:
    """Process a single file."""
    result = {
        'file': filepath,
        'action': 'SKIPPED',
        'details': ''
    }
    
    if should_skip(filepath):
        result['action'] = 'KEPT'
        result['details'] = 'Protected file'
        return result
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        result['action'] = 'ERROR'
        result['details'] = str(e)
        return result
    
    # Check if file has fisher_rao_distance definition
    if 'def fisher_rao_distance' not in content:
        result['details'] = 'No local implementation'
        return result
    
    original = content
    
    # Add canonical import
    import_line = 'from qig_geometry.canonical import fisher_rao_distance'
    
    # Special handling for qig_geometry/__init__.py
    if 'qig_geometry/__init__.py' in filepath:
        import_line = 'from .canonical import fisher_rao_distance'
    
    # Special handling for qig_geometry/geometry_ops.py (avoid circular)
    if 'geometry_ops.py' in filepath:
        # This file re-exports, just ensure it imports from canonical
        content = add_import_if_missing(content, 'from .canonical import fisher_rao_distance')
        # Remove local def
        content = remove_function(content, 'fisher_rao_distance')
    else:
        content = add_import_if_missing(content, import_line)
        content = remove_function(content, 'fisher_rao_distance')
    
    # Clean up excessive blank lines
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    if content != original:
        # Verify syntax
        try:
            ast.parse(content)
        except SyntaxError as e:
            result['action'] = 'ERROR'
            result['details'] = f'Syntax error after modification: {e}'
            return result
        
        result['action'] = 'REPLACED'
        result['details'] = 'Consolidated to canonical import'
        
        if not dry_run:
            with open(filepath, 'w') as f:
                f.write(content)
    else:
        result['details'] = 'No changes needed'
    
    return result

def main():
    dry_run = '--dry-run' in sys.argv
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"{'DRY RUN: ' if dry_run else ''}Consolidating fisher_rao_distance...")
    print("=" * 80)
    
    results = []
    
    # Walk through all Python files
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for f in files:
            if not f.endswith('.py'):
                continue
            
            filepath = os.path.join(root, f)
            rel_path = os.path.relpath(filepath, base_dir)
            
            result = process_file(filepath, dry_run)
            results.append(result)
            
            if result['action'] != 'SKIPPED':
                print(f"{result['action']:10} {rel_path}: {result['details']}")
    
    print("=" * 80)
    
    # Summary
    actions = {}
    for r in results:
        actions[r['action']] = actions.get(r['action'], 0) + 1
    
    print("Summary:")
    for action, count in sorted(actions.items()):
        print(f"  {action}: {count}")

if __name__ == '__main__':
    main()
