#!/usr/bin/env python3
"""
Comprehensive Fisher-Rao Distance Consolidation Script

Consolidates ALL fisher_rao_distance implementations to use the canonical version
from qig_geometry.canonical.

Handles:
1. Standalone functions
2. Functions inside try/except blocks
3. Class methods
4. Re-exports in __init__.py
"""

import os
import re
import sys

# Files to process (excluding canonical.py, tests, and training_chaos)
FILES_TO_PROCESS = [
    "autonomous_improvement.py",
    "geometric_completion.py",
    "geometric_search.py",
    "ocean_qig_core.py",
    "qig_generation.py",
    "olympus/autonomous_moe.py",
    "olympus/domain_geometry.py",
    "olympus/search_strategy_learner.py",
    "olympus/zeus_chat.py",
    "pattern_response_generator.py",
    "qig_core/geometric_completion/completion_criteria.py",
    "qig_core/neuroplasticity/mushroom_mode.py",
    "qig_geometry/__init__.py",
    "qig_geometry/geometry_simplex.py",
    "qigchain/geometric_tools.py",
    "qigkernels/geometry/distances.py",
    "qiggraph/consciousness.py",
    "qiggraph/manifold.py",
    "olympus/qig_rag.py",
    "qig_core/geometric_primitives/canonical_fisher.py",
    "qig_core/geometric_primitives/fisher_metric.py",
]

CANONICAL_IMPORT = "from qig_geometry.canonical import fisher_rao_distance"
CANONICAL_IMPORT_RELATIVE = "from .canonical import fisher_rao_distance"


def add_import_if_missing(content: str, import_line: str) -> str:
    """Add import line if not already present."""
    if import_line in content:
        return content
    
    # Find the best place to add the import
    lines = content.split('\n')
    insert_idx = 0
    
    # Find the last import line
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i + 1
    
    # Insert after docstring if no imports found
    if insert_idx == 0:
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                if '"""' in line or "'''" in line:
                    # Skip docstring
                    continue
                insert_idx = i
                break
    
    lines.insert(insert_idx, import_line)
    return '\n'.join(lines)


def remove_standalone_function(content: str, func_name: str = "fisher_rao_distance") -> str:
    """Remove a standalone function definition."""
    # Pattern for standalone function with any indentation
    pattern = rf'^(def {func_name}\s*\([^)]*\)[^:]*:.*?)(?=\n(?:def |class |@|\n\n[a-zA-Z]|\Z))'
    
    # Try to remove the function
    new_content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Clean up extra blank lines
    new_content = re.sub(r'\n{3,}', '\n\n', new_content)
    
    return new_content


def remove_function_from_try_block(content: str, func_name: str = "fisher_rao_distance") -> str:
    """Remove a function defined inside a try block, keeping the try block."""
    lines = content.split('\n')
    new_lines = []
    skip_until_dedent = False
    func_indent = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is the start of the function we want to remove
        if f'def {func_name}' in line and not skip_until_dedent:
            # Get the indentation of this function
            func_indent = len(line) - len(line.lstrip())
            skip_until_dedent = True
            i += 1
            continue
        
        if skip_until_dedent:
            # Check if we've reached a line with same or less indentation (end of function)
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= func_indent:
                skip_until_dedent = False
                func_indent = None
                new_lines.append(line)
            # Skip empty lines and lines that are part of the function
            i += 1
            continue
        
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines)


def replace_class_method_body(content: str, class_name: str = None) -> str:
    """Replace class method body with call to canonical function."""
    # Pattern for class method
    pattern = r'(def fisher_rao_distance\s*\(self[^)]*\)[^:]*:)\s*\n((?:\s+.*\n)*?)(?=\s*def |\s*class |\Z)'
    
    def replacement(match):
        signature = match.group(1)
        # Extract parameter names from signature
        params = re.findall(r'self\s*,\s*(\w+)[^,)]*,?\s*(\w+)?', signature)
        if params and params[0]:
            p1, p2 = params[0][0], params[0][1] if len(params[0]) > 1 and params[0][1] else 'q'
            return f'''{signature}
        from qig_geometry.canonical import fisher_rao_distance as _canonical_fr
        return _canonical_fr({p1}, {p2})
'''
        return match.group(0)
    
    return re.sub(pattern, replacement, content, flags=re.MULTILINE)


def process_file(filepath: str, is_qig_geometry: bool = False) -> tuple:
    """Process a single file to consolidate fisher_rao_distance."""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    with open(filepath, 'r') as f:
        original = f.read()
    
    content = original
    import_line = CANONICAL_IMPORT_RELATIVE if is_qig_geometry else CANONICAL_IMPORT
    
    # Check if it's a class method (contains 'self' in the function signature)
    if re.search(r'def fisher_rao_distance\s*\(\s*self', content):
        # Replace class method body
        content = replace_class_method_body(content)
    else:
        # Remove standalone or try-block function
        if 'def fisher_rao_distance' in content:
            # First try to remove from try block
            content = remove_function_from_try_block(content)
            
            # If still present, try standalone removal
            if 'def fisher_rao_distance' in content:
                content = remove_standalone_function(content)
    
    # Add import if we made changes
    if content != original:
        content = add_import_if_missing(content, import_line)
    
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
        is_qig_geometry = filepath.startswith("qig_geometry/")
        
        success, message = process_file(full_path, is_qig_geometry)
        results.append((filepath, success, message))
        
        status = "✓" if success else "✗"
        print(f"{status} {filepath}: {message}")
    
    # Summary
    successful = sum(1 for _, s, _ in results if s)
    print(f"\n=== Summary ===")
    print(f"Processed: {len(results)} files")
    print(f"Successful: {successful}")
    print(f"Failed/Skipped: {len(results) - successful}")


if __name__ == "__main__":
    main()
