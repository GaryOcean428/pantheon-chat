#!/usr/bin/env python3
"""
Consolidate all fisher_rao_distance implementations to use the canonical version.

This script:
1. Finds all files with local fisher_rao_distance implementations
2. Replaces them with imports from qig_geometry.canonical
3. Preserves the canonical implementation in qig_geometry/canonical.py
"""

import os
import re
import ast
import sys

# Files that should KEEP their implementation (canonical or special cases)
KEEP_FILES = {
    'qig_geometry/canonical.py',  # THE canonical implementation
    'tests/demo_cosine_vs_fisher.py',  # Test file with intentional comparison
    'tests/test_phi_fixes_standalone.py',  # Standalone test
    'tests/test_no_cosine_in_generation.py',  # Test file checking for function
    'training_chaos/chaos_kernel.py',  # PyTorch version (different signature)
}

# Files to process
TARGET_FILES = [
    'autonomous_improvement.py',
    'geometric_completion.py',
    'geometric_deep_research.py',
    'geometric_search.py',
    'ocean_qig_core.py',
    'olympus/autonomous_moe.py',
    'olympus/domain_geometry.py',
    'olympus/search_strategy_learner.py',
    'olympus/zeus_chat.py',
    'pattern_response_generator.py',
    'qig_core/consciousness_metrics.py',
    'qig_core/geometric_completion/completion_criteria.py',
    'qig_core/geometric_primitives/fisher_metric.py',
    'qig_core/neuroplasticity/mushroom_mode.py',
    'qig_deep_agents/state.py',
    'qig_generation.py',
    'qig_geometry.py',
    'qig_geometry/__init__.py',
    'qig_geometry/geometry_simplex.py',
    'qig_geometry/geometry_ops.py',
    'qig_numerics.py',
    'qigchain/geometric_tools.py',
    'qiggraph/consciousness.py',
    'qiggraph/manifold.py',
    'qigkernels/geometry/distances.py',
    'search/cross_domain_insight_tool.py',
    'training/loss_functions.py',
]

def remove_function_def(content: str, func_name: str) -> str:
    """Remove a function definition from content."""
    # Pattern to match function definition including decorators and docstring
    pattern = rf'^(\s*)def {func_name}\s*\([^)]*\)[^:]*:.*?(?=\n\1(?:def |class |@|\Z)|\Z)'
    
    # Try to remove the function
    new_content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    return new_content

def add_import(content: str, import_stmt: str) -> str:
    """Add import statement after existing imports."""
    # Check if import already exists
    if import_stmt in content:
        return content
    
    # Find the last import line
    lines = content.split('\n')
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            last_import_idx = i
    
    # Insert after last import
    lines.insert(last_import_idx + 1, import_stmt)
    return '\n'.join(lines)

def process_file(filepath: str, dry_run: bool = False) -> dict:
    """Process a single file to consolidate fisher_rao_distance."""
    result = {
        'file': filepath,
        'action': 'SKIPPED',
        'details': ''
    }
    
    # Check if file should be kept
    for keep in KEEP_FILES:
        if filepath.endswith(keep):
            result['action'] = 'KEPT'
            result['details'] = 'Canonical or special case file'
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
        result['details'] = 'No local implementation found'
        return result
    
    original_content = content
    
    # Add canonical import
    import_stmt = 'from qig_geometry.canonical import fisher_rao_distance'
    
    # For qig_geometry/__init__.py, use re-export pattern
    if filepath.endswith('qig_geometry/__init__.py'):
        # Just ensure it imports from canonical
        if 'from .canonical import fisher_rao_distance' not in content:
            content = add_import(content, 'from .canonical import fisher_rao_distance')
    else:
        content = add_import(content, import_stmt)
    
    # Remove local function definition
    # Handle both standalone functions and class methods
    
    # Pattern for standalone function
    standalone_pattern = r'^def fisher_rao_distance\s*\([^)]*\)[^:]*:.*?(?=\n(?:def |class |@|\Z)|\Z)'
    
    # Pattern for class method (indented)
    method_pattern = r'^(\s+)def fisher_rao_distance\s*\([^)]*\)[^:]*:.*?(?=\n\1(?:def |@|\Z)|\n(?:def |class |@)|\Z)'
    
    # Remove standalone functions
    content = re.sub(standalone_pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    # For methods, replace with delegation to canonical
    def replace_method(match):
        indent = match.group(1)
        return f'{indent}def fisher_rao_distance(self, p, q):\n{indent}    """Delegate to canonical implementation."""\n{indent}    from qig_geometry.canonical import fisher_rao_distance as _fr\n{indent}    return _fr(p, q)\n'
    
    content = re.sub(method_pattern, replace_method, content, flags=re.MULTILINE | re.DOTALL)
    
    # Clean up multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    if content != original_content:
        result['action'] = 'REPLACED'
        result['details'] = 'Removed local implementation, added canonical import'
        
        if not dry_run:
            with open(filepath, 'w') as f:
                f.write(content)
    else:
        result['details'] = 'No changes needed'
    
    return result

def main():
    dry_run = '--dry-run' in sys.argv
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"{'DRY RUN: ' if dry_run else ''}Consolidating fisher_rao_distance implementations...")
    print(f"Base directory: {base_dir}")
    print("=" * 80)
    
    results = []
    for target in TARGET_FILES:
        filepath = os.path.join(base_dir, target)
        if os.path.exists(filepath):
            result = process_file(filepath, dry_run)
            results.append(result)
            print(f"{result['action']:10} {target}: {result['details']}")
        else:
            print(f"{'MISSING':10} {target}: File not found")
    
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
