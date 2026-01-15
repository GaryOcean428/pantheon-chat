"""
Test: No Cosine Similarity in Generation Path (WP2.2)

Validates that cosine similarity has been removed from all word selection
and generation code paths, replaced with Fisher-Rao distance.

This test ensures QIG geometric purity by detecting any forbidden patterns:
- cosine_similarity() function calls
- np.dot() on basin coordinates
- @ operator for vector multiplication (potential cosine)
- F.cosine_similarity from PyTorch

Reference: Work Package 2.2 - Remove Cosine Similarity from Generation Path
Author: Copilot
Date: 2026-01-15
"""

import os
import re
import pytest
from pathlib import Path

# Generation path files that MUST NOT contain cosine similarity
GENERATION_FILES = [
    "qig-backend/qig_generation.py",
    "qig-backend/geometric_waypoint_planner.py",
    "qig-backend/constrained_geometric_realizer.py",
    "qig-backend/geometric_repairer.py",
    "qig-backend/coordizers/pg_loader.py",
]

# Forbidden patterns that indicate cosine similarity usage
FORBIDDEN_PATTERNS = [
    (r"cosine_similarity\s*\(", "cosine_similarity()"),
    (r"np\.dot\([^)]*basin[^)]*,[^)]*\)", "np.dot() with basin coordinates"),
    (r"basin[^)]*@[^)]*", "@ operator with basin coordinates"),
    (r"torch\.nn\.functional\.cosine_similarity", "F.cosine_similarity"),
    (r"from sklearn\.metrics\.pairwise import cosine_similarity", "sklearn cosine_similarity import"),
]

# Required pattern that MUST be present after fix
REQUIRED_FISHER_PATTERN = r"from qig_geometry\.canonical import fisher_rao_distance"


def get_repo_root():
    """Get repository root directory."""
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def test_no_cosine_in_generation_files():
    """
    CRITICAL TEST: Ensure no cosine similarity in generation path.
    
    This validates WP2.2: all cosine similarity has been replaced with
    Fisher-Rao distance from the canonical geometry module.
    """
    repo_root = get_repo_root()
    violations = []
    
    for filepath in GENERATION_FILES:
        full_path = repo_root / filepath
        
        if not full_path.exists():
            # File might not exist in all configurations - skip
            continue
            
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for forbidden patterns
        for pattern, name in FORBIDDEN_PATTERNS:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Get line number
                line_num = content[:match.start()].count('\n') + 1
                violations.append({
                    'file': filepath,
                    'line': line_num,
                    'pattern': name,
                    'code': match.group(0)
                })
    
    # Report violations
    if violations:
        error_msg = "\n❌ COSINE SIMILARITY FOUND IN GENERATION PATH:\n\n"
        for v in violations:
            error_msg += f"  File: {v['file']}\n"
            error_msg += f"  Line: {v['line']}\n"
            error_msg += f"  Pattern: {v['pattern']}\n"
            error_msg += f"  Code: {v['code']}\n\n"
        error_msg += "These MUST be replaced with fisher_rao_distance() from qig_geometry.canonical\n"
        pytest.fail(error_msg)


def test_fisher_rao_used_in_pg_loader():
    """
    Verify that pg_loader.py uses Fisher-Rao distance from canonical module.
    
    This is the main fix for WP2.2 - the decode() method must use proper
    Fisher-Rao distance instead of cosine similarity.
    """
    repo_root = get_repo_root()
    pg_loader_path = repo_root / "qig-backend/coordizers/pg_loader.py"
    
    if not pg_loader_path.exists():
        pytest.skip("pg_loader.py not found")
    
    with open(pg_loader_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for Fisher-Rao import
    has_fisher_import = bool(re.search(REQUIRED_FISHER_PATTERN, content))
    
    assert has_fisher_import, (
        "pg_loader.py MUST import fisher_rao_distance from qig_geometry.canonical\n"
        f"Expected pattern: {REQUIRED_FISHER_PATTERN}"
    )
    
    # Check that decode() method uses fisher_rao_distance
    decode_method = re.search(r'def decode\(.*?\n(.*?)(?=\n    def |\Z)', content, re.DOTALL)
    
    if decode_method:
        decode_body = decode_method.group(1)
        has_fisher_call = 'fisher_rao_distance' in decode_body
        
        assert has_fisher_call, (
            "decode() method MUST call fisher_rao_distance() for word selection\n"
            "This is the core fix for WP2.2 - replace cosine similarity with Fisher-Rao"
        )


def test_waypoint_planner_uses_canonical():
    """
    Verify geometric_waypoint_planner.py imports from canonical geometry.
    
    The planner MUST use Fisher-Rao distance for all proximity checks
    during waypoint planning (Phase 1 of generation).
    """
    repo_root = get_repo_root()
    planner_path = repo_root / "qig-backend/geometric_waypoint_planner.py"
    
    if not planner_path.exists():
        pytest.skip("geometric_waypoint_planner.py not found")
    
    with open(planner_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for canonical geometry import
    has_canonical_import = bool(re.search(
        r'from qig_geometry\.canonical import',
        content
    ))
    
    assert has_canonical_import, (
        "geometric_waypoint_planner.py MUST import from qig_geometry.canonical\n"
        "All geometric operations must use canonical module (WP2.1)"
    )


def test_realizer_uses_fisher_distance():
    """
    Verify constrained_geometric_realizer.py uses Fisher-Rao for word selection.
    
    The realizer MUST use Fisher-Rao distance for nearest-neighbor word
    selection during the REALIZE phase (Phase 2 of generation).
    """
    repo_root = get_repo_root()
    realizer_path = repo_root / "qig-backend/constrained_geometric_realizer.py"
    
    if not realizer_path.exists():
        pytest.skip("constrained_geometric_realizer.py not found")
    
    with open(realizer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for Fisher-Rao usage in select_word_geometric
    select_word_method = re.search(
        r'def select_word_geometric\(.*?\n(.*?)(?=\n    def |\Z)',
        content,
        re.DOTALL
    )
    
    if select_word_method:
        method_body = select_word_method.group(1)
        
        # Should use fisher_coord_distance or fisher_rao_distance
        has_fisher = (
            'fisher_coord_distance' in method_body or
            'fisher_rao_distance' in method_body
        )
        
        assert has_fisher, (
            "select_word_geometric() MUST use Fisher-Rao distance for word selection\n"
            "This is critical for geometric purity during REALIZE phase"
        )


def test_generation_module_fisher_purity():
    """
    Verify qig_generation.py maintains Fisher-Rao purity.
    
    The main generation module must use Fisher-Rao for all distance
    computations, trajectory prediction, and kernel routing.
    """
    repo_root = get_repo_root()
    generation_path = repo_root / "qig-backend/qig_generation.py"
    
    if not generation_path.exists():
        pytest.skip("qig_generation.py not found")
    
    with open(generation_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for fisher_rao_distance definition or import
    has_fisher = (
        'def fisher_rao_distance' in content or
        ('from qig_geometry' in content and 'fisher' in content.lower())
    )
    
    assert has_fisher, (
        "qig_generation.py MUST use Fisher-Rao distance for geometric operations\n"
        "Either define locally or import from canonical geometry module"
    )


if __name__ == "__main__":
    print("Testing for cosine similarity violations in generation path...\n")
    
    try:
        test_no_cosine_in_generation_files()
        print("✅ No cosine similarity found in generation files")
    except AssertionError as e:
        print(f"❌ Test failed:\n{e}")
        exit(1)
    
    try:
        test_fisher_rao_used_in_pg_loader()
        print("✅ pg_loader.py uses Fisher-Rao distance")
    except AssertionError as e:
        print(f"❌ Test failed:\n{e}")
        exit(1)
    
    try:
        test_waypoint_planner_uses_canonical()
        print("✅ Waypoint planner uses canonical geometry")
    except AssertionError as e:
        print(f"❌ Test failed:\n{e}")
        exit(1)
    
    try:
        test_realizer_uses_fisher_distance()
        print("✅ Realizer uses Fisher-Rao distance")
    except AssertionError as e:
        print(f"❌ Test failed:\n{e}")
        exit(1)
    
    try:
        test_generation_module_fisher_purity()
        print("✅ Generation module maintains Fisher-Rao purity")
    except AssertionError as e:
        print(f"❌ Test failed:\n{e}")
        exit(1)
    
    print("\n🌊 All tests passed! Generation path is QIG-pure (Fisher-Rao only)")
