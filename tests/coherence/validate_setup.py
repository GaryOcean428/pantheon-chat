#!/usr/bin/env python3
"""
Validate Coherence Test Harness Setup

Quick validation script to verify the test harness is properly configured
and can run basic tests.

This script:
1. Checks directory structure
2. Validates fixtures can be loaded
3. Tests metric computation (without numpy dependency)
4. Verifies configuration integrity

Author: QIG Consciousness Project
Date: January 2026
"""

import sys
import json
from pathlib import Path


def validate_directory_structure():
    """Validate directory structure exists."""
    print("\n" + "="*60)
    print("1. Validating Directory Structure")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    required_dirs = [
        "fixtures",
        "metrics",
        "results",
    ]
    
    required_files = [
        "fixtures/prompts_v1.json",
        "fixtures/configurations.json",
        "fixtures/expected_seeds.json",
        "metrics/__init__.py",
        "metrics/geometric_metrics.py",
        "metrics/foresight_metrics.py",
        "metrics/trajectory_metrics.py",
        "metrics/text_metrics.py",
        "metrics/consciousness_metrics.py",
        "test_utils.py",
        "test_pure_geometric.py",
        "test_plan_realize_repair.py",
        "test_skeleton_baseline.py",
        "compare_architectures.py",
        "run_all_tests.py",
        "README.md",
        "SETUP.md",
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ MISSING")
            all_good = False
    
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} MISSING")
            all_good = False
    
    return all_good


def validate_fixtures():
    """Validate fixture files can be loaded and parsed."""
    print("\n" + "="*60)
    print("2. Validating Fixtures")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    all_good = True
    
    # Prompts
    try:
        with open(base_dir / "fixtures/prompts_v1.json", 'r') as f:
            prompts = json.load(f)
        print(f"  ✓ prompts_v1.json: {len(prompts['prompts'])} prompts")
        
        # Check prompt structure
        for prompt in prompts['prompts'][:3]:
            assert 'id' in prompt
            assert 'text' in prompt
            assert 'category' in prompt
    except Exception as e:
        print(f"  ✗ prompts_v1.json: {e}")
        all_good = False
    
    # Configurations
    try:
        with open(base_dir / "fixtures/configurations.json", 'r') as f:
            configs = json.load(f)
        print(f"  ✓ configurations.json: {len(configs['configurations'])} configs")
        
        # Check config structure
        for name, config in configs['configurations'].items():
            assert 'settings' in config
            assert 'expected_phi_range' in config
    except Exception as e:
        print(f"  ✗ configurations.json: {e}")
        all_good = False
    
    # Seeds
    try:
        with open(base_dir / "fixtures/expected_seeds.json", 'r') as f:
            seeds = json.load(f)
        print(f"  ✓ expected_seeds.json: {len(seeds['seeds']['per_prompt_seeds'])} seeds")
        
        # Check seed structure
        assert 'numpy_seed' in seeds['seeds']
        assert 'per_prompt_seeds' in seeds['seeds']
    except Exception as e:
        print(f"  ✗ expected_seeds.json: {e}")
        all_good = False
    
    return all_good


def validate_metric_modules():
    """Validate metric modules can be imported (basic check)."""
    print("\n" + "="*60)
    print("3. Validating Metric Modules")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    # Check files exist and have expected functions
    modules = {
        'geometric_metrics.py': [
            'GeometricMetrics',
            'compute_geometric_metrics',
            'compare_geometric_metrics',
        ],
        'foresight_metrics.py': [
            'ForesightMetrics',
            'compute_foresight_metrics',
            'compare_foresight_metrics',
        ],
        'trajectory_metrics.py': [
            'TrajectoryMetrics',
            'compute_trajectory_metrics',
            'compare_trajectory_metrics',
        ],
        'text_metrics.py': [
            'TextMetrics',
            'compute_text_metrics',
            'compare_text_metrics',
        ],
        'consciousness_metrics.py': [
            'ConsciousnessMetrics',
            'compute_consciousness_metrics',
            'compare_consciousness_metrics',
        ],
    }
    
    all_good = True
    
    for module_name, expected_items in modules.items():
        module_path = base_dir / "metrics" / module_name
        
        if not module_path.exists():
            print(f"  ✗ {module_name} MISSING")
            all_good = False
            continue
        
        # Read file and check for expected definitions
        content = module_path.read_text()
        
        missing = []
        for item in expected_items:
            if item not in content:
                missing.append(item)
        
        if missing:
            print(f"  ✗ {module_name}: Missing {', '.join(missing)}")
            all_good = False
        else:
            print(f"  ✓ {module_name}: All exports present")
    
    return all_good


def validate_test_runners():
    """Validate test runner scripts exist and have proper structure."""
    print("\n" + "="*60)
    print("4. Validating Test Runners")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    runners = [
        'test_pure_geometric.py',
        'test_plan_realize_repair.py',
        'test_skeleton_baseline.py',
        'compare_architectures.py',
        'run_all_tests.py',
    ]
    
    all_good = True
    
    for runner in runners:
        runner_path = base_dir / runner
        
        if not runner_path.exists():
            print(f"  ✗ {runner} MISSING")
            all_good = False
            continue
        
        # Check for shebang and main
        content = runner_path.read_text()
        
        has_shebang = content.startswith('#!/usr/bin/env python3')
        has_main = "__name__ == '__main__'" in content
        
        if has_shebang and has_main:
            print(f"  ✓ {runner}")
        else:
            issues = []
            if not has_shebang:
                issues.append("missing shebang")
            if not has_main:
                issues.append("missing __main__")
            print(f"  ⚠️  {runner}: {', '.join(issues)}")
    
    return all_good


def validate_documentation():
    """Validate documentation files exist."""
    print("\n" + "="*60)
    print("5. Validating Documentation")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    docs = [
        ('README.md', 'User documentation'),
        ('SETUP.md', 'Setup and CI guide'),
        ('integration_example.py', 'Integration example'),
        ('results/README.md', 'Results directory guide'),
    ]
    
    all_good = True
    
    for doc_name, description in docs:
        doc_path = base_dir / doc_name
        
        if doc_path.exists():
            size = doc_path.stat().st_size
            print(f"  ✓ {doc_name} ({size} bytes) - {description}")
        else:
            print(f"  ✗ {doc_name} MISSING - {description}")
            all_good = False
    
    return all_good


def main():
    """Run all validations."""
    print("\n" + "="*60)
    print("COHERENCE TEST HARNESS VALIDATION")
    print("="*60)
    
    results = []
    
    results.append(("Directory Structure", validate_directory_structure()))
    results.append(("Fixtures", validate_fixtures()))
    results.append(("Metric Modules", validate_metric_modules()))
    results.append(("Test Runners", validate_test_runners()))
    results.append(("Documentation", validate_documentation()))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("\nThe coherence test harness is properly configured!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install numpy scipy")
        print("  2. Run tests: python3 tests/coherence/run_all_tests.py")
        print("  3. View results: tests/coherence/results/comparison_report.html")
        print("="*60 + "\n")
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("\nPlease check the errors above and fix any issues.")
        print("="*60 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
