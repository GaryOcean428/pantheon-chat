"""
Test Helper Utilities for Coherence Testing
============================================

Shared utilities for loading fixtures, running tests, and
collecting metrics across all test configurations.

Author: WP4.3 Coherence Harness
Date: 2026-01-20
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Get path to coherence test directory
COHERENCE_DIR = Path(__file__).parent
FIXTURES_DIR = COHERENCE_DIR / "fixtures"


def load_json_fixture(filename: str) -> Dict[str, Any]:
    """
    Load a JSON fixture file.
    
    Args:
        filename: Name of fixture file
        
    Returns:
        Parsed JSON data
    """
    filepath = FIXTURES_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Fixture not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def load_test_prompts() -> List[Dict[str, Any]]:
    """Load fixed test prompts from fixture."""
    data = load_json_fixture("prompts_v1.json")
    return data['prompts']


def load_test_configurations() -> Dict[str, Any]:
    """Load test configurations from fixture."""
    data = load_json_fixture("configurations.json")
    return data['configurations']


def load_test_seeds() -> Dict[str, Any]:
    """Load test seeds from fixture."""
    return load_json_fixture("expected_seeds.json")


def set_reproducible_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    # Try to set Python's random module if available
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def get_prompt_seed(prompt_id: str) -> int:
    """
    Get the seed for a specific prompt.
    
    Args:
        prompt_id: Prompt ID from fixtures
        
    Returns:
        Seed value for this prompt
    """
    seeds_data = load_test_seeds()
    per_prompt_seeds = seeds_data.get('per_prompt_seeds', {})
    
    return per_prompt_seeds.get(prompt_id, 1000)


def create_mock_basin(dim: int = 64, seed: Optional[int] = None) -> np.ndarray:
    """
    Create a mock basin for testing (when actual generation unavailable).
    
    Args:
        dim: Basin dimension
        seed: Optional random seed
        
    Returns:
        Mock basin coordinates (simplex)
    """
    if seed is not None:
        np.random.seed(seed)
    
    basin = np.random.dirichlet(np.ones(dim))
    return basin


def create_mock_trajectory(
    n_steps: int = 10,
    dim: int = 64,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Create a mock trajectory for testing.
    
    Args:
        n_steps: Number of trajectory steps
        dim: Basin dimension
        seed: Optional random seed
        
    Returns:
        List of basin coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    trajectory = []
    
    # Create smooth trajectory with small random steps
    current = np.random.dirichlet(np.ones(dim))
    
    for _ in range(n_steps):
        # Small random perturbation
        noise = np.random.normal(0, 0.1, size=dim)
        next_basin = current + noise
        next_basin = np.abs(next_basin)
        next_basin = next_basin / next_basin.sum()
        
        trajectory.append(next_basin.copy())
        current = next_basin
    
    return trajectory


def mock_generation_run(
    prompt: str,
    config: Dict[str, Any],
    seed: int
) -> Dict[str, Any]:
    """
    Mock a generation run for testing (when actual generation unavailable).
    
    This creates synthetic data that matches the expected format.
    Real tests should replace this with actual generation calls.
    
    Args:
        prompt: Test prompt
        config: Configuration dict
        seed: Random seed
        
    Returns:
        Mock generation result with metrics
    """
    set_reproducible_seed(seed)
    
    # Determine trajectory length based on config
    if config.get('waypoint_planning', False):
        n_steps = 15  # Longer trajectory with planning
    else:
        n_steps = 8  # Shorter reactive trajectory
    
    # Create mock trajectory
    basins = create_mock_trajectory(n_steps, seed=seed)
    
    # Create mock waypoints if planning enabled
    waypoints = []
    if config.get('waypoint_planning', False):
        waypoints = create_mock_trajectory(n_steps, seed=seed + 1000)
    
    # Mock text output
    text = f"Generated text for prompt: {prompt[:30]}..."
    words = text.split()
    
    # Mock metrics
    result = {
        'text': text,
        'words': words,
        'basins': basins,
        'waypoints': waypoints,
        'phi_scores': [0.5 + 0.2 * np.random.rand() for _ in range(n_steps)],
        'kappa_values': [63.0 + 2.0 * np.random.randn() for _ in range(n_steps)],
        'recursive_depths': [config.get('recursive_integration', 0)] * n_steps,
        'kernel_activations': [['Heart', 'Ocean', 'Gary']] * n_steps if config.get('kernel_coordination', False) else [['Skeleton']] * n_steps,
        'config_name': config.get('name', 'unknown'),
    }
    
    return result


def validate_generation_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a generation result has all required fields.
    
    Args:
        result: Generation result dict
        
    Returns:
        (is_valid, list_of_issues)
    """
    required_fields = [
        'text',
        'words',
        'basins',
        'phi_scores',
        'kappa_values',
    ]
    
    issues = []
    
    for field in required_fields:
        if field not in result:
            issues.append(f"Missing required field: {field}")
    
    # Validate types
    if 'basins' in result:
        if not isinstance(result['basins'], list):
            issues.append("basins must be a list")
        elif len(result['basins']) > 0:
            if not isinstance(result['basins'][0], np.ndarray):
                issues.append("basins must contain numpy arrays")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


def save_results_to_file(
    results: Dict[str, Any],
    filename: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save test results to JSON file.
    
    Args:
        results: Results dictionary
        filename: Output filename
        output_dir: Optional output directory
        
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = COHERENCE_DIR / "results"
    
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir / filename
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_safe_results = convert_for_json(results)
    
    with open(filepath, 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    
    logger.info(f"Results saved to: {filepath}")
    
    return filepath


def load_results_from_file(filename: str, results_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load test results from JSON file.
    
    Args:
        filename: Results filename
        results_dir: Optional results directory
        
    Returns:
        Results dictionary
    """
    if results_dir is None:
        results_dir = COHERENCE_DIR / "results"
    
    filepath = results_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test helper utilities
    print("Testing Coherence Test Helpers")
    print("=" * 70)
    
    # Load fixtures
    print("\nLoading fixtures...")
    prompts = load_test_prompts()
    print(f"Loaded {len(prompts)} test prompts")
    
    configs = load_test_configurations()
    print(f"Loaded {len(configs)} test configurations")
    
    seeds = load_test_seeds()
    print(f"Loaded seed configuration")
    
    # Test mock generation
    print("\nTesting mock generation...")
    test_prompt = prompts[0]
    test_config = configs['plan_realize_repair']['config']
    test_seed = get_prompt_seed(test_prompt['id'])
    
    result = mock_generation_run(test_prompt['text'], test_config, test_seed)
    
    print(f"Generated {len(result['basins'])} trajectory steps")
    print(f"Text length: {len(result['text'])} chars")
    print(f"Mean Φ: {np.mean(result['phi_scores']):.3f}")
    
    # Validate result
    is_valid, issues = validate_generation_result(result)
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    print("\n✅ Test helpers validated")
