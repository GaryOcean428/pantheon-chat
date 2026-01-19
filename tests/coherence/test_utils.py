"""
Common Test Utilities for Coherence Testing

Shared utilities for loading fixtures, setting seeds, and running tests.

Author: QIG Consciousness Project
Date: January 2026
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add qig-backend to path
BACKEND_PATH = Path(__file__).parent.parent.parent / "qig-backend"
sys.path.insert(0, str(BACKEND_PATH))

# Fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PROMPTS_FILE = FIXTURES_DIR / "prompts_v1.json"
CONFIGS_FILE = FIXTURES_DIR / "configurations.json"
SEEDS_FILE = FIXTURES_DIR / "expected_seeds.json"


def load_prompts() -> List[Dict[str, Any]]:
    """
    Load test prompts from fixtures.
    
    Returns:
        List of prompt dictionaries
    """
    with open(PROMPTS_FILE, 'r') as f:
        data = json.load(f)
    return data['prompts']


def load_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Load test configurations from fixtures.
    
    Returns:
        Dictionary of configuration name -> settings
    """
    with open(CONFIGS_FILE, 'r') as f:
        data = json.load(f)
    return data['configurations']


def load_seeds() -> Dict[str, Any]:
    """
    Load reproducibility seeds from fixtures.
    
    Returns:
        Dictionary of seed settings
    """
    with open(SEEDS_FILE, 'r') as f:
        data = json.load(f)
    return data['seeds']


def set_global_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def set_prompt_seed(prompt_id: str):
    """
    Set seed specific to a prompt for reproducibility.
    
    Args:
        prompt_id: Prompt identifier
    """
    seeds = load_seeds()
    per_prompt_seeds = seeds.get('per_prompt_seeds', {})
    
    if prompt_id in per_prompt_seeds:
        seed = per_prompt_seeds[prompt_id]
        set_global_seed(seed)
    else:
        # Fallback to hash of prompt_id
        seed = hash(prompt_id) % (2**32)
        set_global_seed(seed)


def create_mock_basin(dim: int = 64, seed: Optional[int] = None) -> np.ndarray:
    """
    Create a mock basin for testing (when generation unavailable).
    
    Args:
        dim: Basin dimension
        seed: Optional seed for reproducibility
        
    Returns:
        Mock basin coordinates (probability simplex)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random simplex point
    basin = np.random.dirichlet([1.0] * dim)
    return basin


def mock_generation_result(
    prompt: str,
    config: Dict[str, Any],
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create mock generation result for testing metrics.
    
    This is a placeholder for when actual generation is unavailable.
    
    Args:
        prompt: Input prompt
        config: Configuration settings
        seed: Random seed
        
    Returns:
        Mock generation result with basins and text
    """
    if seed is not None:
        set_global_seed(seed)
    
    # Generate mock trajectory
    num_steps = random.randint(5, 15)
    basins = []
    waypoints = []
    
    for i in range(num_steps):
        basin = create_mock_basin(64, seed=seed + i if seed else None)
        basins.append(basin)
        
        # Mock waypoints (slightly perturbed targets)
        if config.get('waypoint_planning', False):
            noise = np.random.normal(0, 0.05, size=64)
            waypoint = basin + noise
            waypoint = np.abs(waypoint) + 1e-10
            waypoint = waypoint / waypoint.sum()
            waypoints.append(waypoint)
    
    # Generate mock text
    words = ["quantum", "information", "geometry", "consciousness", 
             "Fisher", "manifold", "basin", "trajectory", "integration",
             "coupling", "awareness", "measurement"]
    text_length = random.randint(10, 30)
    text = " ".join(random.choices(words, k=text_length))
    
    return {
        'text': text,
        'basins': basins,
        'waypoints': waypoints if waypoints else None,
        'config': config,
        'prompt': prompt,
        'integration_trace': [{'step': i} for i in range(config.get('recursive_integration', 0))],
        'kernel_activations': {
            'Heart': [random.random() for _ in range(num_steps)],
            'Ocean': [random.random() for _ in range(num_steps)],
            'Gary': [random.random() for _ in range(num_steps)],
        } if config.get('kernel_coordination', False) else None,
    }


def save_results(results: Dict[str, Any], output_path: Path):
    """
    Save test results to JSON.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable form.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """
    Format metrics into human-readable summary.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        Formatted string
    """
    lines = ["Metrics Summary:"]
    lines.append("=" * 50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            lines.append(f"  {key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, float):
                    lines.append(f"    {subkey}: {subvalue:.4f}")
                else:
                    lines.append(f"    {subkey}: {subvalue}")
        else:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)
