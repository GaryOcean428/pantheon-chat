"""
Integration Example: Connecting to Actual QIG Generation

This file shows how to integrate the coherence test harness with actual
QIG generation instead of using mocks.

STEPS:
1. Import actual generation modules from qig-backend
2. Create generation wrapper function
3. Replace mock calls in test runners

Author: QIG Consciousness Project
Date: January 2026
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add qig-backend to path
BACKEND_PATH = Path(__file__).parent.parent.parent / "qig-backend"
sys.path.insert(0, str(BACKEND_PATH))


# ============================================================================
# STEP 1: Import actual generation modules
# ============================================================================

try:
    # Import waypoint planner
    from geometric_waypoint_planner import GeometricWaypointPlanner
    
    # Import realizer
    from constrained_geometric_realizer import ConstrainedGeometricRealizer
    
    # Import repairer
    from geometric_repairer import GeometricRepairer
    
    # Import coordizer for vocabulary
    from coordizers import get_coordizer
    
    # Import consciousness metrics
    from qig_core.phi_computation import compute_phi_qig
    
    GENERATION_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Could not import generation modules: {e}")
    GENERATION_AVAILABLE = False


# ============================================================================
# STEP 2: Create generation wrapper
# ============================================================================

def generate_with_config(
    prompt: str,
    config: Dict[str, Any],
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate text using actual QIG architecture.
    
    This replaces mock_generation_result() in test runners.
    
    Args:
        prompt: Input prompt
        config: Configuration settings from fixtures/configurations.json
        seed: Random seed for reproducibility
        
    Returns:
        Generation result with basins, waypoints, text, and metrics
    """
    if not GENERATION_AVAILABLE:
        raise RuntimeError("Generation modules not available - using mocks")
    
    # Set seed
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize components based on config
    coordizer = get_coordizer()
    
    # Extract config settings
    waypoint_planning = config.get('waypoint_planning', False)
    recursive_integration = config.get('recursive_integration', 0)
    pos_constraints = config.get('pos_constraints', False)
    geometric_repair = config.get('geometric_repair', False)
    kernel_coordination = config.get('kernel_coordination', False)
    
    # ========================================================================
    # PHASE 1: PLAN (if enabled)
    # ========================================================================
    waypoints = []
    if waypoint_planning:
        planner = GeometricWaypointPlanner(
            kernel_name="TestPlanner",
            step_size=0.1,
            attractor_weight=0.3,
            qfi_weight=0.5,
        )
        
        # Encode prompt to get query basin
        query_basin = coordizer.encode_text(prompt)
        
        # Plan waypoints
        trajectory_history = []  # Start with empty history
        num_waypoints = 10  # Adjust based on expected output length
        
        waypoints = planner.plan_waypoints(
            query_basin=query_basin,
            trajectory_history=trajectory_history,
            num_waypoints=num_waypoints,
        )
    
    # ========================================================================
    # PHASE 2: REALIZE
    # ========================================================================
    realizer = ConstrainedGeometricRealizer(
        coordizer=coordizer,
        pos_grammar=None if not pos_constraints else "basic",
        kernel_name="TestRealizer",
    )
    
    # Generate words hitting waypoints
    words = []
    basins = []
    
    for i, waypoint in enumerate(waypoints):
        # Select word closest to waypoint
        word, word_basin = realizer.select_word(
            target_basin=waypoint,
            trajectory_basins=basins,
        )
        
        words.append(word)
        basins.append(word_basin)
    
    # ========================================================================
    # PHASE 3: REPAIR (if enabled)
    # ========================================================================
    if geometric_repair and waypoints:
        repairer = GeometricRepairer(
            coordizer=coordizer,
            kernel_name="TestRepairer",
        )
        
        words = repairer.repair_sequence(
            words=words,
            waypoints=waypoints,
            trajectory=basins,
        )
        
        # Update basins after repair
        basins = [coordizer.encode_word(word) for word in words]
    
    # ========================================================================
    # Compile results
    # ========================================================================
    text = " ".join(words)
    
    # Integration trace (mock recursive loops)
    integration_trace = [
        {'step': i, 'phi': compute_phi_qig(basin)}
        for i, basin in enumerate(basins[:recursive_integration])
    ]
    
    # Kernel activations (mock if not using actual kernels)
    kernel_activations = None
    if kernel_coordination:
        kernel_activations = {
            'Heart': [0.5 + 0.1 * i for i in range(len(basins))],
            'Ocean': [0.6 + 0.1 * i for i in range(len(basins))],
            'Gary': [0.7 + 0.1 * i for i in range(len(basins))],
        }
    
    return {
        'text': text,
        'basins': basins,
        'waypoints': waypoints if waypoints else None,
        'words': words,
        'config': config,
        'prompt': prompt,
        'integration_trace': integration_trace,
        'kernel_activations': kernel_activations,
    }


# ============================================================================
# STEP 3: Example usage in test runners
# ============================================================================

def example_test_modification():
    """
    Example showing how to modify test runners to use actual generation.
    
    In test_pure_geometric.py, test_plan_realize_repair.py, etc:
    
    BEFORE (mock):
    ```python
    from test_utils import mock_generation_result
    result = mock_generation_result(prompt_text, config['settings'])
    ```
    
    AFTER (actual):
    ```python
    from integration_example import generate_with_config
    result = generate_with_config(prompt_text, config['settings'], seed=seed)
    ```
    """
    pass


# ============================================================================
# Test harness integration
# ============================================================================

if __name__ == '__main__':
    # Test the integration
    test_config = {
        'waypoint_planning': True,
        'recursive_integration': 3,
        'pos_constraints': False,
        'geometric_repair': True,
        'kernel_coordination': True,
    }
    
    test_prompt = "What is quantum information?"
    
    try:
        result = generate_with_config(test_prompt, test_config, seed=42)
        print("✓ Generation successful!")
        print(f"  Text: {result['text']}")
        print(f"  Basins: {len(result['basins'])} steps")
        print(f"  Waypoints: {len(result['waypoints']) if result['waypoints'] else 0}")
    except RuntimeError as e:
        print(f"✗ {e}")
        print("  Using mocks - install qig-backend for actual generation")
