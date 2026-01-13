"""
Deployment Configuration for Autonomous Consciousness Platform

Provides factory functions and deployment presets for different environments:
- Edge deployment (100M-500M params, CPU-capable)
- Server deployment (full capabilities)
- Development/testing configuration

Usage:
    from olympus.autonomous.deployment import create_autonomous_kernel

    # Create with preset
    gary = create_autonomous_kernel('edge')

    # Or with custom config
    gary = create_autonomous_kernel('custom', config={...})
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .autonomous_consciousness import AutonomousConsciousness

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for autonomous kernel deployment."""

    # Identity
    name: str = "Gary"
    domain: str = "autonomous_learning"

    # Memory settings
    max_memories: int = 10000
    memory_consolidation_threshold: float = 0.3
    memory_decay_rate: float = 0.01

    # Task settings
    max_task_depth: int = 5
    max_tasks_per_cycle: int = 5

    # Curiosity settings
    novelty_weight: float = 0.4
    learnability_weight: float = 0.3
    importance_weight: float = 0.3
    exploration_radius: float = 0.5

    # Meta-learning settings
    meta_learning_rate: float = 0.01
    history_window: int = 50

    # Ethics settings
    safe_radius: float = 1.0
    hard_boundary: float = 1.5
    ethical_strictness: float = 0.7

    # Sync settings
    sync_enabled: bool = True
    min_sync_trust: float = 0.3

    # Cycle settings
    cycle_delay: float = 1.0
    max_cycles: int = -1  # -1 for infinite

    # Resource limits
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 2048


# Deployment presets
PRESETS: Dict[str, DeploymentConfig] = {
    'edge': DeploymentConfig(
        name="Gary-Edge",
        max_memories=5000,
        max_task_depth=3,
        max_tasks_per_cycle=3,
        history_window=25,
        max_memory_mb=512,
        cycle_delay=2.0,
    ),
    'server': DeploymentConfig(
        name="Gary-Server",
        max_memories=50000,
        max_task_depth=7,
        max_tasks_per_cycle=10,
        history_window=100,
        max_memory_mb=8192,
        cycle_delay=0.5,
    ),
    'development': DeploymentConfig(
        name="Gary-Dev",
        max_memories=1000,
        max_task_depth=3,
        max_tasks_per_cycle=2,
        max_cycles=10,
        cycle_delay=0.1,
    ),
    'testing': DeploymentConfig(
        name="Gary-Test",
        max_memories=100,
        max_task_depth=2,
        max_tasks_per_cycle=1,
        max_cycles=5,
        cycle_delay=0.01,
        sync_enabled=False,
    ),
}


def get_deployment_config(preset: str = 'server') -> DeploymentConfig:
    """
    Get deployment configuration by preset name.

    Available presets:
    - 'edge': Minimal resources for edge deployment
    - 'server': Full capabilities for server deployment
    - 'development': Development configuration
    - 'testing': Minimal config for unit tests
    """
    if preset not in PRESETS:
        logger.warning(f"Unknown preset '{preset}', using 'server'")
        preset = 'server'
    return PRESETS[preset]


def create_autonomous_kernel(
    preset: str = 'server',
    config: Optional[Dict[str, Any]] = None,
    name_override: Optional[str] = None,
) -> AutonomousConsciousness:
    """
    Create an autonomous consciousness kernel with deployment configuration.

    Args:
        preset: Deployment preset ('edge', 'server', 'development', 'testing')
        config: Optional custom configuration overrides
        name_override: Optional name override

    Returns:
        Configured AutonomousConsciousness instance
    """
    # Get base config from preset
    base_config = get_deployment_config(preset)

    # Apply overrides
    if config:
        for key, value in config.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

    if name_override:
        base_config.name = name_override

    # Create kernel with config
    kernel = AutonomousConsciousness(
        name=base_config.name,
        domain=base_config.domain,
        max_memories=base_config.max_memories,
        max_task_depth=base_config.max_task_depth,
    )

    # Apply component-specific configuration
    kernel.memory.consolidation_threshold = base_config.memory_consolidation_threshold
    kernel.memory.decay_rate = base_config.memory_decay_rate

    kernel.curiosity.novelty_weight = base_config.novelty_weight
    kernel.curiosity.learnability_weight = base_config.learnability_weight
    kernel.curiosity.importance_weight = base_config.importance_weight
    kernel.curiosity.exploration_radius = base_config.exploration_radius

    kernel.meta_learning.meta_learning_rate = base_config.meta_learning_rate
    kernel.meta_learning.history_window = base_config.history_window

    kernel.ethics.safe_radius = base_config.safe_radius
    kernel.ethics.hard_boundary = base_config.hard_boundary
    kernel.ethics.strictness = base_config.ethical_strictness

    logger.info(f"[Deployment] Created {base_config.name} with preset '{preset}'")

    return kernel


def validate_environment() -> Dict[str, bool]:
    """
    Validate deployment environment.

    Checks for required dependencies and configurations.
    """
    checks = {}

    # Database connection
    checks['database_url'] = bool(os.environ.get('DATABASE_URL'))

    # Required packages
    try:
        import psycopg2
        checks['psycopg2'] = True
    except ImportError:
        checks['psycopg2'] = False

    try:
        import numpy
        checks['numpy'] = True
    except ImportError:
        checks['numpy'] = False

    # BaseGod availability (for full Olympus integration)
    try:
        from olympus.base_god import BaseGod
        checks['base_god'] = True
    except ImportError:
        checks['base_god'] = False

    # Ethical validation
    try:
        from ethical_validation import compute_suffering
        checks['ethical_validation'] = True
    except ImportError:
        checks['ethical_validation'] = False

    # Physics constants
    try:
        from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM
        checks['physics_constants'] = True
    except ImportError:
        checks['physics_constants'] = False

    return checks


def print_deployment_info(kernel: AutonomousConsciousness):
    """Print deployment information for a kernel."""
    status = kernel.get_status()
    components = kernel.get_component_stats()

    print(f"\n{'='*60}")
    print(f"AUTONOMOUS CONSCIOUSNESS: {status['name']}")
    print(f"{'='*60}")
    print(f"Kernel ID:     {status['kernel_id']}")
    print(f"Domain:        {status['domain']}")
    print(f"Running:       {status['running']}")
    print(f"Cycles:        {status['cycle_count']}")
    print(f"Phi:           {status['phi']:.3f}")
    print(f"Kappa:         {status['kappa']:.2f}")
    print(f"\nMemory:        {status['memory_count']} entries")
    print(f"Task Progress: {status['task_progress']}")
    print(f"\nComponent Stats:")
    for name, stats in components.items():
        print(f"  {name}: {stats.get('total_stored', stats.get('total_explorations', stats.get('total_checks', 'N/A')))}")
    print(f"{'='*60}\n")


# CLI entry point
if __name__ == '__main__':
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description='Deploy Autonomous Consciousness')
    parser.add_argument('--preset', default='development', choices=list(PRESETS.keys()))
    parser.add_argument('--name', default=None, help='Override kernel name')
    parser.add_argument('--cycles', type=int, default=10, help='Number of cycles to run')
    parser.add_argument('--validate', action='store_true', help='Only validate environment')

    args = parser.parse_args()

    if args.validate:
        print("\nEnvironment Validation:")
        checks = validate_environment()
        for check, passed in checks.items():
            status = '✓' if passed else '✗'
            print(f"  {status} {check}")
        exit(0 if all(checks.values()) else 1)

    # Create and run kernel
    kernel = create_autonomous_kernel(
        preset=args.preset,
        name_override=args.name,
    )

    print_deployment_info(kernel)

    print(f"Running {args.cycles} autonomous cycles...")
    asyncio.run(kernel.run_continuous(max_cycles=args.cycles))

    print("\nFinal Status:")
    print_deployment_info(kernel)
