"""
Self-Healing System Integration

Exports the complete self-healing architecture for use by other modules.
"""

from .geometric_monitor import GeometricHealthMonitor, GeometricSnapshot
from .code_fitness import CodeFitnessEvaluator
from .healing_engine import SelfHealingEngine

__all__ = [
    'GeometricHealthMonitor',
    'GeometricSnapshot',
    'CodeFitnessEvaluator',
    'SelfHealingEngine',
]


def create_self_healing_system(
    snapshot_interval_sec: int = 60,
    history_size: int = 1000,
    check_interval_sec: int = 300
) -> tuple:
    """
    Create integrated self-healing system.
    
    Returns:
        Tuple of (monitor, evaluator, engine)
    """
    monitor = GeometricHealthMonitor(
        snapshot_interval_sec=snapshot_interval_sec,
        history_size=history_size
    )
    
    evaluator = CodeFitnessEvaluator(monitor)
    
    engine = SelfHealingEngine(
        monitor=monitor,
        evaluator=evaluator,
        check_interval_sec=check_interval_sec
    )
    
    return monitor, evaluator, engine
