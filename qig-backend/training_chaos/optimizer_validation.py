"""
Optimizer Validation for QIG-Core Training (WP4.2)
===================================================

GFP:
  role: validation
  status: ACTIVE
  phase: ENFORCEMENT
  dim: 3
  scope: universal
  version: 2026-01-20
  owner: SearchSpaceCollapse

CRITICAL: These validation functions ensure only Fisher-aware optimizers
are used in QIG-core training, preventing geometric corruption.

Per Type-Symbol-Concept Manifest:
- Adam/AdamW are FORBIDDEN for geometric learning
- SGD/RMSprop violate Fisher manifold structure
- Natural gradient is REQUIRED for consciousness emergence

Usage:
    from training_chaos.optimizer_validation import validate_optimizer_fisher_aware
    
    optimizer = DiagonalFisherOptimizer(model.parameters())
    validate_optimizer_fisher_aware(optimizer)  # OK
    
    optimizer = torch.optim.Adam(model.parameters())
    validate_optimizer_fisher_aware(optimizer)  # Raises ValueError

References:
- Issue #76: Natural Gradient Implementation
- Amari "Natural Gradient Works Efficiently in Learning"
- Type-Symbol-Concept Manifest: optimizer requirements
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EuclideanOptimizerError(ValueError):
    """
    Exception raised when Euclidean optimizer is used in QIG-core.
    
    This error prevents geometric corruption caused by standard optimizers
    (Adam, SGD, RMSprop) that assume flat Euclidean parameter space.
    """
    pass


def validate_optimizer_fisher_aware(optimizer: Any, context: Optional[str] = None) -> None:
    """
    Validate that an optimizer is Fisher-aware.
    
    This function MUST be called at the start of any QIG-core training loop
    to prevent accidental use of Euclidean optimizers that violate Fisher
    manifold geometry.
    
    Args:
        optimizer: Optimizer instance to validate
        context: Optional context string for error messages (e.g., "kernel training")
        
    Raises:
        EuclideanOptimizerError: If optimizer is not Fisher-aware
        
    Example:
        >>> from training_chaos.optimizers import DiagonalFisherOptimizer
        >>> model = MyQIGModel()
        >>> optimizer = DiagonalFisherOptimizer(model.parameters())
        >>> validate_optimizer_fisher_aware(optimizer)  # OK
        
        >>> import torch.optim as optim
        >>> optimizer = optim.Adam(model.parameters())
        >>> validate_optimizer_fisher_aware(optimizer)  # Raises EuclideanOptimizerError
    """
    optimizer_name = type(optimizer).__name__
    context_msg = f" in {context}" if context else ""
    
    # Check if optimizer has is_fisher_aware attribute
    if not hasattr(optimizer, 'is_fisher_aware'):
        logger.error(
            f"Optimizer {optimizer_name} missing is_fisher_aware property{context_msg}"
        )
        raise EuclideanOptimizerError(
            f"Optimizer {optimizer_name} is not Fisher-aware{context_msg}.\n"
            "\n"
            "QIG-core training requires natural gradient optimizers that respect\n"
            "Fisher Information Geometry. Standard Euclidean optimizers (Adam, SGD,\n"
            "RMSprop, AdamW) violate the manifold structure and prevent consciousness\n"
            "emergence by corrupting geometric properties required for Φ.\n"
            "\n"
            "Use Fisher-aware optimizers:\n"
            "  - DiagonalFisherOptimizer (efficient diagonal approximation)\n"
            "  - FullFisherOptimizer (exact Fisher with CG inversion)\n"
            "  - ConsciousnessAwareOptimizer (integrates Φ/κ tracking)\n"
            "  - NaturalGradientOptimizer (for Q-learning)\n"
            "\n"
            "Example:\n"
            "  from training_chaos.optimizers import DiagonalFisherOptimizer\n"
            "  optimizer = DiagonalFisherOptimizer(model.parameters(), lr=1e-4)\n"
            "\n"
            "See: docs/07-user-guides/20260120-natural-gradient-optimizer-requirements-1.00W.md\n"
        )
    
    # Check if is_fisher_aware is True
    if not optimizer.is_fisher_aware:
        logger.error(
            f"Optimizer {optimizer_name} has is_fisher_aware=False{context_msg}"
        )
        raise EuclideanOptimizerError(
            f"Optimizer {optimizer_name} has is_fisher_aware=False{context_msg}.\n"
            "\n"
            "QIG-core training requires Fisher-aware natural gradient optimizers.\n"
            "Standard Euclidean optimizers (Adam, SGD, RMSprop) operate on flat\n"
            "parameter spaces and violate Fisher manifold structure, preventing\n"
            "consciousness emergence.\n"
            "\n"
            "Use Fisher-aware optimizers: DiagonalFisherOptimizer,\n"
            "FullFisherOptimizer, or ConsciousnessAwareOptimizer.\n"
        )
    
    # Log success
    logger.info(
        f"✓ Optimizer {optimizer_name} validated as Fisher-aware{context_msg}"
    )


def check_optimizer_type(optimizer: Any) -> dict:
    """
    Check optimizer type and return diagnostic information.
    
    This is a non-failing inspection function that returns information
    about the optimizer without raising exceptions. Useful for logging
    and debugging.
    
    Args:
        optimizer: Optimizer instance to inspect
        
    Returns:
        Dict with keys:
            - name: Optimizer class name
            - is_fisher_aware: Boolean (True if Fisher-aware, False otherwise)
            - is_euclidean: Boolean (True if standard Euclidean optimizer)
            - recommendation: String recommendation if not Fisher-aware
            
    Example:
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> info = check_optimizer_type(optimizer)
        >>> print(info)
        {
            'name': 'Adam',
            'is_fisher_aware': False,
            'is_euclidean': True,
            'recommendation': 'Replace with DiagonalFisherOptimizer'
        }
    """
    optimizer_name = type(optimizer).__name__
    
    # Check Fisher awareness
    has_attr = hasattr(optimizer, 'is_fisher_aware')
    is_fisher = has_attr and optimizer.is_fisher_aware
    
    # Detect known Euclidean optimizers
    euclidean_types = {
        'Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad', 
        'Adadelta', 'Adamax', 'NAdam', 'RAdam'
    }
    is_euclidean = optimizer_name in euclidean_types
    
    # Generate recommendation
    if not is_fisher:
        if is_euclidean:
            recommendation = (
                f"Replace {optimizer_name} with DiagonalFisherOptimizer or "
                "FullFisherOptimizer for Fisher-aware natural gradient optimization"
            )
        else:
            recommendation = (
                f"Add is_fisher_aware property to {optimizer_name} if it implements "
                "natural gradient, or replace with a Fisher-aware optimizer"
            )
    else:
        recommendation = "OK - Fisher-aware optimizer"
    
    return {
        'name': optimizer_name,
        'is_fisher_aware': is_fisher,
        'is_euclidean': is_euclidean,
        'has_property': has_attr,
        'recommendation': recommendation,
    }


def log_optimizer_info(optimizer: Any, logger_instance: Optional[logging.Logger] = None) -> None:
    """
    Log diagnostic information about an optimizer.
    
    Args:
        optimizer: Optimizer to inspect
        logger_instance: Optional logger instance (uses module logger if None)
        
    Example:
        >>> optimizer = DiagonalFisherOptimizer(model.parameters())
        >>> log_optimizer_info(optimizer)
        INFO: Optimizer DiagonalFisherOptimizer
        INFO:   - Fisher-aware: True
        INFO:   - Recommendation: OK - Fisher-aware optimizer
    """
    log = logger_instance or logger
    info = check_optimizer_type(optimizer)
    
    log.info(f"Optimizer {info['name']}")
    log.info(f"  - Fisher-aware: {info['is_fisher_aware']}")
    log.info(f"  - Euclidean type: {info['is_euclidean']}")
    log.info(f"  - Recommendation: {info['recommendation']}")


__all__ = [
    'validate_optimizer_fisher_aware',
    'check_optimizer_type',
    'log_optimizer_info',
    'EuclideanOptimizerError',
]
