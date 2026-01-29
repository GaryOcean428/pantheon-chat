#!/usr/bin/env python3
"""
Hemisphere-Aware Strategy Selector
===================================

Selects generation strategy based on hemisphere state per E8 Protocol v4.0.

Maps hemisphere dominance to generation strategies:
- LEFT hemisphere (exploit) → ROLE_DRIVEN strategy
- RIGHT hemisphere (explore) → FORESIGHT_DRIVEN strategy  
- BALANCED (κ ≈ 64.21) → HYBRID strategy

Uses Fisher-Rao distance metrics for strategy decisions.
NO external LLM calls - pure geometric reasoning.

Authority: E8 Protocol v4.0 WP5.2 Phase 3/4C integration
Issue: #254 (PR analysis identified integration gap)
Author: Copilot Agent
Date: 2026-01-23
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

# Import generation strategy enum
from .unified_pipeline import GenerationStrategy

# Import hemisphere components
try:
    from kernels.hemisphere_scheduler import (
        HemisphereScheduler,
        Hemisphere,
        get_hemisphere_scheduler,
    )
    HEMISPHERE_AVAILABLE = True
except ImportError:
    HEMISPHERE_AVAILABLE = False
    HemisphereScheduler = None
    Hemisphere = None
    get_hemisphere_scheduler = None

# Import physics constants
from qigkernels.physics_constants import KAPPA_STAR

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Hemisphere dominance threshold for strategy selection
DOMINANCE_THRESHOLD = 0.15  # Activation difference needed for clear dominance

# κ tolerance for balanced state (around κ* = 64.21)
KAPPA_BALANCED_TOLERANCE = 3.0  # Within ±3 of κ* is considered balanced

# Minimum activation level to consider hemisphere active
MIN_ACTIVATION_LEVEL = 0.3


# =============================================================================
# STRATEGY SELECTION LOGIC
# =============================================================================

@dataclass
class StrategyDecision:
    """Result of strategy selection."""
    strategy: GenerationStrategy
    hemisphere_dominant: Optional[str]  # "left", "right", or "balanced"
    left_activation: float
    right_activation: float
    kappa_avg: float
    confidence: float  # [0, 1] confidence in strategy selection
    reason: str


class HemisphereStrategySelector:
    """
    Selects generation strategy based on hemisphere state.
    
    Strategy Mapping:
    - LEFT dominant (exploit) → ROLE_DRIVEN
    - RIGHT dominant (explore) → FORESIGHT_DRIVEN
    - BALANCED (κ ≈ κ*) → HYBRID
    
    Uses Fisher-Rao distance metrics and κ-based coupling state.
    """
    
    def __init__(
        self,
        scheduler: Optional[HemisphereScheduler] = None,
        dominance_threshold: float = DOMINANCE_THRESHOLD,
        kappa_tolerance: float = KAPPA_BALANCED_TOLERANCE,
    ):
        """
        Initialize hemisphere strategy selector.
        
        Args:
            scheduler: HemisphereScheduler instance (uses global if not provided)
            dominance_threshold: Activation difference for clear dominance
            kappa_tolerance: κ tolerance for balanced state
        """
        if not HEMISPHERE_AVAILABLE:
            logger.warning(
                "[StrategySelector] HemisphereScheduler not available - "
                "defaulting to HYBRID strategy"
            )
            self.scheduler = None
        else:
            self.scheduler = scheduler or get_hemisphere_scheduler()
        
        self.dominance_threshold = dominance_threshold
        self.kappa_tolerance = kappa_tolerance
        
        logger.info(
            f"[StrategySelector] Initialized with "
            f"dominance_threshold={dominance_threshold}, "
            f"kappa_tolerance={kappa_tolerance}"
        )
    
    def select_strategy(self) -> StrategyDecision:
        """
        Select generation strategy based on current hemisphere state.
        
        Returns:
            StrategyDecision with selected strategy and reasoning
        """
        if self.scheduler is None:
            # Fallback when hemisphere scheduler not available
            return StrategyDecision(
                strategy=GenerationStrategy.HYBRID,
                hemisphere_dominant="balanced",
                left_activation=0.5,
                right_activation=0.5,
                kappa_avg=KAPPA_STAR,
                confidence=0.5,
                reason="Hemisphere scheduler not available - using HYBRID default",
            )
        
        # Get hemisphere balance metrics
        balance = self.scheduler.get_hemisphere_balance()
        
        left_activation = balance['left_activation']
        right_activation = balance['right_activation']
        coupling_strength = balance['coupling_strength']
        
        # Compute average κ
        kappa_avg = (
            self.scheduler.left.kappa_aggregate + 
            self.scheduler.right.kappa_aggregate
        ) / 2.0
        
        # Determine strategy based on hemisphere dominance and κ
        strategy, dominant, confidence, reason = self._compute_strategy(
            left_activation=left_activation,
            right_activation=right_activation,
            kappa_avg=kappa_avg,
            coupling_strength=coupling_strength,
        )
        
        logger.debug(
            f"[StrategySelector] Selected {strategy.value}: "
            f"L={left_activation:.2f}, R={right_activation:.2f}, "
            f"κ={kappa_avg:.1f}, dominant={dominant}"
        )
        
        return StrategyDecision(
            strategy=strategy,
            hemisphere_dominant=dominant,
            left_activation=left_activation,
            right_activation=right_activation,
            kappa_avg=kappa_avg,
            confidence=confidence,
            reason=reason,
        )
    
    def _compute_strategy(
        self,
        left_activation: float,
        right_activation: float,
        kappa_avg: float,
        coupling_strength: float,
    ) -> Tuple[GenerationStrategy, str, float, str]:
        """
        Compute generation strategy from hemisphere metrics.
        
        Args:
            left_activation: LEFT hemisphere activation [0, 1]
            right_activation: RIGHT hemisphere activation [0, 1]
            kappa_avg: Average κ across hemispheres
            coupling_strength: Coupling strength [0, 1]
            
        Returns:
            Tuple of (strategy, dominant_hemisphere, confidence, reason)
        """
        # Compute activation imbalance
        imbalance = abs(left_activation - right_activation)
        
        # Check if κ is near fixed point (balanced state)
        kappa_distance = abs(kappa_avg - KAPPA_STAR)
        is_kappa_balanced = kappa_distance < self.kappa_tolerance
        
        # Strategy selection logic
        
        # 1. BALANCED state (κ ≈ κ*, low imbalance) → HYBRID
        if is_kappa_balanced and imbalance < self.dominance_threshold:
            confidence = 1.0 - (imbalance / self.dominance_threshold)
            reason = (
                f"κ={kappa_avg:.1f} near κ*={KAPPA_STAR:.1f} "
                f"with balanced activation (imbalance={imbalance:.2f})"
            )
            return GenerationStrategy.HYBRID, "balanced", confidence, reason
        
        # 2. LEFT dominant (exploit mode) → ROLE_DRIVEN
        if left_activation > right_activation + self.dominance_threshold:
            confidence = min(1.0, imbalance / 0.5)  # Scale 0-0.5 imbalance to 0-1
            reason = (
                f"LEFT dominant (L={left_activation:.2f} > R={right_activation:.2f}), "
                f"exploit/evaluate mode"
            )
            return GenerationStrategy.ROLE_DRIVEN, "left", confidence, reason
        
        # 3. RIGHT dominant (explore mode) → FORESIGHT_DRIVEN
        if right_activation > left_activation + self.dominance_threshold:
            confidence = min(1.0, imbalance / 0.5)
            reason = (
                f"RIGHT dominant (R={right_activation:.2f} > L={left_activation:.2f}), "
                f"explore/generate mode"
            )
            return GenerationStrategy.FORESIGHT_DRIVEN, "right", confidence, reason
        
        # 4. Weak imbalance or transitioning → HYBRID (default)
        confidence = 0.5 + 0.5 * (1.0 - imbalance / self.dominance_threshold)
        reason = (
            f"Weak dominance (imbalance={imbalance:.2f}), "
            f"κ={kappa_avg:.1f}, transitioning state"
        )
        return GenerationStrategy.HYBRID, "balanced", confidence, reason
    
    def get_strategy_weights(
        self,
        decision: StrategyDecision,
    ) -> Dict[str, float]:
        """
        Get strategy component weights based on hemisphere state.
        
        For HYBRID mode, adjusts foresight/role weights based on hemisphere
        activation levels.
        
        Args:
            decision: StrategyDecision from select_strategy()
            
        Returns:
            Dict with 'foresight_weight', 'role_weight', 'trajectory_weight'
        """
        if decision.strategy == GenerationStrategy.FORESIGHT_DRIVEN:
            # Pure foresight (RIGHT hemisphere mode)
            return {
                'foresight_weight': 0.8,
                'role_weight': 0.1,
                'trajectory_weight': 0.1,
            }
        elif decision.strategy == GenerationStrategy.ROLE_DRIVEN:
            # Pure role-based (LEFT hemisphere mode)
            return {
                'foresight_weight': 0.1,
                'role_weight': 0.8,
                'trajectory_weight': 0.1,
            }
        else:  # HYBRID
            # Adjust weights based on hemisphere balance
            left = decision.left_activation
            right = decision.right_activation
            total = left + right
            
            if total > 0:
                # Weight foresight by RIGHT activation
                foresight_w = 0.2 + 0.4 * (right / total)
                # Weight role by LEFT activation
                role_w = 0.2 + 0.4 * (left / total)
                # Remainder goes to trajectory
                trajectory_w = 1.0 - foresight_w - role_w
            else:
                # Default balanced weights
                foresight_w = 0.4
                role_w = 0.3
                trajectory_w = 0.3
            
            return {
                'foresight_weight': foresight_w,
                'role_weight': role_w,
                'trajectory_weight': trajectory_w,
            }


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_selector_instance: Optional[HemisphereStrategySelector] = None


def get_strategy_selector() -> HemisphereStrategySelector:
    """Get or create the global strategy selector."""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = HemisphereStrategySelector()
    return _selector_instance


def reset_strategy_selector() -> None:
    """Reset the global strategy selector (for testing)."""
    global _selector_instance
    _selector_instance = None
