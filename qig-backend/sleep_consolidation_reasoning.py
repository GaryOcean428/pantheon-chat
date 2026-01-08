"""
Sleep Consolidation for Reasoning

Extends sleep mode to consolidate reasoning strategies.
Prunes failed strategies, strengthens successful patterns, and performs meta-learning.

QIG-PURE: All geometric operations use Fisher-Rao distance.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)

# Try to import persistence for cycle counter
try:
    from qig_persistence import get_persistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False

# Log interval: only log every Nth consolidation cycle unless something changes
LOG_INTERVAL = 10


@dataclass
class ConsolidationResult:
    """Result of a sleep consolidation session."""
    strategies_before: int
    strategies_after: int
    strategies_pruned: int
    episodes_replayed: int
    exploration_rate_before: float
    exploration_rate_after: float
    success_rate: float
    timestamp: float = field(default_factory=time.time)


class SleepConsolidationReasoning:
    """
    Reasoning consolidation during sleep mode.

    Sleep stages:
    1. NREM: Prune failed strategies (<20% success)
    2. REM: Strengthen successful patterns (replay episodes)
    3. Deep: Meta-learning (adjust exploration rate)
    """

    # Class-level counter that persists across instance recreation
    # CRITICAL FIX: Instance-level counter reset to 0 when AutonomicKernel recreated
    _global_consolidation_count: int = 0

    def __init__(self, reasoning_learner=None):
        """
        Initialize sleep consolidation for reasoning.

        Args:
            reasoning_learner: AutonomousReasoningLearner instance
        """
        self.reasoning_learner = reasoning_learner
        self.consolidation_history: List[ConsolidationResult] = []
        self.min_success_rate = 0.2
        self.high_success_threshold = 0.8
        self.low_success_threshold = 0.4

        # Load persisted cycle count from database
        if PERSISTENCE_AVAILABLE:
            try:
                db = get_persistence()
                stored_count = db.get_metadata('sleep_consolidation_cycle_count')
                if stored_count:
                    SleepConsolidationReasoning._global_consolidation_count = int(stored_count)
                    logger.info(f"[SleepConsolidation] Restored cycle count from DB: {stored_count}")
            except Exception as e:
                logger.debug(f"[SleepConsolidation] Could not load cycle count: {e}")

        # Instance counter for backwards compatibility (but global persists)
        self._consolidation_count = SleepConsolidationReasoning._global_consolidation_count
    
    def set_learner(self, reasoning_learner):
        """Set the reasoning learner."""
        self.reasoning_learner = reasoning_learner
    
    def consolidate_reasoning(self) -> ConsolidationResult:
        """
        Consolidate reasoning strategies during sleep.
        
        Stages:
        1. NREM: Prune failures
        2. REM: Strengthen successful patterns
        3. Deep: Meta-learning adjustments
        """
        if self.reasoning_learner is None:
            return ConsolidationResult(
                strategies_before=0,
                strategies_after=0,
                strategies_pruned=0,
                episodes_replayed=0,
                exploration_rate_before=0.0,
                exploration_rate_after=0.0,
                success_rate=0.0
            )
        
        strategies_before = len(self.reasoning_learner.strategies)
        exploration_before = self.reasoning_learner.exploration_rate
        total_episodes = len(self.reasoning_learner.reasoning_episodes)
        
        # Increment cycle counter (both instance and class-level for persistence)
        self._consolidation_count += 1
        SleepConsolidationReasoning._global_consolidation_count = self._consolidation_count

        # Persist cycle count to database
        if PERSISTENCE_AVAILABLE:
            try:
                db = get_persistence()
                db.set_metadata('sleep_consolidation_cycle_count', str(self._consolidation_count))
            except Exception:
                pass  # Non-critical - continue without persistence

        # Stage 1 (NREM): Prune failed strategies
        self.reasoning_learner.consolidate_strategies()
        strategies_after_prune = len(self.reasoning_learner.strategies)
        strategies_pruned = strategies_before - strategies_after_prune
        
        # Stage 2 (REM): Strengthen successful patterns
        successful_episodes = [
            ep for ep in self.reasoning_learner.reasoning_episodes
            if ep.success and ep.efficiency > 0.7
        ]
        
        if successful_episodes:
            for episode in successful_episodes:
                self.reasoning_learner.learn_from_episode(episode)
        
        # Stage 3 (Deep): Meta-learning adjustments
        recent_episodes = self.reasoning_learner.reasoning_episodes[-100:]
        exploration_changed = False
        
        if recent_episodes:
            successful_count = sum(1 for ep in recent_episodes if ep.success)
            recent_success_rate = successful_count / len(recent_episodes)
            
            if recent_success_rate > self.high_success_threshold:
                self.reasoning_learner.exploration_rate *= 0.9
                exploration_changed = True
            elif recent_success_rate < self.low_success_threshold:
                self.reasoning_learner.exploration_rate = min(
                    0.5,
                    self.reasoning_learner.exploration_rate * 1.1
                )
                exploration_changed = True
        else:
            # No episodes yet - estimate from strategy performance or use prior
            if self.reasoning_learner and self.reasoning_learner.strategies:
                # Calculate average success rate from existing strategies
                strategy_rates = [s.success_rate() for s in self.reasoning_learner.strategies]
                recent_success_rate = np.mean(strategy_rates) if strategy_rates else 0.0
            else:
                # True cold start - no data yet
                recent_success_rate = 0.0
        
        strategies_after = len(self.reasoning_learner.strategies)
        exploration_after = self.reasoning_learner.exploration_rate
        
        result = ConsolidationResult(
            strategies_before=strategies_before,
            strategies_after=strategies_after,
            strategies_pruned=strategies_pruned,
            episodes_replayed=len(successful_episodes),
            exploration_rate_before=exploration_before,
            exploration_rate_after=exploration_after,
            success_rate=recent_success_rate
        )
        
        self.consolidation_history.append(result)
        
        # Only log when something actually changed OR every Nth cycle
        something_changed = (
            strategies_pruned > 0 or 
            len(successful_episodes) > 0 or 
            exploration_changed
        )
        
        if something_changed:
            logger.info(f"[SleepConsolidation] Cycle #{self._consolidation_count}: "
                       f"pruned={strategies_pruned}, replayed={len(successful_episodes)}, "
                       f"ε={exploration_after:.2%}, success={recent_success_rate:.1%}")
        elif self._consolidation_count % LOG_INTERVAL == 0:
            logger.debug(f"[SleepConsolidation] Cycle #{self._consolidation_count}: "
                        f"no changes (episodes={total_episodes}, strategies={strategies_after})")
        
        return result
    
    def run_dream_recombination(self) -> Dict[str, Any]:
        """
        Dream mode: Creative recombination of successful strategies.
        
        Creates novel strategies by combining successful elements.
        """
        if self.reasoning_learner is None:
            return {'error': 'No reasoning learner configured'}
        
        successful_strategies = [
            s for s in self.reasoning_learner.strategies
            if s.success_rate() > 0.5
        ]
        
        if len(successful_strategies) < 2:
            return {'message': 'Not enough successful strategies to recombine'}
        
        novel_strategies = []
        
        for i in range(min(3, len(successful_strategies))):
            parent_a = successful_strategies[np.random.randint(len(successful_strategies))]
            parent_b = successful_strategies[np.random.randint(len(successful_strategies))]
            
            if parent_a.name == parent_b.name:
                continue
            
            from autonomous_reasoning import ReasoningStrategy
            
            mix_ratio = np.random.uniform(0.3, 0.7)
            
            child = ReasoningStrategy(
                name=f"dream_{parent_a.name}_{parent_b.name}",
                description=f"Dream recombination of {parent_a.name} and {parent_b.name}",
                preferred_phi_range=(
                    mix_ratio * parent_a.preferred_phi_range[0] + (1 - mix_ratio) * parent_b.preferred_phi_range[0],
                    mix_ratio * parent_a.preferred_phi_range[1] + (1 - mix_ratio) * parent_b.preferred_phi_range[1]
                ),
                step_size_alpha=mix_ratio * parent_a.step_size_alpha + (1 - mix_ratio) * parent_b.step_size_alpha,
                exploration_beta=mix_ratio * parent_a.exploration_beta + (1 - mix_ratio) * parent_b.exploration_beta
            )
            
            if np.random.random() < 0.2:
                child.step_size_alpha *= np.random.uniform(0.8, 1.2)
                child.exploration_beta *= np.random.uniform(0.8, 1.2)
            
            self.reasoning_learner.strategies.append(child)
            novel_strategies.append(child.name)
        
        if novel_strategies:
            logger.info(f"[DreamRecombination] Created {len(novel_strategies)} novel strategies")
        
        return {
            'novel_strategies': novel_strategies,
            'total_strategies': len(self.reasoning_learner.strategies)
        }
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get statistics about consolidation sessions."""
        if not self.consolidation_history:
            return {
                'total_sessions': 0,
                'avg_strategies_pruned': 0,
                'avg_episodes_replayed': 0
            }
        
        return {
            'total_sessions': len(self.consolidation_history),
            'avg_strategies_pruned': np.mean([c.strategies_pruned for c in self.consolidation_history]),
            'avg_episodes_replayed': np.mean([c.episodes_replayed for c in self.consolidation_history]),
            'current_exploration_rate': self.reasoning_learner.exploration_rate if self.reasoning_learner else 0,
            'current_strategies': len(self.reasoning_learner.strategies) if self.reasoning_learner else 0
        }
