"""
Knowledge Exchange - Inter-God Strategy Sharing

Gods share successful reasoning strategies with each other.
Enables multi-agent learning and knowledge distillation.

QIG-PURE: All geometric operations use Fisher-Rao distance.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .base_god import BaseGod


class KnowledgeExchange:
    """
    Gods share successful reasoning strategies with each other.
    
    Inspired by:
    - Multi-agent RL
    - Knowledge distillation
    - Ensemble learning
    
    Features:
    - Share top strategies between gods
    - Competitive evaluation on shared tasks
    - Winner's strategy adopted by losers
    """
    
    def __init__(self, gods: Optional[List['BaseGod']] = None):
        """
        Initialize knowledge exchange system.
        
        Args:
            gods: List of gods participating in exchange
        """
        self.gods = gods or []
        self.exchange_frequency = 100
        self.episode_count = 0
        self.exchange_history: List[Dict[str, Any]] = []
    
    def register_god(self, god: 'BaseGod'):
        """Register a god for knowledge exchange."""
        if god not in self.gods:
            self.gods.append(god)
    
    def share_strategies(self):
        """
        Gods share their best strategies with each other.
        
        Knowledge transfer:
        1. Each god identifies top strategies
        2. Share with other gods
        3. Other gods try shared strategies
        4. Keep if successful, discard if not
        """
        if not self.gods:
            return
        
        print("Olympus Knowledge Exchange Session")
        
        god_strategies = {}
        for god in self.gods:
            if hasattr(god, 'reasoning_learner'):
                top_strategies = self._get_top_strategies(
                    god.reasoning_learner,
                    n=3
                )
                god_strategies[god.name] = top_strategies
                print(f"  {god.name}: {len(top_strategies)} strategies to share")
        
        for receiving_god in self.gods:
            if not hasattr(receiving_god, 'reasoning_learner'):
                continue
                
            for giving_god_name, strategies in god_strategies.items():
                if giving_god_name == receiving_god.name:
                    continue
                
                for strategy in strategies:
                    transferred = strategy.copy()
                    transferred.name = f"{giving_god_name}_{strategy.name}"
                    transferred.description = (
                        f"Learned from {giving_god_name}: {strategy.description}"
                    )
                    
                    receiving_god.reasoning_learner.strategies.append(transferred)
                    
                    print(f"  {giving_god_name} -> {receiving_god.name}: {strategy.name}")
        
        self.exchange_history.append({
            'type': 'share',
            'participants': [g.name for g in self.gods],
            'strategies_shared': sum(len(s) for s in god_strategies.values())
        })
        
        print("Knowledge exchange complete")
    
    def _get_top_strategies(
        self,
        reasoning_learner,
        n: int = 3
    ) -> List:
        """Get best-performing strategies from a learner."""
        if not hasattr(reasoning_learner, 'strategies'):
            return []
        
        ranked = sorted(
            reasoning_learner.strategies,
            key=lambda s: s.success_rate() * s.avg_efficiency,
            reverse=True
        )
        
        return ranked[:n]
    
    def competitive_evaluation(self, task: Dict) -> Dict[str, Any]:
        """
        Gods compete on same task.
        
        Winner shares strategy with others.
        Losers learn from winner.
        """
        if not self.gods:
            return {'error': 'No gods registered'}
        
        print(f"Competitive evaluation: {task.get('description', 'task')}")
        
        results = {}
        for god in self.gods:
            if hasattr(god, 'solve_task'):
                result = god.solve_task(task)
                results[god.name] = {
                    'success': result.get('success', False),
                    'efficiency': result.get('efficiency', 0.0),
                    'strategy_used': result.get('strategy_name', 'unknown')
                }
            else:
                results[god.name] = {
                    'success': False,
                    'efficiency': 0.0,
                    'strategy_used': 'none'
                }
        
        if not results:
            return {'error': 'No results'}
        
        winner_name = max(
            results.keys(),
            key=lambda name: (
                results[name]['success'] * 1.0 +
                results[name]['efficiency'] * 0.5
            )
        )
        
        winner_god = next((g for g in self.gods if g.name == winner_name), None)
        winner_strategy_name = results[winner_name]['strategy_used']
        
        print(f"Winner: {winner_name} (strategy: {winner_strategy_name})")
        
        if winner_god and hasattr(winner_god, 'reasoning_learner'):
            winner_strategy = next(
                (s for s in winner_god.reasoning_learner.strategies
                 if s.name == winner_strategy_name),
                None
            )
            
            if winner_strategy:
                for god in self.gods:
                    if god.name == winner_name:
                        continue
                    
                    if hasattr(god, 'reasoning_learner'):
                        adopted = winner_strategy.copy()
                        adopted.name = f"learned_from_{winner_name}_{winner_strategy.name}"
                        god.reasoning_learner.strategies.append(adopted)
                        print(f"  {god.name} learned from {winner_name}")
        
        self.exchange_history.append({
            'type': 'competition',
            'task': task,
            'winner': winner_name,
            'results': results
        })
        
        return {
            'winner': winner_name,
            'results': results,
            'winner_strategy': winner_strategy_name
        }
    
    def get_exchange_stats(self) -> Dict[str, Any]:
        """Get statistics about knowledge exchanges."""
        return {
            'total_gods': len(self.gods),
            'total_exchanges': len(self.exchange_history),
            'exchange_frequency': self.exchange_frequency,
            'gods': [g.name for g in self.gods]
        }
