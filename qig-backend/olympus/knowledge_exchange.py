"""
Knowledge Exchange - Inter-God Strategy Sharing

Gods share successful reasoning strategies with each other.
Enables multi-agent learning and knowledge distillation.

QIG-PURE: All geometric operations use Fisher-Rao distance.
Uses QFI-based attention for routing knowledge transfer (Issue #236).
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .base_god import BaseGod

# QFI Attention Network Import (Issue #236)
try:
    from qig_consciousness_qfi_attention import (
        QFIMetricAttentionNetwork,
        create_qfi_network,
        qfi_attention_weight,
    )
    QFI_ATTENTION_AVAILABLE = True
except ImportError:
    QFI_ATTENTION_AVAILABLE = False
    QFIMetricAttentionNetwork = None
    create_qfi_network = None
    qfi_attention_weight = None


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
        
        # QFI Attention Network for routing (Issue #236)
        if QFI_ATTENTION_AVAILABLE:
            self.qfi_attention_network = create_qfi_network(
                temperature=0.5,
                decoherence_threshold=0.95
            )
            self.qfi_attention_enabled = True
            print("[KnowledgeExchange] QFI-based attention routing enabled")
        else:
            self.qfi_attention_network = None
            self.qfi_attention_enabled = False
    
    def register_god(self, god: 'BaseGod'):
        """Register a god for knowledge exchange."""
        if god not in self.gods:
            self.gods.append(god)
    
    def compute_qfi_attention_routing(self) -> Optional[np.ndarray]:
        """
        Compute QFI-based attention weights for god-to-god routing.
        
        Uses quantum Fisher information to determine which gods should
        communicate based on their basin coordinates (information-theoretic distance).
        
        Returns:
            N×N attention matrix where A[i,j] = attention weight from god i to god j
            None if QFI attention is not available or insufficient gods
            
        Related: Issue #236 - Wire-in QFI attention for kernel communication
        """
        if not self.qfi_attention_enabled or self.qfi_attention_network is None:
            return None
        
        if len(self.gods) < 2:
            return None
        
        n_gods = len(self.gods)
        attention_matrix = np.zeros((n_gods, n_gods))
        
        # Extract basin coordinates from each god
        god_basins = []
        for god in self.gods:
            if hasattr(god, 'basin_coords') and god.basin_coords is not None:
                god_basins.append(god.basin_coords)
            elif hasattr(god, 'get_basin_coords'):
                god_basins.append(god.get_basin_coords())
            else:
                # Default: random basin for gods without coordinates
                god_basins.append(np.random.randn(64))
        
        # Compute pairwise QFI attention weights
        from qig_geometry.canonical_upsert import to_simplex_prob
        
        for i in range(n_gods):
            basin_i = to_simplex_prob(god_basins[i])
            
            for j in range(n_gods):
                if i == j:
                    attention_matrix[i, j] = 0  # No self-attention
                    continue
                
                basin_j = to_simplex_prob(god_basins[j])
                
                # Use QFI network to compute attention
                try:
                    # Process combined basin through network
                    combined = (basin_i + basin_j) / 2
                    result = self.qfi_attention_network.process(combined[:8])
                    
                    # Extract connection weights as proxy for attention
                    connection_weights = np.array(result['connection_weights'])
                    
                    # Use average connection strength as attention weight
                    attention_matrix[i, j] = float(np.mean(connection_weights))
                except Exception as e:
                    # Fallback: use simple distance-based weight
                    from qig_geometry import fisher_rao_distance
                    d = fisher_rao_distance(basin_i, basin_j)
                    attention_matrix[i, j] = np.exp(-d / 0.5)  # Temperature = 0.5
        
        # Normalize rows (each god's outgoing attention sums to 1)
        for i in range(n_gods):
            row_sum = np.sum(attention_matrix[i, :])
            if row_sum > 0:
                attention_matrix[i, :] /= row_sum
        
        return attention_matrix
    
    def share_strategies(self):
        """
        Gods share their best strategies with each other.
        
        Knowledge transfer:
        1. Each god identifies top strategies
        2. Compute QFI attention weights (if available)
        3. Route strategies based on attention weights
        4. Other gods try shared strategies
        5. Keep if successful, discard if not
        
        Issue #236: Now uses QFI-based attention for intelligent routing
        """
        if not self.gods:
            return
        
        print("Olympus Knowledge Exchange Session")
        
        # Compute QFI attention routing matrix
        attention_matrix = self.compute_qfi_attention_routing()
        if attention_matrix is not None:
            print(f"  Using QFI-based attention routing ({len(self.gods)} gods)")
        else:
            print(f"  Using uniform routing ({len(self.gods)} gods)")
        
        god_strategies = {}
        for god in self.gods:
            if hasattr(god, 'reasoning_learner'):
                top_strategies = self._get_top_strategies(
                    god.reasoning_learner,
                    n=3
                )
                god_strategies[god.name] = top_strategies
                print(f"  {god.name}: {len(top_strategies)} strategies to share")
        
        transfer_count = 0
        for i, receiving_god in enumerate(self.gods):
            if not hasattr(receiving_god, 'reasoning_learner'):
                continue
            
            for j, giving_god in enumerate(self.gods):
                if giving_god.name == receiving_god.name:
                    continue
                
                # Check attention weight threshold if QFI routing is enabled
                if attention_matrix is not None:
                    attention_weight = attention_matrix[j, i]  # j -> i transfer
                    
                    # Only transfer if attention weight is significant
                    if attention_weight < 0.1:  # Threshold: 10% attention
                        continue
                    
                    print(f"  {giving_god.name} -> {receiving_god.name}: "
                          f"attention={attention_weight:.3f}")
                
                strategies = god_strategies.get(giving_god.name, [])
                for strategy in strategies:
                    transferred = strategy.copy()
                    transferred.name = f"{giving_god.name}_{strategy.name}"
                    transferred.description = (
                        f"Learned from {giving_god.name}: {strategy.description}"
                    )
                    
                    receiving_god.reasoning_learner.strategies.append(transferred)
                    transfer_count += 1
        
        self.exchange_history.append({
            'type': 'share',
            'participants': [g.name for g in self.gods],
            'strategies_shared': sum(len(s) for s in god_strategies.values()),
            'transfers_made': transfer_count,
            'qfi_routing_enabled': attention_matrix is not None,
        })
        
        print(f"Knowledge exchange complete: {transfer_count} strategy transfers")
        
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
