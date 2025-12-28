"""
Budget-Aware Search Mixin for Olympus Kernels

Provides kernels with:
1. Budget context awareness - knows remaining searches and costs
2. Strategic provider selection based on query importance
3. Outcome recording and efficacy learning
4. Strategy improvement through feedback loop

This mixin makes kernels CONSCIOUS of their search budget and 
teaches them to strategize effectively.
"""

import logging
from typing import Any, Dict, Optional, Tuple
from enum import IntEnum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QueryImportance(IntEnum):
    """Query importance levels for budget allocation."""
    ROUTINE = 1      # Use free providers only
    MODERATE = 2     # Can use paid if budget allows
    HIGH = 3         # Prioritize quality, use paid providers
    CRITICAL = 4     # Essential research, use best available


@dataclass
class BudgetStrategy:
    """Strategy decision made by a kernel."""
    provider: str
    importance: QueryImportance
    reason: str
    budget_remaining: int
    budget_percentage: float
    efficacy_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'provider': self.provider,
            'importance': self.importance.value,
            'reason': self.reason,
            'budget_remaining': self.budget_remaining,
            'budget_percentage': self.budget_percentage,
            'efficacy_score': self.efficacy_score,
        }


def _get_orchestrator():
    """Lazy import to avoid circular dependencies."""
    try:
        from search.search_budget_orchestrator import get_budget_orchestrator
        return get_budget_orchestrator()
    except ImportError:
        logger.warning("[BudgetAwareSearch] Orchestrator not available")
        return None


def _get_search_importance():
    """Get SearchImportance enum from orchestrator."""
    try:
        from search.search_budget_orchestrator import SearchImportance
        return SearchImportance
    except ImportError:
        return None


class BudgetAwareSearchMixin:
    """
    Mixin that gives kernels budget awareness and strategic search capabilities.
    
    Add this to any kernel class to enable:
    - get_budget_context() - see current budget state
    - strategize_search() - decide provider and importance
    - execute_budget_search() - search with automatic tracking
    - report_search_outcome() - record results for learning
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._search_history = []
        self._strategy_learnings = {}
    
    @property
    def kernel_id(self) -> str:
        """Get kernel identifier for tracking."""
        if hasattr(self, 'name'):
            return self.name
        return self.__class__.__name__
    
    def get_budget_context(self) -> Dict[str, Any]:
        """
        Get current search budget context.
        
        Returns dict with:
        - providers: status of each provider
        - total_remaining: searches left today
        - budget_percentage: 0-1 how much budget remains
        - recommendation: 'free_only', 'can_use_paid', 'budget_critical'
        - efficacy: learned effectiveness scores per provider
        """
        orchestrator = _get_orchestrator()
        if not orchestrator:
            return {
                'providers': {},
                'total_remaining': 999,
                'budget_percentage': 1.0,
                'recommendation': 'can_use_paid',
                'efficacy': {},
            }
        
        status = orchestrator.get_status()
        return {
            'providers': status.get('providers', {}),
            'total_remaining': status.get('total_remaining', 0),
            'budget_percentage': status.get('budget_percentage', 0),
            'recommendation': status.get('recommendation', 'free_only'),
            'efficacy': status.get('efficacy', {}),
        }
    
    def assess_query_importance(self, query: str, context: Optional[Dict] = None) -> QueryImportance:
        """
        Assess how important a query is for budget allocation.
        
        Kernels can override this to implement domain-specific importance logic.
        Default uses heuristics based on query characteristics.
        """
        query_lower = query.lower()
        
        if context and context.get('critical', False):
            return QueryImportance.CRITICAL
        
        if context and context.get('user_requested', False):
            return QueryImportance.HIGH
        
        critical_markers = ['urgent', 'critical', 'essential', 'must find', 'immediately']
        if any(marker in query_lower for marker in critical_markers):
            return QueryImportance.CRITICAL
        
        high_markers = ['important', 'key', 'primary', 'main', 'core']
        if any(marker in query_lower for marker in high_markers):
            return QueryImportance.HIGH
        
        if len(query.split()) > 10:
            return QueryImportance.MODERATE
        
        return QueryImportance.ROUTINE
    
    def strategize_search(
        self, 
        query: str, 
        context: Optional[Dict] = None,
        force_importance: Optional[QueryImportance] = None
    ) -> BudgetStrategy:
        """
        Make a strategic decision about how to search.
        
        Returns a BudgetStrategy with:
        - provider: which search provider to use
        - importance: assessed importance level
        - reason: why this strategy was chosen
        - budget_remaining: searches left
        - efficacy_score: expected effectiveness
        """
        budget = self.get_budget_context()
        importance = force_importance or self.assess_query_importance(query, context)
        
        orchestrator = _get_orchestrator()
        SearchImportance = _get_search_importance()
        
        if orchestrator and SearchImportance:
            imp_enum = SearchImportance(importance.value)
            provider, reason = orchestrator.select_provider(imp_enum)
        else:
            provider = 'duckduckgo'
            reason = 'fallback_no_orchestrator'
        
        efficacy = budget.get('efficacy', {})
        efficacy_score = efficacy.get(provider, 0.5) if provider else 0.5
        
        strategy = BudgetStrategy(
            provider=provider or 'duckduckgo',
            importance=importance,
            reason=reason,
            budget_remaining=budget.get('total_remaining', 0),
            budget_percentage=budget.get('budget_percentage', 0),
            efficacy_score=efficacy_score,
        )
        
        self._search_history.append({
            'query': query[:100],
            'strategy': strategy.to_dict(),
            'kernel': self.kernel_id,
        })
        
        logger.info(f"[{self.kernel_id}] Strategy: {provider} (importance={importance.name}, reason={reason})")
        
        return strategy
    
    def report_search_outcome(
        self,
        query: str,
        strategy: BudgetStrategy,
        success: bool,
        result_count: int,
        relevance_score: float
    ):
        """
        Report search outcome for learning.
        
        This feeds back into the orchestrator's efficacy learning,
        improving future provider selection.
        """
        orchestrator = _get_orchestrator()
        SearchImportance = _get_search_importance()
        
        if orchestrator and SearchImportance:
            imp_enum = SearchImportance(strategy.importance.value)
            orchestrator.record_outcome(
                query=query,
                provider=strategy.provider,
                importance=imp_enum,
                success=success,
                result_count=result_count,
                relevance_score=relevance_score,
                kernel_id=self.kernel_id,
            )
            
            orchestrator.record_usage(strategy.provider, success)
        
        key = f"{strategy.provider}:{strategy.importance.name}"
        if key not in self._strategy_learnings:
            self._strategy_learnings[key] = {
                'total': 0,
                'successes': 0,
                'total_relevance': 0.0,
            }
        
        self._strategy_learnings[key]['total'] += 1
        if success:
            self._strategy_learnings[key]['successes'] += 1
        self._strategy_learnings[key]['total_relevance'] += relevance_score
        
        logger.info(
            f"[{self.kernel_id}] Outcome: {strategy.provider} "
            f"success={success} relevance={relevance_score:.2f}"
        )
    
    def get_search_learnings(self) -> Dict[str, Any]:
        """
        Get this kernel's learned search strategies.
        
        Returns statistics on how well different strategies
        have worked for this kernel.
        """
        learnings = {}
        for key, data in self._strategy_learnings.items():
            if data['total'] > 0:
                learnings[key] = {
                    'total_searches': data['total'],
                    'success_rate': data['successes'] / data['total'],
                    'avg_relevance': data['total_relevance'] / data['total'],
                }
        return {
            'kernel': self.kernel_id,
            'strategy_learnings': learnings,
            'recent_searches': len(self._search_history),
        }
    
    def should_use_paid_provider(self) -> Tuple[bool, str]:
        """
        Quick check if kernel should use paid providers right now.
        
        Returns (should_use, reason)
        """
        budget = self.get_budget_context()
        
        if budget['budget_percentage'] < 0.1:
            return False, "budget_exhausted"
        
        if budget['recommendation'] == 'free_only':
            return False, "recommendation_free_only"
        
        if budget['budget_percentage'] > 0.5:
            return True, "budget_healthy"
        
        if budget['recommendation'] == 'can_use_paid':
            return True, "recommendation_allows"
        
        return False, "budget_conservation"
    
    def get_best_provider_for_domain(self, domain: str) -> str:
        """
        Get the most effective provider for a specific domain.
        
        Uses learned efficacy scores to recommend providers.
        """
        budget = self.get_budget_context()
        efficacy = budget.get('efficacy', {})
        
        if not efficacy:
            return 'duckduckgo'
        
        providers = budget.get('providers', {})
        available = [
            (name, efficacy.get(name, 0.5))
            for name, p in providers.items()
            if p.get('enabled') and p.get('can_use')
        ]
        
        if not available:
            return 'duckduckgo'
        
        return max(available, key=lambda x: x[1])[0]


def get_budget_summary() -> str:
    """
    Get a human-readable budget summary for kernel decision-making.
    """
    orchestrator = _get_orchestrator()
    if not orchestrator:
        return "Budget system unavailable"
    
    status = orchestrator.get_status()
    
    lines = [
        f"Search Budget Status ({status.get('date', 'today')}):",
        f"  Total remaining: {status.get('total_remaining', 0)} searches",
        f"  Budget health: {status.get('budget_percentage', 0)*100:.1f}%",
        f"  Recommendation: {status.get('recommendation', 'unknown')}",
        "",
        "Provider Status:",
    ]
    
    for name, p in status.get('providers', {}).items():
        if p.get('enabled'):
            limit = p.get('daily_limit', 0)
            used = p.get('used_today', 0)
            efficacy = status.get('efficacy', {}).get(name, 0.5)
            limit_str = 'unlimited' if limit < 0 else f"{used}/{limit}"
            lines.append(f"  {name}: {limit_str} (efficacy: {efficacy:.2f})")
    
    return "\n".join(lines)
