"""
Search Budget Orchestrator

Manages daily search quotas with strategic allocation:
- DuckDuckGo: FREE, unlimited
- Google: 100/day limit (configurable)
- Perplexity: 100/day limit (configurable)
- Tavily: Toggle-only (expensive, user must enable)

Kernels receive budget context to make strategic decisions.
Search outcomes feed into learning for kernel evolution.

Extended Features:
- Per-kernel quota tracking
- Time-bound override with expiry
- UI override toggle support
- Broadcast notifications for limit changes
"""

import os
import json
import logging
import hashlib
from datetime import datetime, date, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from capability_telemetry import CapabilityEventBus
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False
    CapabilityEventBus = None

# Database persistence for search outcomes
try:
    import psycopg2
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


def _persist_search_outcome(
    query: str,
    provider: str,
    success: bool,
    result_count: int,
    relevance_score: float,
    kernel_id: Optional[str] = None,
    cost_cents: float = 0.0,
    importance: int = 1,
) -> bool:
    """Persist search outcome to search_outcomes table."""
    if not DB_AVAILABLE:
        return False

    try:
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            return False

        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()

        # Hash the query for grouping/deduplication
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:32]
        # Create query preview (first 100 chars)
        query_preview = query[:100] if query else None

        cursor.execute("""
            INSERT INTO search_outcomes (
                date, query_hash, query_preview, provider, importance, success, result_count,
                relevance_score, cost_cents, kernel_id, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            date.today(),
            query_hash,
            query_preview,
            provider,
            importance,
            success,
            result_count,
            relevance_score,
            cost_cents,
            kernel_id,
            datetime.now(timezone.utc)
        ))

        conn.commit()
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.debug(f"[SearchBudget] Failed to persist search outcome: {e}")
        return False


def _persist_search_feedback(
    query: str,
    provider: str,
    relevance_score: float,
    outcome_quality: float,
    kernel_id: Optional[str] = None,
) -> bool:
    """Persist search feedback to search_feedback table for learning."""
    if not DB_AVAILABLE:
        return False

    try:
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            return False

        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()

        record_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(query.encode()).hexdigest()[:8]}"

        cursor.execute("""
            INSERT INTO search_feedback (
                id, query, feedback_type, relevance_score, source, domain,
                outcome_quality, record_id, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT DO NOTHING
        """, (
            record_id,
            query[:500],  # Truncate long queries
            'auto',
            relevance_score,
            provider,
            kernel_id or 'unknown',
            outcome_quality,
            record_id
        ))

        conn.commit()
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        logger.debug(f"[SearchBudget] Failed to persist search feedback: {e}")
        return False


class SearchImportance(Enum):
    """Query importance levels for budget allocation."""
    ROUTINE = 1      # Use free providers only
    MODERATE = 2     # Can use paid if budget allows
    HIGH = 3         # Prioritize quality, use paid providers
    CRITICAL = 4     # Essential research, use best available


@dataclass
class ProviderBudget:
    """Budget state for a single provider."""
    name: str
    daily_limit: int  # -1 = unlimited, 0 = disabled/toggle-only
    used_today: int = 0
    cost_per_query: float = 0.0
    enabled: bool = True
    toggle_only: bool = False  # Requires explicit user toggle
    
    @property
    def remaining(self) -> int:
        if self.daily_limit < 0:
            return 999999  # Unlimited
        return max(0, self.daily_limit - self.used_today)
    
    @property
    def budget_percentage(self) -> float:
        if self.daily_limit <= 0:
            return 1.0 if self.daily_limit < 0 else 0.0
        return self.remaining / self.daily_limit
    
    def can_use(self) -> bool:
        if not self.enabled:
            return False
        if self.toggle_only and not self.enabled:
            return False
        if self.daily_limit == 0:
            return False
        if self.daily_limit > 0 and self.used_today >= self.daily_limit:
            return False
        return True


@dataclass
class BudgetContext:
    """Context passed to kernels for budget-aware decisions."""
    providers: Dict[str, Dict[str, Any]]
    total_budget_remaining: int
    total_budget_used: int
    budget_percentage: float
    recommendation: str  # 'free_only', 'can_use_paid', 'budget_critical'
    allow_overage: bool
    date: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchOutcome:
    """Tracks search execution for learning."""
    query: str
    provider: str
    importance: int
    success: bool
    result_count: int
    relevance_score: float  # 0-1, how relevant results were
    kernel_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SearchBudgetOrchestrator:
    """
    Orchestrates search budget across providers with strategic allocation.
    
    Features:
    - Daily limits with Redis persistence
    - Strategic provider selection based on importance
    - Learning integration for outcome tracking
    - User overrides for budget limits
    - Per-kernel quota tracking
    - Time-bound override with expiry
    - Broadcast notifications for limit changes
    """
    
    DEFAULT_LIMITS = {
        'duckduckgo': -1,   # Unlimited (free)
        'google': 100,      # 100/day
        'perplexity': 100,  # 100/day
        'tavily': 100,      # 100/day (auto-enabled if API key present)
    }
    
    COST_PER_QUERY = {
        'duckduckgo': 0.0,
        'google': 0.005,
        'perplexity': 0.005,
        'tavily': 0.01,
    }
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.budgets: Dict[str, ProviderBudget] = {}
        self.allow_overage = False
        self.outcomes: List[SearchOutcome] = []
        self._provider_efficacy: Dict[str, float] = {}
        
        self.kernel_allocations: Dict[str, Dict[str, int]] = {}
        self._kernel_usage: Dict[str, Dict[str, int]] = {}
        
        self.override_expires_at: Optional[datetime] = None
        self.override_approved_by: Optional[str] = None
        
        self._event_bus: Optional[Any] = None
        if EVENT_BUS_AVAILABLE:
            try:
                self._event_bus = CapabilityEventBus()
            except Exception:
                pass
        
        self._init_budgets()
        self._load_state()
        
        logger.info("[SearchBudget] Orchestrator initialized")
    
    def _init_budgets(self):
        """Initialize provider budgets with auto-enable for available API keys."""
        import os
        
        # Map providers to their API key environment variables
        api_key_map = {
            'tavily': 'TAVILY_API_KEY',
            'perplexity': 'PERPLEXITY_API_KEY',
            'google': 'GOOGLE_API_KEY',
        }
        
        for provider, limit in self.DEFAULT_LIMITS.items():
            # Auto-enable if API key is available
            has_api_key = bool(provider in api_key_map and os.environ.get(api_key_map[provider]))
            should_enable = (provider == 'duckduckgo') or has_api_key
            
            self.budgets[provider] = ProviderBudget(
                name=provider,
                daily_limit=limit if has_api_key or provider == 'duckduckgo' else 0,
                cost_per_query=self.COST_PER_QUERY.get(provider, 0.0),
                enabled=bool(should_enable),
                toggle_only=False,  # Auto-enable all providers with API keys
            )
            
            if has_api_key and provider != 'duckduckgo':
                logger.info(f"[SearchBudget] Auto-enabled {provider} (API key detected)")
    
    def _get_redis_key(self, provider: str, kernel_id: Optional[str] = None) -> str:
        """Get Redis key for daily counter."""
        today = date.today().isoformat()
        if kernel_id:
            return f"search_budget:{today}:{provider}:{kernel_id}"
        return f"search_budget:{today}:{provider}"
    
    def _load_state(self):
        """Load today's usage from Redis."""
        if not self.redis:
            return
        
        try:
            today = date.today().isoformat()
            
            for provider in self.budgets:
                key = self._get_redis_key(provider)
                count = self.redis.get(key)
                if count:
                    self.budgets[provider].used_today = int(count)
            
            prefs_key = f"search_budget:preferences"
            prefs = self.redis.get(prefs_key)
            if prefs:
                prefs_data = json.loads(prefs)
                self.allow_overage = prefs_data.get('allow_overage', False)
                
                for provider, enabled in prefs_data.get('enabled', {}).items():
                    if provider in self.budgets:
                        self.budgets[provider].enabled = enabled
                
                for provider, limit in prefs_data.get('limits', {}).items():
                    if provider in self.budgets:
                        self.budgets[provider].daily_limit = limit
                
                expires_at_str = prefs_data.get('override_expires_at')
                if expires_at_str:
                    self.override_expires_at = datetime.fromisoformat(expires_at_str)
                else:
                    self.override_expires_at = None
                self.override_approved_by = prefs_data.get('override_approved_by')
                
                self.kernel_allocations = prefs_data.get('kernel_allocations', {})
            
            kernel_usage_key = f"search_budget:kernel_usage:{today}"
            kernel_usage_data = self.redis.get(kernel_usage_key)
            if kernel_usage_data:
                self._kernel_usage = json.loads(kernel_usage_data)
            
            logger.info(f"[SearchBudget] Loaded state for {today}")
            
        except Exception as e:
            logger.error(f"[SearchBudget] Failed to load state: {e}")
    
    def _save_state(self):
        """Save current state to Redis."""
        if not self.redis:
            return
        
        try:
            for provider, budget in self.budgets.items():
                key = self._get_redis_key(provider)
                self.redis.set(key, budget.used_today)
                self.redis.expire(key, 86400 * 2)  # Expire after 2 days
            
            prefs_key = f"search_budget:preferences"
            prefs_data = {
                'allow_overage': self.allow_overage,
                'enabled': {p: b.enabled for p, b in self.budgets.items()},
                'limits': {p: b.daily_limit for p, b in self.budgets.items()},
                'override_expires_at': self.override_expires_at.isoformat() if self.override_expires_at else None,
                'override_approved_by': self.override_approved_by,
                'kernel_allocations': self.kernel_allocations,
            }
            self.redis.set(prefs_key, json.dumps(prefs_data))
            
            today = date.today().isoformat()
            kernel_usage_key = f"search_budget:kernel_usage:{today}"
            self.redis.set(kernel_usage_key, json.dumps(self._kernel_usage))
            self.redis.expire(kernel_usage_key, 86400 * 2)
            
        except Exception as e:
            logger.error(f"[SearchBudget] Failed to save state: {e}")
    
    def _is_override_active(self) -> bool:
        """Check if override is currently active (considering expiry).
        
        IMPORTANT: Also resets allow_overage flag when override expires to keep
        downstream logic consistent. Uses a flag to prevent repeated expiry events.
        """
        if not self.allow_overage:
            return False
        if self.override_expires_at is None:
            return True
        
        now = datetime.now(timezone.utc)
        expires_at = self.override_expires_at.replace(tzinfo=timezone.utc) if self.override_expires_at.tzinfo is None else self.override_expires_at
        
        if now >= expires_at:
            expired_at_iso = expires_at.isoformat()
            self.allow_overage = False
            self.override_expires_at = None
            self.override_approved_by = None
            self._save_state()
            logger.info("[SearchBudget] Override expired - automatically disabled")
            if not getattr(self, '_override_expiry_broadcast_done', False):
                self._override_expiry_broadcast_done = True
                self._broadcast_limit_change('override_expired', {
                    'expired_at': expired_at_iso,
                })
            return False
        
        self._override_expiry_broadcast_done = False
        return True
    
    def _broadcast_limit_change(self, event_type: str, details: Dict[str, Any]):
        """Broadcast limit change event via CapabilityEventBus if available."""
        event_data = {
            'event_type': event_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **details,
        }
        
        logger.info(f"[SearchBudget] Limit change: {event_type} - {details}")
        
        if self._event_bus:
            try:
                self._event_bus.emit('search_budget_change', event_data)
            except Exception as e:
                logger.debug(f"[SearchBudget] Failed to broadcast event: {e}")
    
    def get_provider_quota(self, provider: str, kernel_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get quota information for a provider.
        
        Args:
            provider: The provider name (e.g., 'google', 'tavily')
            kernel_id: Optional kernel ID to get kernel-specific quota
            
        Returns:
            Dict with: total_limit, used, remaining, override_active, override_expires_at
        """
        if provider not in self.budgets:
            return {
                'error': f'Unknown provider: {provider}',
                'total_limit': 0,
                'used': 0,
                'remaining': 0,
                'override_active': False,
                'override_expires_at': None,
            }
        
        budget = self.budgets[provider]
        override_active = self._is_override_active()
        override_expires_at = self.override_expires_at.isoformat() if self.override_expires_at else None
        
        if kernel_id is None:
            return {
                'provider': provider,
                'total_limit': budget.daily_limit,
                'used': budget.used_today,
                'remaining': budget.remaining,
                'override_active': override_active,
                'override_expires_at': override_expires_at,
                'enabled': budget.enabled,
            }
        
        kernel_allocation = self.kernel_allocations.get(kernel_id, {}).get(provider)
        kernel_used = self._kernel_usage.get(kernel_id, {}).get(provider, 0)
        
        if kernel_allocation is None:
            return {
                'provider': provider,
                'kernel_id': kernel_id,
                'total_limit': None,
                'used': kernel_used,
                'remaining': None,
                'override_active': override_active,
                'override_expires_at': override_expires_at,
                'enabled': budget.enabled,
                'note': 'No specific allocation for this kernel',
            }
        
        return {
            'provider': provider,
            'kernel_id': kernel_id,
            'total_limit': kernel_allocation,
            'used': kernel_used,
            'remaining': max(0, kernel_allocation - kernel_used),
            'override_active': override_active,
            'override_expires_at': override_expires_at,
            'enabled': budget.enabled,
        }
    
    def set_kernel_allocation(self, kernel_id: str, provider: str, limit: int) -> bool:
        """Set a soft allocation limit for a specific kernel."""
        if provider not in self.budgets:
            return False
        
        if kernel_id not in self.kernel_allocations:
            self.kernel_allocations[kernel_id] = {}
        
        self.kernel_allocations[kernel_id][provider] = limit
        self._save_state()
        
        self._broadcast_limit_change('kernel_allocation_set', {
            'kernel_id': kernel_id,
            'provider': provider,
            'limit': limit,
        })
        
        return True
    
    def set_override(
        self,
        enabled: bool,
        expires_in_minutes: Optional[int] = None,
        approved_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set or clear the budget override.
        
        Args:
            enabled: Whether to enable the override
            expires_in_minutes: Optional expiry time in minutes (None = no expiry)
            approved_by: Optional string indicating who approved the override
            
        Returns:
            Dict with current override state
        """
        self.allow_overage = enabled
        
        if enabled:
            if expires_in_minutes is not None:
                self.override_expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)
            else:
                self.override_expires_at = None
            self.override_approved_by = approved_by
        else:
            self.override_expires_at = None
            self.override_approved_by = None
        
        self._save_state()
        
        self._broadcast_limit_change('override_changed', {
            'enabled': enabled,
            'expires_at': self.override_expires_at.isoformat() if self.override_expires_at else None,
            'approved_by': approved_by,
        })
        
        return self.get_override_status()
    
    def get_override_status(self) -> Dict[str, Any]:
        """
        Get current override status for UI display.
        
        Returns:
            Dict with: enabled, active, expires_at, approved_by, expires_in_seconds
        """
        is_active = self._is_override_active()
        expires_in_seconds = None
        
        if self.override_expires_at:
            now = datetime.now(timezone.utc)
            expires_at = self.override_expires_at.replace(tzinfo=timezone.utc) if self.override_expires_at.tzinfo is None else self.override_expires_at
            diff = (expires_at - now).total_seconds()
            expires_in_seconds = max(0, int(diff))
        
        return {
            'enabled': self.allow_overage,
            'active': is_active,
            'expires_at': self.override_expires_at.isoformat() if self.override_expires_at else None,
            'approved_by': self.override_approved_by,
            'expires_in_seconds': expires_in_seconds,
        }
    
    def get_budget_context(self) -> BudgetContext:
        """Get current budget context for kernel decisions."""
        total_remaining = sum(
            b.remaining for b in self.budgets.values() 
            if b.daily_limit > 0
        )
        total_used = sum(b.used_today for b in self.budgets.values())
        total_limit = sum(
            b.daily_limit for b in self.budgets.values() 
            if b.daily_limit > 0
        )
        
        budget_pct = total_remaining / total_limit if total_limit > 0 else 1.0
        
        if budget_pct > 0.5:
            recommendation = 'can_use_paid'
        elif budget_pct > 0.1:
            recommendation = 'budget_critical'
        else:
            recommendation = 'free_only'
        
        return BudgetContext(
            providers={
                name: {
                    'remaining': b.remaining,
                    'daily_limit': b.daily_limit,
                    'used_today': b.used_today,
                    'enabled': b.enabled,
                    'can_use': b.can_use(),
                    'cost': b.cost_per_query,
                }
                for name, b in self.budgets.items()
            },
            total_budget_remaining=total_remaining,
            total_budget_used=total_used,
            budget_percentage=budget_pct,
            recommendation=recommendation,
            allow_overage=self.allow_overage,
            date=date.today().isoformat(),
        )
    
    def select_provider(
        self, 
        importance: SearchImportance = SearchImportance.ROUTINE,
        preferred_provider: Optional[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        Select best provider based on importance and budget.
        
        Now properly filters out exhausted premium providers (remaining=0) unless
        override is active. This ensures callers never receive a provider that
        cannot actually be used.
        
        Returns: (provider_name, reason)
        """
        override_active = self._is_override_active()
        premium_providers = {'tavily', 'perplexity', 'google'}
        
        def _can_use_provider(name: str, budget: ProviderBudget) -> bool:
            """Check if a provider can actually be used (considering quota)."""
            if not budget.enabled:
                return False
            if budget.daily_limit == 0:
                return False
            if budget.daily_limit == -1:
                return True
            if name in premium_providers:
                if budget.remaining <= 0:
                    return override_active
            else:
                return budget.can_use()
            return budget.remaining > 0 or override_active
        
        if preferred_provider and preferred_provider in self.budgets:
            budget = self.budgets[preferred_provider]
            if _can_use_provider(preferred_provider, budget):
                return preferred_provider, "user_preferred"
            elif preferred_provider in premium_providers and budget.remaining <= 0:
                logger.info(f"[SearchBudget] Preferred provider {preferred_provider} exhausted, falling back")
        
        available = [
            (name, b) for name, b in self.budgets.items()
            if _can_use_provider(name, b)
        ]
        
        if not available:
            if self.budgets['duckduckgo'].enabled:
                return 'duckduckgo', "fallback_free"
            return None, "no_providers_available"
        
        if importance == SearchImportance.ROUTINE:
            for name, b in available:
                if b.cost_per_query == 0:
                    return name, "routine_free"
            return available[0][0], "routine_first_available"
        
        if importance == SearchImportance.CRITICAL:
            paid = [(n, b) for n, b in available if b.cost_per_query > 0 and n in premium_providers]
            if paid:
                best = max(paid, key=lambda x: x[1].cost_per_query)
                return best[0], "critical_best_quality"
        
        if importance in (SearchImportance.MODERATE, SearchImportance.HIGH):
            ctx = self.get_budget_context()
            
            if ctx.budget_percentage > 0.3 or importance == SearchImportance.HIGH:
                paid = [(n, b) for n, b in available if b.cost_per_query > 0 and n in premium_providers]
                if paid:
                    efficacy_scores = [
                        (n, self._provider_efficacy.get(n, 0.5))
                        for n, _ in paid
                    ]
                    best = max(efficacy_scores, key=lambda x: x[1])
                    return best[0], f"strategic_efficacy_{best[1]:.2f}"
            
            for name, b in available:
                if b.cost_per_query == 0:
                    return name, "budget_conscious_free"
        
        return available[0][0], "default_first_available"
    
    def consume_quota(self, provider: str, kernel_id: Optional[str] = None) -> bool:
        """
        Consume quota BEFORE executing a search.
        
        This ensures failed premium requests count against the limit.
        Call this BEFORE dispatching the search, not after.
        
        Args:
            provider: The provider to consume quota for
            kernel_id: Optional kernel ID to track per-kernel usage
            
        Returns:
            True if quota was consumed successfully, False if no quota available
        """
        if provider not in self.budgets:
            return False
        
        budget = self.budgets[provider]
        override_active = self._is_override_active()
        
        if budget.daily_limit == -1:
            return True
        
        if budget.remaining <= 0 and not override_active:
            logger.warning(f"[SearchBudget] Cannot consume quota: {provider} exhausted (remaining=0, override={override_active})")
            return False
        
        budget.used_today += 1
        
        if kernel_id:
            if kernel_id not in self._kernel_usage:
                self._kernel_usage[kernel_id] = {}
            current = self._kernel_usage[kernel_id].get(provider, 0)
            self._kernel_usage[kernel_id][provider] = current + 1
            
            if self.redis:
                try:
                    kernel_key = self._get_redis_key(provider, kernel_id)
                    self.redis.incr(kernel_key)
                    self.redis.expire(kernel_key, 86400 * 2)
                except Exception as e:
                    logger.debug(f"[SearchBudget] Failed to update kernel usage in Redis: {e}")
        
        self._save_state()
        logger.debug(f"[SearchBudget] Consumed quota: {provider}: {budget.used_today}/{budget.daily_limit}" + (f" (kernel: {kernel_id})" if kernel_id else ""))
        return True
    
    def record_usage(self, provider: str, success: bool = True, kernel_id: Optional[str] = None):
        """
        Record that a search was executed (DEPRECATED - use consume_quota before search).
        
        This method is kept for backward compatibility but consume_quota should be
        preferred for new code as it ensures quota is consumed BEFORE the search.
        
        Args:
            provider: The provider used
            success: Whether the search was successful
            kernel_id: Optional kernel ID to track per-kernel usage
        """
        if provider in self.budgets:
            self.budgets[provider].used_today += 1
            
            if kernel_id:
                if kernel_id not in self._kernel_usage:
                    self._kernel_usage[kernel_id] = {}
                current = self._kernel_usage[kernel_id].get(provider, 0)
                self._kernel_usage[kernel_id][provider] = current + 1
                
                if self.redis:
                    try:
                        kernel_key = self._get_redis_key(provider, kernel_id)
                        self.redis.incr(kernel_key)
                        self.redis.expire(kernel_key, 86400 * 2)
                    except Exception as e:
                        logger.debug(f"[SearchBudget] Failed to update kernel usage in Redis: {e}")
            
            self._save_state()
            logger.debug(f"[SearchBudget] {provider}: {self.budgets[provider].used_today}/{self.budgets[provider].daily_limit}" + (f" (kernel: {kernel_id})" if kernel_id else ""))
    
    def record_outcome(
        self,
        query: str,
        provider: str,
        importance: SearchImportance,
        success: bool,
        result_count: int,
        relevance_score: float,
        kernel_id: Optional[str] = None
    ):
        """Record search outcome for learning."""
        outcome = SearchOutcome(
            query=query,
            provider=provider,
            importance=importance.value,
            success=success,
            result_count=result_count,
            relevance_score=relevance_score,
            kernel_id=kernel_id,
        )
        
        self.outcomes.append(outcome)
        
        if len(self.outcomes) > 1000:
            self.outcomes = self.outcomes[-500:]
        
        self._update_efficacy(provider, relevance_score)

        # Persist to PostgreSQL search_outcomes table (for analytics and learning)
        cost_cents = self.budgets.get(provider, ProviderBudget(name=provider, daily_limit=0)).cost_per_query * 100
        _persist_search_outcome(
            query=query,
            provider=provider,
            success=success,
            result_count=result_count,
            relevance_score=relevance_score,
            kernel_id=kernel_id,
            cost_cents=cost_cents,
            importance=importance.value
        )
        
        # Also persist to search_feedback for learning
        _persist_search_feedback(
            query=query,
            provider=provider,
            relevance_score=relevance_score,
            outcome_quality=relevance_score if success else 0.0,
            kernel_id=kernel_id
        )

        if self.redis:
            try:
                key = f"search_outcomes:{date.today().isoformat()}"
                self.redis.rpush(key, json.dumps(outcome.to_dict()))
                self.redis.expire(key, 86400 * 7)
            except Exception as e:
                logger.error(f"[SearchBudget] Failed to save outcome: {e}")
    
    def _update_efficacy(self, provider: str, relevance: float):
        """Update provider efficacy with exponential moving average."""
        alpha = 0.1  # Learning rate
        current = self._provider_efficacy.get(provider, 0.5)
        self._provider_efficacy[provider] = current * (1 - alpha) + relevance * alpha
        
        # Persist to provider_efficacy table
        self._persist_provider_efficacy(provider)
    
    def _persist_provider_efficacy(self, provider: str):
        """Persist provider efficacy metrics to PostgreSQL."""
        try:
            # Aggregate metrics from outcomes for this provider
            provider_outcomes = [o for o in self.outcomes if o.provider == provider]
            if not provider_outcomes:
                return
            
            total_queries = len(provider_outcomes)
            successful_queries = sum(1 for o in provider_outcomes if o.success)
            avg_relevance = sum(o.relevance for o in provider_outcomes) / total_queries if total_queries > 0 else 0.0
            efficacy_score = self._provider_efficacy.get(provider, 0.5)
            
            # Calculate cost from budget
            budget = self.budgets.get(provider)
            cost_per_query = budget.cost_per_query if budget else 0.0
            total_cost_cents = int(total_queries * cost_per_query * 100)
            cost_per_successful = (total_cost_cents / successful_queries) if successful_queries > 0 else 0.0
            
            database_url = os.environ.get('DATABASE_URL')
            if database_url and DB_AVAILABLE:
                conn = psycopg2.connect(database_url)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO provider_efficacy (
                        provider, total_queries, successful_queries, avg_relevance,
                        efficacy_score, total_cost_cents, cost_per_successful_query, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (provider) DO UPDATE SET
                        total_queries = EXCLUDED.total_queries,
                        successful_queries = EXCLUDED.successful_queries,
                        avg_relevance = EXCLUDED.avg_relevance,
                        efficacy_score = EXCLUDED.efficacy_score,
                        total_cost_cents = EXCLUDED.total_cost_cents,
                        cost_per_successful_query = EXCLUDED.cost_per_successful_query,
                        updated_at = NOW()
                """, (
                    provider, total_queries, successful_queries, avg_relevance,
                    efficacy_score, total_cost_cents, cost_per_successful
                ))
                conn.commit()
                cur.close()
                conn.close()
        except Exception as e:
            logger.debug(f"[SearchBudget] Failed to persist provider efficacy: {e}")
    
    def set_provider_enabled(self, provider: str, enabled: bool) -> bool:
        """
        Enable or disable a provider.
        
        Also syncs with GeometricProviderSelector for premium providers (tavily, perplexity)
        to ensure they are included/excluded from geometric provider selection.
        """
        if provider not in self.budgets:
            return False
        
        self.budgets[provider].enabled = enabled
        self._save_state()
        self._persist_budget_preferences()
        
        # Sync with GeometricProviderSelector for premium providers
        try:
            from search.provider_selector import get_provider_selector
            selector = get_provider_selector('regular')
            if provider in selector.PREMIUM_PROVIDERS:
                selector.enable_premium_provider(provider, enabled)
                logger.info(f"[SearchBudget] Synced {provider} enabled={enabled} with GeometricProviderSelector")
        except Exception as e:
            logger.warning(f"[SearchBudget] Failed to sync with provider selector: {e}")
        
        self._broadcast_limit_change('provider_enabled_changed', {
            'provider': provider,
            'enabled': enabled,
        })
        
        return True
    
    def set_daily_limit(self, provider: str, limit: int) -> bool:
        """Set daily limit for a provider."""
        if provider not in self.budgets:
            return False
        
        old_limit = self.budgets[provider].daily_limit
        self.budgets[provider].daily_limit = limit
        self._save_state()
        self._persist_budget_preferences()
        
        self._broadcast_limit_change('daily_limit_changed', {
            'provider': provider,
            'old_limit': old_limit,
            'new_limit': limit,
        })
        
        return True
    
    def set_allow_overage(self, allow: bool):
        """Set whether to allow exceeding daily limits."""
        self.allow_overage = allow
        self._save_state()
        self._persist_budget_preferences()
        logger.info(f"[SearchBudget] allow_overage={allow}")
    
    def _persist_budget_preferences(self):
        """Persist current budget preferences to PostgreSQL."""
        try:
            database_url = os.environ.get('DATABASE_URL')
            if not database_url or not DB_AVAILABLE:
                return
            
            conn = psycopg2.connect(database_url)
            cur = conn.cursor()
            
            # Get current limits and enabled status for each provider
            google_limit = self.budgets.get('google', ProviderBudget(name='google', daily_limit=0)).daily_limit
            perplexity_limit = self.budgets.get('perplexity', ProviderBudget(name='perplexity', daily_limit=0)).daily_limit
            tavily_limit = self.budgets.get('tavily', ProviderBudget(name='tavily', daily_limit=0)).daily_limit
            
            google_enabled = self.budgets.get('google', ProviderBudget(name='google', daily_limit=0)).enabled
            perplexity_enabled = self.budgets.get('perplexity', ProviderBudget(name='perplexity', daily_limit=0)).enabled
            tavily_enabled = self.budgets.get('tavily', ProviderBudget(name='tavily', daily_limit=0)).enabled
            
            cur.execute("""
                INSERT INTO search_budget_preferences (
                    id, google_daily_limit, perplexity_daily_limit, tavily_daily_limit,
                    google_enabled, perplexity_enabled, tavily_enabled, allow_overage,
                    created_at, updated_at
                )
                VALUES (1, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (id) DO UPDATE SET
                    google_daily_limit = EXCLUDED.google_daily_limit,
                    perplexity_daily_limit = EXCLUDED.perplexity_daily_limit,
                    tavily_daily_limit = EXCLUDED.tavily_daily_limit,
                    google_enabled = EXCLUDED.google_enabled,
                    perplexity_enabled = EXCLUDED.perplexity_enabled,
                    tavily_enabled = EXCLUDED.tavily_enabled,
                    allow_overage = EXCLUDED.allow_overage,
                    updated_at = NOW()
            """, (
                google_limit, perplexity_limit, tavily_limit,
                google_enabled, perplexity_enabled, tavily_enabled,
                self.allow_overage
            ))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.debug(f"[SearchBudget] Failed to persist preferences: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get full budget status."""
        ctx = self.get_budget_context()
        override_status = self.get_override_status()
        
        return {
            'date': ctx.date,
            'providers': ctx.providers,
            'total_remaining': ctx.total_budget_remaining,
            'total_used': ctx.total_budget_used,
            'budget_percentage': ctx.budget_percentage,
            'recommendation': ctx.recommendation,
            'allow_overage': ctx.allow_overage,
            'override': override_status,
            'kernel_allocations': self.kernel_allocations,
            'kernel_usage': self._kernel_usage,
            'efficacy': self._provider_efficacy,
            'recent_outcomes': len(self.outcomes),
        }
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get metrics for kernel learning/evolution."""
        if not self.outcomes:
            return {'message': 'No outcomes recorded yet'}
        
        by_provider: Dict[str, List[SearchOutcome]] = {}
        for o in self.outcomes:
            by_provider.setdefault(o.provider, []).append(o)
        
        by_importance: Dict[int, List[SearchOutcome]] = {}
        for o in self.outcomes:
            by_importance.setdefault(o.importance, []).append(o)
        
        return {
            'provider_stats': {
                provider: {
                    'count': len(outcomes),
                    'avg_relevance': sum(o.relevance_score for o in outcomes) / len(outcomes),
                    'success_rate': sum(1 for o in outcomes if o.success) / len(outcomes),
                }
                for provider, outcomes in by_provider.items()
            },
            'importance_stats': {
                imp: {
                    'count': len(outcomes),
                    'avg_relevance': sum(o.relevance_score for o in outcomes) / len(outcomes),
                }
                for imp, outcomes in by_importance.items()
            },
            'efficacy_scores': self._provider_efficacy,
            'total_outcomes': len(self.outcomes),
        }
    
    def reset_daily(self):
        """Reset daily counters (called at midnight)."""
        for budget in self.budgets.values():
            budget.used_today = 0
        self._save_state()
        logger.info("[SearchBudget] Daily counters reset")


_orchestrator: Optional[SearchBudgetOrchestrator] = None


def get_budget_orchestrator(redis_client=None) -> SearchBudgetOrchestrator:
    """Get or create singleton budget orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SearchBudgetOrchestrator(redis_client)
    return _orchestrator


def get_budget_context() -> BudgetContext:
    """Convenience function to get current budget context."""
    return get_budget_orchestrator().get_budget_context()
