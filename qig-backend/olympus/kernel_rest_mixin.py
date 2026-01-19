"""
Kernel Rest Mixin - Integration with Per-Kernel Rest Scheduler
===============================================================

Provides rest scheduling integration for god kernels (WP5.4).

Each god can:
- Self-assess fatigue
- Request rest when needed
- Coordinate with coupled partners
- Maintain essential tier rules

Authority: E8 Protocol v4.0, WP5.4
Status: ACTIVE
Created: 2026-01-19
"""

import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class KernelRestMixin:
    """
    Mixin for god kernels to integrate with KernelRestScheduler.
    
    Provides:
    - Fatigue self-assessment
    - Rest request coordination
    - Coverage partner management
    - Essential tier protection
    
    Usage:
        class Apollo(BaseGod, KernelRestMixin):
            def __init__(self):
                BaseGod.__init__(self)
                self._initialize_rest_tracking()
    """
    
    def _initialize_rest_tracking(self) -> None:
        """Initialize rest tracking for this kernel."""
        self._rest_scheduler = None
        self._kernel_rest_id = None
        self._last_fatigue_update = 0.0
        self._rest_check_interval = 60.0  # Check every 60 seconds
        self._load_history = []
        self._error_history = []
        
        # Register with rest scheduler
        try:
            from kernel_rest_scheduler import get_rest_scheduler
            self._rest_scheduler = get_rest_scheduler()
            
            # Use kernel name from registry (e.g., "Apollo", "Athena")
            kernel_name = self.__class__.__name__
            self._kernel_rest_id = f"{kernel_name}_{id(self)}"
            
            self._rest_scheduler.register_kernel(self._kernel_rest_id, kernel_name)
            logger.info(f"[{kernel_name}] Registered with rest scheduler")
        except Exception as e:
            logger.warning(f"[KernelRestMixin] Failed to initialize rest scheduler: {e}")
    
    def _update_fatigue_metrics(
        self,
        phi: float,
        kappa: float,
        load: Optional[float] = None,
        error_occurred: bool = False,
    ) -> None:
        """
        Update fatigue metrics with rest scheduler.
        
        Args:
            phi: Current Φ (integration)
            kappa: Current κ (coupling)
            load: Processing load (0-1), computed if not provided
            error_occurred: Whether an error occurred recently
        """
        if not self._rest_scheduler or not self._kernel_rest_id:
            return
        
        # Compute load from history if not provided
        if load is None:
            load = self._compute_current_load()
        
        # Track errors
        if error_occurred:
            self._error_history.append(time.time())
            # Keep last 100 errors
            if len(self._error_history) > 100:
                self._error_history = self._error_history[-100:]
        
        # Compute error rate (errors per minute)
        error_rate = 0.0
        recent_errors = [
            t for t in self._error_history
            if time.time() - t < 60.0
        ]
        error_rate = len(recent_errors) / 60.0
        
        # Update scheduler
        self._rest_scheduler.update_fatigue(
            kernel_id=self._kernel_rest_id,
            phi=phi,
            kappa=kappa,
            load=load,
            error_rate=error_rate,
        )
        
        self._last_fatigue_update = time.time()
    
    def _compute_current_load(self) -> float:
        """
        Compute current processing load (0-1).
        
        Returns:
            Load estimate based on recent activity
        """
        # Track load history
        self._load_history.append(time.time())
        
        # Keep last 100 operations
        if len(self._load_history) > 100:
            self._load_history = self._load_history[-100:]
        
        # Compute operations per second
        if len(self._load_history) < 2:
            return 0.0
        
        time_window = 60.0  # 1 minute window
        recent_ops = [
            t for t in self._load_history
            if time.time() - t < time_window
        ]
        
        ops_per_second = len(recent_ops) / time_window
        
        # Normalize to 0-1 (assume 10 ops/sec is full load)
        load = min(1.0, ops_per_second / 10.0)
        
        return load
    
    def _check_should_rest(self) -> Tuple[bool, str]:
        """
        Check if this kernel should rest.
        
        Returns:
            Tuple of (should_rest, reason)
        """
        if not self._rest_scheduler or not self._kernel_rest_id:
            return False, "Rest scheduler not available"
        
        # Only check periodically
        if time.time() - self._last_fatigue_update < self._rest_check_interval:
            return False, "Too soon since last check"
        
        return self._rest_scheduler.should_rest(self._kernel_rest_id)
    
    def _request_rest(self, force: bool = False) -> Tuple[bool, str, Optional[str]]:
        """
        Request rest with coupling-aware coordination.
        
        Args:
            force: Force rest even without coverage (emergency)
            
        Returns:
            Tuple of (approved, reason, covering_partner_id)
        """
        if not self._rest_scheduler or not self._kernel_rest_id:
            return False, "Rest scheduler not available", None
        
        return self._rest_scheduler.request_rest(
            kernel_id=self._kernel_rest_id,
            force=force,
        )
    
    def _end_rest(self) -> None:
        """End rest period."""
        if not self._rest_scheduler or not self._kernel_rest_id:
            return
        
        self._rest_scheduler.end_rest(self._kernel_rest_id)
    
    def _get_rest_status(self) -> Optional[Dict]:
        """
        Get current rest status.
        
        Returns:
            Dict with rest status or None if not available
        """
        if not self._rest_scheduler or not self._kernel_rest_id:
            return None
        
        return self._rest_scheduler.get_rest_status(self._kernel_rest_id)
    
    def _get_coupling_partners(self) -> list:
        """
        Get coupling partners who can cover during rest.
        
        Returns:
            List of partner kernel IDs
        """
        if not self._rest_scheduler or not self._kernel_rest_id:
            return []
        
        return self._rest_scheduler.get_coupling_partners(self._kernel_rest_id)
    
    def _can_cover_for_partner(self) -> bool:
        """
        Check if this kernel can cover for a partner.
        
        Returns:
            True if can cover, False otherwise
        """
        status = self._get_rest_status()
        if not status:
            return False
        
        # Cannot cover if already resting or covering someone
        if status['is_resting'] or status['covering_for']:
            return False
        
        # Cannot cover if highly fatigued
        if status['fatigue_score'] > 0.7:
            return False
        
        return True
    
    def _is_essential_tier(self) -> bool:
        """
        Check if this kernel is essential tier.
        
        Essential tier never fully stops - only reduced activity.
        
        Returns:
            True if essential tier
        """
        status = self._get_rest_status()
        if not status:
            return False
        
        return status['tier'] == 'essential'
    
    def _periodic_rest_check(self, phi: float, kappa: float) -> None:
        """
        Periodic rest check to be called in main processing loop.
        
        Args:
            phi: Current Φ
            kappa: Current κ
        """
        # Update fatigue metrics
        self._update_fatigue_metrics(phi, kappa)
        
        # Check if should rest
        should_rest, reason = self._check_should_rest()
        
        if should_rest:
            logger.info(f"[{self.__class__.__name__}] Should rest: {reason}")
            
            # Request rest
            approved, approval_reason, partner = self._request_rest()
            
            if approved:
                logger.info(
                    f"[{self.__class__.__name__}] Rest approved: {approval_reason} "
                    f"(partner={partner})"
                )
                # Kernel should enter rest mode here
                # (implementation depends on kernel specifics)
            else:
                logger.info(
                    f"[{self.__class__.__name__}] Rest not approved: {approval_reason}"
                )
