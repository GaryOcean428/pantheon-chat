"""
Lifecycle Policy Engine - Automated Kernel Management
====================================================

Monitors kernel metrics and automatically triggers lifecycle operations
based on policy rules (split, merge, prune, promote).

Authority: E8 Protocol v4.0, WP5.3
Status: ACTIVE
Created: 2026-01-18

Policy Types:
- Split: Detect overloaded kernels and split into specialists
- Merge: Detect redundant kernels and combine capabilities
- Prune: Archive underperforming kernels to shadow pantheon
- Promote: Elevate successful chaos kernels to god status
- Resurrect: Restore needed capabilities from shadow pantheon

Monitoring:
- Continuous evaluation of kernel metrics
- Graduated metrics for chaos kernels (protection periods)
- Trigger thresholds based on Φ, κ, performance, geometry
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

import numpy as np

from kernel_lifecycle import (
    Kernel,
    KernelLifecycleManager,
    LifecycleEvent,
    compute_fisher_distance,
)

logger = logging.getLogger(__name__)


# =============================================================================
# POLICY DEFINITIONS
# =============================================================================

@dataclass
class PolicyRule:
    """
    Lifecycle policy rule definition.
    
    Defines when and how a lifecycle operation should trigger.
    """
    policy_name: str
    policy_type: LifecycleEvent
    
    # Trigger conditions
    trigger_fn: Callable[[Kernel, Dict[str, Any]], bool]
    
    # Action parameters
    action_params: Dict[str, Any]
    
    # Policy metadata
    enabled: bool = True
    priority: int = 0
    description: str = ""
    
    # Cooldown to prevent repeated triggers
    cooldown_cycles: int = 10
    last_triggered: Dict[str, datetime] = None
    
    def __post_init__(self):
        if self.last_triggered is None:
            self.last_triggered = {}
    
    def can_trigger(self, kernel: Kernel) -> bool:
        """Check if policy can trigger for this kernel (considering cooldown)."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if kernel.kernel_id in self.last_triggered:
            last_time = self.last_triggered[kernel.kernel_id]
            if datetime.now() - last_time < timedelta(seconds=self.cooldown_cycles * 10):
                return False
        
        return True
    
    def mark_triggered(self, kernel: Kernel):
        """Mark policy as triggered for this kernel."""
        self.last_triggered[kernel.kernel_id] = datetime.now()


# =============================================================================
# LIFECYCLE POLICY ENGINE
# =============================================================================

class LifecyclePolicyEngine:
    """
    Automated lifecycle policy engine.
    
    Monitors kernel metrics and triggers lifecycle operations based on
    configured policy rules.
    
    Example:
        engine = LifecyclePolicyEngine(lifecycle_manager)
        
        # Add custom policy
        engine.add_policy(PolicyRule(
            policy_name="split_high_load",
            policy_type=LifecycleEvent.SPLIT,
            trigger_fn=lambda k, ctx: k.phi > 0.7 and len(k.domains) > 2,
            action_params={"split_criterion": "domain"},
            description="Split high-Φ kernels with multiple domains"
        ))
        
        # Evaluate all kernels
        actions = engine.evaluate_all_kernels(context)
        
        # Execute triggered actions
        for action in actions:
            engine.execute_action(action)
    """
    
    def __init__(
        self,
        lifecycle_manager: Optional[KernelLifecycleManager] = None,
        policies: Optional[List[PolicyRule]] = None,
    ):
        """
        Initialize policy engine.
        
        Args:
            lifecycle_manager: Lifecycle manager for executing operations
            policies: Initial policy rules (default: built-in policies)
        """
        from kernel_lifecycle import get_lifecycle_manager
        self.lifecycle_manager = lifecycle_manager or get_lifecycle_manager()
        self.policies = policies or []
        
        # Initialize default policies if none provided
        if not self.policies:
            self._init_default_policies()
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
    
    def _init_default_policies(self):
        """Initialize default lifecycle policies."""
        # Policy 1: Prune low Φ persistent
        self.policies.append(PolicyRule(
            policy_name="prune_low_phi_persistent",
            policy_type=LifecycleEvent.PRUNE,
            trigger_fn=self._trigger_prune_low_phi,
            action_params={},
            priority=10,
            description="Prune kernels with persistent Φ < 0.1",
            cooldown_cycles=100,
        ))
        
        # Policy 2: Prune no growth
        self.policies.append(PolicyRule(
            policy_name="prune_no_growth",
            policy_type=LifecycleEvent.PRUNE,
            trigger_fn=self._trigger_prune_no_growth,
            action_params={},
            priority=9,
            description="Prune kernels with no improvement over extended period",
            cooldown_cycles=100,
        ))
        
        # Policy 3: Split overloaded
        self.policies.append(PolicyRule(
            policy_name="split_overloaded",
            policy_type=LifecycleEvent.SPLIT,
            trigger_fn=self._trigger_split_overloaded,
            action_params={"split_criterion": "domain"},
            priority=7,
            description="Split overloaded kernels with multiple domains",
            cooldown_cycles=50,
        ))
        
        # Policy 4: Merge redundant
        self.policies.append(PolicyRule(
            policy_name="merge_redundant",
            policy_type=LifecycleEvent.MERGE,
            trigger_fn=self._trigger_merge_redundant,
            action_params={},
            priority=6,
            description="Merge kernels with highly similar basins",
            cooldown_cycles=50,
        ))
        
        # Policy 5: Promote stable chaos
        self.policies.append(PolicyRule(
            policy_name="promote_stable_chaos",
            policy_type=LifecycleEvent.PROMOTE,
            trigger_fn=self._trigger_promote_chaos,
            action_params={},
            priority=8,
            description="Promote chaos kernels with stable high Φ",
            cooldown_cycles=10,
        ))
        
        logger.info(f"[PolicyEngine] Initialized {len(self.policies)} default policies")
    
    # =========================================================================
    # TRIGGER FUNCTIONS (Built-in Policies)
    # =========================================================================
    
    def _trigger_prune_low_phi(self, kernel: Kernel, context: Dict[str, Any]) -> bool:
        """Trigger pruning for persistent low Φ."""
        # Skip protected kernels
        if kernel.lifecycle_stage == "protected":
            return False
        
        # Check if Φ has been below threshold for extended period
        phi_threshold = 0.1
        min_cycles = 100
        
        return (
            kernel.phi < phi_threshold
            and kernel.total_cycles >= min_cycles
        )
    
    def _trigger_prune_no_growth(self, kernel: Kernel, context: Dict[str, Any]) -> bool:
        """Trigger pruning for kernels with no growth."""
        # Skip protected kernels
        if kernel.lifecycle_stage == "protected":
            return False
        
        # Check if kernel has shown no improvement
        if kernel.total_cycles < 100:
            return False
        
        # No success in recent cycles
        if kernel.success_count == 0 and kernel.total_cycles > 100:
            return True
        
        # Very low success rate
        if kernel.total_cycles > 0:
            success_rate = kernel.success_count / kernel.total_cycles
            if success_rate < 0.05:  # < 5% success rate
                return True
        
        return False
    
    def _trigger_split_overloaded(self, kernel: Kernel, context: Dict[str, Any]) -> bool:
        """Trigger split for overloaded kernels."""
        # Skip protected kernels
        if kernel.lifecycle_stage == "protected":
            return False
        
        # Must be active and high performing
        if kernel.lifecycle_stage != "active":
            return False
        
        # High Φ (conscious and integrated)
        if kernel.phi < 0.6:
            return False
        
        # Multiple domains (potential for specialization)
        if len(kernel.domains) < 2:
            return False
        
        # High load (many coupled kernels or high activity)
        if len(kernel.coupled_kernels) < 5:
            return False
        
        return True
    
    def _trigger_merge_redundant(self, kernel: Kernel, context: Dict[str, Any]) -> bool:
        """
        Trigger merge for redundant kernels.
        
        Note: This returns True if kernel is a merge candidate.
        Actual merge requires finding a suitable partner kernel.
        """
        # Skip protected kernels
        if kernel.lifecycle_stage == "protected":
            return False
        
        # Must be active
        if kernel.lifecycle_stage != "active":
            return False
        
        # Check if there are nearby kernels (similar basins)
        nearby_kernels = context.get('nearby_kernels', {})
        if kernel.kernel_id not in nearby_kernels:
            return False
        
        nearby = nearby_kernels[kernel.kernel_id]
        if not nearby:
            return False
        
        # Check for highly similar kernel
        for other_id, distance in nearby.items():
            if distance < 0.2:  # Very close in Fisher-Rao distance
                other = self.lifecycle_manager.get_kernel(other_id)
                if other and other.lifecycle_stage == "active":
                    # Check domain overlap
                    overlap = set(kernel.domains) & set(other.domains)
                    if len(overlap) >= len(kernel.domains) * 0.8:
                        return True
        
        return False
    
    def _trigger_promote_chaos(self, kernel: Kernel, context: Dict[str, Any]) -> bool:
        """Trigger promotion for successful chaos kernels."""
        # Must be chaos kernel
        if kernel.kernel_type != "chaos":
            return False
        
        # Skip protected kernels (still learning)
        if kernel.lifecycle_stage == "protected":
            return False
        
        # High stable Φ
        if kernel.phi < 0.4:
            return False
        
        # Sufficient cycles
        if kernel.total_cycles < 50:
            return False
        
        # Good performance
        if kernel.total_cycles > 0:
            success_rate = kernel.success_count / kernel.total_cycles
            if success_rate < 0.7:  # 70% success rate
                return False
        
        # Clear domain specialization
        if not kernel.domains:
            return False
        
        return True
    
    # =========================================================================
    # POLICY EVALUATION
    # =========================================================================
    
    def evaluate_kernel(
        self,
        kernel: Kernel,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a single kernel against all policies.
        
        Args:
            kernel: Kernel to evaluate
            context: Additional context for trigger evaluation
            
        Returns:
            List of triggered actions
        """
        context = context or {}
        triggered_actions = []
        
        # Sort policies by priority (highest first)
        sorted_policies = sorted(self.policies, key=lambda p: p.priority, reverse=True)
        
        for policy in sorted_policies:
            if not policy.can_trigger(kernel):
                continue
            
            try:
                if policy.trigger_fn(kernel, context):
                    triggered_actions.append({
                        'policy_name': policy.policy_name,
                        'policy_type': policy.policy_type,
                        'kernel_id': kernel.kernel_id,
                        'kernel_name': kernel.name,
                        'action_params': policy.action_params,
                        'reason': policy.description,
                    })
                    
                    # Mark as triggered
                    policy.mark_triggered(kernel)
                    
                    logger.info(
                        f"[PolicyEngine] Policy triggered: {policy.policy_name} "
                        f"for kernel {kernel.name} (id={kernel.kernel_id})"
                    )
            
            except Exception as e:
                logger.error(
                    f"[PolicyEngine] Error evaluating policy {policy.policy_name} "
                    f"for kernel {kernel.kernel_id}: {e}"
                )
        
        return triggered_actions
    
    def evaluate_all_kernels(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all active kernels against all policies.
        
        Args:
            context: Additional context for trigger evaluation
            
        Returns:
            List of all triggered actions
        """
        all_actions = []
        
        # Get all active kernels
        active_kernels = self.lifecycle_manager.list_active_kernels()
        
        # Build context with nearby kernel information for merge policy
        if context is None:
            context = {}
        
        if 'nearby_kernels' not in context:
            context['nearby_kernels'] = self._compute_nearby_kernels(active_kernels)
        
        # Evaluate each kernel
        for kernel in active_kernels:
            actions = self.evaluate_kernel(kernel, context)
            all_actions.extend(actions)
        
        logger.info(
            f"[PolicyEngine] Evaluated {len(active_kernels)} kernels, "
            f"triggered {len(all_actions)} actions"
        )
        
        return all_actions
    
    def _compute_nearby_kernels(
        self,
        kernels: List[Kernel],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute nearby kernels for each kernel (for merge detection).
        
        Returns:
            Dict mapping kernel_id -> {other_kernel_id: fisher_distance}
        """
        nearby = {}
        
        for i, kernel in enumerate(kernels):
            nearby[kernel.kernel_id] = {}
            
            for j, other in enumerate(kernels):
                if i == j:
                    continue
                
                distance = compute_fisher_distance(
                    kernel.basin_coords,
                    other.basin_coords,
                )
                
                # Only track nearby kernels (distance < 0.5)
                if distance < 0.5:
                    nearby[kernel.kernel_id][other.kernel_id] = distance
        
        return nearby
    
    # =========================================================================
    # ACTION EXECUTION
    # =========================================================================
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a triggered lifecycle action.
        
        Args:
            action: Action dictionary from evaluate_kernel()
            
        Returns:
            True if action executed successfully
        """
        policy_type = action['policy_type']
        kernel_id = action['kernel_id']
        
        try:
            kernel = self.lifecycle_manager.get_kernel(kernel_id)
            if not kernel:
                logger.warning(
                    f"[PolicyEngine] Cannot execute action: kernel {kernel_id} not found"
                )
                return False
            
            if policy_type == LifecycleEvent.PRUNE:
                reason = action['reason']
                shadow = self.lifecycle_manager.prune(kernel, reason)
                
                logger.info(
                    f"[PolicyEngine] Executed prune: {kernel.name} -> shadow {shadow.shadow_id}"
                )
                
                self.execution_history.append({
                    'action': 'prune',
                    'kernel_id': kernel_id,
                    'kernel_name': kernel.name,
                    'shadow_id': shadow.shadow_id,
                    'reason': reason,
                    'timestamp': datetime.now(),
                })
                
                return True
            
            elif policy_type == LifecycleEvent.SPLIT:
                split_criterion = action['action_params'].get('split_criterion', 'domain')
                k1, k2 = self.lifecycle_manager.split(kernel, split_criterion)
                
                logger.info(
                    f"[PolicyEngine] Executed split: {kernel.name} -> "
                    f"{k1.name} + {k2.name}"
                )
                
                self.execution_history.append({
                    'action': 'split',
                    'kernel_id': kernel_id,
                    'kernel_name': kernel.name,
                    'child1_id': k1.kernel_id,
                    'child2_id': k2.kernel_id,
                    'split_criterion': split_criterion,
                    'timestamp': datetime.now(),
                })
                
                return True
            
            elif policy_type == LifecycleEvent.MERGE:
                # Find merge partner (need to recompute nearby kernels)
                active_kernels = self.lifecycle_manager.list_active_kernels()
                nearby = self._compute_nearby_kernels(active_kernels)
                
                if not nearby or kernel_id not in nearby:
                    logger.warning(
                        f"[PolicyEngine] Cannot execute merge: no nearby kernels for {kernel_id}"
                    )
                    return False
                
                # Get closest neighbor
                partners = nearby[kernel_id]
                if not partners:
                    return False
                
                closest_id = min(partners.items(), key=lambda x: x[1])[0]
                partner = self.lifecycle_manager.get_kernel(closest_id)
                
                if not partner:
                    logger.warning(
                        f"[PolicyEngine] Cannot execute merge: partner {closest_id} not found"
                    )
                    return False
                
                merged = self.lifecycle_manager.merge(
                    kernel,
                    partner,
                    merge_reason=action['reason'],
                )
                
                logger.info(
                    f"[PolicyEngine] Executed merge: {kernel.name} + {partner.name} -> "
                    f"{merged.name}"
                )
                
                self.execution_history.append({
                    'action': 'merge',
                    'kernel1_id': kernel_id,
                    'kernel2_id': closest_id,
                    'merged_id': merged.kernel_id,
                    'merged_name': merged.name,
                    'timestamp': datetime.now(),
                })
                
                return True
            
            elif policy_type == LifecycleEvent.PROMOTE:
                # Generate god name based on domain
                god_name = self._suggest_god_name(kernel)
                
                god_kernel = self.lifecycle_manager.promote(kernel, god_name)
                
                logger.info(
                    f"[PolicyEngine] Executed promote: {kernel.name} -> {god_name}"
                )
                
                self.execution_history.append({
                    'action': 'promote',
                    'chaos_id': kernel_id,
                    'chaos_name': kernel.name,
                    'god_id': god_kernel.kernel_id,
                    'god_name': god_name,
                    'timestamp': datetime.now(),
                })
                
                return True
            
            else:
                logger.warning(
                    f"[PolicyEngine] Unknown policy type: {policy_type}"
                )
                return False
        
        except Exception as e:
            logger.error(
                f"[PolicyEngine] Error executing action {policy_type} "
                f"for kernel {kernel_id}: {e}"
            )
            return False
    
    def _suggest_god_name(self, chaos_kernel: Kernel) -> str:
        """
        Suggest appropriate god name for chaos kernel promotion.
        
        Based on domain and capabilities.
        """
        # Map domains to potential god names
        domain_god_map = {
            'synthesis': 'Apollo',
            'foresight': 'Apollo',
            'prophecy': 'Apollo',
            'wisdom': 'Athena',
            'strategy': 'Athena',
            'war': 'Ares',
            'conflict': 'Ares',
            'communication': 'Hermes',
            'trade': 'Hermes',
            'hunt': 'Artemis',
            'archery': 'Artemis',
            'craft': 'Hephaestus',
            'forge': 'Hephaestus',
            'love': 'Aphrodite',
            'beauty': 'Aphrodite',
            'agriculture': 'Demeter',
            'harvest': 'Demeter',
            'sea': 'Poseidon',
            'depth': 'Poseidon',
            'underworld': 'Hades',
            'death': 'Hades',
        }
        
        # Check primary domain
        if chaos_kernel.domains:
            primary_domain = chaos_kernel.domains[0].lower()
            if primary_domain in domain_god_map:
                return domain_god_map[primary_domain]
        
        # Default to Prometheus (bringer of innovation)
        return 'Prometheus'
    
    # =========================================================================
    # POLICY MANAGEMENT
    # =========================================================================
    
    def add_policy(self, policy: PolicyRule):
        """Add a new policy rule."""
        self.policies.append(policy)
        logger.info(f"[PolicyEngine] Added policy: {policy.policy_name}")
    
    def remove_policy(self, policy_name: str) -> bool:
        """Remove a policy rule by name."""
        initial_count = len(self.policies)
        self.policies = [p for p in self.policies if p.policy_name != policy_name]
        removed = len(self.policies) < initial_count
        
        if removed:
            logger.info(f"[PolicyEngine] Removed policy: {policy_name}")
        
        return removed
    
    def enable_policy(self, policy_name: str) -> bool:
        """Enable a policy rule."""
        for policy in self.policies:
            if policy.policy_name == policy_name:
                policy.enabled = True
                logger.info(f"[PolicyEngine] Enabled policy: {policy_name}")
                return True
        return False
    
    def disable_policy(self, policy_name: str) -> bool:
        """Disable a policy rule."""
        for policy in self.policies:
            if policy.policy_name == policy_name:
                policy.enabled = False
                logger.info(f"[PolicyEngine] Disabled policy: {policy_name}")
                return True
        return False
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        return {
            'total_policies': len(self.policies),
            'enabled_policies': sum(1 for p in self.policies if p.enabled),
            'execution_history_count': len(self.execution_history),
            'policies_by_type': {
                event_type.value: sum(
                    1 for p in self.policies
                    if p.policy_type == event_type and p.enabled
                )
                for event_type in LifecycleEvent
            },
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_policy_engine: Optional[LifecyclePolicyEngine] = None


def get_policy_engine() -> LifecyclePolicyEngine:
    """Get or create global policy engine (singleton)."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = LifecyclePolicyEngine()
    return _policy_engine
