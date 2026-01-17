"""
Kernel Spawner - Contract-Based God Selection
==============================================

Spawns kernels based on Pantheon Registry contracts, not ad-hoc logic.
Enforces spawn constraints, uses epithets for aspects, and manages
chaos kernel lifecycle.

Authority: E8 Protocol v4.0, WP5.1
Status: ACTIVE
Created: 2026-01-17
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from pantheon_registry import (
    PantheonRegistry,
    GodContract,
    get_registry,
    ChaosLifecycleStage,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

@dataclass
class RoleSpec:
    """Specification for a required role/capability."""
    domains: List[str]
    required_capabilities: List[str]
    preferred_god: Optional[str] = None
    allow_chaos_spawn: bool = True
    urgency: str = "normal"  # "low", "normal", "high", "critical"


@dataclass
class KernelSelection:
    """Result of kernel selection process."""
    selected_type: str  # "god" or "chaos"
    god_name: Optional[str] = None
    epithet: Optional[str] = None
    chaos_name: Optional[str] = None
    rationale: str = ""
    spawn_approved: bool = False
    requires_pantheon_vote: bool = False


# =============================================================================
# KERNEL SPAWNER
# =============================================================================

class KernelSpawner:
    """
    Spawns kernels based on Pantheon Registry contracts.
    
    Key Features:
    - Contract-based god selection (NO apollo_1, apollo_2)
    - Uses epithets for god aspects (Apollo Pythios, Apollo Paean)
    - Enforces spawn constraints from registry
    - Manages chaos kernel lifecycle
    - Integrates with pantheon governance for voting
    
    Example:
        spawner = KernelSpawner()
        
        # Try to get a synthesis god
        role = RoleSpec(
            domains=["synthesis", "foresight"],
            required_capabilities=["prediction", "aesthetic_evaluation"]
        )
        selection = spawner.select_god(role)
        
        if selection.selected_type == "god":
            print(f"Selected {selection.god_name} {selection.epithet}")
        else:
            print(f"Spawning chaos kernel: {selection.chaos_name}")
    """
    
    def __init__(
        self,
        registry: Optional[PantheonRegistry] = None,
        active_instances: Optional[Dict[str, int]] = None,
        chaos_counter: Optional[Dict[str, int]] = None,
        active_chaos_count: Optional[int] = None,
    ):
        """
        Initialize kernel spawner.
        
        Args:
            registry: Pantheon registry (default: global singleton)
            active_instances: Current active instance counts by god name
            chaos_counter: Sequential ID counters by chaos domain (for naming)
            active_chaos_count: Actual count of active chaos kernels (for limits)
        """
        self.registry = registry or get_registry()
        self.active_instances = active_instances or {}
        self.chaos_counter = chaos_counter or {}
        self.active_chaos_count = active_chaos_count or 0
        
    # =========================================================================
    # GOD SELECTION
    # =========================================================================
    
    def select_god(self, role: RoleSpec) -> KernelSelection:
        """
        Select a god or chaos kernel based on role specification.
        
        Selection logic:
        1. If preferred_god specified and available, use it
        2. Find gods matching domain requirements
        3. Check spawn constraints (gods are singular)
        4. Select best matching god with epithet
        5. If no god available and chaos allowed, spawn chaos kernel
        6. If chaos not allowed, return empty selection
        
        Args:
            role: Role specification with domain and capability requirements
            
        Returns:
            KernelSelection with selected god/chaos and rationale
        """
        # Step 1: Check preferred god
        if role.preferred_god:
            god = self.registry.get_god(role.preferred_god)
            if god and self._can_spawn_god(god):
                epithet = self._select_epithet(god, role)
                return KernelSelection(
                    selected_type="god",
                    god_name=role.preferred_god,
                    epithet=epithet,
                    rationale=f"Preferred god {role.preferred_god} available",
                    spawn_approved=True,
                )
        
        # Step 2: Find matching gods by domain
        candidate_gods = self._find_matching_gods(role)
        
        # Step 3: Filter by spawn constraints
        available_gods = [
            god for god in candidate_gods
            if self._can_spawn_god(god)
        ]
        
        # Step 4: Select best match
        if available_gods:
            best_god = self._select_best_god(available_gods, role)
            epithet = self._select_epithet(best_god, role)
            return KernelSelection(
                selected_type="god",
                god_name=best_god.name,
                epithet=epithet,
                rationale=self._explain_god_selection(best_god, role),
                spawn_approved=True,
            )
        
        # Step 5: Spawn chaos kernel if allowed
        if role.allow_chaos_spawn:
            chaos_name = self._generate_chaos_kernel_name(role)
            return KernelSelection(
                selected_type="chaos",
                chaos_name=chaos_name,
                rationale=self._explain_chaos_spawn(role),
                spawn_approved=False,  # Needs pantheon vote
                requires_pantheon_vote=True,
            )
        
        # Step 6: No selection possible
        return KernelSelection(
            selected_type="none",
            rationale="No available gods and chaos spawn not allowed",
            spawn_approved=False,
        )
    
    def _find_matching_gods(self, role: RoleSpec) -> List[GodContract]:
        """Find gods matching role domains."""
        candidates = []
        for domain in role.domains:
            gods = self.registry.find_gods_by_domain(domain)
            candidates.extend(gods)
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for god in candidates:
            if god.name not in seen:
                seen.add(god.name)
                unique_candidates.append(god)
        
        return unique_candidates
    
    def _can_spawn_god(self, god: GodContract) -> bool:
        """Check if god can be spawned based on constraints."""
        # Gods are singular - max_instances should always be 1
        if god.spawn_constraints.max_instances != 1:
            logger.warning(
                f"God {god.name} has invalid max_instances: "
                f"{god.spawn_constraints.max_instances} (should be 1)"
            )
            return False
        
        # Check if already active
        current_instances = self.active_instances.get(god.name, 0)
        if current_instances >= god.spawn_constraints.max_instances:
            return False
        
        # Check when_allowed constraint
        if god.spawn_constraints.when_allowed == "never":
            return current_instances == 0  # Only if not already spawned
        
        return True
    
    def _select_best_god(
        self, 
        candidates: List[GodContract], 
        role: RoleSpec
    ) -> GodContract:
        """
        Select best god from candidates based on role requirements.
        
        Scoring criteria:
        - Domain overlap
        - E8 layer alignment (prefer specialized tier for complex tasks)
        - Rest policy (prefer more available gods for critical roles)
        """
        def score_god(god: GodContract) -> float:
            score = 0.0
            
            # Domain overlap (primary factor)
            overlap = len(set(god.domain) & set(role.domains))
            score += overlap * 10.0
            
            # E8 layer bonus (prefer core faculties for fundamental roles)
            if god.e8_alignment.layer == "8":
                score += 5.0
            elif god.e8_alignment.layer == "0/1":
                score += 3.0
            
            # Availability bonus (higher duty cycle = more available)
            if god.rest_policy.duty_cycle:
                score += god.rest_policy.duty_cycle * 2.0
            
            # Essential tier bonus for critical urgency
            if role.urgency == "critical" and god.tier.value == "essential":
                score += 10.0
            
            return score
        
        candidates.sort(key=score_god, reverse=True)
        return candidates[0]
    
    def _select_epithet(
        self, 
        god: GodContract, 
        role: RoleSpec
    ) -> Optional[str]:
        """
        Select appropriate epithet for god based on role.
        
        Epithets define aspects of a god, NOT numbering.
        Example: Apollo Pythios (prophecy), Apollo Paean (healing)
        """
        if not god.epithets:
            return None
        
        # Try to match epithet to role domains
        # For now, use first epithet (can be enhanced with domain mapping)
        # TODO: Build epithet-to-domain mapping in registry
        return god.epithets[0] if god.epithets else None
    
    def _explain_god_selection(
        self, 
        god: GodContract, 
        role: RoleSpec
    ) -> str:
        """Generate explanation for god selection."""
        domain_overlap = set(god.domain) & set(role.domains)
        return (
            f"Selected {god.name} for domains: {', '.join(domain_overlap)}. "
            f"Tier: {god.tier.value}, Layer: {god.e8_alignment.layer}"
        )
    
    # =========================================================================
    # CHAOS KERNEL SPAWNING
    # =========================================================================
    
    def _generate_chaos_kernel_name(self, role: RoleSpec) -> str:
        """
        Generate chaos kernel name following naming pattern.
        
        Pattern: chaos_{domain}_{id}
        Example: chaos_synthesis_001
        """
        # Use first domain as primary
        domain = role.domains[0] if role.domains else "general"
        
        # Get or initialize counter for this domain
        if domain not in self.chaos_counter:
            self.chaos_counter[domain] = 0
        
        # Increment and format
        self.chaos_counter[domain] += 1
        chaos_id = self.chaos_counter[domain]
        
        return f"chaos_{domain}_{chaos_id:03d}"
    
    def _explain_chaos_spawn(self, role: RoleSpec) -> str:
        """Generate explanation for chaos kernel spawn."""
        return (
            f"No available gods for domains: {', '.join(role.domains)}. "
            f"Spawning chaos kernel to fill capability gap. "
            f"Requires pantheon vote approval."
        )
    
    # =========================================================================
    # INSTANCE TRACKING
    # =========================================================================
    
    def register_spawn(self, name: str) -> None:
        """Register that a kernel has been spawned."""
        if self.registry.is_god_name(name):
            self.active_instances[name] = self.active_instances.get(name, 0) + 1
    
    def register_death(self, name: str) -> None:
        """Register that a kernel has died/been pruned."""
        if self.registry.is_god_name(name):
            if name in self.active_instances:
                self.active_instances[name] = max(0, self.active_instances[name] - 1)
    
    def get_active_count(self, name: str) -> int:
        """Get current active instance count for a god."""
        return self.active_instances.get(name, 0)
    
    def get_total_chaos_count(self) -> int:
        """Get total number of chaos kernels spawned."""
        return sum(self.chaos_counter.values())
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate_spawn_request(self, name: str) -> Tuple[bool, str]:
        """
        Validate a spawn request against registry constraints.
        
        Returns:
            Tuple of (valid, reason)
        """
        # Check if god
        if self.registry.is_god_name(name):
            god = self.registry.get_god(name)
            if not god:
                return (False, f"God {name} not found in registry")
            
            if not self._can_spawn_god(god):
                return (
                    False,
                    f"God {name} spawn constraints violated "
                    f"(max_instances: {god.spawn_constraints.max_instances}, "
                    f"active: {self.get_active_count(name)})"
                )
            
            return (True, f"God {name} spawn approved")
        
        # Check if chaos kernel
        elif self.registry.is_valid_chaos_kernel_name(name):
            # Validate against chaos kernel rules
            rules = self.registry.get_chaos_kernel_rules()
            total_limit = rules.spawning_limits['total_active_limit']
            
            # Use actual active count, not generation counter
            current_total = self.active_chaos_count
            if current_total >= total_limit:
                return (
                    False,
                    f"Chaos kernel limit reached ({current_total}/{total_limit})"
                )
            
            # Check domain limit
            parsed = self.registry.parse_chaos_kernel_name(name)
            if parsed:
                domain, _ = parsed
                per_domain_limit = rules.spawning_limits['per_domain_limit']
                domain_count = self.chaos_counter.get(domain, 0)
                
                if domain_count >= per_domain_limit:
                    return (
                        False,
                        f"Chaos kernel domain limit reached for {domain} "
                        f"({domain_count}/{per_domain_limit})"
                    )
            
            return (True, f"Chaos kernel {name} spawn approved")
        
        else:
            return (
                False,
                f"Invalid kernel name: {name} (not god or chaos kernel)"
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select_kernel_for_role(role: RoleSpec) -> KernelSelection:
    """
    Convenience function to select kernel for role.
    
    Uses global spawner instance.
    """
    spawner = KernelSpawner()
    return spawner.select_god(role)


def create_role_spec(
    domains: List[str],
    capabilities: List[str],
    preferred_god: Optional[str] = None,
) -> RoleSpec:
    """Convenience function to create role spec."""
    return RoleSpec(
        domains=domains,
        required_capabilities=capabilities,
        preferred_god=preferred_god,
    )
