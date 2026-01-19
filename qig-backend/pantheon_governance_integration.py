"""
Pantheon Governance Integration
================================

Integrates Pantheon Registry with Pantheon Governance for lifecycle control.
Ensures all kernel spawning uses contracts and requires proper approval.

Authority: E8 Protocol v4.0, WP5.1
Status: ACTIVE
Created: 2026-01-17
"""

import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from pantheon_registry import get_registry, GodContract
from kernel_spawner import KernelSpawner, RoleSpec, KernelSelection

logger = logging.getLogger(__name__)

# Import governance if available
try:
    from olympus.pantheon_governance import (
        PantheonGovernance,
        ProposalType,
        LifecycleProposal,
    )
    GOVERNANCE_AVAILABLE = True
except ImportError:
    logger.warning("PantheonGovernance not available")
    GOVERNANCE_AVAILABLE = False
    PantheonGovernance = None
    ProposalType = None
    LifecycleProposal = None


# =============================================================================
# INTEGRATION LAYER
# =============================================================================

class PantheonGovernanceIntegration:
    """
    Integration layer between Registry and Governance.
    
    Ensures all kernel spawning:
    1. Uses registry contracts (NO apollo_1 naming)
    2. Enforces spawn constraints
    3. Requires pantheon vote for chaos kernels
    4. Tracks active instances
    5. Uses epithets for god aspects
    
    Example:
        integration = PantheonGovernanceIntegration()
        
        # Request kernel for role
        role = RoleSpec(domains=["synthesis"], required_capabilities=["prediction"])
        result = integration.request_kernel_for_role(role, requester="Zeus")
        
        if result.approved:
            print(f"Spawn approved: {result.kernel_name}")
        else:
            print(f"Spawn requires vote: {result.proposal_id}")
    """
    
    def __init__(
        self,
        governance: Optional[PantheonGovernance] = None,
        spawner: Optional[KernelSpawner] = None,
    ):
        """
        Initialize integration layer.
        
        Args:
            governance: PantheonGovernance instance (optional)
            spawner: KernelSpawner instance (optional)
        """
        self.governance = governance
        self.spawner = spawner or KernelSpawner()
        self.registry = self.spawner.registry
    
    @dataclass
    class SpawnResult:
        """Result of spawn request."""
        approved: bool
        kernel_name: Optional[str]
        epithet: Optional[str]
        proposal_id: Optional[str]
        requires_vote: bool
        rationale: str
    
    def request_kernel_for_role(
        self,
        role: RoleSpec,
        requester: str = "System",
    ) -> "PantheonGovernanceIntegration.SpawnResult":
        """
        Request kernel spawn for role using registry contracts.
        
        Process:
        1. Use KernelSpawner to select god or chaos kernel
        2. If god selected and available, auto-approve
        3. If chaos kernel needed, create governance proposal
        4. Return spawn result with approval status
        
        Args:
            role: Role specification
            requester: Who is requesting the kernel
            
        Returns:
            SpawnResult with approval status and kernel info
        """
        # Step 1: Use spawner to select kernel
        selection = self.spawner.select_god(role)
        
        # Step 2: Handle god selection
        if selection.selected_type == "god":
            god_name = selection.god_name
            
            # Validate spawn constraints
            valid, reason = self.spawner.validate_spawn_request(god_name)
            if not valid:
                return self.SpawnResult(
                    approved=False,
                    kernel_name=None,
                    epithet=None,
                    proposal_id=None,
                    requires_vote=False,
                    rationale=f"Spawn validation failed: {reason}",
                )
            
            # Auto-approve god spawn
            self.spawner.register_spawn(god_name)
            
            return self.SpawnResult(
                approved=True,
                kernel_name=god_name,
                epithet=selection.epithet,
                proposal_id=None,
                requires_vote=False,
                rationale=selection.rationale,
            )
        
        # Step 3: Handle chaos kernel spawn
        elif selection.selected_type == "chaos":
            chaos_name = selection.chaos_name
            
            # Validate spawn constraints
            valid, reason = self.spawner.validate_spawn_request(chaos_name)
            if not valid:
                return self.SpawnResult(
                    approved=False,
                    kernel_name=None,
                    epithet=None,
                    proposal_id=None,
                    requires_vote=False,
                    rationale=f"Spawn validation failed: {reason}",
                )
            
            # Create governance proposal if governance available
            if self.governance and GOVERNANCE_AVAILABLE:
                proposal_id = self._create_chaos_spawn_proposal(
                    chaos_name=chaos_name,
                    role=role,
                    requester=requester,
                )
                
                return self.SpawnResult(
                    approved=False,
                    kernel_name=chaos_name,
                    epithet=None,
                    proposal_id=proposal_id,
                    requires_vote=True,
                    rationale=selection.rationale,
                )
            else:
                # No governance, auto-approve (dev mode)
                logger.warning(
                    f"Chaos kernel {chaos_name} auto-approved (no governance)"
                )
                return self.SpawnResult(
                    approved=True,
                    kernel_name=chaos_name,
                    epithet=None,
                    proposal_id=None,
                    requires_vote=False,
                    rationale=selection.rationale,
                )
        
        # Step 4: No selection possible
        else:
            return self.SpawnResult(
                approved=False,
                kernel_name=None,
                epithet=None,
                proposal_id=None,
                requires_vote=False,
                rationale=selection.rationale,
            )
    
    def _create_chaos_spawn_proposal(
        self,
        chaos_name: str,
        role: RoleSpec,
        requester: str,
    ) -> str:
        """
        Create governance proposal for chaos kernel spawn.
        
        Args:
            chaos_name: Name of chaos kernel (chaos_{domain}_{id})
            role: Role specification
            requester: Who requested the spawn
            
        Returns:
            Proposal ID
        """
        if not self.governance or not GOVERNANCE_AVAILABLE:
            raise RuntimeError("Governance not available")
        
        reason = (
            f"Spawn chaos kernel {chaos_name} for domains: {', '.join(role.domains)}. "
            f"Requested by: {requester}. "
            f"Required capabilities: {', '.join(role.required_capabilities)}"
        )
        
        proposal = self.governance.propose_action(
            proposal_type=ProposalType.CHAOS_SPAWN,
            reason=reason,
            count=1,
            parent_id=requester,
            parent_phi=0.5,  # Default phi for spawn request
        )
        
        logger.info(
            f"Created chaos spawn proposal {proposal.proposal_id} for {chaos_name}"
        )
        
        return proposal.proposal_id
    
    def get_god_with_epithet(
        self,
        god_name: str,
        domain: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Get god name with appropriate epithet for domain.
        
        Args:
            god_name: Base god name (e.g., "Apollo")
            domain: Optional domain to select epithet for
            
        Returns:
            Tuple of (god_name, epithet)
            
        Example:
            name, epithet = integration.get_god_with_epithet("Apollo", "prophecy")
            # Returns: ("Apollo", "Pythios")
        """
        god = self.registry.get_god(god_name)
        if not god:
            return (god_name, None)
        
        if not god.epithets:
            return (god_name, None)
        
        # TODO: Build epithet-to-domain mapping in registry
        # For now, return first epithet
        return (god_name, god.epithets[0])
    
    def validate_kernel_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate kernel name against registry rules.
        
        Returns:
            Tuple of (valid, reason)
        """
        # Check if god
        if self.registry.is_god_name(name):
            return (True, f"Valid god name: {name}")
        
        # Check if chaos kernel
        elif self.registry.is_valid_chaos_kernel_name(name):
            return (True, f"Valid chaos kernel name: {name}")
        
        # Invalid
        else:
            return (
                False,
                f"Invalid kernel name: {name}. "
                "Must be a registered god or follow chaos_{domain}_{id} pattern"
            )
    
    def get_kernel_info(self, name: str) -> Dict:
        """
        Get information about a kernel.
        
        Args:
            name: Kernel name
            
        Returns:
            Dict with kernel information
        """
        # Check if god
        if self.registry.is_god_name(name):
            god = self.registry.get_god(name)
            return {
                "type": "god",
                "name": name,
                "tier": god.tier.value,
                "domain": god.domain,
                "epithets": god.epithets,
                "layer": god.e8_alignment.layer,
                "simple_root": god.e8_alignment.simple_root,
                "max_instances": god.spawn_constraints.max_instances,
            }
        
        # Check if chaos kernel
        elif self.registry.is_valid_chaos_kernel_name(name):
            parsed = self.registry.parse_chaos_kernel_name(name)
            if parsed:
                domain, chaos_id = parsed
                return {
                    "type": "chaos",
                    "name": name,
                    "domain": domain,
                    "id": chaos_id,
                    "lifecycle": "unknown",  # Would need to query from DB
                }
        
        return {
            "type": "unknown",
            "name": name,
            "error": "Invalid kernel name",
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def request_kernel(
    domains: list,
    capabilities: list,
    preferred_god: Optional[str] = None,
    requester: str = "System",
) -> PantheonGovernanceIntegration.SpawnResult:
    """
    Convenience function to request kernel spawn.
    
    Example:
        result = request_kernel(
            domains=["synthesis", "foresight"],
            capabilities=["prediction"],
            preferred_god="Apollo",
            requester="Zeus"
        )
    """
    integration = PantheonGovernanceIntegration()
    role = RoleSpec(
        domains=domains,
        required_capabilities=capabilities,
        preferred_god=preferred_god,
    )
    return integration.request_kernel_for_role(role, requester)


def validate_kernel_name(name: str) -> Tuple[bool, str]:
    """
    Convenience function to validate kernel name.
    
    Example:
        valid, reason = validate_kernel_name("Apollo")
        valid, reason = validate_kernel_name("chaos_synthesis_001")
        valid, reason = validate_kernel_name("apollo_1")  # Should be invalid
    """
    integration = PantheonGovernanceIntegration()
    return integration.validate_kernel_name(name)


def get_god_with_epithet(god_name: str, domain: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Convenience function to get god with epithet.
    
    Example:
        name, epithet = get_god_with_epithet("Apollo", "prophecy")
        # Returns: ("Apollo", "Pythios")
    """
    integration = PantheonGovernanceIntegration()
    return integration.get_god_with_epithet(god_name, domain)
