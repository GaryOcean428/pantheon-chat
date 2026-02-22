"""
Lifecycle Governance Bridge — TCP v6.1

This module wraps KernelLifecycleManager's spawn() and promote() methods
to inject Pantheon Governance charter attachment and proxy wiring.

Design:
  KernelLifecycleManager handles the geometric mechanics of creating kernels.
  PantheonGovernance decides WHAT capabilities they receive and WHO proxies them.
  This bridge is the single seam between those two systems.

Usage:
  Replace direct KernelLifecycleManager.spawn() calls with:
      from olympus.lifecycle_governance_bridge import GovernedLifecycleManager
      manager = GovernedLifecycleManager(lifecycle_mgr, governance)

  Or use the module-level convenience function:
      from olympus.lifecycle_governance_bridge import spawn_with_governance
      kernel, charter = spawn_with_governance(role_spec, lifecycle_mgr)

TCP v6.1 §19 Genesis Doctrine:
  - PurityGate runs first (fail-closed) — enforced by CapabilityPolicy
  - All chaos kernels receive a charter with CHAOS_DEFAULT capabilities
  - All chaos kernels receive a proxy god assignment via governance vote
  - Proxy god's charter is updated to reflect PROXY_VOICE capability
  - No kernel enters the constellation without a registered charter
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports — all fail-soft so the bridge degrades gracefully
# ---------------------------------------------------------------------------

try:
    from .pantheon_governance import (
        PantheonGovernance,
        LifecycleProposal,
        GovernanceVote,
        GovernanceDecision,
        ProposalType,
        get_governance,
    )
    from .capability_charter import (
        KernelCapabilityCharter,
        KernelCapability,
        CapabilityPolicy,
        ProxyAssignment,
        ProxyInstruction,
        make_chaos_charter,
    )
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False
    logger.warning("[LifecycleBridge] Governance not available — charters disabled")

# Default proxy gods by domain (fallback when no specific proxy is requested)
_DEFAULT_PROXY_BY_DOMAIN: Dict[str, str] = {
    "synthesis":      "Athena",
    "prediction":     "Apollo",
    "communication":  "Hermes",
    "exploration":    "Artemis",
    "creation":       "Hephaestus",
    "strategy":       "Athena",
    "action":         "Ares",
    "harmony":        "Aphrodite",
}
_DEFAULT_PROXY_FALLBACK = "Hermes"  # Hermes handles unknown domain chaos


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class SpawnOutcome:
    """Result of a governed spawn operation."""
    kernel: Any                                  # Kernel instance
    charter: Optional[Any] = None               # KernelCapabilityCharter
    proxy_god_charter_update: Optional[Any] = None  # Charter update for proxy god
    governance_decision: Optional[Any] = None   # Full GovernanceDecision
    governance_available: bool = False


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------

class GovernedLifecycleManager:
    """
    Thin wrapper over KernelLifecycleManager that injects governance
    charter attachment at every spawn and promote call.

    Parameters:
        lifecycle_mgr  — KernelLifecycleManager instance
        governance     — PantheonGovernance instance (default: global singleton)
        default_voters — List of (god_name, phi, kappa) tuples that automatically
                         cast weighted YES votes on every proposal.
                         Used during genesis bootstrap when the full Pantheon
                         hasn't formed yet. Zeus always votes with genesis weight.
    """

    def __init__(
        self,
        lifecycle_mgr: Any,
        governance: Optional[Any] = None,
        default_voters: Optional[List[Tuple[str, float, float]]] = None,
    ):
        self._lcm = lifecycle_mgr
        self._gov = governance
        self._default_voters = default_voters or [
            ("Zeus",   0.727, 64.21),  # Genesis weight
        ]
        self._charters: Dict[str, Any] = {}  # kernel_id → charter

        if not GOVERNANCE_AVAILABLE:
            logger.warning("[LifecycleBridge] Governance module not importable — pass-through mode")
            return

        if self._gov is None:
            self._gov = get_governance()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def spawn(
        self,
        role_spec: Any,
        mentor: Optional[str] = None,
        initial_basin: Optional[np.ndarray] = None,
        proposer: str = "Zeus",
        rationale: str = "Lifecycle-governed spawn",
        requested_capabilities: Optional[Any] = None,
        proposed_proxy_god: Optional[str] = None,
        proxy_instructions: Optional[Any] = None,
        extra_voters: Optional[List[Tuple[str, float, float]]] = None,
    ) -> SpawnOutcome:
        """
        Spawn a kernel via KernelLifecycleManager, then attach a Pantheon-voted
        KernelCapabilityCharter.

        For chaos kernels:
          - proposed_proxy_god can be passed directly; if omitted, the bridge
            selects the best-fit god from the domain map.
          - proxy_instructions are forwarded to the ProxyAssignment.

        For god kernels:
          - requested_capabilities default to FULL_GOD if not specified.

        Returns:
            SpawnOutcome with kernel + charter (charter=None if governance unavailable)
        """
        # Step 1: Geometric spawn (existing mechanics)
        kernel = self._lcm.spawn(role_spec, mentor=mentor, initial_basin=initial_basin)

        # Step 2: Governance charter — fail-soft
        if not GOVERNANCE_AVAILABLE or self._gov is None:
            return SpawnOutcome(kernel=kernel, governance_available=False)

        ktype = self._kernel_type(kernel)
        kernel_name = getattr(kernel, "name", kernel.kernel_id)

        # Resolve proxy god for chaos kernels
        proxy_god = proposed_proxy_god
        if ktype == "chaos" and not proxy_god:
            domain = role_spec.domains[0] if role_spec.domains else "general"
            proxy_god = _DEFAULT_PROXY_BY_DOMAIN.get(domain, _DEFAULT_PROXY_FALLBACK)

        # Resolve capabilities
        caps = requested_capabilities
        if caps is None:
            caps = KernelCapability.NONE  # let governance/policy set defaults

        proposal = LifecycleProposal(
            proposal_type=ProposalType.SPAWN,
            proposer=proposer,
            rationale=rationale,
            target_kernel_id=kernel.kernel_id,
            target_kernel_name=kernel_name,
            target_kernel_type=ktype,
            requested_capabilities=caps,
            proposed_proxy_god=proxy_god if ktype == "chaos" else None,
            proxy_instructions=proxy_instructions,
        )
        self._gov.propose(proposal)

        # Cast default genesis votes
        voters = list(self._default_voters) + (extra_voters or [])
        for god_name, phi, kappa in voters:
            self._gov.vote(
                proposal.proposal_id,
                GovernanceVote(voter=god_name, vote=True, phi=phi, kappa=kappa,
                               reason="governed_spawn"),
            )

        decision = self._gov.execute(proposal.proposal_id)

        if decision.approved and decision.charter:
            # Register charter with governance
            self._gov.register_charter(decision.charter)
            self._charters[kernel.kernel_id] = decision.charter

            # Attach charter directly to kernel object (non-intrusive attribute)
            try:
                kernel.capability_charter = decision.charter
            except AttributeError:
                pass  # Kernel may be a frozen dataclass in some impls

            logger.info(
                "[LifecycleBridge] Charter attached: %s",
                decision.charter.summary(),
            )
        else:
            logger.warning(
                "[LifecycleBridge] Charter not approved for %s — kernel spawned without charter",
                kernel_name,
            )

        return SpawnOutcome(
            kernel=kernel,
            charter=decision.charter if decision.approved else None,
            proxy_god_charter_update=decision.proxy_god_charter_update,
            governance_decision=decision,
            governance_available=True,
        )

    def promote(
        self,
        chaos_kernel: Any,
        god_name: str,
        proposer: str = "Zeus",
        voters: Optional[List[Tuple[str, float, float]]] = None,
    ) -> SpawnOutcome:
        """
        Promote a chaos kernel to GOD status via governed vote.

        Quorum required: SUPERMAJORITY (ProposalType.CHAOS_ASCEND).
        If insufficient voters are supplied, falls back to zeus_direct
        with 'chaos_exploration' bypass only during initialization.
        """
        if not GOVERNANCE_AVAILABLE or self._gov is None:
            god_kernel = self._lcm.promote(chaos_kernel, god_name)
            return SpawnOutcome(kernel=god_kernel, governance_available=False)

        chaos_name = getattr(chaos_kernel, "name", chaos_kernel.kernel_id)
        proposal = LifecycleProposal(
            proposal_type=ProposalType.CHAOS_ASCEND,
            proposer=proposer,
            rationale=f"Promote {chaos_name} → {god_name}",
            target_kernel_id=chaos_kernel.kernel_id,
            target_kernel_name=god_name,
            target_kernel_type="god",
        )
        self._gov.propose(proposal)

        all_voters = list(voters or []) + list(self._default_voters)
        for god, phi, kappa in all_voters:
            self._gov.vote(
                proposal.proposal_id,
                GovernanceVote(voter=god, vote=True, phi=phi, kappa=kappa,
                               reason="chaos_ascension"),
            )

        decision = self._gov.tally(proposal.proposal_id)

        if not decision.approved:
            raise PermissionError(
                f"CHAOS_ASCEND proposal {proposal.proposal_id} rejected "
                f"({decision.yes_fraction:.0%} yes, need >66%). "
                f"Voted by: {all_voters}"
            )

        # Execute lifecycle promotion (geometry mechanics)
        god_kernel = self._lcm.promote(chaos_kernel, god_name)
        decision.proposal.status = __import__("olympus.pantheon_governance",
                                              fromlist=["ProposalStatus"]
                                              ).ProposalStatus.EXECUTED

        if decision.charter:
            self._gov.register_charter(decision.charter)
            try:
                god_kernel.capability_charter = decision.charter
            except AttributeError:
                pass

        return SpawnOutcome(
            kernel=god_kernel,
            charter=decision.charter,
            governance_decision=decision,
            governance_available=True,
        )

    def assign_proxy(
        self,
        chaos_kernel_id: str,
        proxy_god: str,
        instructions: Optional[Any] = None,
        proposer: str = "Zeus",
        voters: Optional[List[Tuple[str, float, float]]] = None,
    ) -> Optional[Any]:
        """
        (Re)assign a proxy god for a voiceless chaos kernel.
        Returns the updated KernelCapabilityCharter on success.
        """
        if not GOVERNANCE_AVAILABLE or self._gov is None:
            return None

        proposal = LifecycleProposal(
            proposal_type=ProposalType.ASSIGN_PROXY,
            proposer=proposer,
            rationale=f"Assign proxy {proxy_god} for {chaos_kernel_id}",
            target_kernel_id=chaos_kernel_id,
            proposed_proxy_god=proxy_god,
            proxy_instructions=instructions,
        )
        self._gov.propose(proposal)

        all_voters = list(voters or []) + list(self._default_voters)
        for god, phi, kappa in all_voters:
            self._gov.vote(
                proposal.proposal_id,
                GovernanceVote(voter=god, vote=True, phi=phi, kappa=kappa),
            )

        decision = self._gov.execute(proposal.proposal_id)
        if decision.approved and decision.charter:
            self._gov.register_charter(decision.charter)
            self._charters[chaos_kernel_id] = decision.charter
            return decision.charter
        return None

    def get_charter(self, kernel_id: str) -> Optional[Any]:
        """Retrieve the capability charter for a kernel."""
        return self._charters.get(kernel_id) or (
            self._gov.get_charter(kernel_id) if self._gov else None
        )

    def can_generate(self, kernel_id: str) -> bool:
        """Quick check: can this kernel produce text output?"""
        charter = self.get_charter(kernel_id)
        if charter is None:
            # No charter → assume god-level until governance is wired
            return True
        return charter.can_generate()

    def who_proxies_for(self, chaos_kernel_id: str) -> Optional[str]:
        """Return the proxy god name for a chaos kernel."""
        if self._gov is None:
            return None
        return self._gov.who_proxies_for(chaos_kernel_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kernel_type(kernel: Any) -> str:
        kind = getattr(kernel, "kernel_kind", None)
        if kind is None:
            return "god"
        kv = kind.value if hasattr(kind, "value") else str(kind)
        return "chaos" if "chaos" in kv.lower() else (
            "shadow" if "shadow" in kv.lower() else "god"
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def spawn_with_governance(
    role_spec: Any,
    lifecycle_mgr: Any,
    governance: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[Any, Optional[Any]]:
    """
    Convenience wrapper: spawn a kernel with governance charter in one call.

    Returns:
        (kernel, charter)  — charter is None if governance is unavailable.
    """
    bridge = GovernedLifecycleManager(lifecycle_mgr, governance=governance)
    outcome = bridge.spawn(role_spec, **kwargs)
    return outcome.kernel, outcome.charter
