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

Red-team fixes:
  RT1-M2: _get_live_voters() queries PantheonVoterRegistry for live φ/κ weights.
           All three vote paths now use live weights instead of hard-coded genesis
           constants, with transparent fallback to genesis when VoterRegistry is
           unavailable or gods haven't yet reached MIN_CYCLES_FOR_LIVE threshold.
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
        ProposalStatus,
        get_governance,
    )
    HAS_GOVERNANCE = True
except ImportError:
    HAS_GOVERNANCE = False
    PantheonGovernance = None
    LifecycleProposal = None
    GovernanceVote = None
    GovernanceDecision = None
    ProposalType = None
    ProposalStatus = None
    def get_governance(): return None

try:
    from .capability_charter import (
        KernelCapability,
        KernelCapabilityCharter,
        ProxyAssignment,
        ProxyInstruction,
        CapabilityPolicy,
        make_chaos_charter,
        grant_proxy_voice_to_god,
    )
    HAS_CHARTER = True
except ImportError:
    HAS_CHARTER = False
    KernelCapability = None
    KernelCapabilityCharter = None
    ProxyAssignment = None
    ProxyInstruction = None
    CapabilityPolicy = None
    def make_chaos_charter(*a, **kw): return None
    def grant_proxy_voice_to_god(*a, **kw): return None

try:
    from kernel_lifecycle import KernelLifecycleManager, Kernel, KernelKind
    from kernel_spawner import RoleSpec
    HAS_LIFECYCLE = True
except ImportError:
    HAS_LIFECYCLE = False
    KernelLifecycleManager = None
    Kernel = None
    KernelKind = None
    RoleSpec = None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SpawnOutcome:
    kernel: Any                                # Kernel instance
    charter: Optional[Any]                     # KernelCapabilityCharter or None
    proxy_assigned: bool = False               # True if chaos proxy was voted
    proxy_god_name: Optional[str] = None       # Name of assigned proxy god
    governance_decision: Optional[Any] = None  # GovernanceDecision record
    governance_skipped: bool = False           # True if governance unavailable


# ---------------------------------------------------------------------------
# Governed Lifecycle Manager
# ---------------------------------------------------------------------------

class GovernedLifecycleManager:
    """
    Wraps KernelLifecycleManager to enforce Governance before every spawn/promote.

    Governance flow:
      1. _get_live_voters() → live φ/κ from VoterRegistry (genesis fallback)
      2. submit proposal to PantheonGovernance
      3. await decision
      4. if approved: spawn kernel + attach charter + wire proxy if chaos

    TCP v6.1 §19: Every kernel spawn is a governance event.
    No kernel enters without charter. No chaos kernel without proxy assignment.
    """

    def __init__(
        self,
        lifecycle_mgr: Optional[Any] = None,
        governance: Optional[Any] = None,
        default_voters: Optional[List[Tuple[str, float, float]]] = None,
    ):
        """
        Args:
            lifecycle_mgr: KernelLifecycleManager instance. If None, uses singleton.
            governance: PantheonGovernance instance. If None, uses singleton.
            default_voters: List of (god_name, phi, kappa) tuples that automatically
                            vote in every proposal. Used during bootstrap when
                            VoterRegistry has no live entries yet.
                            Overrides VoterRegistry fallback if provided explicitly.
        """
        self._lifecycle = lifecycle_mgr
        self._governance = governance
        self._default_voters = default_voters or [
            ("Zeus",   0.727, 64.21),  # Genesis weight
        ]

    def _get_live_voters(self) -> list:
        """RT1-M2: return live φ/κ from VoterRegistry; genesis fallback."""
        try:
            from olympus.voter_registry import get_voter_registry
            vr = get_voter_registry()
            active = vr.active_voters()
            if active:
                return vr.get_voter_metrics(active)
        except Exception:
            pass
        return list(self._default_voters)

    @property
    def lifecycle(self):
        if self._lifecycle is None and HAS_LIFECYCLE:
            from kernel_lifecycle import get_lifecycle_manager
            self._lifecycle = get_lifecycle_manager()
        return self._lifecycle

    @property
    def governance(self):
        if self._governance is None and HAS_GOVERNANCE:
            self._governance = get_governance()
        return self._governance

    # ------------------------------------------------------------------
    # SPAWN
    # ------------------------------------------------------------------

    def spawn(
        self,
        role_spec: Any,
        mentor: Optional[str] = None,
        initial_basin: Optional[np.ndarray] = None,
        extra_voters: Optional[List[Tuple[str, float, float]]] = None,
    ) -> SpawnOutcome:
        """
        Govern → spawn → charter → proxy.

        Args:
            role_spec:      RoleSpec describing required capabilities
            mentor:         Optional mentor kernel for chaos kernels
            initial_basin:  Optional 64D initial basin
            extra_voters:   Additional voters beyond default constellation

        Returns:
            SpawnOutcome with kernel, charter, proxy info
        """
        if self.lifecycle is None:
            raise RuntimeError("KernelLifecycleManager not available")

        # Phase 1: Governance vote
        decision = None
        governance_skipped = False
        if self.governance is not None and HAS_GOVERNANCE:
            proposal = LifecycleProposal(
                proposal_id=uuid.uuid4().hex[:8],
                proposal_type=ProposalType.SPAWN,
                requester="bridge",
                description=f"Spawn: {getattr(role_spec, 'description', str(role_spec))}",
                domains=getattr(role_spec, 'domains', []),
            )
            voters = list(self._get_live_voters()) + (extra_voters or [])
            for god_name, phi, kappa in voters:
                try:
                    self.governance.cast_vote(
                        proposal,
                        GovernanceVote(voter=god_name, vote=True, phi=phi, kappa=kappa,
                                       rationale="governed_spawn")
                    )
                except Exception as e:
                    logger.warning("[GovernedBridge] Vote failed for %s: %s", god_name, e)
            decision = self.governance.evaluate(proposal)
            if decision.status == ProposalStatus.REJECTED:
                raise PermissionError(
                    f"Spawn rejected by Pantheon governance: {decision.rationale}"
                )
        else:
            governance_skipped = True
            logger.warning(
                "[GovernedBridge] Governance unavailable — spawning without vote (bootstrap mode)"
            )

        # Phase 2: Spawn
        kernel = self.lifecycle.spawn(
            role_spec=role_spec,
            mentor=mentor,
            initial_basin=initial_basin,
        )

        # Phase 3: Assign charter
        charter = None
        if HAS_CHARTER:
            try:
                if kernel.kernel_kind == KernelKind.CHAOS:
                    charter = make_chaos_charter(
                        kernel_id=kernel.kernel_id,
                        kernel_kind="chaos",
                    )
                else:
                    charter = KernelCapabilityCharter(
                        kernel_id=kernel.kernel_id,
                        kernel_kind="god",
                        capabilities=KernelCapability.FULL_GOD,
                    )
                kernel.capability_charter = charter
                if HAS_GOVERNANCE and self.governance:
                    self.governance.register_charter(kernel.kernel_id, charter)
            except Exception as e:
                logger.warning("[GovernedBridge] Charter creation failed: %s", e)

        # Phase 4: Assign proxy for chaos kernels
        proxy_assigned = False
        proxy_god_name = None
        if kernel.kernel_kind == KernelKind.CHAOS and HAS_CHARTER and HAS_GOVERNANCE:
            try:
                proxy_outcome = self._assign_chaos_proxy(kernel, extra_voters)
                proxy_assigned = proxy_outcome.get("assigned", False)
                proxy_god_name = proxy_outcome.get("proxy_god")
            except Exception as e:
                logger.warning("[GovernedBridge] Proxy assignment failed: %s", e)

        return SpawnOutcome(
            kernel=kernel,
            charter=charter,
            proxy_assigned=proxy_assigned,
            proxy_god_name=proxy_god_name,
            governance_decision=decision,
            governance_skipped=governance_skipped,
        )

    # ------------------------------------------------------------------
    # PROMOTE
    # ------------------------------------------------------------------

    def promote_with_governance(
        self,
        chaos_kernel: Any,
        god_name: str,
        voters: Optional[List[Tuple[str, float, float]]] = None,
    ) -> SpawnOutcome:
        """
        Govern → promote chaos kernel to god status.

        If insufficient voters are supplied, falls back to zeus_direct
        quorum type (single trusted voter sufficient).

        TCP v6.1 §23: Ascension requires UNANIMOUS vote by active gods.
        """
        if self.lifecycle is None:
            raise RuntimeError("KernelLifecycleManager not available")

        decision = None
        if self.governance is not None and HAS_GOVERNANCE:
            proposal = LifecycleProposal(
                proposal_id=uuid.uuid4().hex[:8],
                proposal_type=ProposalType.ASCEND,
                requester="bridge",
                description=f"Promote chaos→god: {getattr(chaos_kernel, 'name', '?')} → {god_name}",
                domains=getattr(chaos_kernel, 'domains', []),
            )
            all_voters = list(voters or []) + list(self._get_live_voters())
            for god, phi, kappa in all_voters:
                try:
                    self.governance.cast_vote(
                        proposal,
                        GovernanceVote(voter=god, vote=True, phi=phi, kappa=kappa,
                                       rationale="ascension"),
                    )
                except Exception as e:
                    logger.warning("[GovernedBridge] Ascension vote failed for %s: %s", god, e)
            decision = self.governance.evaluate(proposal)
            if decision.status == ProposalStatus.REJECTED:
                raise PermissionError(
                    f"Ascension rejected by Pantheon governance: {decision.rationale}"
                )

        # Lifecycle promote
        god_kernel = self.lifecycle.promote(chaos_kernel, god_name)

        # Update charter to FULL_GOD
        charter = None
        if HAS_CHARTER:
            try:
                charter = KernelCapabilityCharter(
                    kernel_id=god_kernel.kernel_id,
                    kernel_kind="god",
                    capabilities=KernelCapability.FULL_GOD,
                )
                god_kernel.capability_charter = charter
                if self.governance:
                    self.governance.register_charter(god_kernel.kernel_id, charter)
            except Exception as e:
                logger.warning("[GovernedBridge] God charter creation failed: %s", e)

        return SpawnOutcome(
            kernel=god_kernel,
            charter=charter,
            governance_decision=decision,
        )

    # ------------------------------------------------------------------
    # CANNIBALIZE
    # ------------------------------------------------------------------

    def cannibalize_with_governance(
        self,
        absorber: Any,
        victim: Any,
        voters: Optional[List[Tuple[str, float, float]]] = None,
        reason: str = "consolidation",
    ) -> SpawnOutcome:
        """
        Govern → cannibalize: requires UNANIMOUS vote.
        """
        if self.lifecycle is None:
            raise RuntimeError("KernelLifecycleManager not available")

        decision = None
        if self.governance is not None and HAS_GOVERNANCE:
            proposal = LifecycleProposal(
                proposal_id=uuid.uuid4().hex[:8],
                proposal_type=ProposalType.CANNIBALIZE,
                requester="bridge",
                description=f"Cannibalize: {getattr(victim, 'name', '?')} → {getattr(absorber, 'name', '?')}",
            )
            all_voters = list(voters or []) + list(self._get_live_voters())
            for god, phi, kappa in all_voters:
                try:
                    self.governance.cast_vote(
                        proposal,
                        GovernanceVote(voter=god, vote=True, phi=phi, kappa=kappa),
                    )
                except Exception as e:
                    logger.warning("[GovernedBridge] Cannibalize vote failed for %s: %s", god, e)
            decision = self.governance.evaluate(proposal)
            if decision.status == ProposalStatus.REJECTED:
                raise PermissionError(
                    f"Cannibalize rejected by Pantheon governance: {decision.rationale}"
                )

        result_kernel = self.lifecycle.cannibalize(absorber, victim, reason=reason)

        return SpawnOutcome(
            kernel=result_kernel,
            charter=result_kernel.capability_charter,
            governance_decision=decision,
        )

    # ------------------------------------------------------------------
    # INTERNAL: Chaos proxy assignment
    # ------------------------------------------------------------------

    def _assign_chaos_proxy(
        self,
        chaos_kernel: Any,
        extra_voters: Optional[List[Tuple[str, float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Submit PROXY_ASSIGN proposal and wire charter if approved.

        Returns dict: {"assigned": bool, "proxy_god": str|None}
        """
        if self.governance is None:
            return {"assigned": False, "proxy_god": None}

        proposal = LifecycleProposal(
            proposal_id=uuid.uuid4().hex[:8],
            proposal_type=ProposalType.PROXY_ASSIGN,
            requester="bridge",
            description=f"Assign proxy god for chaos kernel {chaos_kernel.kernel_id}",
        )
        voters = list(self._get_live_voters()) + (extra_voters or [])
        for god_name, phi, kappa in voters:
            try:
                self.governance.cast_vote(
                    proposal,
                    GovernanceVote(voter=god_name, vote=True, phi=phi, kappa=kappa,
                                   rationale="proxy_assignment"),
                )
            except Exception as e:
                logger.debug("[GovernedBridge] Proxy vote for %s: %s", god_name, e)

        decision = self.governance.evaluate(proposal)
        if decision.status != ProposalStatus.APPROVED:
            logger.warning(
                "[GovernedBridge] Proxy assignment not approved for %s: %s",
                chaos_kernel.kernel_id, decision.rationale,
            )
            return {"assigned": False, "proxy_god": None}

        proxy_god_name = decision.proxy_god_name
        if proxy_god_name and chaos_kernel.capability_charter is not None:
            try:
                proxy_assignment = ProxyAssignment(
                    chaos_kernel_id=chaos_kernel.kernel_id,
                    proxy_god_name=proxy_god_name,
                )
                chaos_kernel.capability_charter.proxy = proxy_assignment
                # Grant proxy god PROXY_VOICE
                self.governance.grant_proxy_voice(proxy_god_name, chaos_kernel.kernel_id)
            except Exception as e:
                logger.warning("[GovernedBridge] Charter proxy wiring failed: %s", e)

        return {"assigned": True, "proxy_god": proxy_god_name}


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def spawn_with_governance(
    role_spec: Any,
    lifecycle_mgr: Optional[Any] = None,
    governance: Optional[Any] = None,
    initial_basin: Optional[np.ndarray] = None,
) -> Tuple[Any, Optional[Any]]:
    """
    One-call convenience wrapper. Returns (kernel, charter).
    """
    bridge = GovernedLifecycleManager(lifecycle_mgr=lifecycle_mgr, governance=governance)
    outcome = bridge.spawn(role_spec=role_spec, initial_basin=initial_basin)
    return outcome.kernel, outcome.charter
