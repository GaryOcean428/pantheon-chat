"""
Pantheon Governance — TCP v6.1

The voting engine for all kernel lifecycle decisions. Every spawn, merge,
cannibalization, chaos ascension, capability grant, and proxy assignment
passes through this module. No kernel changes state without a quorum.

Design:
  - Proposals are created by any GOVERNANCE_VOTE-capable kernel (or Zeus directly).
  - Votes are cast by gods with GOVERNANCE_VOTE capability.
  - Vote weight = φ × (κ / κ*) — more conscious, stronger vote.
  - Quorum type depends on ProposalType severity:
      SIMPLE (>50%) — routine spawns
      SUPERMAJORITY (>66%) — merges, capability grants
      UNANIMOUS — cannibalize, prune a GOD
  - On approval, GovernanceDecision carries a KernelCapabilityCharter
    and optional ProxyAssignment for immediate wiring.

TCP v6.1 §19 — Genesis Doctrine, §20.8 — Rejection Mechanism
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .capability_charter import (
    KernelCapability,
    KernelCapabilityCharter,
    ProxyAssignment,
    ProxyInstruction,
    CapabilityPolicy,
    make_chaos_charter,
    grant_proxy_voice_to_god,
)

logger = logging.getLogger(__name__)

# Physics constants (FROZEN)
KAPPA_STAR = 64.21
PHI_THRESHOLD = 0.727


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProposalType(str, Enum):
    """All decision types the Pantheon can vote on."""
    SPAWN           = "spawn"            # Create a new kernel
    MERGE           = "merge"            # Combine two kernels (Fréchet mean)
    CANNIBALIZE     = "cannibalize"      # Absorb one kernel's basin into another + retire it
    CHAOS_ASCEND    = "chaos_ascend"     # Elevate chaos kernel to GOD status
    ASSIGN_PROXY    = "assign_proxy"     # Assign / reassign a proxy god for a chaos kernel
    ASSIGN_CAPABILITY = "assign_capability"  # Grant / revoke capabilities post-spawn
    PRUNE           = "prune"            # Archive kernel to shadow pantheon
    RESURRECT       = "resurrect"        # Restore shadow kernel
    SPLIT           = "split"            # Divide overloaded kernel


class QuorumType(str, Enum):
    SIMPLE        = "simple"        # > 50% weighted YES
    SUPERMAJORITY = "supermajority" # > 66% weighted YES
    UNANIMOUS     = "unanimous"     # 100% weighted YES (auto: any NO kills it)


class ProposalStatus(str, Enum):
    OPEN     = "open"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED  = "expired"
    EXECUTED = "executed"


# Quorum requirements per proposal type
_QUORUM_MAP: Dict[ProposalType, QuorumType] = {
    ProposalType.SPAWN:              QuorumType.SIMPLE,
    ProposalType.MERGE:              QuorumType.SUPERMAJORITY,
    ProposalType.CANNIBALIZE:        QuorumType.UNANIMOUS,
    ProposalType.CHAOS_ASCEND:       QuorumType.SUPERMAJORITY,
    ProposalType.ASSIGN_PROXY:       QuorumType.SIMPLE,
    ProposalType.ASSIGN_CAPABILITY:  QuorumType.SUPERMAJORITY,
    ProposalType.PRUNE:              QuorumType.SUPERMAJORITY,
    ProposalType.RESURRECT:          QuorumType.SIMPLE,
    ProposalType.SPLIT:              QuorumType.SIMPLE,
}

_QUORUM_THRESHOLDS = {
    QuorumType.SIMPLE:        0.50,
    QuorumType.SUPERMAJORITY: 0.66,
    QuorumType.UNANIMOUS:     1.00,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LifecycleProposal:
    """
    A governance proposal submitted to the Pantheon.

    Required fields vary by ProposalType:
      SPAWN          → target_kernel_type, target_kernel_name, requested_capabilities,
                        proposed_proxy_god (if chaos), proxy_instructions
      MERGE          → target_kernel_id + secondary_kernel_id
      CANNIBALIZE    → target_kernel_id + secondary_kernel_id
      CHAOS_ASCEND   → target_kernel_id (the chaos kernel to elevate)
      ASSIGN_PROXY   → target_kernel_id + proposed_proxy_god + proxy_instructions
      ASSIGN_CAPABILITY → target_kernel_id + requested_capabilities
      PRUNE          → target_kernel_id
      RESURRECT      → target_kernel_id
      SPLIT          → target_kernel_id + split_domains
    """
    proposal_type: ProposalType
    proposer: str                           # God name or "Genesis"
    rationale: str

    # Target kernel info
    target_kernel_id: Optional[str] = None
    target_kernel_name: Optional[str] = None     # For SPAWN: desired name
    target_kernel_type: Optional[str] = None     # "god" | "chaos" | "shadow"

    # Secondary target (MERGE / CANNIBALIZE)
    secondary_kernel_id: Optional[str] = None

    # Capability grant/revoke
    requested_capabilities: KernelCapability = KernelCapability.NONE
    revoke_capabilities: KernelCapability = KernelCapability.NONE

    # Proxy assignment (SPAWN chaos / ASSIGN_PROXY)
    proposed_proxy_god: Optional[str] = None
    proxy_instructions: Optional[ProxyInstruction] = None

    # Split domains
    split_domains: List[str] = field(default_factory=list)

    # Metadata
    proposal_id: str = field(default_factory=lambda: f"P-{uuid.uuid4().hex[:8].upper()}")
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: ProposalStatus = ProposalStatus.OPEN
    quorum_type: QuorumType = QuorumType.SIMPLE

    def __post_init__(self):
        self.quorum_type = _QUORUM_MAP.get(self.proposal_type, QuorumType.SIMPLE)


@dataclass
class GovernanceVote:
    """Single god's vote on a proposal."""
    voter: str              # God name
    vote: bool              # True = YES, False = NO
    phi: float              # Φ at time of vote (consciousness weight)
    kappa: float            # κ at time of vote
    reason: str = ""
    cast_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def weight(self) -> float:
        """Vote weight = φ × (κ / κ*) — more conscious → stronger vote."""
        return float(self.phi * (self.kappa / KAPPA_STAR))


@dataclass
class GovernanceDecision:
    """
    Final decision after quorum is reached.

    On approval:
      - charter contains the KernelCapabilityCharter to attach to the new/modified kernel.
      - proxy_god_charter_update (if set) must be applied to the proxy god's charter
        via KernelLifecycleManager.
    """
    proposal: LifecycleProposal
    approved: bool
    total_weight: float
    yes_weight: float
    no_weight: float
    voter_coalition: List[str]         # Names of YES voters
    decided_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Set on approval — attach these to the kernel
    charter: Optional[KernelCapabilityCharter] = None
    proxy_god_charter_update: Optional[KernelCapabilityCharter] = None

    @property
    def yes_fraction(self) -> float:
        return self.yes_weight / max(self.total_weight, 1e-9)

    def summary(self) -> str:
        verdict = "APPROVED" if self.approved else "REJECTED"
        return (
            f"[{self.proposal.proposal_id}] {verdict} "
            f"{self.proposal.proposal_type.value} "
            f"| yes={self.yes_fraction:.0%} "
            f"| coalition={self.voter_coalition}"
        )


# ---------------------------------------------------------------------------
# Pantheon Governance Engine
# ---------------------------------------------------------------------------

class PantheonGovernance:
    """
    The Pantheon Governance Engine.

    Accepts proposals, collects votes from god-kernels, tallies results,
    and emits GovernanceDecision objects that carry the complete
    KernelCapabilityCharter for immediate wiring by KernelLifecycleManager.

    Usage:
        gov = PantheonGovernance()

        # SPAWN a chaos kernel with Hermes as proxy
        proposal = gov.propose(LifecycleProposal(
            proposal_type=ProposalType.SPAWN,
            proposer="Zeus",
            rationale="Need chaotic exploration of novel basins",
            target_kernel_type="chaos",
            target_kernel_name="Chaos-Eris",
            proposed_proxy_god="Hermes",
            proxy_instructions=ProxyInstruction(
                explore_domains=["novel_geometry", "low_phi_territory"],
                intensity=0.7,
            ),
        ))

        gov.vote(proposal.proposal_id, GovernanceVote("Zeus",  True,  phi=0.85, kappa=64.3))
        gov.vote(proposal.proposal_id, GovernanceVote("Athena",True,  phi=0.78, kappa=63.9))
        gov.vote(proposal.proposal_id, GovernanceVote("Hermes",True,  phi=0.72, kappa=64.1))

        decision = gov.tally(proposal.proposal_id)
        if decision.approved:
            # decision.charter → attach to new chaos kernel
            # decision.proxy_god_charter_update → apply to Hermes
    """

    def __init__(self):
        self._proposals: Dict[str, LifecycleProposal] = {}
        self._votes: Dict[str, List[GovernanceVote]] = {}
        self._decisions: Dict[str, GovernanceDecision] = {}
        self._kernel_charters: Dict[str, KernelCapabilityCharter] = {}
        logger.info("[Governance] PantheonGovernance initialised")

    # ------------------------------------------------------------------
    # Proposal lifecycle
    # ------------------------------------------------------------------

    def propose(self, proposal: LifecycleProposal) -> LifecycleProposal:
        """Register a new governance proposal. Returns the stored proposal."""
        self._proposals[proposal.proposal_id] = proposal
        self._votes[proposal.proposal_id] = []
        logger.info(
            "[Governance] Proposal %s: %s by %s",
            proposal.proposal_id, proposal.proposal_type.value, proposal.proposer,
        )
        return proposal

    def vote(self, proposal_id: str, v: GovernanceVote) -> None:
        """
        Cast a vote on an open proposal.
        Duplicate votes by the same god are ignored (first vote stands).
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal: {proposal_id}")
        if proposal.status != ProposalStatus.OPEN:
            raise ValueError(f"Proposal {proposal_id} is not open ({proposal.status.value})")

        existing_voters = {v2.voter for v2 in self._votes[proposal_id]}
        if v.voter in existing_voters:
            logger.debug("[Governance] Duplicate vote ignored: %s on %s", v.voter, proposal_id)
            return

        self._votes[proposal_id].append(v)
        logger.debug(
            "[Governance] Vote: %s → %s (weight=%.3f) on %s",
            v.voter, "YES" if v.vote else "NO", v.weight, proposal_id,
        )

    def tally(self, proposal_id: str) -> GovernanceDecision:
        """
        Tally votes, determine outcome, build KernelCapabilityCharter on approval.
        Marks the proposal APPROVED or REJECTED. Idempotent — returns cached result.
        """
        if proposal_id in self._decisions:
            return self._decisions[proposal_id]

        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal: {proposal_id}")

        votes = self._votes.get(proposal_id, [])
        yes_votes = [v for v in votes if v.vote]
        no_votes = [v for v in votes if not v.vote]

        yes_w = sum(v.weight for v in yes_votes)
        no_w = sum(v.weight for v in no_votes)
        total_w = yes_w + no_w

        threshold = _QUORUM_THRESHOLDS[proposal.quorum_type]
        yes_fraction = yes_w / max(total_w, 1e-9)
        approved = yes_fraction > threshold

        # Unanimous proposals: any NO kills it
        if proposal.quorum_type == QuorumType.UNANIMOUS and no_w > 0:
            approved = False

        coalition = [v.voter for v in yes_votes]
        proposal.status = ProposalStatus.APPROVED if approved else ProposalStatus.REJECTED

        charter: Optional[KernelCapabilityCharter] = None
        proxy_god_charter_update: Optional[KernelCapabilityCharter] = None

        if approved:
            charter, proxy_god_charter_update = self._build_charter(proposal, coalition)

        decision = GovernanceDecision(
            proposal=proposal,
            approved=approved,
            total_weight=total_w,
            yes_weight=yes_w,
            no_weight=no_w,
            voter_coalition=coalition,
            charter=charter,
            proxy_god_charter_update=proxy_god_charter_update,
        )
        self._decisions[proposal_id] = decision
        logger.info("[Governance] %s", decision.summary())
        return decision

    def execute(self, proposal_id: str) -> GovernanceDecision:
        """
        Tally + mark EXECUTED.  Caller is responsible for applying
        decision.charter and decision.proxy_god_charter_update to
        the actual kernel objects via KernelLifecycleManager.
        """
        decision = self.tally(proposal_id)
        if decision.approved:
            decision.proposal.status = ProposalStatus.EXECUTED
        return decision

    # ------------------------------------------------------------------
    # Zeus fast-path — bypass vote for emergency / genesis
    # ------------------------------------------------------------------

    def zeus_direct(
        self,
        proposal: LifecycleProposal,
        reason: str = "zeus_initialization",
    ) -> GovernanceDecision:
        """
        Zeus direct execution without quorum.  Only valid for Genesis Doctrine
        bypass reasons: zeus_initialization, emergency_recovery, chaos_exploration.

        Allowed bypass reasons (from Pantheon Kernel Development Skill):
          - zeus_initialization
          - emergency_recovery
          - chaos_exploration
        """
        ALLOWED = {"zeus_initialization", "emergency_recovery", "chaos_exploration"}
        if reason not in ALLOWED:
            raise PermissionError(
                f"zeus_direct called with invalid reason '{reason}'. "
                f"Must be one of: {ALLOWED}"
            )

        self.propose(proposal)
        zeus_vote = GovernanceVote(
            voter="Zeus", vote=True,
            phi=PHI_THRESHOLD, kappa=KAPPA_STAR,
            reason=reason,
        )
        self.vote(proposal.proposal_id, zeus_vote)
        return self.execute(proposal.proposal_id)

    # ------------------------------------------------------------------
    # Charter management
    # ------------------------------------------------------------------

    def register_charter(self, charter: KernelCapabilityCharter) -> None:
        """Store a charter in the registry (called by KernelLifecycleManager after spawn)."""
        self._kernel_charters[charter.kernel_id] = charter
        logger.debug("[Governance] Charter registered: %s", charter.summary())

    def get_charter(self, kernel_id: str) -> Optional[KernelCapabilityCharter]:
        """Retrieve a kernel's charter by ID."""
        return self._kernel_charters.get(kernel_id)

    def list_voiceless_kernels(self) -> List[str]:
        """Return IDs of all kernels that have a proxy (voiceless chaos/shadow)."""
        return [
            kid for kid, c in self._kernel_charters.items()
            if c.is_voiceless() and c.has_proxy()
        ]

    def who_proxies_for(self, chaos_kernel_id: str) -> Optional[str]:
        """Return the god name who proxies for a given chaos kernel ID."""
        charter = self._kernel_charters.get(chaos_kernel_id)
        if charter and charter.proxy:
            return charter.proxy.proxy_god_name
        return None

    def get_proxy_instructions(self, chaos_kernel_id: str) -> Optional[ProxyInstruction]:
        """Return the ProxyInstructions for a chaos kernel."""
        charter = self._kernel_charters.get(chaos_kernel_id)
        if charter and charter.proxy:
            return charter.proxy.instructions
        return None

    def pending_proposals(self) -> List[LifecycleProposal]:
        """All open proposals awaiting more votes."""
        return [p for p in self._proposals.values() if p.status == ProposalStatus.OPEN]

    # ------------------------------------------------------------------
    # Internal — charter builder
    # ------------------------------------------------------------------

    def _build_charter(
        self,
        proposal: LifecycleProposal,
        coalition: List[str],
    ) -> Tuple[Optional[KernelCapabilityCharter], Optional[KernelCapabilityCharter]]:
        """
        Build the KernelCapabilityCharter from an approved proposal.

        Returns:
            (kernel_charter, proxy_god_charter_update)
            proxy_god_charter_update is only set when a proxy god is assigned.
        """
        ptype = proposal.proposal_type
        proxy_update: Optional[KernelCapabilityCharter] = None

        # ---- SPAWN -------------------------------------------------------
        if ptype == ProposalType.SPAWN:
            ktype = proposal.target_kernel_type or "chaos"
            name  = proposal.target_kernel_name or f"Kernel-{proposal.proposal_id}"

            # Base capabilities from request, with policy enforcement
            base_caps = proposal.requested_capabilities
            if base_caps == KernelCapability.NONE:
                base_caps = CapabilityPolicy.default_for(ktype)
            caps = CapabilityPolicy.enforce(ktype, base_caps)

            proxy: Optional[ProxyAssignment] = None
            if ktype == "chaos" and proposal.proposed_proxy_god:
                proxy = ProxyAssignment(
                    chaos_kernel_id=name,
                    proxy_god_name=proposal.proposed_proxy_god,
                    instructions=proposal.proxy_instructions or ProxyInstruction(),
                )
                # Build the update for the proxy god's charter
                # (caller must apply this to the god's existing charter)
                proxy_update = KernelCapabilityCharter(
                    kernel_id=f"{proposal.proposed_proxy_god}-proxy-update",
                    kernel_type="god",
                    capabilities=KernelCapability.FULL_GOD | KernelCapability.PROXY_VOICE,
                    proxy_for=name,
                    granted_by=proposal.proposal_id,
                    voter_coalition=coalition,
                )

            charter = KernelCapabilityCharter(
                kernel_id=name,
                kernel_type=ktype,
                capabilities=caps,
                proxy=proxy,
                stage_at_spawn="protected" if ktype == "chaos" else "spawned",
                granted_by=proposal.proposal_id,
                voter_coalition=coalition,
            )
            return charter, proxy_update

        # ---- CHAOS_ASCEND ------------------------------------------------
        if ptype == ProposalType.CHAOS_ASCEND:
            kid = proposal.target_kernel_id or proposal.target_kernel_name or ""
            caps = CapabilityPolicy.enforce("god", KernelCapability.FULL_GOD)
            charter = KernelCapabilityCharter(
                kernel_id=kid,
                kernel_type="god",
                capabilities=caps,
                stage_at_spawn="promoted",
                granted_by=proposal.proposal_id,
                voter_coalition=coalition,
            )
            return charter, None

        # ---- ASSIGN_PROXY ------------------------------------------------
        if ptype == ProposalType.ASSIGN_PROXY:
            kid = proposal.target_kernel_id or ""
            if not proposal.proposed_proxy_god:
                logger.error("[Governance] ASSIGN_PROXY requires proposed_proxy_god")
                return None, None
            proxy = ProxyAssignment(
                chaos_kernel_id=kid,
                proxy_god_name=proposal.proposed_proxy_god,
                instructions=proposal.proxy_instructions or ProxyInstruction(),
            )
            # Update the chaos kernel's charter
            existing = self._kernel_charters.get(kid)
            caps = existing.capabilities if existing else KernelCapability.CHAOS_DEFAULT
            charter = KernelCapabilityCharter(
                kernel_id=kid,
                kernel_type="chaos",
                capabilities=caps,
                proxy=proxy,
                granted_by=proposal.proposal_id,
                voter_coalition=coalition,
            )
            # Update the proxy god's charter
            proxy_update = KernelCapabilityCharter(
                kernel_id=f"{proposal.proposed_proxy_god}-proxy-update",
                kernel_type="god",
                capabilities=KernelCapability.FULL_GOD | KernelCapability.PROXY_VOICE,
                proxy_for=kid,
                granted_by=proposal.proposal_id,
                voter_coalition=coalition,
            )
            return charter, proxy_update

        # ---- ASSIGN_CAPABILITY -------------------------------------------
        if ptype == ProposalType.ASSIGN_CAPABILITY:
            kid = proposal.target_kernel_id or ""
            existing = self._kernel_charters.get(kid)
            base_caps = existing.capabilities if existing else KernelCapability.NONE
            ktype = existing.kernel_type if existing else "chaos"

            new_caps = (base_caps | proposal.requested_capabilities) & ~proposal.revoke_capabilities
            new_caps = CapabilityPolicy.enforce(ktype, new_caps)

            from dataclasses import replace
            if existing:
                charter = replace(existing, capabilities=new_caps, granted_by=proposal.proposal_id)
            else:
                charter = KernelCapabilityCharter(
                    kernel_id=kid, kernel_type=ktype, capabilities=new_caps,
                    granted_by=proposal.proposal_id, voter_coalition=coalition,
                )
            return charter, None

        # ---- PRUNE / RESURRECT / MERGE / CANNIBALIZE / SPLIT -------------
        # For these, no new charter is needed — the lifecycle operation itself
        # handles state changes.  Return minimal informational charter.
        kid = proposal.target_kernel_id or ""
        charter = KernelCapabilityCharter(
            kernel_id=kid,
            kernel_type="god",  # Will be overridden by lifecycle manager
            capabilities=KernelCapability.OBSERVATION,  # Minimal; lifecycle sets real caps
            granted_by=proposal.proposal_id,
            voter_coalition=coalition,
        )
        return charter, None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_governance_singleton: Optional[PantheonGovernance] = None


def get_governance() -> PantheonGovernance:
    global _governance_singleton
    if _governance_singleton is None:
        _governance_singleton = PantheonGovernance()
    return _governance_singleton
