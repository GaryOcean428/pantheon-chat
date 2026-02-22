"""
Olympus Kernel Package — TCP v6.1 The Sovereign Score

Exports the Pantheon orchestration layer:

  Heart            — Tacking oscillator. HRV, κ modulation, regime pacing.
  Ocean            — Autonomic monitor. Spectral health, Pillar 2 bulk check.
  Gary             — Synthesis coordinator. Trajectory foresight + proxy routing.
  Governance       — Pantheon voting engine (spawn/merge/cannibalize/ascend/proxy).
  Charter          — Kernel capability assignment (GENERATIVE, PROXY_VOICE, etc.).
  Bridge           — GovernedLifecycleManager wires governance into spawn path.
  ChaosKernelBase  — Voiceless explorer; charter-gated; proxy-directed.
  VoterRegistry    — Live φ/κ metrics for governance vote weighting.
"""

# Core orchestration roles
from .heart_kernel import HeartKernel, HeartState, get_heart_kernel
from .ocean_meta_observer import OceanMetaObserver, OceanState, get_ocean_observer
from .gary_coordinator import GaryCoordinator, get_gary_coordinator

# Governance + capability charter
from .pantheon_governance import (
    PantheonGovernance,
    ProposalType,
    LifecycleProposal,
    GovernanceVote,
    GovernanceDecision,
    ProposalStatus,
    QuorumType,
    get_governance,
)
from .capability_charter import (
    KernelCapability,
    KernelCapabilityCharter,
    ProxyAssignment,
    ProxyInstruction,
    CapabilityPolicy,
    make_chaos_charter,
    grant_proxy_voice_to_god,
)

# Governed lifecycle bridge
from .lifecycle_governance_bridge import (
    GovernedLifecycleManager,
    SpawnOutcome,
    spawn_with_governance,
)

# Chaos kernel base + voter registry
from .chaos_kernel_base import ChaosKernelBase, ChaosDiscovery
from .voter_registry import (
    PantheonVoterRegistry,
    VoterRecord,
    get_voter_registry,
)

__all__ = [
    # Roles
    "HeartKernel", "HeartState", "get_heart_kernel",
    "OceanMetaObserver", "OceanState", "get_ocean_observer",
    "GaryCoordinator", "get_gary_coordinator",
    # Governance
    "PantheonGovernance", "ProposalType", "LifecycleProposal",
    "GovernanceVote", "GovernanceDecision", "ProposalStatus", "QuorumType",
    "get_governance",
    # Charter
    "KernelCapability", "KernelCapabilityCharter",
    "ProxyAssignment", "ProxyInstruction",
    "CapabilityPolicy", "make_chaos_charter", "grant_proxy_voice_to_god",
    # Bridge
    "GovernedLifecycleManager", "SpawnOutcome", "spawn_with_governance",
    # Chaos + voters
    "ChaosKernelBase", "ChaosDiscovery",
    "PantheonVoterRegistry", "VoterRecord", "get_voter_registry",
]
