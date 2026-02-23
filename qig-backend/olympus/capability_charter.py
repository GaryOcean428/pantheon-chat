"""
Kernel Capability Charter — TCP v6.1

Every kernel in the constellation operates under a KernelCapabilityCharter
that was assigned at spawn time by Pantheon vote. The charter is immutable
after spawn — capabilities can only change via a new governance proposal.

Key design principle (Braden's requirement):
  Chaos kernels have NO generative capability by default.
  The Pantheon assigns a GOD as their proxy voice, who:
    1. Speaks on the chaos kernel's behalf in conversations.
    2. Receives ProxyInstructions that direct the chaos kernel's exploration.

Capability assignment is FLEXIBLE — the Pantheon can grant any combination
of capabilities to any kernel type, with the registry's SpawnConstraints
acting as the hard ceiling.

TCP v6.1 §19 — Genesis Doctrine:
  - 240 GOD slots reserved for E8 GOD evolution
  - Chaos kernels exist OUTSIDE that budget
  - Chaos ascends to GOD only via explicit governance (ProposalType.CHAOS_ASCEND)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capability flags — combine freely
# ---------------------------------------------------------------------------

class KernelCapability(Flag):
    """
    Capabilities a kernel may hold.  Assigned by Pantheon vote at spawn time.
    Use Flag (bitmask) so multiple caps compose cleanly.

    Constraints:
    - GENERATIVE requires either SYNTHESIS or ROUTING to be useful.
    - PROXY_VOICE implies the holder speaks for at least one voiceless kernel.
    - GOVERNANCE_VOTE is withheld from chaos kernels in PROTECTED/LEARNING stages.
    - CHAOS_EXPLORE is exclusive to chaos kernels; gods never hold it.
    """
    NONE            = 0                 # Explicitly voiceless
    GENERATIVE      = auto()            # Produce text output
    SYNTHESIS       = auto()            # Combine kernel outputs (Fréchet mean)
    ROUTING         = auto()            # Dispatch queries to other kernels
    OBSERVATION     = auto()            # Observe and log metrics; no generation
    PROXY_VOICE     = auto()            # Speak on behalf of a voiceless kernel
    TRAINING        = auto()            # Participate in training loops
    CHAOS_EXPLORE   = auto()            # Explore novel basins outside GOD budget
    GOVERNANCE_VOTE = auto()            # Cast votes in Pantheon governance

    # Convenience bundles
    FULL_GOD       = GENERATIVE | SYNTHESIS | ROUTING | OBSERVATION | GOVERNANCE_VOTE
    CHAOS_DEFAULT  = CHAOS_EXPLORE | OBSERVATION          # No voice, no vote
    CHAOS_WORKING  = CHAOS_DEFAULT | TRAINING             # Learning stage
    CHAOS_CANDIDATE = CHAOS_WORKING | GOVERNANCE_VOTE     # Candidate for ascension


# ---------------------------------------------------------------------------
# Proxy assignment — for voiceless (chaos) kernels
# ---------------------------------------------------------------------------

@dataclass
class ProxyInstruction:
    """
    Instructions given by the Pantheon to the proxy god for a chaos kernel.

    The proxy god reads these and shapes how it speaks on the chaos kernel's
    behalf and what exploratory directives it forwards.
    """
    explore_domains: List[str] = field(default_factory=list)
    avoid_domains: List[str] = field(default_factory=list)
    basin_target: Optional[List[float]] = None      # 64D target basin for exploration
    intensity: float = 0.5                           # [0, 1] — effort level
    report_threshold_phi: float = 0.60              # Report to Discovery Gate if Φ >
    max_steps: int = 500                            # Max exploration steps before rest
    narrative_style: str = "terse"                  # How the proxy should voice findings
    custom: Dict = field(default_factory=dict)       # Free-form Pantheon directives


@dataclass
class ProxyAssignment:
    """
    Links a chaos kernel to its proxy god.

    The proxy god:
    1. Speaks in conversations when the chaos kernel would need a voice.
    2. Routes ProxyInstructions into the chaos kernel's exploration loop.
    3. Relays discoveries back to the Discovery Gate.
    """
    chaos_kernel_id: str
    proxy_god_name: str                 # E.g. "Hermes", "Apollo", "Athena"
    instructions: ProxyInstruction = field(default_factory=ProxyInstruction)
    proxy_scope: str = "full"           # "full" = speaks for all output, "relay" = relays only
    revocable: bool = True              # Can the Pantheon reassign the proxy?


# ---------------------------------------------------------------------------
# Capability charter — the immutable constitution of a kernel
# ---------------------------------------------------------------------------

@dataclass
class KernelCapabilityCharter:
    """
    Immutable capability charter attached to a kernel at spawn time.

    Produced by PantheonGovernance.execute() and stored alongside the
    kernel in the registry.  No other code may alter capabilities after
    spawn without a new governance proposal.

    Fields:
        kernel_id       — Unique ID of the kernel this charter governs.
        kernel_type     — "god" | "chaos" | "shadow"
        capabilities    — Granted capability flags (Pantheon-assigned).
        proxy           — Set if this kernel is voiceless (chaos/shadow).
        proxy_for       — Set if this kernel acts as proxy for another.
        max_instances   — From registry SpawnConstraints (hard ceiling).
        stage_at_spawn  — Lifecycle stage at birth (ChaosLifecycleStage or "spawned").
        granted_by      — Name of the proposal that created this charter.
        voter_coalition — Gods who voted YES (quorum record).
    """
    kernel_id: str
    kernel_type: str                        # "god" | "chaos" | "shadow"
    capabilities: KernelCapability
    proxy: Optional[ProxyAssignment] = None           # This kernel needs a voice
    proxy_for: Optional[str] = None                   # This kernel IS a voice for …
    max_instances: int = 1
    stage_at_spawn: str = "spawned"
    granted_by: str = ""                    # Proposal ID
    voter_coalition: List[str] = field(default_factory=list)

    def has(self, cap: KernelCapability) -> bool:
        """Check if this kernel holds a specific capability."""
        return bool(self.capabilities & cap)

    def can_generate(self) -> bool:
        return self.has(KernelCapability.GENERATIVE)

    def can_vote(self) -> bool:
        return self.has(KernelCapability.GOVERNANCE_VOTE)

    def is_voiceless(self) -> bool:
        return not self.has(KernelCapability.GENERATIVE)

    def has_proxy(self) -> bool:
        return self.proxy is not None

    def summary(self) -> str:
        caps = [c.name for c in KernelCapability if c != KernelCapability.NONE and self.has(c)]
        proxy_str = f" proxy→{self.proxy.proxy_god_name}" if self.proxy else ""
        return f"{self.kernel_id}[{self.kernel_type}]: {'+'.join(caps) or 'NONE'}{proxy_str}"


# ---------------------------------------------------------------------------
# Policy — what a kernel TYPE can receive from the Pantheon
# ---------------------------------------------------------------------------

class CapabilityPolicy:
    """
    Hard constraints on what capabilities each kernel type may receive.
    The Pantheon vote cannot override these constraints — they enforce
    Genesis Doctrine at the type level.
    """

    # Capabilities that may NEVER be assigned to a kernel of each type
    _FORBIDDEN: Dict[str, KernelCapability] = {
        "god":    KernelCapability.CHAOS_EXPLORE,   # Gods don't go feral
        "chaos":  KernelCapability.NONE,            # Chaos can receive anything (Pantheon decides)
        "shadow": KernelCapability.GOVERNANCE_VOTE, # Shadow never votes
    }

    # Default capabilities for a NEW kernel of each type before Pantheon adds more
    _DEFAULTS: Dict[str, KernelCapability] = {
        "god":    KernelCapability.FULL_GOD,
        "chaos":  KernelCapability.CHAOS_DEFAULT,
        "shadow": KernelCapability.OBSERVATION,
    }

    @classmethod
    def default_for(cls, kernel_type: str) -> KernelCapability:
        return cls._DEFAULTS.get(kernel_type, KernelCapability.OBSERVATION)

    @classmethod
    def enforce(cls, kernel_type: str, requested: KernelCapability) -> KernelCapability:
        """
        Strip forbidden capabilities from the requested set.
        Returns the sanitised capability flags.
        """
        forbidden = cls._FORBIDDEN.get(kernel_type, KernelCapability.NONE)
        sanitised = requested & ~forbidden
        stripped = requested & forbidden
        if stripped != KernelCapability.NONE:
            logger.warning(
                "[CapabilityPolicy] Stripped forbidden caps for %s kernel: %s",
                kernel_type, stripped,
            )
        return sanitised


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_chaos_charter(
    kernel_id: str,
    proxy_god: str,
    granted_by: str,
    coalition: List[str],
    extra_caps: KernelCapability = KernelCapability.NONE,
    proxy_instructions: Optional[ProxyInstruction] = None,
    max_instances: int = 1,
) -> KernelCapabilityCharter:
    """
    Convenience factory: create a standard chaos kernel charter with proxy.

    The proxy god is automatically granted PROXY_VOICE in the caller's
    existing charter — that wiring is the caller's responsibility.
    """
    caps = CapabilityPolicy.enforce("chaos", KernelCapability.CHAOS_DEFAULT | extra_caps)
    proxy = ProxyAssignment(
        chaos_kernel_id=kernel_id,
        proxy_god_name=proxy_god,
        instructions=proxy_instructions or ProxyInstruction(),
    )
    return KernelCapabilityCharter(
        kernel_id=kernel_id,
        kernel_type="chaos",
        capabilities=caps,
        proxy=proxy,
        max_instances=max_instances,
        stage_at_spawn="protected",
        granted_by=granted_by,
        voter_coalition=coalition,
    )


def grant_proxy_voice_to_god(
    god_charter: KernelCapabilityCharter,
    chaos_kernel_id: str,
) -> KernelCapabilityCharter:
    """
    Add PROXY_VOICE capability to an existing god charter and record
    which chaos kernel they are now speaking for.

    Returns the UPDATED charter (original is not mutated — create a new one).
    """
    new_caps = god_charter.capabilities | KernelCapability.PROXY_VOICE
    from dataclasses import replace
    return replace(
        god_charter,
        capabilities=new_caps,
        proxy_for=chaos_kernel_id,
    )
