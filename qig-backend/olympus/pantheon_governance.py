"""
Pantheon Governance: Centralized Lifecycle Control
===================================================

Enforces that ALL kernel spawning, breeding, and death decisions
go through Pantheon approval. No kernel can be born or die without
the gods' consent.

This prevents uncontrolled population explosions and ensures
proper oversight of the kernel ecosystem.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import os

try:
    import psycopg2
    from psycopg2.extras import Json as PsycopgJson
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None
    PsycopgJson = None

# Import capability mesh for event emission
try:
    from .capability_mesh import (
        CapabilityEvent,
        CapabilityType,
        EventType,
        emit_event,
    )
    CAPABILITY_MESH_AVAILABLE = True
except ImportError:
    CAPABILITY_MESH_AVAILABLE = False
    CapabilityEvent = None
    CapabilityType = None
    EventType = None
    emit_event = None

# Import ActivityBroadcaster for kernel visibility
try:
    from .activity_broadcaster import get_broadcaster, ActivityType
    ACTIVITY_BROADCASTER_AVAILABLE = True
except ImportError:
    ACTIVITY_BROADCASTER_AVAILABLE = False
    get_broadcaster = None
    ActivityType = None


class ProposalType(Enum):
    """Types of lifecycle proposals."""
    SPAWN = "spawn"
    BREED = "breed"
    DEATH = "death"
    TURBO_SPAWN = "turbo_spawn"
    # Extended lifecycle proposals - ALL kernel decisions through Pantheon
    EVOLVE = "evolve"              # Kernel mutation/evolution
    MERGE = "merge"                # Combine two kernels into one
    CANNIBALIZE = "cannibalize"    # Strong kernel absorbs weak kernel
    NEW_GOD = "new_god"            # Promote chaos kernel to god status
    CHAOS_SPAWN = "chaos_spawn"    # Worker kernel creation


class ProposalStatus(Enum):
    """Status of a governance proposal."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


@dataclass
class LifecycleProposal:
    """A proposal for kernel lifecycle action."""
    proposal_id: str
    proposal_type: ProposalType
    created_at: datetime
    status: ProposalStatus = ProposalStatus.PENDING
    
    # Proposal details
    reason: Optional[str] = None
    parent_id: Optional[str] = None
    parent_phi: Optional[float] = None
    count: int = 1
    
    # Voting
    votes_for: List[str] = field(default_factory=list)
    votes_against: List[str] = field(default_factory=list)
    
    # Audit trail
    approved_by: Optional[str] = None
    rejected_by: Optional[str] = None
    audit_log: List[Dict] = field(default_factory=list)
    
    # Result
    executed: bool = False
    execution_result: Optional[Dict] = None


# Allowed bypass reasons (no governance required)
# NOTE: Keep this minimal - Pantheon should approve most lifecycle events
ALLOWED_BYPASS_REASONS = {
    'minimum_population',  # Prevent extinction (pop < 3)
    'initial_population',  # Bootstrap on empty startup (pop == 0)
    'test_mode',          # Testing only
}
# REMOVED: 'zeus_initialization' - Zeus should use 'initial_population' like everyone else

# Emergency bypass reasons (require manual confirmation)
EMERGENCY_BYPASS_REASONS = {
    'emergency_recovery',  # System recovery
    'critical_failure',    # Catastrophic state
}


class PantheonGovernance:
    """
    Centralized governance for kernel lifecycle.
    
    All spawning, breeding, and death decisions must go through
    this governance layer. Gods vote on proposals, and audit
    trails are maintained in the database.
    """
    
    def __init__(self):
        """Initialize governance system."""
        self.proposals: Dict[str, LifecycleProposal] = {}
        self.proposal_counter = 0
        
        # Database connection
        self.db_url = os.environ.get('DATABASE_URL')
        self._conn = None
        
        if POSTGRES_AVAILABLE and self.db_url:
            try:
                self._conn = psycopg2.connect(self.db_url)
                self._conn.autocommit = True
                self._ensure_tables_exist()
                print("[PantheonGovernance] ⚖️ Governance system initialized with PostgreSQL")
            except Exception as e:
                print(f"[PantheonGovernance] ⚠️ Database unavailable ({e}), using in-memory only")
                self._conn = None
        else:
            print("[PantheonGovernance] ⚖️ Governance system initialized (in-memory mode)")
    
    def _emit_proposal_event(
        self,
        proposal: LifecycleProposal,
        action: str,
        actor: str
    ) -> None:
        """
        Emit a governance proposal event for visibility.
        
        Broadcasts to ActivityBroadcaster (UI) and CapabilityEventBus (internal routing).
        
        Args:
            proposal: The lifecycle proposal
            action: Action taken (approved, rejected, created)
            actor: Who performed the action
        """
        try:
            if ACTIVITY_BROADCASTER_AVAILABLE and get_broadcaster is not None:
                broadcaster = get_broadcaster()
                broadcaster.broadcast_message(
                    from_god="Governance",
                    to_god=None,
                    content=f"Proposal {proposal.proposal_id} {action}: {proposal.proposal_type.value} - {proposal.reason or 'No reason'}",
                    activity_type=ActivityType.SPAWN_PROPOSAL,
                    phi=proposal.parent_phi if proposal.parent_phi else 0.7,
                    kappa=64.21,
                    importance=0.8,
                    metadata={
                        'proposal_id': proposal.proposal_id,
                        'proposal_type': proposal.proposal_type.value,
                        'action': action,
                        'actor': actor,
                        'parent_id': proposal.parent_id,
                        'count': proposal.count,
                    }
                )
            
            if CAPABILITY_MESH_AVAILABLE and emit_event is not None:
                emit_event(
                    source=CapabilityType.KERNELS,
                    event_type=EventType.KERNEL_SPAWN,
                    content={
                        'proposal_id': proposal.proposal_id,
                        'proposal_type': proposal.proposal_type.value,
                        'action': action,
                        'actor': actor,
                        'reason': proposal.reason[:200] if proposal.reason else None,
                    },
                    phi=proposal.parent_phi if proposal.parent_phi else 0.7,
                    basin_coords=None,
                    priority=8
                )
                
        except Exception as e:
            print(f"[PantheonGovernance] Event emission failed: {e}")

    def _ensure_tables_exist(self):
        """Create required tables if they don't exist."""
        if not self._conn:
            return

        try:
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS governance_audit_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    details TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS governance_proposals (
                    id SERIAL PRIMARY KEY,
                    proposal_id VARCHAR(64) UNIQUE NOT NULL,
                    proposal_type VARCHAR(32) NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'pending',
                    reason TEXT,
                    parent_id VARCHAR(64),
                    parent_phi FLOAT,
                    count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    votes_for JSONB DEFAULT '{}',
                    votes_against JSONB DEFAULT '{}',
                    audit_log JSONB DEFAULT '[]'
                )
            """)
            cur.close()
        except Exception as e:
            print(f"[PantheonGovernance] Warning: Could not ensure tables exist: {e}")
    
    def check_spawn_permission(
        self,
        reason: str = "",
        parent_id: Optional[str] = None,
        parent_phi: Optional[float] = None,
        pantheon_approved: bool = False
    ) -> bool:
        """
        Check if spawning is allowed.
        
        Args:
            reason: Why spawning is needed
            parent_id: Parent kernel ID (if spawning from parent)
            parent_phi: Parent's Φ value
            pantheon_approved: Explicit Pantheon approval
            
        Returns:
            True if spawning is allowed
            
        Raises:
            PermissionError: If spawning not allowed
        """
        # Explicit approval
        if pantheon_approved:
            self._log_audit("spawn", "allowed", f"Explicit Pantheon approval (reason: {reason})")
            return True
        
        # Check for allowed bypass reasons
        if reason in ALLOWED_BYPASS_REASONS:
            self._log_audit("spawn", "auto_approved", f"Bypass reason: {reason}")
            print(f"[PantheonGovernance] ✅ Auto-approved spawn (reason: {reason})")
            return True
        
        # Check for emergency bypass (requires confirmation)
        if reason in EMERGENCY_BYPASS_REASONS:
            print(f"[PantheonGovernance] ⚠️ EMERGENCY: Spawn requested for reason: {reason}")
            print("[PantheonGovernance] Emergency spawns require manual confirmation!")
            self._log_audit("spawn", "emergency_pending", f"Emergency reason: {reason}")
            raise PermissionError(
                f"Emergency spawn reason '{reason}' requires manual confirmation. "
                f"Set pantheon_approved=True after review."
            )
        
        # Auto-approval for high-Φ parents
        if parent_phi is not None and parent_phi >= 0.7:
            self._log_audit("spawn", "auto_approved", f"High-Φ parent (Φ={parent_phi:.3f})")
            print(f"[PantheonGovernance] ✅ Auto-approved spawn from high-Φ parent (Φ={parent_phi:.3f})")
            return True
        
        # Create proposal for Pantheon voting
        proposal = self._create_proposal(
            ProposalType.SPAWN,
            reason=reason,
            parent_id=parent_id,
            parent_phi=parent_phi
        )
        
        print(f"[PantheonGovernance] 📋 Spawn proposal created: {proposal.proposal_id}")
        print(f"[PantheonGovernance] Reason: {reason}")
        if parent_id:
            print(f"[PantheonGovernance] Parent: {parent_id} (Φ={parent_phi:.3f if parent_phi else 0:.3f})")
        print("[PantheonGovernance] ⚠️ Waiting for Pantheon approval...")
        
        raise PermissionError(
            f"Spawning requires Pantheon approval. "
            f"Proposal ID: {proposal.proposal_id}. "
            f"Use approve_proposal() to vote."
        )
    
    def check_breed_permission(
        self,
        parent1_id: str,
        parent2_id: str,
        parent1_phi: float,
        parent2_phi: float,
        pantheon_approved: bool = False
    ) -> bool:
        """
        Check if breeding is allowed.
        
        Args:
            parent1_id: First parent ID
            parent2_id: Second parent ID
            parent1_phi: First parent's Φ
            parent2_phi: Second parent's Φ
            pantheon_approved: Explicit approval
            
        Returns:
            True if breeding is allowed
            
        Raises:
            PermissionError: If breeding not allowed
        """
        if pantheon_approved:
            self._log_audit("breed", "allowed", f"Explicit Pantheon approval ({parent1_id} × {parent2_id})")
            return True
        
        # Auto-approval for high-Φ parents
        avg_phi = (parent1_phi + parent2_phi) / 2
        if avg_phi >= 0.7:
            self._log_audit("breed", "auto_approved", f"High-Φ parents (avg Φ={avg_phi:.3f})")
            print(f"[PantheonGovernance] ✅ Auto-approved breeding (avg Φ={avg_phi:.3f})")
            return True
        
        # Create proposal
        proposal = self._create_proposal(
            ProposalType.BREED,
            reason=f"Breed {parent1_id} (Φ={parent1_phi:.3f}) × {parent2_id} (Φ={parent2_phi:.3f})",
            parent_id=parent1_id,
            parent_phi=parent1_phi
        )
        
        print(f"[PantheonGovernance] 📋 Breeding proposal created: {proposal.proposal_id}")
        print(f"[PantheonGovernance] Parents: {parent1_id} (Φ={parent1_phi:.3f}) × {parent2_id} (Φ={parent2_phi:.3f})")
        print("[PantheonGovernance] ⚠️ Waiting for Pantheon approval...")
        
        raise PermissionError(
            f"Breeding requires Pantheon approval. "
            f"Proposal ID: {proposal.proposal_id}. "
            f"Average parent Φ={avg_phi:.3f}"
        )
    
    def check_turbo_spawn_permission(
        self,
        count: int,
        pantheon_approved: bool = False,
        emergency_override: bool = False
    ) -> bool:
        """
        Check if turbo spawning is allowed.
        
        Turbo spawning creates many kernels at once and is
        very dangerous. Requires explicit approval.
        
        Args:
            count: Number of kernels to spawn
            pantheon_approved: Explicit approval
            emergency_override: Manual emergency override
            
        Returns:
            True if allowed
            
        Raises:
            PermissionError: If not allowed
        """
        if pantheon_approved:
            self._log_audit("turbo_spawn", "allowed", f"Explicit approval for {count} kernels")
            return True
        
        if emergency_override:
            print(f"[PantheonGovernance] ⚠️ EMERGENCY OVERRIDE: Turbo spawn {count} kernels")
            self._log_audit("turbo_spawn", "emergency_override", f"Manual override for {count} kernels")
            return True
        
        # Turbo spawn always requires approval
        proposal = self._create_proposal(
            ProposalType.TURBO_SPAWN,
            reason=f"Mass spawn {count} kernels",
            count=count
        )
        
        print(f"[PantheonGovernance] 🚨 TURBO SPAWN PROPOSAL: {proposal.proposal_id}")
        print(f"[PantheonGovernance] Requesting mass spawn of {count} kernels")
        print("[PantheonGovernance] ⚠️ This is potentially dangerous. Review carefully!")
        
        raise PermissionError(
            f"Turbo spawning {count} kernels requires Pantheon approval. "
            f"Proposal ID: {proposal.proposal_id}. "
            f"This is a mass spawning operation!"
        )

    def check_evolve_permission(
        self,
        kernel_id: str,
        mutation_type: str,
        kernel_phi: float = 0.0,
        pantheon_approved: bool = False
    ) -> bool:
        """
        Check if kernel evolution/mutation is allowed.

        Args:
            kernel_id: Kernel to evolve
            mutation_type: Type of mutation (e.g., 'gradient', 'random', 'targeted')
            kernel_phi: Current Phi value
            pantheon_approved: Explicit approval

        Returns:
            True if evolution allowed

        Raises:
            PermissionError: If evolution not allowed
        """
        if pantheon_approved:
            self._log_audit("evolve", "allowed", f"Explicit approval for {kernel_id}")
            return True

        # Auto-approve evolution for high-Phi kernels (they know what they're doing)
        if kernel_phi >= 0.7:
            self._log_audit("evolve", "auto_approved", f"High-Phi kernel {kernel_id} (Phi={kernel_phi:.3f})")
            print(f"[PantheonGovernance] Auto-approved evolution for high-Phi {kernel_id}")
            return True

        proposal = self._create_proposal(
            ProposalType.EVOLVE,
            reason=f"Evolution of {kernel_id} via {mutation_type}",
            parent_id=kernel_id,
            parent_phi=kernel_phi
        )

        print(f"[PantheonGovernance] Evolution proposal created: {proposal.proposal_id}")
        raise PermissionError(
            f"Evolution requires Pantheon approval. Proposal ID: {proposal.proposal_id}"
        )

    def check_merge_permission(
        self,
        kernel1_id: str,
        kernel2_id: str,
        kernel1_phi: float = 0.0,
        kernel2_phi: float = 0.0,
        pantheon_approved: bool = False
    ) -> bool:
        """
        Check if merging two kernels is allowed.

        Merging combines two kernels into one, destroying both originals.

        Args:
            kernel1_id: First kernel
            kernel2_id: Second kernel
            kernel1_phi: First kernel's Phi
            kernel2_phi: Second kernel's Phi
            pantheon_approved: Explicit approval

        Returns:
            True if merge allowed

        Raises:
            PermissionError: If merge not allowed
        """
        if pantheon_approved:
            self._log_audit("merge", "allowed", f"Explicit approval for {kernel1_id} + {kernel2_id}")
            return True

        # Auto-approve for high-Phi pairs
        avg_phi = (kernel1_phi + kernel2_phi) / 2
        if avg_phi >= 0.7:
            self._log_audit("merge", "auto_approved", f"High-Phi merge (avg={avg_phi:.3f})")
            print(f"[PantheonGovernance] Auto-approved merge (avg Phi={avg_phi:.3f})")
            return True

        proposal = self._create_proposal(
            ProposalType.MERGE,
            reason=f"Merge {kernel1_id} (Phi={kernel1_phi:.3f}) + {kernel2_id} (Phi={kernel2_phi:.3f})",
            parent_id=kernel1_id,
            parent_phi=kernel1_phi
        )

        print(f"[PantheonGovernance] Merge proposal created: {proposal.proposal_id}")
        raise PermissionError(
            f"Merge requires Pantheon approval. Proposal ID: {proposal.proposal_id}"
        )

    def check_cannibalize_permission(
        self,
        strong_id: str,
        weak_id: str,
        strong_phi: float = 0.0,
        weak_phi: float = 0.0,
        pantheon_approved: bool = False
    ) -> bool:
        """
        Check if cannibalization (absorption) is allowed.

        Cannibalization kills the weak kernel and absorbs its knowledge
        into the strong kernel.

        Args:
            strong_id: Absorbing kernel
            weak_id: Kernel to be absorbed (will die)
            strong_phi: Absorber's Phi
            weak_phi: Victim's Phi
            pantheon_approved: Explicit approval

        Returns:
            True if cannibalization allowed

        Raises:
            PermissionError: If cannibalization not allowed
        """
        if pantheon_approved:
            self._log_audit("cannibalize", "allowed", f"Explicit approval: {strong_id} absorbs {weak_id}")
            return True

        # Auto-approve if strong kernel has high Phi AND weak kernel is failing
        if strong_phi >= 0.7 and weak_phi < 0.3:
            self._log_audit(
                "cannibalize", "auto_approved",
                f"Strong {strong_id} (Phi={strong_phi:.3f}) absorbs weak {weak_id} (Phi={weak_phi:.3f})"
            )
            print(f"[PantheonGovernance] Auto-approved cannibalization (clear strength differential)")
            return True

        proposal = self._create_proposal(
            ProposalType.CANNIBALIZE,
            reason=f"{strong_id} (Phi={strong_phi:.3f}) absorbs {weak_id} (Phi={weak_phi:.3f})",
            parent_id=strong_id,
            parent_phi=strong_phi
        )

        print(f"[PantheonGovernance] Cannibalization proposal created: {proposal.proposal_id}")
        print(f"[PantheonGovernance] WARNING: {weak_id} will be terminated if approved")
        raise PermissionError(
            f"Cannibalization requires Pantheon approval. Proposal ID: {proposal.proposal_id}. "
            f"WARNING: This will terminate {weak_id}!"
        )

    def check_new_god_permission(
        self,
        kernel_id: str,
        proposed_name: str,
        kernel_phi: float = 0.0,
        achievements: Optional[List[str]] = None,
        pantheon_approved: bool = False
    ) -> bool:
        """
        Check if promoting a chaos kernel to god status is allowed.

        This is a major lifecycle event - creates a new god in the Pantheon.

        Args:
            kernel_id: Kernel to promote
            proposed_name: Proposed god name
            kernel_phi: Kernel's current Phi
            achievements: List of achievements justifying promotion
            pantheon_approved: Explicit approval

        Returns:
            True if promotion allowed

        Raises:
            PermissionError: If promotion not allowed
        """
        if pantheon_approved:
            self._log_audit("new_god", "allowed", f"Explicit approval: {kernel_id} -> {proposed_name}")
            return True

        # High bar for god promotion - Phi >= 0.8
        if kernel_phi >= 0.8:
            self._log_audit(
                "new_god", "auto_approved",
                f"Exceptional Phi ({kernel_phi:.3f}): {kernel_id} -> {proposed_name}"
            )
            print(f"[PantheonGovernance] Auto-approved god promotion (exceptional Phi={kernel_phi:.3f})")
            return True

        achievements_str = ", ".join(achievements or []) or "none listed"
        proposal = self._create_proposal(
            ProposalType.NEW_GOD,
            reason=f"Promote {kernel_id} to god '{proposed_name}'. Achievements: {achievements_str}",
            parent_id=kernel_id,
            parent_phi=kernel_phi
        )

        print(f"[PantheonGovernance] NEW GOD proposal created: {proposal.proposal_id}")
        print(f"[PantheonGovernance] {kernel_id} seeks to become '{proposed_name}'")
        raise PermissionError(
            f"God promotion requires Pantheon approval. Proposal ID: {proposal.proposal_id}"
        )

    def check_chaos_spawn_permission(
        self,
        reason: str,
        connectivity_target: Optional[str] = None,
        parent_phi: Optional[float] = None,
        pantheon_approved: bool = False
    ) -> bool:
        """
        Check if spawning a chaos (worker) kernel is allowed.

        Chaos kernels must have connectivity coupling and generative capability.

        Args:
            reason: Why the chaos kernel is needed
            connectivity_target: God or kernel this chaos kernel will connect to
            parent_phi: Parent's Phi (if spawned from parent)
            pantheon_approved: Explicit approval

        Returns:
            True if chaos spawn allowed

        Raises:
            PermissionError: If chaos spawn not allowed
        """
        if pantheon_approved:
            self._log_audit("chaos_spawn", "allowed", f"Explicit approval: {reason}")
            return True

        # Check for allowed bypass reasons
        if reason in ALLOWED_BYPASS_REASONS:
            self._log_audit("chaos_spawn", "auto_approved", f"Bypass reason: {reason}")
            return True

        # Auto-approve if parent has high Phi
        if parent_phi is not None and parent_phi >= 0.7:
            self._log_audit("chaos_spawn", "auto_approved", f"High-Phi parent: {parent_phi:.3f}")
            return True

        proposal = self._create_proposal(
            ProposalType.CHAOS_SPAWN,
            reason=f"Chaos kernel: {reason}. Connectivity: {connectivity_target or 'unassigned'}",
            parent_phi=parent_phi
        )

        print(f"[PantheonGovernance] Chaos spawn proposal created: {proposal.proposal_id}")
        raise PermissionError(
            f"Chaos kernel spawn requires Pantheon approval. Proposal ID: {proposal.proposal_id}"
        )

    def _create_proposal(
        self,
        proposal_type: ProposalType,
        reason: Optional[str] = None,
        parent_id: Optional[str] = None,
        parent_phi: Optional[float] = None,
        count: int = 1
    ) -> LifecycleProposal:
        """Create a new proposal."""
        self.proposal_counter += 1
        proposal_id = f"prop_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.proposal_counter}"
        
        proposal = LifecycleProposal(
            proposal_id=proposal_id,
            proposal_type=proposal_type,
            created_at=datetime.now(),
            reason=reason,
            parent_id=parent_id,
            parent_phi=parent_phi,
            count=count
        )
        
        self.proposals[proposal_id] = proposal
        self._persist_proposal(proposal)
        
        # 🔗 WIRE: Emit creation event for kernel visibility
        self._emit_proposal_event(proposal, 'created', 'system')
        
        return proposal
    
    def approve_proposal(self, proposal_id: str, approver: str = "system") -> Dict:
        """
        Approve a proposal.
        
        Args:
            proposal_id: Proposal to approve
            approver: Who is approving (god name or "system")
            
        Returns:
            Approval result
        """
        if proposal_id not in self.proposals:
            return {"success": False, "error": "Proposal not found"}
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.PENDING:
            return {"success": False, "error": f"Proposal already {proposal.status.value}"}
        
        proposal.votes_for.append(approver)
        proposal.status = ProposalStatus.APPROVED
        proposal.approved_by = approver
        proposal.audit_log.append({
            "action": "approved",
            "by": approver,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"[PantheonGovernance] ✅ Proposal {proposal_id} APPROVED by {approver}")
        print(f"[PantheonGovernance] Type: {proposal.proposal_type.value}")
        print(f"[PantheonGovernance] Reason: {proposal.reason}")
        
        self._persist_proposal(proposal)
        
        # 🔗 WIRE: Emit approval event for kernel visibility
        self._emit_proposal_event(proposal, 'approved', approver)
        
        return {"success": True, "proposal": proposal}
    
    def reject_proposal(self, proposal_id: str, rejector: str = "system") -> Dict:
        """Reject a proposal."""
        if proposal_id not in self.proposals:
            return {"success": False, "error": "Proposal not found"}
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != ProposalStatus.PENDING:
            return {"success": False, "error": f"Proposal already {proposal.status.value}"}
        
        proposal.votes_against.append(rejector)
        proposal.status = ProposalStatus.REJECTED
        proposal.rejected_by = rejector
        proposal.audit_log.append({
            "action": "rejected",
            "by": rejector,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"[PantheonGovernance] ❌ Proposal {proposal_id} REJECTED by {rejector}")
        
        self._persist_proposal(proposal)
        
        # 🔗 WIRE: Emit rejection event for kernel visibility
        self._emit_proposal_event(proposal, 'rejected', rejector)
        
        return {"success": True, "proposal": proposal}
    
    def get_pending_proposals(self) -> List[LifecycleProposal]:
        """Get all pending proposals."""
        return [
            p for p in self.proposals.values()
            if p.status == ProposalStatus.PENDING
        ]
    
    def _log_audit(self, action: str, status: str, details: str):
        """Log governance action to audit trail (full details, no truncation)."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {action.upper()} - {status}: {details}"
        print(f"[PantheonGovernance] {log_entry}")
        
        if self._conn:
            try:
                cur = self._conn.cursor()
                cur.execute("""
                    INSERT INTO governance_audit_log (timestamp, action, status, details)
                    VALUES (%s, %s, %s, %s)
                """, (timestamp, action, status, details))
                cur.close()
            except Exception as e:
                print(f"[PantheonGovernance] Failed to persist audit log: {e}")
    
    def _persist_proposal(self, proposal: LifecycleProposal):
        """Persist proposal to database."""
        if not self._conn:
            return
        
        try:
            cur = self._conn.cursor()
            cur.execute("""
                INSERT INTO governance_proposals (
                    proposal_id, proposal_type, status, reason,
                    parent_id, parent_phi, count, created_at,
                    votes_for, votes_against, audit_log
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (proposal_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    votes_for = EXCLUDED.votes_for,
                    votes_against = EXCLUDED.votes_against,
                    audit_log = EXCLUDED.audit_log
            """, (
                proposal.proposal_id,
                proposal.proposal_type.value,
                proposal.status.value,
                proposal.reason,
                proposal.parent_id,
                proposal.parent_phi,
                proposal.count,
                proposal.created_at,
                PsycopgJson(proposal.votes_for) if PsycopgJson else None,
                PsycopgJson(proposal.votes_against) if PsycopgJson else None,
                PsycopgJson(proposal.audit_log) if PsycopgJson else None,
            ))
            cur.close()
        except Exception as e:
            print(f"[PantheonGovernance] Failed to persist proposal: {e}")
    
    def close(self):
        """Close database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# Global governance instance
_governance = None


def get_governance() -> PantheonGovernance:
    """Get or create the global governance instance."""
    global _governance
    if _governance is None:
        _governance = PantheonGovernance()
    return _governance
