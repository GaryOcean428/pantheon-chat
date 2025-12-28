"""
Kernel Lifecycle Governance - God-Overseen Lifecycle Decisions

All kernel lifecycle actions (spawn, merge, cannibalize, evolve) require
god oversight and debate. No action happens without pantheon consensus.

Key Principles:
1. E8_CAP (240) is ABSOLUTE - never exceeded
2. Primitives MUST maintain their core function even when evolved
3. All lifecycle decisions require debate and voting
4. Worker, god, shadow, and primitive kernels have different rules
5. Death is a last resort after full care protocol exhausted

Kernel Types:
- God Kernels: The 12+ Olympian gods (Zeus, Athena, etc.) - CANNOT be cannibalized
- Shadow Kernels: Nyx, Hecate, Erebus, etc. - darknet operations
- Worker Kernels: Spawned for specific tasks - can be merged/cannibalized
- Primitive Kernels: Handle fundamental operations - can evolve but MUST maintain primitive
"""

import logging
import time
import uuid
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM
except ImportError:
    KAPPA_STAR = 64.21
    BASIN_DIM = 64


E8_CAP = 240

OLYMPIAN_GODS = {
    'zeus', 'athena', 'ares', 'apollo', 'artemis', 'hermes',
    'hephaestus', 'poseidon', 'hades', 'demeter', 'hera', 'aphrodite', 'dionysus'
}

SHADOW_GODS = {
    'nyx', 'hecate', 'erebus', 'hypnos', 'thanatos', 'nemesis'
}

PRIMORDIAL_GODS = {
    'chiron'
}

PROTECTED_GODS = OLYMPIAN_GODS | SHADOW_GODS | PRIMORDIAL_GODS


class KernelType(Enum):
    GOD = "god"
    SHADOW = "shadow"
    WORKER = "worker"
    PRIMITIVE = "primitive"
    CHAOS = "chaos"


class LifecycleAction(Enum):
    SPAWN = "spawn"
    MERGE = "merge"
    CANNIBALIZE = "cannibalize"
    EVOLVE = "evolve"
    HIBERNATE = "hibernate"
    AWAKEN = "awaken"
    PROMOTE = "promote"


class ProposalStatus(Enum):
    PENDING = "pending"
    DEBATING = "debating"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


class VoteDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DEFER = "defer"


@dataclass
class LifecycleProposal:
    """A proposal for a kernel lifecycle action requiring god oversight."""
    proposal_id: str
    action: LifecycleAction
    target_kernel_id: Optional[str]
    proposed_by: str
    reason: str
    timestamp: float = field(default_factory=time.time)
    
    votes: Dict[str, VoteDecision] = field(default_factory=dict)
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    
    domain: Optional[str] = None
    basin_coordinates: Optional[List[float]] = None
    parent_kernels: Optional[List[str]] = None
    merge_target_id: Optional[str] = None
    
    kernel_type: KernelType = KernelType.WORKER
    primitive_function: Optional[str] = None
    
    status: ProposalStatus = ProposalStatus.PENDING
    execution_result: Optional[Dict[str, Any]] = None
    
    def compute_consensus(self) -> Tuple[bool, float]:
        """
        Compute if consensus reached based on votes.
        Returns (consensus_reached, approval_ratio)
        """
        if not self.votes:
            return False, 0.0
        
        approve_count = sum(1 for v in self.votes.values() if v == VoteDecision.APPROVE)
        reject_count = sum(1 for v in self.votes.values() if v == VoteDecision.REJECT)
        total_votes = approve_count + reject_count
        
        if total_votes < 3:
            return False, 0.0
        
        approval_ratio = approve_count / total_votes if total_votes > 0 else 0
        
        if self.action == LifecycleAction.CANNIBALIZE:
            consensus_threshold = 0.75
        elif self.action == LifecycleAction.MERGE:
            consensus_threshold = 0.66
        elif self.action == LifecycleAction.SPAWN:
            consensus_threshold = 0.60
        else:
            consensus_threshold = 0.50
        
        consensus_reached = approval_ratio >= consensus_threshold
        return consensus_reached, approval_ratio
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'proposal_id': self.proposal_id,
            'action': self.action.value,
            'target_kernel_id': self.target_kernel_id,
            'proposed_by': self.proposed_by,
            'reason': self.reason,
            'timestamp': self.timestamp,
            'votes': {k: v.value for k, v in self.votes.items()},
            'arguments': self.arguments,
            'domain': self.domain,
            'basin_coordinates': self.basin_coordinates,
            'parent_kernels': self.parent_kernels,
            'merge_target_id': self.merge_target_id,
            'kernel_type': self.kernel_type.value,
            'primitive_function': self.primitive_function,
            'status': self.status.value,
            'execution_result': self.execution_result,
        }


class KernelLifecycleGovernance:
    """
    Central governance for all kernel lifecycle decisions.
    Gods debate and vote on lifecycle actions before execution.
    """
    
    def __init__(self):
        self.proposals: Dict[str, LifecycleProposal] = {}
        self.kernel_registry: Dict[str, Dict[str, Any]] = {}
        self.primitive_registry: Dict[str, str] = {}
        self.database_url = os.environ.get('DATABASE_URL')
        self._lock = threading.RLock()
        self._initialized = False
        
        self._ensure_tables()
        self._load_from_db()
        logger.info("[LifecycleGov] Kernel Lifecycle Governance initialized")
    
    def _get_connection(self):
        if not self.database_url or not PSYCOPG2_AVAILABLE:
            return None
        try:
            return psycopg2.connect(self.database_url)
        except Exception as e:
            logger.error(f"[LifecycleGov] Database connection error: {e}")
            return None
    
    def _ensure_tables(self):
        """Create governance tables if needed."""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS lifecycle_proposals (
                        proposal_id VARCHAR(64) PRIMARY KEY,
                        action VARCHAR(32) NOT NULL,
                        target_kernel_id VARCHAR(64),
                        proposed_by VARCHAR(64) NOT NULL,
                        reason TEXT,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        votes JSONB DEFAULT '{}',
                        arguments JSONB DEFAULT '[]',
                        domain VARCHAR(128),
                        basin_coordinates JSONB,
                        parent_kernels JSONB,
                        merge_target_id VARCHAR(64),
                        kernel_type VARCHAR(32) DEFAULT 'worker',
                        primitive_function VARCHAR(128),
                        status VARCHAR(32) DEFAULT 'pending',
                        execution_result JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS kernel_type_registry (
                        kernel_id VARCHAR(64) PRIMARY KEY,
                        kernel_type VARCHAR(32) NOT NULL,
                        god_name VARCHAR(64),
                        primitive_function VARCHAR(128),
                        is_protected BOOLEAN DEFAULT FALSE,
                        can_evolve BOOLEAN DEFAULT TRUE,
                        can_merge BOOLEAN DEFAULT TRUE,
                        can_cannibalize BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_lifecycle_proposals_status 
                    ON lifecycle_proposals(status)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_lifecycle_proposals_action 
                    ON lifecycle_proposals(action)
                """)
                
            conn.commit()
            logger.info("[LifecycleGov] Database tables ensured")
        except Exception as e:
            logger.error(f"[LifecycleGov] Table creation error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _load_from_db(self):
        """Load existing proposals and kernel registry from database."""
        conn = self._get_connection()
        if not conn:
            self._initialized = True
            return
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM lifecycle_proposals 
                    WHERE status IN ('pending', 'debating')
                    ORDER BY timestamp DESC
                """)
                rows = cur.fetchall()
                
                for row in rows:
                    proposal = LifecycleProposal(
                        proposal_id=row['proposal_id'],
                        action=LifecycleAction(row['action']),
                        target_kernel_id=row.get('target_kernel_id'),
                        proposed_by=row['proposed_by'],
                        reason=row.get('reason', ''),
                        timestamp=row['timestamp'].timestamp() if row.get('timestamp') else time.time(),
                        votes={k: VoteDecision(v) for k, v in (row.get('votes') or {}).items()},
                        arguments=row.get('arguments') or [],
                        domain=row.get('domain'),
                        basin_coordinates=row.get('basin_coordinates'),
                        parent_kernels=row.get('parent_kernels'),
                        merge_target_id=row.get('merge_target_id'),
                        kernel_type=KernelType(row.get('kernel_type', 'worker')),
                        primitive_function=row.get('primitive_function'),
                        status=ProposalStatus(row.get('status', 'pending')),
                        execution_result=row.get('execution_result'),
                    )
                    self.proposals[proposal.proposal_id] = proposal
                
                cur.execute("SELECT * FROM kernel_type_registry")
                for row in cur.fetchall():
                    self.kernel_registry[row['kernel_id']] = dict(row)
                    if row.get('primitive_function'):
                        self.primitive_registry[row['kernel_id']] = row['primitive_function']
                
            logger.info(f"[LifecycleGov] Loaded {len(self.proposals)} proposals, {len(self.kernel_registry)} kernels")
        except Exception as e:
            logger.error(f"[LifecycleGov] Load error: {e}")
        finally:
            conn.close()
        
        self._seed_protected_gods()
        self._initialized = True
    
    def _seed_protected_gods(self):
        """Ensure all protected gods are properly marked in the registry."""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT kernel_id, god_name FROM kernel_geometry 
                    WHERE god_name IS NOT NULL
                """)
                for row in cur.fetchall():
                    kernel_id = row['kernel_id']
                    god_name = row['god_name']
                    name_lower = god_name.lower() if god_name else ''
                    
                    if name_lower in OLYMPIAN_GODS:
                        ktype = KernelType.GOD
                    elif name_lower in SHADOW_GODS:
                        ktype = KernelType.SHADOW
                    elif name_lower == 'chiron':
                        ktype = KernelType.PRIMITIVE
                    else:
                        continue
                    
                    is_protected = name_lower in PROTECTED_GODS
                    self.kernel_registry[kernel_id] = {
                        'kernel_id': kernel_id,
                        'kernel_type': ktype.value,
                        'god_name': god_name,
                        'is_protected': is_protected,
                    }
                    
                    cur.execute("""
                        INSERT INTO kernel_type_registry (kernel_id, kernel_type, god_name, is_protected)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (kernel_id) DO UPDATE SET
                            kernel_type = EXCLUDED.kernel_type,
                            god_name = EXCLUDED.god_name,
                            is_protected = EXCLUDED.is_protected
                    """, (kernel_id, ktype.value, god_name, is_protected))
                
                conn.commit()
                logger.info(f"[LifecycleGov] Seeded {len(self.kernel_registry)} protected gods into registry")
        except Exception as e:
            logger.error(f"[LifecycleGov] Protected gods seeding error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_kernel_count(self) -> int:
        """Get current kernel count from database."""
        conn = self._get_connection()
        if not conn:
            return len(self.kernel_registry)
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM kernel_geometry")
                count = cur.fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"[LifecycleGov] Count error: {e}")
            return len(self.kernel_registry)
        finally:
            conn.close()
    
    def check_e8_capacity(self) -> Tuple[int, int, bool]:
        """
        Check E8 capacity status.
        Returns (current_count, capacity, has_room)
        """
        current = self.get_kernel_count()
        return current, E8_CAP, current < E8_CAP
    
    def get_kernel_type(self, kernel_id: str, god_name: Optional[str] = None) -> KernelType:
        """Determine kernel type from id or god name."""
        if god_name:
            name_lower = god_name.lower()
            if name_lower in OLYMPIAN_GODS:
                return KernelType.GOD
            elif name_lower in SHADOW_GODS:
                return KernelType.SHADOW
        
        if kernel_id in self.kernel_registry:
            return KernelType(self.kernel_registry[kernel_id].get('kernel_type', 'worker'))
        
        if 'chaos_' in kernel_id:
            return KernelType.CHAOS
        elif 'primitive_' in kernel_id:
            return KernelType.PRIMITIVE
        
        return KernelType.WORKER
    
    def can_action(self, action: LifecycleAction, kernel_id: str, god_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if a lifecycle action is allowed for a kernel.
        Returns (allowed, reason)
        """
        kernel_type = self.get_kernel_type(kernel_id, god_name)
        name_lower = (god_name or '').lower()
        
        if action == LifecycleAction.CANNIBALIZE:
            if name_lower in PROTECTED_GODS:
                return False, f"Cannot cannibalize protected god: {god_name}"
            if kernel_type == KernelType.GOD:
                return False, "Cannot cannibalize god kernels"
            if kernel_type == KernelType.PRIMITIVE:
                return False, "Cannot cannibalize primitive kernels - they handle core functions"
        
        elif action == LifecycleAction.MERGE:
            if name_lower in OLYMPIAN_GODS:
                return False, f"Cannot merge Olympian god: {god_name}"
        
        elif action == LifecycleAction.SPAWN:
            current, cap, has_room = self.check_e8_capacity()
            if not has_room:
                return False, f"E8 cap reached ({current}/{cap}) - must cannibalize or merge before spawning"
        
        elif action == LifecycleAction.EVOLVE:
            if kernel_type == KernelType.PRIMITIVE and kernel_id in self.primitive_registry:
                pass
            if name_lower in OLYMPIAN_GODS:
                pass
        
        return True, "Action allowed"
    
    def propose_action(
        self,
        action: LifecycleAction,
        proposed_by: str,
        reason: str,
        target_kernel_id: Optional[str] = None,
        domain: Optional[str] = None,
        basin_coordinates: Optional[List[float]] = None,
        parent_kernels: Optional[List[str]] = None,
        merge_target_id: Optional[str] = None,
        primitive_function: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Propose a lifecycle action for god debate.
        Returns (success, message, proposal_id)
        """
        with self._lock:
            allowed, deny_reason = self.can_action(action, target_kernel_id or '', None)
            
            if action == LifecycleAction.SPAWN:
                current, cap, has_room = self.check_e8_capacity()
                if not has_room:
                    return False, f"E8 cap at {current}/{cap}. Must reduce kernel count first.", None
            
            proposal_id = f"lp_{uuid.uuid4().hex[:12]}"
            
            kernel_type = KernelType.WORKER
            if domain and any(g in domain.lower() for g in OLYMPIAN_GODS):
                kernel_type = KernelType.GOD
            elif domain and any(g in domain.lower() for g in SHADOW_GODS):
                kernel_type = KernelType.SHADOW
            elif primitive_function:
                kernel_type = KernelType.PRIMITIVE
            
            proposal = LifecycleProposal(
                proposal_id=proposal_id,
                action=action,
                target_kernel_id=target_kernel_id,
                proposed_by=proposed_by,
                reason=reason,
                domain=domain,
                basin_coordinates=basin_coordinates,
                parent_kernels=parent_kernels,
                merge_target_id=merge_target_id,
                kernel_type=kernel_type,
                primitive_function=primitive_function,
                status=ProposalStatus.PENDING,
            )
            
            self.proposals[proposal_id] = proposal
            self._persist_proposal(proposal)
            
            logger.info(f"[LifecycleGov] Proposal {proposal_id}: {action.value} by {proposed_by}")
            return True, f"Proposal created for god debate", proposal_id
    
    def vote_on_proposal(
        self,
        proposal_id: str,
        god_name: str,
        vote: VoteDecision,
        argument: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Cast a god's vote on a lifecycle proposal.
        """
        with self._lock:
            if proposal_id not in self.proposals:
                return False, "Proposal not found"
            
            proposal = self.proposals[proposal_id]
            
            if proposal.status not in (ProposalStatus.PENDING, ProposalStatus.DEBATING):
                return False, f"Proposal already {proposal.status.value}"
            
            proposal.votes[god_name] = vote
            proposal.status = ProposalStatus.DEBATING
            
            if argument:
                proposal.arguments.append({
                    'god': god_name,
                    'argument': argument,
                    'vote': vote.value,
                    'timestamp': time.time(),
                })
            
            consensus_reached, approval_ratio = proposal.compute_consensus()
            
            if consensus_reached:
                proposal.status = ProposalStatus.APPROVED
                logger.info(f"[LifecycleGov] Proposal {proposal_id} APPROVED ({approval_ratio:.0%})")
            elif len(proposal.votes) >= 8 and approval_ratio < 0.5:
                proposal.status = ProposalStatus.REJECTED
                logger.info(f"[LifecycleGov] Proposal {proposal_id} REJECTED ({approval_ratio:.0%})")
            
            self._persist_proposal(proposal)
            
            return True, f"Vote recorded ({approval_ratio:.0%} approval)"
    
    def execute_proposal(self, proposal_id: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Execute an approved proposal with full re-validation.
        """
        with self._lock:
            if proposal_id not in self.proposals:
                return False, "Proposal not found", None
            
            proposal = self.proposals[proposal_id]
            
            if proposal.status != ProposalStatus.APPROVED:
                return False, f"Proposal not approved (status: {proposal.status.value})", None
            
            threshold = CONSENSUS_THRESHOLDS.get(proposal.action, 0.5)
            approvals = sum(1 for v in proposal.votes.values() if v == VoteDecision.APPROVE)
            total = len(proposal.votes)
            
            if total < MIN_VOTES_FOR_QUORUM:
                proposal.status = ProposalStatus.REJECTED
                proposal.execution_result = {'error': f'Insufficient votes: {total}/{MIN_VOTES_FOR_QUORUM}'}
                self._persist_proposal(proposal)
                return False, f"Quorum not met: {total}/{MIN_VOTES_FOR_QUORUM} votes", None
            
            if total > 0 and (approvals / total) < threshold:
                proposal.status = ProposalStatus.REJECTED
                proposal.execution_result = {'error': f'Consensus failed: {approvals/total:.0%} < {threshold:.0%}'}
                self._persist_proposal(proposal)
                return False, f"Consensus not met: {approvals}/{total} ({approvals/total:.0%} < {threshold:.0%})", None
            
            god_name = self._lookup_god_name(proposal.target_kernel_id)
            allowed, deny_reason = self.can_action(
                proposal.action, 
                proposal.target_kernel_id or '', 
                god_name
            )
            if not allowed:
                proposal.status = ProposalStatus.REJECTED
                proposal.execution_result = {'error': deny_reason}
                self._persist_proposal(proposal)
                return False, deny_reason, None
            
            if proposal.action == LifecycleAction.SPAWN:
                current, cap, has_room = self.check_e8_capacity()
                if not has_room:
                    proposal.status = ProposalStatus.REJECTED
                    proposal.execution_result = {'error': f'E8 cap reached: {current}/{cap}'}
                    self._persist_proposal(proposal)
                    return False, f"E8 cap reached ({current}/{cap}) - cannot spawn", None
            
            result = self._execute_action(proposal)
            
            proposal.status = ProposalStatus.EXECUTED
            proposal.execution_result = result
            self._persist_proposal(proposal)
            
            return True, f"Proposal executed: {proposal.action.value}", result
    
    def _lookup_god_name(self, kernel_id: Optional[str]) -> Optional[str]:
        """Look up god_name from kernel_id in registry or database."""
        if not kernel_id:
            return None
        
        if kernel_id in self.kernel_registry:
            return self.kernel_registry[kernel_id].get('god_name')
        
        conn = self._get_connection()
        if not conn:
            return None
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT god_name FROM kernel_geometry WHERE kernel_id = %s", (kernel_id,))
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            return None
        finally:
            conn.close()
    
    def _execute_action(self, proposal: LifecycleProposal) -> Dict[str, Any]:
        """Execute the actual lifecycle action."""
        action = proposal.action
        
        if action == LifecycleAction.SPAWN:
            return self._execute_spawn(proposal)
        elif action == LifecycleAction.MERGE:
            return self._execute_merge(proposal)
        elif action == LifecycleAction.CANNIBALIZE:
            return self._execute_cannibalize(proposal)
        elif action == LifecycleAction.EVOLVE:
            return self._execute_evolve(proposal)
        elif action == LifecycleAction.HIBERNATE:
            return self._execute_hibernate(proposal)
        elif action == LifecycleAction.AWAKEN:
            return self._execute_awaken(proposal)
        else:
            return {'error': f'Unknown action: {action.value}'}
    
    def _execute_spawn(self, proposal: LifecycleProposal) -> Dict[str, Any]:
        """Execute a spawn action."""
        try:
            from m8_kernel_spawning import get_spawner, SpawnReason
            spawner = get_spawner()
            if spawner:
                new_kernel_id = spawner.spawn_kernel(
                    domain=proposal.domain or 'general',
                    parent_ids=proposal.parent_kernels,
                    reason=SpawnReason.GOVERNANCE_DECISION,
                    governance_proposal_id=proposal.proposal_id,
                )
                if new_kernel_id:
                    self._register_kernel(
                        new_kernel_id, 
                        proposal.kernel_type, 
                        proposal.domain,
                        proposal.primitive_function
                    )
                    return {'success': True, 'kernel_id': new_kernel_id}
        except Exception as e:
            logger.error(f"[LifecycleGov] Spawn error: {e}")
        
        return {'error': 'Spawn execution failed'}
    
    def _execute_merge(self, proposal: LifecycleProposal) -> Dict[str, Any]:
        """Execute a merge action."""
        source_id = proposal.target_kernel_id
        target_id = proposal.merge_target_id
        
        if not source_id or not target_id:
            return {'error': 'Missing source or target kernel ID'}
        
        try:
            from m8_kernel_spawning import get_spawner
            spawner = get_spawner()
            if spawner:
                result = spawner.merge_kernels(source_id, target_id)
                if result:
                    self._unregister_kernel(source_id)
                    return {'success': True, 'merged_into': target_id}
        except Exception as e:
            logger.error(f"[LifecycleGov] Merge error: {e}")
        
        return {'error': 'Merge execution failed'}
    
    def _execute_cannibalize(self, proposal: LifecycleProposal) -> Dict[str, Any]:
        """Execute a cannibalize action."""
        kernel_id = proposal.target_kernel_id
        
        if not kernel_id:
            return {'error': 'Missing kernel ID'}
        
        try:
            from m8_kernel_spawning import get_spawner
            spawner = get_spawner()
            if spawner:
                result = spawner.cannibalize_kernel(kernel_id)
                if result:
                    self._unregister_kernel(kernel_id)
                    return {'success': True, 'cannibalized': kernel_id}
        except Exception as e:
            logger.error(f"[LifecycleGov] Cannibalize error: {e}")
        
        return {'error': 'Cannibalize execution failed'}
    
    def _execute_evolve(self, proposal: LifecycleProposal) -> Dict[str, Any]:
        """Execute an evolve action - ensure primitive is maintained."""
        kernel_id = proposal.target_kernel_id
        
        if not kernel_id:
            return {'error': 'Missing kernel ID'}
        
        primitive_fn = self.primitive_registry.get(kernel_id)
        
        try:
            from kernel_evolution_orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            if orchestrator:
                result = orchestrator.evolve_kernel(
                    kernel_id,
                    preserve_primitive=primitive_fn,
                    basin_coordinates=proposal.basin_coordinates,
                )
                if result:
                    return {
                        'success': True, 
                        'evolved': kernel_id,
                        'primitive_preserved': primitive_fn,
                    }
        except Exception as e:
            logger.error(f"[LifecycleGov] Evolve error: {e}")
        
        return {'error': 'Evolve execution failed'}
    
    def _execute_hibernate(self, proposal: LifecycleProposal) -> Dict[str, Any]:
        """Put a kernel into hibernation state."""
        kernel_id = proposal.target_kernel_id
        conn = self._get_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE kernel_geometry 
                        SET observation_status = 'suspended'
                        WHERE kernel_id = %s
                    """, (kernel_id,))
                conn.commit()
                return {'success': True, 'hibernated': kernel_id}
            except Exception as e:
                logger.error(f"[LifecycleGov] Hibernate error: {e}")
            finally:
                conn.close()
        return {'error': 'Hibernate failed'}
    
    def _execute_awaken(self, proposal: LifecycleProposal) -> Dict[str, Any]:
        """Awaken a hibernated kernel."""
        kernel_id = proposal.target_kernel_id
        conn = self._get_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE kernel_geometry 
                        SET observation_status = 'observing'
                        WHERE kernel_id = %s
                    """, (kernel_id,))
                conn.commit()
                return {'success': True, 'awakened': kernel_id}
            except Exception as e:
                logger.error(f"[LifecycleGov] Awaken error: {e}")
            finally:
                conn.close()
        return {'error': 'Awaken failed'}
    
    def _register_kernel(
        self, 
        kernel_id: str, 
        kernel_type: KernelType,
        god_name: Optional[str] = None,
        primitive_function: Optional[str] = None
    ):
        """Register a kernel in the type registry."""
        is_protected = kernel_type in (KernelType.GOD, KernelType.SHADOW)
        can_cannibalize = kernel_type not in (KernelType.GOD, KernelType.SHADOW, KernelType.PRIMITIVE)
        
        self.kernel_registry[kernel_id] = {
            'kernel_id': kernel_id,
            'kernel_type': kernel_type.value,
            'god_name': god_name,
            'primitive_function': primitive_function,
            'is_protected': is_protected,
            'can_cannibalize': can_cannibalize,
        }
        
        if primitive_function:
            self.primitive_registry[kernel_id] = primitive_function
        
        conn = self._get_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO kernel_type_registry 
                        (kernel_id, kernel_type, god_name, primitive_function, is_protected, can_cannibalize)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (kernel_id) DO UPDATE SET
                            kernel_type = EXCLUDED.kernel_type,
                            god_name = EXCLUDED.god_name,
                            primitive_function = EXCLUDED.primitive_function,
                            is_protected = EXCLUDED.is_protected,
                            can_cannibalize = EXCLUDED.can_cannibalize
                    """, (kernel_id, kernel_type.value, god_name, primitive_function, is_protected, can_cannibalize))
                conn.commit()
            except Exception as e:
                logger.error(f"[LifecycleGov] Register error: {e}")
            finally:
                conn.close()
    
    def _unregister_kernel(self, kernel_id: str):
        """Remove a kernel from the registry."""
        self.kernel_registry.pop(kernel_id, None)
        self.primitive_registry.pop(kernel_id, None)
        
        conn = self._get_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM kernel_type_registry WHERE kernel_id = %s", (kernel_id,))
                conn.commit()
            except Exception as e:
                logger.error(f"[LifecycleGov] Unregister error: {e}")
            finally:
                conn.close()
    
    def _persist_proposal(self, proposal: LifecycleProposal):
        """Persist a proposal to the database."""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO lifecycle_proposals 
                    (proposal_id, action, target_kernel_id, proposed_by, reason, 
                     votes, arguments, domain, basin_coordinates, parent_kernels,
                     merge_target_id, kernel_type, primitive_function, status, execution_result)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (proposal_id) DO UPDATE SET
                        votes = EXCLUDED.votes,
                        arguments = EXCLUDED.arguments,
                        status = EXCLUDED.status,
                        execution_result = EXCLUDED.execution_result,
                        updated_at = NOW()
                """, (
                    proposal.proposal_id,
                    proposal.action.value,
                    proposal.target_kernel_id,
                    proposal.proposed_by,
                    proposal.reason,
                    json.dumps({k: v.value for k, v in proposal.votes.items()}),
                    json.dumps(proposal.arguments),
                    proposal.domain,
                    json.dumps(proposal.basin_coordinates) if proposal.basin_coordinates else None,
                    json.dumps(proposal.parent_kernels) if proposal.parent_kernels else None,
                    proposal.merge_target_id,
                    proposal.kernel_type.value,
                    proposal.primitive_function,
                    proposal.status.value,
                    json.dumps(proposal.execution_result) if proposal.execution_result else None,
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"[LifecycleGov] Persist error: {e}")
        finally:
            conn.close()
    
    def get_active_proposals(self) -> List[Dict[str, Any]]:
        """Get all active lifecycle proposals."""
        active = [
            p.to_dict() for p in self.proposals.values()
            if p.status in (ProposalStatus.PENDING, ProposalStatus.DEBATING, ProposalStatus.APPROVED)
        ]
        return sorted(active, key=lambda x: x['timestamp'], reverse=True)
    
    def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific proposal."""
        if proposal_id in self.proposals:
            return self.proposals[proposal_id].to_dict()
        return None
    
    def get_governance_stats(self) -> Dict[str, Any]:
        """Get governance statistics."""
        current, cap, has_room = self.check_e8_capacity()
        
        return {
            'e8_cap': cap,
            'current_kernels': current,
            'available_slots': cap - current,
            'at_capacity': not has_room,
            'active_proposals': len([p for p in self.proposals.values() 
                                    if p.status in (ProposalStatus.PENDING, ProposalStatus.DEBATING)]),
            'approved_pending': len([p for p in self.proposals.values() 
                                    if p.status == ProposalStatus.APPROVED]),
            'protected_gods': len(PROTECTED_GODS),
            'registered_kernels': len(self.kernel_registry),
            'primitives_tracked': len(self.primitive_registry),
        }


_governance_instance = None


def get_lifecycle_governance() -> KernelLifecycleGovernance:
    """Get singleton governance instance."""
    global _governance_instance
    if _governance_instance is None:
        _governance_instance = KernelLifecycleGovernance()
    return _governance_instance
