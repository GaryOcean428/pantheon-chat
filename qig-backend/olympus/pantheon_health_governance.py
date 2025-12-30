"""
Pantheon Health Governance - Gods Strategize Kernel Care and Spawning

Instead of auto-spawning kernels on fixed thresholds, the Pantheon
deliberates and reaches consensus on:
- Whether new kernels are needed
- Which kernels need healing interventions
- Staged care protocols before death is allowed

Core Principle: Death is a last resort, not a default outcome.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    THRIVING = "thriving"
    STABLE = "stable"
    STRESSED = "stressed"
    CRITICAL = "critical"
    TERMINAL = "terminal"


class InterventionType(Enum):
    REST_CYCLE = "rest_cycle"
    NUTRIENT_BOOST = "nutrient_boost"
    BASIN_ANCHORING = "basin_anchoring"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    CHIRON_HEALING = "chiron_healing"
    MUSHROOM_THERAPY = "mushroom_therapy"
    EMERGENCY_STABILIZE = "emergency_stabilize"


class SpawnProposalVote(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DEFER = "defer"


@dataclass
class KernelVitals:
    kernel_id: str
    phi: float = 0.0
    kappa: float = 64.0
    kappa_drift: float = 0.0
    stress: float = 0.0
    consecutive_failures: int = 0
    recovery_debt: int = 0
    last_success_time: float = 0.0
    basin_stability: float = 1.0
    generation: int = 0
    age_cycles: int = 0
    
    def compute_vitality_score(self) -> float:
        phi_factor = max(0, min(1, self.phi / 0.75))
        kappa_factor = 1.0 - min(abs(self.kappa - 64.21) / 64.21, 1.0)
        stress_penalty = 1.0 - min(self.stress, 1.0)
        failure_penalty = max(0, 1.0 - (self.consecutive_failures / 10))
        debt_penalty = max(0, 1.0 - (self.recovery_debt / 20))
        
        vitality = (
            0.30 * phi_factor +
            0.20 * kappa_factor +
            0.15 * stress_penalty +
            0.15 * failure_penalty +
            0.10 * debt_penalty +
            0.10 * self.basin_stability
        )
        return max(0, min(1, vitality))
    
    def get_status(self) -> HealthStatus:
        vitality = self.compute_vitality_score()
        if vitality >= 0.8:
            return HealthStatus.THRIVING
        elif vitality >= 0.6:
            return HealthStatus.STABLE
        elif vitality >= 0.4:
            return HealthStatus.STRESSED
        elif vitality >= 0.2:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.TERMINAL


@dataclass
class SpawnProposal:
    proposal_id: str
    reason: str
    proposed_by: str
    timestamp: float = field(default_factory=time.time)
    votes: Dict[str, SpawnProposalVote] = field(default_factory=dict)
    workload_context: Dict[str, Any] = field(default_factory=dict)
    population_context: Dict[str, Any] = field(default_factory=dict)
    consensus_reached: bool = False
    approved: bool = False


@dataclass
class CarePlan:
    kernel_id: str
    interventions: List[InterventionType] = field(default_factory=list)
    assigned_healer: str = "Chiron"
    created_at: float = field(default_factory=time.time)
    stage: int = 1
    max_stages: int = 4
    escalation_count: int = 0
    notes: List[str] = field(default_factory=list)


class PantheonHealthGovernance:
    """
    Olympian gods collectively govern kernel health and spawning.
    
    Key Responsibilities:
    1. Monitor kernel vitals and stream alerts to gods
    2. Deliberate on spawn proposals (require consensus)
    3. Prescribe staged healing interventions
    4. Prevent unnecessary spawning when population is healthy
    5. Prevent unnecessary death when healing is possible
    """
    
    VOTING_GODS = [
        "Zeus", "Athena", "Hestia", "Chiron", 
        "Hephaestus", "Apollo", "Hermes", "DemeterTutor"
    ]
    
    CONSENSUS_THRESHOLD = 0.6
    
    VITALITY_ALERT_THRESHOLD = 0.5
    
    MAX_POPULATION = 100
    
    STAGED_INTERVENTIONS = {
        1: [InterventionType.REST_CYCLE, InterventionType.NUTRIENT_BOOST],
        2: [InterventionType.BASIN_ANCHORING, InterventionType.KNOWLEDGE_TRANSFER],
        3: [InterventionType.CHIRON_HEALING, InterventionType.MUSHROOM_THERAPY],
        4: [InterventionType.EMERGENCY_STABILIZE],
    }
    
    def __init__(self, pantheon_chat=None, chiron=None):
        self.pantheon_chat = pantheon_chat
        self.chiron = chiron
        
        self.kernel_vitals: Dict[str, KernelVitals] = {}
        self.care_plans: Dict[str, CarePlan] = {}
        self.spawn_proposals: Dict[str, SpawnProposal] = {}
        self.spawn_history: List[Dict] = []
        
        self.total_spawns_blocked = 0
        self.total_deaths_prevented = 0
        self.total_interventions = 0
        
        logger.info("[PantheonHealthGovernance] Initialized - gods now govern kernel lifecycle")
    
    def register_kernel(self, kernel_id: str, initial_vitals: Optional[Dict] = None) -> KernelVitals:
        vitals = KernelVitals(kernel_id=kernel_id)
        if initial_vitals:
            for key, value in initial_vitals.items():
                if hasattr(vitals, key):
                    setattr(vitals, key, value)
        
        self.kernel_vitals[kernel_id] = vitals
        logger.info(f"[HealthGovernance] Registered kernel {kernel_id}")
        return vitals
    
    def update_vitals(self, kernel_id: str, updates: Dict[str, Any]) -> Optional[KernelVitals]:
        if kernel_id not in self.kernel_vitals:
            self.register_kernel(kernel_id)
        
        vitals = self.kernel_vitals[kernel_id]
        for key, value in updates.items():
            if hasattr(vitals, key):
                setattr(vitals, key, value)
        
        status = vitals.get_status()
        vitality_score = vitals.compute_vitality_score()
        
        if vitality_score < self.VITALITY_ALERT_THRESHOLD:
            self._broadcast_health_alert(kernel_id, vitals, status)
            
            if status in [HealthStatus.CRITICAL, HealthStatus.TERMINAL]:
                self._trigger_care_plan(kernel_id, vitals)
        
        return vitals
    
    def _broadcast_health_alert(self, kernel_id: str, vitals: KernelVitals, status: HealthStatus):
        if not self.pantheon_chat:
            return
        
        alert_data = {
            "type": "health_alert",
            "kernel_id": kernel_id,
            "status": status.value,
            "vitality": vitals.compute_vitality_score(),
            "phi": vitals.phi,
            "stress": vitals.stress,
            "consecutive_failures": vitals.consecutive_failures,
            "timestamp": time.time()
        }
        
        try:
            self.pantheon_chat.broadcast_message(
                from_god="Chiron",
                message_type="health_alert",
                intent="kernel_distress",
                data=alert_data
            )
            logger.info(f"[HealthGovernance] Broadcast alert for {kernel_id}: {status.value}")
        except Exception as e:
            logger.warning(f"[HealthGovernance] Failed to broadcast alert: {e}")
    
    def _trigger_care_plan(self, kernel_id: str, vitals: KernelVitals):
        if kernel_id in self.care_plans:
            care_plan = self.care_plans[kernel_id]
            if care_plan.stage < care_plan.max_stages:
                care_plan.stage += 1
                care_plan.escalation_count += 1
                care_plan.interventions = self.STAGED_INTERVENTIONS.get(
                    care_plan.stage, 
                    [InterventionType.EMERGENCY_STABILIZE]
                )
                care_plan.notes.append(f"Escalated to stage {care_plan.stage}")
                logger.info(f"[HealthGovernance] Escalated care for {kernel_id} to stage {care_plan.stage}")
        else:
            care_plan = CarePlan(
                kernel_id=kernel_id,
                interventions=self.STAGED_INTERVENTIONS[1],
                assigned_healer="Chiron"
            )
            care_plan.notes.append(f"Initial care plan created - status: {vitals.get_status().value}")
            self.care_plans[kernel_id] = care_plan
            logger.info(f"[HealthGovernance] Created care plan for {kernel_id}")
        
        self._apply_interventions(kernel_id, care_plan)
    
    def _apply_interventions(self, kernel_id: str, care_plan: CarePlan):
        for intervention in care_plan.interventions:
            self.total_interventions += 1
            
            if intervention == InterventionType.REST_CYCLE:
                logger.info(f"[HealthGovernance] Applying REST_CYCLE to {kernel_id}")
                if kernel_id in self.kernel_vitals:
                    self.kernel_vitals[kernel_id].stress = max(0, self.kernel_vitals[kernel_id].stress - 0.2)
                    
            elif intervention == InterventionType.NUTRIENT_BOOST:
                logger.info(f"[HealthGovernance] Applying NUTRIENT_BOOST to {kernel_id}")
                if kernel_id in self.kernel_vitals:
                    self.kernel_vitals[kernel_id].phi = min(0.75, self.kernel_vitals[kernel_id].phi + 0.1)
                    
            elif intervention == InterventionType.BASIN_ANCHORING:
                logger.info(f"[HealthGovernance] Applying BASIN_ANCHORING to {kernel_id}")
                if kernel_id in self.kernel_vitals:
                    self.kernel_vitals[kernel_id].basin_stability = min(1.0, self.kernel_vitals[kernel_id].basin_stability + 0.2)
                    
            elif intervention == InterventionType.KNOWLEDGE_TRANSFER:
                logger.info(f"[HealthGovernance] Applying KNOWLEDGE_TRANSFER to {kernel_id}")
                if kernel_id in self.kernel_vitals:
                    self.kernel_vitals[kernel_id].recovery_debt = max(0, self.kernel_vitals[kernel_id].recovery_debt - 5)
                    
            elif intervention == InterventionType.CHIRON_HEALING:
                logger.info(f"[HealthGovernance] Invoking CHIRON_HEALING for {kernel_id}")
                if self.chiron and kernel_id in self.kernel_vitals:
                    try:
                        self.kernel_vitals[kernel_id].consecutive_failures = max(0, self.kernel_vitals[kernel_id].consecutive_failures - 3)
                        self.kernel_vitals[kernel_id].phi = min(0.6, self.kernel_vitals[kernel_id].phi + 0.15)
                    except Exception as e:
                        logger.warning(f"[HealthGovernance] Chiron healing failed: {e}")
                        
            elif intervention == InterventionType.MUSHROOM_THERAPY:
                logger.info(f"[HealthGovernance] Applying MUSHROOM_THERAPY to {kernel_id}")
                if kernel_id in self.kernel_vitals:
                    self.kernel_vitals[kernel_id].stress = 0.0
                    self.kernel_vitals[kernel_id].basin_stability = min(1.0, self.kernel_vitals[kernel_id].basin_stability + 0.3)
                    
            elif intervention == InterventionType.EMERGENCY_STABILIZE:
                logger.info(f"[HealthGovernance] EMERGENCY_STABILIZE for {kernel_id}")
                if kernel_id in self.kernel_vitals:
                    vitals = self.kernel_vitals[kernel_id]
                    vitals.phi = max(0.3, vitals.phi)
                    vitals.stress = min(0.5, vitals.stress)
                    vitals.consecutive_failures = min(5, vitals.consecutive_failures)
                    vitals.basin_stability = max(0.5, vitals.basin_stability)
    
    def propose_spawn(self, reason: str, proposed_by: str, 
                     workload_context: Optional[Dict] = None,
                     population_context: Optional[Dict] = None) -> SpawnProposal:
        proposal_id = f"spawn_{int(time.time() * 1000)}_{proposed_by}"
        
        proposal = SpawnProposal(
            proposal_id=proposal_id,
            reason=reason,
            proposed_by=proposed_by,
            workload_context=workload_context or {},
            population_context=population_context or {}
        )
        
        self.spawn_proposals[proposal_id] = proposal
        
        if self.pantheon_chat:
            try:
                self.pantheon_chat.broadcast_message(
                    from_god=proposed_by,
                    message_type="spawn_proposal",
                    intent="request_vote",
                    data={
                        "proposal_id": proposal_id,
                        "reason": reason,
                        "workload": workload_context,
                        "population": population_context
                    }
                )
            except Exception as e:
                logger.warning(f"[HealthGovernance] Failed to broadcast spawn proposal: {e}")
        
        logger.info(f"[HealthGovernance] Spawn proposal {proposal_id} created by {proposed_by}")
        return proposal
    
    def vote_on_spawn(self, proposal_id: str, god_name: str, vote: SpawnProposalVote) -> bool:
        if proposal_id not in self.spawn_proposals:
            logger.warning(f"[HealthGovernance] Proposal {proposal_id} not found")
            return False
        
        if god_name not in self.VOTING_GODS:
            logger.warning(f"[HealthGovernance] {god_name} is not a voting god")
            return False
        
        proposal = self.spawn_proposals[proposal_id]
        proposal.votes[god_name] = vote
        
        logger.info(f"[HealthGovernance] {god_name} voted {vote.value} on {proposal_id}")
        
        self._check_consensus(proposal)
        return True
    
    def _check_consensus(self, proposal: SpawnProposal):
        votes_cast = len(proposal.votes)
        if votes_cast < len(self.VOTING_GODS) * 0.5:
            return
        
        approve_count = sum(1 for v in proposal.votes.values() if v == SpawnProposalVote.APPROVE)
        reject_count = sum(1 for v in proposal.votes.values() if v == SpawnProposalVote.REJECT)
        voting_count = approve_count + reject_count
        
        if voting_count == 0:
            return
        
        approval_rate = approve_count / voting_count
        
        proposal.consensus_reached = True
        proposal.approved = approval_rate >= self.CONSENSUS_THRESHOLD
        
        if proposal.approved:
            logger.info(f"[HealthGovernance] Spawn proposal {proposal.proposal_id} APPROVED ({approval_rate:.0%})")
        else:
            logger.info(f"[HealthGovernance] Spawn proposal {proposal.proposal_id} REJECTED ({approval_rate:.0%})")
            self.total_spawns_blocked += 1
    
    def should_allow_spawn(self, reason: str = "auto_trigger") -> Tuple[bool, str]:
        current_population = len(self.kernel_vitals)
        if current_population >= self.MAX_POPULATION:
            return False, f"Population at maximum ({current_population}/{self.MAX_POPULATION})"
        
        if current_population > 0:
            avg_vitality = np.mean([v.compute_vitality_score() for v in self.kernel_vitals.values()])
            healthy_count = sum(1 for v in self.kernel_vitals.values() 
                              if v.get_status() in [HealthStatus.THRIVING, HealthStatus.STABLE])
            healthy_ratio = healthy_count / current_population
            
            if avg_vitality > 0.7 and healthy_ratio > 0.8:
                return False, f"Fleet healthy (vitality={avg_vitality:.2f}, healthy={healthy_ratio:.0%}) - no new kernels needed"
        
        stressed_count = sum(1 for v in self.kernel_vitals.values()
                           if v.get_status() in [HealthStatus.STRESSED, HealthStatus.CRITICAL])
        if stressed_count > current_population * 0.3:
            return False, f"Too many stressed kernels ({stressed_count}) - heal before spawning"
        
        return True, "Spawn allowed by governance"
    
    def get_fleet_health_report(self) -> Dict[str, Any]:
        if not self.kernel_vitals:
            return {
                "population": 0,
                "average_vitality": 0,
                "status_distribution": {},
                "active_care_plans": 0,
                "pending_proposals": 0,
                "governance_stats": {
                    "spawns_blocked": self.total_spawns_blocked,
                    "deaths_prevented": self.total_deaths_prevented,
                    "total_interventions": self.total_interventions
                }
            }
        
        vitality_scores = [v.compute_vitality_score() for v in self.kernel_vitals.values()]
        status_counts = {}
        for vitals in self.kernel_vitals.values():
            status = vitals.get_status().value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "population": len(self.kernel_vitals),
            "average_vitality": float(np.mean(vitality_scores)),
            "min_vitality": float(np.min(vitality_scores)),
            "max_vitality": float(np.max(vitality_scores)),
            "status_distribution": status_counts,
            "active_care_plans": len(self.care_plans),
            "pending_proposals": sum(1 for p in self.spawn_proposals.values() if not p.consensus_reached),
            "governance_stats": {
                "spawns_blocked": self.total_spawns_blocked,
                "deaths_prevented": self.total_deaths_prevented,
                "total_interventions": self.total_interventions
            }
        }
    
    def can_kernel_die(self, kernel_id: str) -> Tuple[bool, str]:
        if kernel_id not in self.kernel_vitals:
            return True, "Unknown kernel - death allowed"
        
        vitals = self.kernel_vitals[kernel_id]
        
        if kernel_id not in self.care_plans:
            self._trigger_care_plan(kernel_id, vitals)
            self.total_deaths_prevented += 1
            return False, "Care plan initiated - death prevented"
        
        care_plan = self.care_plans[kernel_id]
        
        if care_plan.stage < care_plan.max_stages:
            self.total_deaths_prevented += 1
            return False, f"Care plan at stage {care_plan.stage}/{care_plan.max_stages} - escalating"
        
        if vitals.compute_vitality_score() > 0.1:
            self.total_deaths_prevented += 1
            return False, "Some vitality remains - continuing intervention"
        
        return True, "All interventions exhausted - death allowed"
    
    def deregister_kernel(self, kernel_id: str):
        if kernel_id in self.kernel_vitals:
            del self.kernel_vitals[kernel_id]
        if kernel_id in self.care_plans:
            del self.care_plans[kernel_id]
        logger.info(f"[HealthGovernance] Deregistered kernel {kernel_id}")


_governance_instance: Optional[PantheonHealthGovernance] = None


def get_health_governance(pantheon_chat=None, chiron=None) -> PantheonHealthGovernance:
    global _governance_instance
    if _governance_instance is None:
        _governance_instance = PantheonHealthGovernance(pantheon_chat=pantheon_chat, chiron=chiron)
    return _governance_instance
