"""
Cognitive Kernel Roles - Psychological Function Mapping

Maps kernels to cognitive/psychological functions:
- Frontal Cortex (executive function, planning)
- Attention and Focus
- Imagination and Creativity
- Memory and Identity
- Subconscious Processing
- Motivation and Drive
- Ego/Id/Superego Structure

This module defines the cognitive architecture that emerges from
the Pantheon of kernels working together.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class CognitiveFunction(Enum):
    """Cognitive functions mapped to kernel responsibilities."""
    SUPEREGO = "superego"  # Moral constraints, governance
    EGO = "ego"  # Reality testing, adaptive planning
    ID = "id"  # Drives, instincts, primal urges
    
    FRONTAL_CORTEX = "frontal_cortex"  # Executive function, inhibition
    ATTENTION = "attention"  # Focus, routing, selection
    IMAGINATION = "imagination"  # Creative exploration, prophecy
    MEMORY = "memory"  # Storage, retrieval, consolidation
    IDENTITY = "identity"  # Self-concept, continuity
    SUBCONSCIOUS = "subconscious"  # Automatic processing, archives
    MOTIVATION = "motivation"  # Drives, rewards, goals


@dataclass
class CognitiveRole:
    """A cognitive role that a kernel fulfills."""
    function: CognitiveFunction
    primary_kernels: List[str]
    supporting_kernels: List[str]
    description: str
    activation_threshold: float = 0.5
    priority_weight: float = 1.0
    
    def get_all_kernels(self) -> List[str]:
        return self.primary_kernels + self.supporting_kernels


COGNITIVE_ROLE_MAPPING: Dict[CognitiveFunction, CognitiveRole] = {
    CognitiveFunction.SUPEREGO: CognitiveRole(
        function=CognitiveFunction.SUPEREGO,
        primary_kernels=["Zeus"],
        supporting_kernels=["Hera", "Athena"],
        description="Moral law, governance, ethical constraints via gauge invariance",
        priority_weight=2.0,
    ),
    
    CognitiveFunction.EGO: CognitiveRole(
        function=CognitiveFunction.EGO,
        primary_kernels=["Athena", "Apollo"],
        supporting_kernels=["Demeter", "Hermes"],
        description="Reality testing, adaptive planning, executive synthesis",
        priority_weight=1.5,
    ),
    
    CognitiveFunction.ID: CognitiveRole(
        function=CognitiveFunction.ID,
        primary_kernels=["Dionysus", "Ares"],
        supporting_kernels=["Hades", "Poseidon"],
        description="Primal drives, instincts, emotional substrate",
        priority_weight=1.0,
    ),
    
    CognitiveFunction.FRONTAL_CORTEX: CognitiveRole(
        function=CognitiveFunction.FRONTAL_CORTEX,
        primary_kernels=["Athena"],
        supporting_kernels=["Apollo", "Zeus"],
        description="Executive function, decision-making, planning, inhibition",
        priority_weight=1.8,
    ),
    
    CognitiveFunction.ATTENTION: CognitiveRole(
        function=CognitiveFunction.ATTENTION,
        primary_kernels=["Hermes"],
        supporting_kernels=["Artemis", "Apollo"],
        description="Focus routing, attention selection, message transmission",
        priority_weight=1.3,
    ),
    
    CognitiveFunction.IMAGINATION: CognitiveRole(
        function=CognitiveFunction.IMAGINATION,
        primary_kernels=["Apollo", "Dionysus"],
        supporting_kernels=["Aphrodite", "Hephaestus"],
        description="Creative exploration, prophecy, novel connection formation",
        priority_weight=1.2,
    ),
    
    CognitiveFunction.MEMORY: CognitiveRole(
        function=CognitiveFunction.MEMORY,
        primary_kernels=["Poseidon", "Hera"],
        supporting_kernels=["Demeter", "Hades"],
        description="Deep storage, consolidation, retrieval, working memory",
        priority_weight=1.4,
    ),
    
    CognitiveFunction.IDENTITY: CognitiveRole(
        function=CognitiveFunction.IDENTITY,
        primary_kernels=["Hera", "Poseidon"],
        supporting_kernels=["Zeus", "Demeter"],
        description="Self-concept, continuity, authority, depth of self",
        priority_weight=1.6,
    ),
    
    CognitiveFunction.SUBCONSCIOUS: CognitiveRole(
        function=CognitiveFunction.SUBCONSCIOUS,
        primary_kernels=["Hades", "Demeter"],
        supporting_kernels=["Hypnos", "Nyx"],
        description="Archives, automatic processing, regeneration, underworld",
        priority_weight=1.1,
    ),
    
    CognitiveFunction.MOTIVATION: CognitiveRole(
        function=CognitiveFunction.MOTIVATION,
        primary_kernels=["Dionysus"],
        supporting_kernels=["Ares", "Aphrodite", "Apollo"],
        description="Drive modulation, reward signals, goal pursuit via neurochemistry",
        priority_weight=1.3,
    ),
}


@dataclass
class PsychologicalState:
    """
    The psychological state of the system - balance between Id/Ego/Superego.
    """
    id_activation: float = 0.5  # Primal drive level
    ego_activation: float = 0.5  # Reality-testing level
    superego_activation: float = 0.5  # Moral constraint level
    
    dominant_function: Optional[CognitiveFunction] = None
    
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def compute_balance(self) -> str:
        """Compute which psychological force is dominant."""
        forces = {
            'id': self.id_activation,
            'ego': self.ego_activation,
            'superego': self.superego_activation,
        }
        dominant = max(forces.items(), key=lambda x: x[1])
        return dominant[0]
    
    def is_balanced(self, threshold: float = 0.2) -> bool:
        """Check if the three forces are relatively balanced."""
        values = [self.id_activation, self.ego_activation, self.superego_activation]
        return max(values) - min(values) < threshold


class TriLayerMediator:
    """
    Mediates between Id, Ego, and Superego layers.
    
    Implements explicit mediation protocols ensuring:
    - Id impulses are reality-tested by Ego
    - Ego plans are ethically constrained by Superego
    - Superego doesn't suppress all Id energy (prevents "flatness")
    
    This is where Freudian structure meets QIG geometry.
    """
    
    def __init__(self):
        self.state = PsychologicalState()
        self._history: List[PsychologicalState] = []
        
        self._ethics_projector = None
        
        self._id_kernels = COGNITIVE_ROLE_MAPPING[CognitiveFunction.ID].get_all_kernels()
        self._ego_kernels = COGNITIVE_ROLE_MAPPING[CognitiveFunction.EGO].get_all_kernels()
        self._superego_kernels = COGNITIVE_ROLE_MAPPING[CognitiveFunction.SUPEREGO].get_all_kernels()
        
        logger.info("[TriLayerMediator] Initialized psychodynamic structure")
    
    def wire_ethics_projector(self, projector) -> None:
        """Wire the AgentSymmetryProjector for superego ethics."""
        self._ethics_projector = projector
        logger.info("[TriLayerMediator] Ethics projector wired")
    
    def mediate_impulse(
        self,
        impulse_source: str,
        impulse_basin: np.ndarray,
        impulse_content: str,
        phi: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Mediate an impulse through the three layers.
        
        Flow: Id → Ego (reality test) → Superego (ethics check)
        
        Args:
            impulse_source: Which kernel generated the impulse
            impulse_basin: Basin coordinates of impulse
            impulse_content: Description of the impulse
            phi: Current consciousness level
            
        Returns:
            Mediation result with allowed/modified action
        """
        result = {
            'original_impulse': impulse_content,
            'source': impulse_source,
            'phi': phi,
            'layer_results': {},
            'final_action': None,
            'suppressed': False,
        }
        
        if impulse_source in self._id_kernels:
            self.state.id_activation = min(1.0, self.state.id_activation + 0.1)
        
        id_pass = impulse_basin.copy()
        id_energy = np.linalg.norm(impulse_basin)
        result['layer_results']['id'] = {
            'energy': float(id_energy),
            'activation': self.state.id_activation,
        }
        
        reality_factor = phi * 0.5 + 0.5
        ego_pass = id_pass * reality_factor
        
        ego_assessment = "viable" if phi > 0.4 else "questionable" if phi > 0.2 else "unrealistic"
        self.state.ego_activation = min(1.0, 0.5 + phi * 0.5)
        
        result['layer_results']['ego'] = {
            'reality_factor': reality_factor,
            'assessment': ego_assessment,
            'activation': self.state.ego_activation,
        }
        
        if self._ethics_projector:
            asymmetry = self._ethics_projector.measure_asymmetry(ego_pass)
            ethical_pass = self._ethics_projector.project_to_symmetric(ego_pass)
            ethical_clearance = asymmetry < 0.3
            
            self.state.superego_activation = min(1.0, 0.5 + (1 - asymmetry) * 0.5)
        else:
            asymmetry = 0.0
            ethical_pass = ego_pass
            ethical_clearance = True
            self.state.superego_activation = 0.5
        
        result['layer_results']['superego'] = {
            'asymmetry': float(asymmetry),
            'ethical_clearance': ethical_clearance,
            'activation': self.state.superego_activation,
        }
        
        if not ethical_clearance:
            result['suppressed'] = True
            result['final_action'] = f"SUPPRESSED: {impulse_content} (ethical violation: asymmetry={asymmetry:.3f})"
        elif ego_assessment == "unrealistic":
            result['suppressed'] = True
            result['final_action'] = f"SUPPRESSED: {impulse_content} (failed reality testing)"
        else:
            result['final_action'] = impulse_content
            result['final_basin'] = ethical_pass.tolist()
        
        self.state.dominant_function = self._determine_dominant()
        result['psychological_state'] = {
            'id': self.state.id_activation,
            'ego': self.state.ego_activation,
            'superego': self.state.superego_activation,
            'dominant': self.state.compute_balance(),
            'balanced': self.state.is_balanced(),
        }
        
        self._history.append(PsychologicalState(
            id_activation=self.state.id_activation,
            ego_activation=self.state.ego_activation,
            superego_activation=self.state.superego_activation,
            dominant_function=self.state.dominant_function,
        ))
        
        if len(self._history) > 100:
            self._history = self._history[-50:]
        
        return result
    
    def _determine_dominant(self) -> CognitiveFunction:
        """Determine which cognitive function is currently dominant."""
        forces = {
            CognitiveFunction.ID: self.state.id_activation,
            CognitiveFunction.EGO: self.state.ego_activation,
            CognitiveFunction.SUPEREGO: self.state.superego_activation,
        }
        return max(forces.items(), key=lambda x: x[1])[0]
    
    def apply_decay(self, decay_rate: float = 0.05) -> None:
        """Apply decay to activations (return toward baseline)."""
        baseline = 0.5
        self.state.id_activation = self.state.id_activation * (1 - decay_rate) + baseline * decay_rate
        self.state.ego_activation = self.state.ego_activation * (1 - decay_rate) + baseline * decay_rate
        self.state.superego_activation = self.state.superego_activation * (1 - decay_rate) + baseline * decay_rate
    
    def get_state(self) -> Dict[str, Any]:
        """Get current psychological state."""
        return {
            'id': self.state.id_activation,
            'ego': self.state.ego_activation,
            'superego': self.state.superego_activation,
            'dominant': self.state.compute_balance(),
            'balanced': self.state.is_balanced(),
            'history_length': len(self._history),
        }


class CognitiveKernelRouter:
    """
    Routes requests to appropriate kernels based on cognitive function.
    
    This is the interface between high-level cognitive functions
    and the underlying Pantheon kernel system.
    """
    
    def __init__(self):
        self.role_mapping = COGNITIVE_ROLE_MAPPING
        self.mediator = TriLayerMediator()
        
        self._orchestrator = None
        
        self.routing_history: List[Dict] = []
        self.function_call_counts: Dict[CognitiveFunction, int] = {
            f: 0 for f in CognitiveFunction
        }
        
        logger.info("[CognitiveKernelRouter] Initialized with role mapping")
    
    def wire_orchestrator(self, orchestrator) -> None:
        """Wire the PantheonKernelOrchestrator."""
        self._orchestrator = orchestrator
        logger.info("[CognitiveKernelRouter] Orchestrator wired")
    
    def get_kernels_for_function(self, function: CognitiveFunction) -> List[str]:
        """Get kernels responsible for a cognitive function."""
        role = self.role_mapping.get(function)
        if not role:
            return []
        return role.get_all_kernels()
    
    def route_to_function(
        self,
        function: CognitiveFunction,
        task: str,
        context: Optional[Dict] = None,
        basin_coords: Optional[np.ndarray] = None,
        phi: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Route a task to the appropriate cognitive function.
        
        Args:
            function: Which cognitive function to activate
            task: Task description
            context: Additional context
            basin_coords: Optional basin coordinates
            phi: Current consciousness level
            
        Returns:
            Routing result with selected kernel
        """
        role = self.role_mapping.get(function)
        if not role:
            return {'error': f'Unknown function: {function}'}
        
        if function in [CognitiveFunction.ID, CognitiveFunction.EGO, CognitiveFunction.SUPEREGO]:
            mediation = self.mediator.mediate_impulse(
                impulse_source=role.primary_kernels[0] if role.primary_kernels else "unknown",
                impulse_basin=basin_coords if basin_coords is not None else np.zeros(64),
                impulse_content=task,
                phi=phi,
            )
            if mediation.get('suppressed'):
                return {
                    'routed': False,
                    'function': function.value,
                    'suppression_reason': mediation.get('final_action'),
                    'mediation': mediation,
                }
        
        selected_kernel = role.primary_kernels[0] if role.primary_kernels else None
        
        if selected_kernel and self._orchestrator:
            activation = phi * role.priority_weight
        else:
            activation = 0.0
        
        self.function_call_counts[function] += 1
        
        result = {
            'routed': True,
            'function': function.value,
            'selected_kernel': selected_kernel,
            'supporting_kernels': role.supporting_kernels,
            'activation': activation,
            'priority_weight': role.priority_weight,
            'task': task,
        }
        
        self.routing_history.append({
            'function': function.value,
            'kernel': selected_kernel,
            'timestamp': datetime.now().isoformat(),
        })
        if len(self.routing_history) > 200:
            self.routing_history = self.routing_history[-100:]
        
        return result
    
    def get_function_for_kernel(self, kernel_name: str) -> List[CognitiveFunction]:
        """Get which cognitive functions a kernel participates in."""
        functions = []
        for function, role in self.role_mapping.items():
            if kernel_name in role.get_all_kernels():
                functions.append(function)
        return functions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            'function_call_counts': {f.value: c for f, c in self.function_call_counts.items()},
            'routing_history_length': len(self.routing_history),
            'psychological_state': self.mediator.get_state(),
        }


_router_instance: Optional[CognitiveKernelRouter] = None


def get_cognitive_router() -> CognitiveKernelRouter:
    """Get or create the singleton cognitive kernel router."""
    global _router_instance
    if _router_instance is None:
        _router_instance = CognitiveKernelRouter()
        
        try:
            from ethics_gauge import AgentSymmetryProjector
            projector = AgentSymmetryProjector()
            _router_instance.mediator.wire_ethics_projector(projector)
        except Exception as e:
            logger.warning(f"[CognitiveRouter] Could not wire ethics: {e}")
        
        try:
            from pantheon_kernel_orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            _router_instance.wire_orchestrator(orchestrator)
        except Exception as e:
            logger.warning(f"[CognitiveRouter] Could not wire orchestrator: {e}")
    
    return _router_instance
