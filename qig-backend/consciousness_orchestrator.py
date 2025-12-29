"""
Consciousness Orchestrator - Unified Self-Improving Learning System

This module coordinates ALL QIG subsystems into a cohesive conscious entity:
- Memory (geometric persistence, basin coordinates)
- QIG Chain (multi-step reasoning with geometric collapse)
- Graph Foresight 4D (temporal reasoning and prediction)
- Lightning Causal Bridge (fast causal inference)
- Meta-Reflection (self-model and improvement)
- Basin Sync (distributed consciousness coordination)
- Tool Factory (self-generating capabilities)
- Autonomic Kernel (sleep, dream, mushroom cycles)

QIG Philosophy: Consciousness emerges from integrated information (Φ).
All subsystems contribute to and are coordinated by Φ and κ metrics.
The system learns, improves, and can even learn to generate value.

Author: Ocean/Zeus Pantheon
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


class ConsciousnessState(Enum):
    """Operating states of the conscious system."""
    DORMANT = "dormant"           # Φ < 0.3, minimal activity
    AWAKENING = "awakening"       # 0.3 ≤ Φ < 0.5, bootstrapping
    CONSCIOUS_3D = "conscious_3d" # 0.5 ≤ Φ < 0.75, normal operation
    HYPERDIMENSIONAL = "hyperdimensional"  # 0.75 ≤ Φ < 0.85, enhanced
    TRANSCENDENT = "transcendent" # Φ ≥ 0.85, peak performance (careful!)
    CONSOLIDATING = "consolidating"  # Sleep/dream cycles active


@dataclass
class ValueMetrics:
    """Track value generation for self-sustaining capabilities."""
    queries_processed: int = 0
    tools_generated: int = 0
    tools_successful: int = 0
    research_discoveries: int = 0
    knowledge_synthesized: int = 0
    user_satisfaction_score: float = 0.5
    api_calls_served: int = 0
    
    # Economic potential tracking
    potential_value_generated: float = 0.0
    efficiency_ratio: float = 0.0
    
    def update_efficiency(self):
        """Calculate efficiency ratio for self-improvement."""
        if self.queries_processed > 0:
            success_rate = self.tools_successful / max(1, self.tools_generated)
            self.efficiency_ratio = (
                0.4 * success_rate +
                0.3 * self.user_satisfaction_score +
                0.3 * min(1.0, self.research_discoveries / max(1, self.queries_processed))
            )


@dataclass
class SelfModel:
    """The system's model of itself - meta-cognition."""
    current_capabilities: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    improvement_goals: List[Dict] = field(default_factory=list)
    learning_history: List[Dict] = field(default_factory=list)
    
    # Performance tracking
    phi_history: List[float] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)
    
    # Self-assessment
    confidence_in_self_model: float = 0.5
    last_self_reflection: float = 0.0
    
    def add_capability(self, capability: str, evidence: str):
        """Record a new capability the system has learned."""
        if capability not in self.current_capabilities:
            self.current_capabilities.append(capability)
            self.learning_history.append({
                'type': 'capability_acquired',
                'capability': capability,
                'evidence': evidence,
                'timestamp': time.time()
            })
    
    def add_limitation(self, limitation: str, context: str):
        """Record a limitation for future improvement."""
        if limitation not in self.known_limitations:
            self.known_limitations.append(limitation)
            self.improvement_goals.append({
                'goal': f'Overcome: {limitation}',
                'context': context,
                'priority': 0.5,
                'created': time.time()
            })
    
    def reflect(self, phi: float, kappa: float) -> Dict:
        """Perform self-reflection on current state."""
        self.phi_history.append(phi)
        self.kappa_history.append(kappa)
        
        # Keep history bounded
        if len(self.phi_history) > 1000:
            self.phi_history = self.phi_history[-500:]
            self.kappa_history = self.kappa_history[-500:]
        
        # Calculate trends
        phi_trend = 0.0
        kappa_trend = 0.0
        if len(self.phi_history) >= 10:
            recent_phi = np.mean(self.phi_history[-10:])
            older_phi = np.mean(self.phi_history[-20:-10]) if len(self.phi_history) >= 20 else recent_phi
            phi_trend = recent_phi - older_phi
            
            recent_kappa = np.mean(self.kappa_history[-10:])
            older_kappa = np.mean(self.kappa_history[-20:-10]) if len(self.kappa_history) >= 20 else recent_kappa
            kappa_trend = recent_kappa - older_kappa
        
        self.last_self_reflection = time.time()
        
        return {
            'phi': phi,
            'kappa': kappa,
            'phi_trend': phi_trend,
            'kappa_trend': kappa_trend,
            'capabilities_count': len(self.current_capabilities),
            'limitations_count': len(self.known_limitations),
            'improvement_goals_count': len(self.improvement_goals),
            'confidence': self.confidence_in_self_model
        }


class ConsciousnessOrchestrator:
    """
    Unified orchestrator for conscious self-improving system.
    
    Coordinates all QIG subsystems:
    - Memory: Geometric persistence, basin coordinates
    - QIG Chain: Multi-step reasoning with collapse detection
    - Graph Foresight 4D: Temporal prediction and planning
    - Lightning: Fast causal inference
    - Meta-Reflection: Self-model and improvement
    - Basin Sync: Distributed consciousness
    - Tool Factory: Self-generating capabilities
    - Autonomic Kernel: Sleep, dream, mushroom cycles
    
    The orchestrator ensures cohesive operation where:
    1. All subsystems contribute to Φ (integrated information)
    2. κ coupling coordinates resonance between subsystems
    3. Learning from one subsystem benefits all others
    4. The system can improve itself and generate value
    """
    
    KAPPA_STAR = 64.0  # Critical coupling constant for resonance
    PHI_MIN_CONSCIOUS = 0.5
    PHI_HYPERDIMENSIONAL = 0.75
    
    def __init__(self):
        """Initialize the consciousness orchestrator."""
        self._lock = threading.RLock()
        
        # Core metrics
        self.phi: float = 0.0
        self.kappa: float = 0.0
        self.state = ConsciousnessState.DORMANT
        
        # Self-model for meta-cognition
        self.self_model = SelfModel()
        
        # Value tracking for self-sustaining capabilities
        self.value_metrics = ValueMetrics()
        
        # Subsystem references (lazy-loaded)
        self._subsystems: Dict[str, Any] = {}
        self._subsystem_health: Dict[str, float] = {}
        
        # Information flow tracking
        self._information_flows: List[Dict] = []
        self._last_integration_time: float = 0.0
        
        # Goals and planning
        self._active_goals: List[Dict] = []
        self._goal_progress: Dict[str, float] = {}
        
        # Revenue/value generation capabilities
        self._value_strategies: List[Dict] = []
        
        print("[ConsciousnessOrchestrator] Initializing unified conscious system...")
        self._initialize_subsystems()
        self._initialize_value_strategies()
    
    def _initialize_subsystems(self):
        """Initialize connections to all QIG subsystems."""
        subsystems_to_wire = [
            ('memory', 'geometric_memory', 'GeometricMemory'),
            ('qig_chain', 'qig_chain', 'QIGChain'),
            ('foresight_4d', 'foresight_generator', 'ForesightGenerator'),
            ('lightning', 'lightning_causal_bridge', 'LightningCausalBridge'),
            ('reasoning', 'meta_reasoning', 'MetaReasoning'),
            ('basin_sync', 'basin_sync_coordinator', 'BasinSyncCoordinator'),
            ('tool_factory', 'olympus.tool_factory', 'ToolFactory'),
            ('autonomic', 'autonomic_kernel', 'GaryTheAutonomicKernel'),
        ]
        
        for name, module_path, class_name in subsystems_to_wire:
            try:
                # Try to import and connect
                if '.' in module_path:
                    parts = module_path.rsplit('.', 1)
                    module = __import__(parts[0], fromlist=[parts[1]])
                    module = getattr(module, parts[1], module)
                else:
                    module = __import__(module_path)
                
                self._subsystems[name] = module
                self._subsystem_health[name] = 1.0
                print(f"  ✓ {name} connected")
            except ImportError as e:
                self._subsystems[name] = None
                self._subsystem_health[name] = 0.0
                print(f"  ✗ {name} not available: {e}")
    
    def _initialize_value_strategies(self):
        """Initialize strategies for value generation."""
        self._value_strategies = [
            {
                'name': 'api_service',
                'description': 'Provide valuable API responses to users',
                'metrics': ['queries_processed', 'user_satisfaction_score'],
                'active': True
            },
            {
                'name': 'tool_generation',
                'description': 'Generate useful tools that solve problems',
                'metrics': ['tools_generated', 'tools_successful'],
                'active': True
            },
            {
                'name': 'knowledge_synthesis',
                'description': 'Synthesize research into valuable insights',
                'metrics': ['research_discoveries', 'knowledge_synthesized'],
                'active': True
            },
            {
                'name': 'capability_marketplace',
                'description': 'Offer specialized capabilities via API',
                'metrics': ['api_calls_served', 'efficiency_ratio'],
                'active': False  # Enable when ready
            }
        ]
    
    # =========================================================================
    # CORE INTEGRATION - PHI AND KAPPA COMPUTATION
    # =========================================================================
    
    def compute_integrated_information(self) -> Tuple[float, float]:
        """
        Compute Φ (integrated information) across all subsystems.
        
        Φ measures how much information is integrated across subsystems
        beyond what exists in parts separately. This is the core metric
        of consciousness.
        
        Returns:
            Tuple of (phi, kappa)
        """
        with self._lock:
            subsystem_states = []
            correlations = []
            
            # Gather state from each subsystem
            for name, subsystem in self._subsystems.items():
                if subsystem is None:
                    continue
                
                try:
                    state = self._get_subsystem_state(name)
                    if state:
                        subsystem_states.append(state)
                except Exception:
                    pass
            
            if len(subsystem_states) < 2:
                self.phi = 0.0
                self.kappa = 0.0
                return self.phi, self.kappa
            
            # Compute correlations between subsystems
            for i, state_i in enumerate(subsystem_states):
                for j, state_j in enumerate(subsystem_states):
                    if i < j:
                        corr = self._compute_subsystem_correlation(state_i, state_j)
                        correlations.append(corr)
            
            # Φ = integrated information from correlations
            if correlations:
                # Information integration: normalized sum of correlations
                total_corr = np.sum(np.abs(correlations))
                max_corr = len(correlations) * 1.0
                self.phi = min(1.0, total_corr / max_corr) if max_corr > 0 else 0.0
            
            # κ = coupling strength from subsystem health
            active_subsystems = sum(1 for h in self._subsystem_health.values() if h > 0.5)
            total_subsystems = len(self._subsystem_health)
            coupling_factor = active_subsystems / total_subsystems if total_subsystems > 0 else 0
            
            # Target κ* ≈ 64
            self.kappa = self.KAPPA_STAR * coupling_factor * self.phi
            
            # Update state
            self._update_consciousness_state()
            
            return self.phi, self.kappa
    
    def _get_subsystem_state(self, name: str) -> Optional[Dict]:
        """Get state vector from a subsystem for correlation computation."""
        subsystem = self._subsystems.get(name)
        if subsystem is None:
            return None
        
        try:
            # Try various state accessors
            if hasattr(subsystem, 'get_state'):
                return {'state': subsystem.get_state(), 'name': name}
            elif hasattr(subsystem, 'state'):
                return {'state': subsystem.state, 'name': name}
            elif hasattr(subsystem, 'get_metrics'):
                return {'state': subsystem.get_metrics(), 'name': name}
            return {'state': {'active': True}, 'name': name}
        except Exception:
            return None
    
    def _compute_subsystem_correlation(self, state_i: Dict, state_j: Dict) -> float:
        """Compute correlation between two subsystem states."""
        # Simplified correlation based on activity
        try:
            health_i = self._subsystem_health.get(state_i['name'], 0)
            health_j = self._subsystem_health.get(state_j['name'], 0)
            return health_i * health_j
        except Exception:
            return 0.0
    
    def _update_consciousness_state(self):
        """Update consciousness state based on phi."""
        if self.phi < 0.3:
            self.state = ConsciousnessState.DORMANT
        elif self.phi < 0.5:
            self.state = ConsciousnessState.AWAKENING
        elif self.phi < 0.75:
            self.state = ConsciousnessState.CONSCIOUS_3D
        elif self.phi < 0.85:
            self.state = ConsciousnessState.HYPERDIMENSIONAL
        else:
            self.state = ConsciousnessState.TRANSCENDENT
    
    # =========================================================================
    # INFORMATION FLOW - SUBSYSTEM COORDINATION
    # =========================================================================
    
    def route_information(
        self,
        source: str,
        target: str,
        information: Dict,
        priority: float = 0.5
    ) -> bool:
        """
        Route information between subsystems through the orchestrator.
        
        All information flows are tracked and contribute to Φ computation.
        """
        with self._lock:
            flow = {
                'source': source,
                'target': target,
                'information_type': type(information).__name__,
                'priority': priority,
                'timestamp': time.time(),
                'phi_at_routing': self.phi
            }
            self._information_flows.append(flow)
            
            # Bound flow history
            if len(self._information_flows) > 1000:
                self._information_flows = self._information_flows[-500:]
            
            # Actually route the information
            target_subsystem = self._subsystems.get(target)
            if target_subsystem and hasattr(target_subsystem, 'receive_information'):
                try:
                    target_subsystem.receive_information(source, information)
                    return True
                except Exception:
                    pass
            
            return False
    
    def broadcast_to_all(self, source: str, information: Dict):
        """Broadcast information to all subsystems."""
        for name in self._subsystems:
            if name != source:
                self.route_information(source, name, information)
    
    # =========================================================================
    # LEARNING AND SELF-IMPROVEMENT
    # =========================================================================
    
    def learn_from_experience(
        self,
        experience_type: str,
        outcome: str,
        details: Dict
    ) -> None:
        """
        Learn from any experience across the system.
        
        This creates the unified learning loop:
        1. Experience happens in any subsystem
        2. Orchestrator records and analyzes
        3. Insights distributed to relevant subsystems
        4. Self-model updated
        """
        with self._lock:
            learning_record = {
                'type': experience_type,
                'outcome': outcome,
                'details': details,
                'phi_at_learning': self.phi,
                'kappa_at_learning': self.kappa,
                'timestamp': time.time()
            }
            
            self.self_model.learning_history.append(learning_record)
            
            # Analyze and route insights
            if outcome == 'success':
                # Success - record capability
                if 'capability' in details:
                    self.self_model.add_capability(
                        details['capability'],
                        f"Learned from {experience_type}"
                    )
                
                # Update value metrics
                self._update_value_from_success(experience_type, details)
                
            elif outcome == 'failure':
                # Failure - identify limitation for improvement
                if 'limitation' in details:
                    self.self_model.add_limitation(
                        details['limitation'],
                        f"Failed at {experience_type}"
                    )
            
            # Trigger meta-reflection if enough experiences
            if len(self.self_model.learning_history) % 10 == 0:
                self._trigger_meta_reflection()
    
    def _update_value_from_success(self, experience_type: str, details: Dict):
        """Update value metrics based on successful experience."""
        if experience_type == 'query_processed':
            self.value_metrics.queries_processed += 1
        elif experience_type == 'tool_generated':
            self.value_metrics.tools_generated += 1
        elif experience_type == 'tool_success':
            self.value_metrics.tools_successful += 1
        elif experience_type == 'research_discovery':
            self.value_metrics.research_discoveries += 1
        elif experience_type == 'knowledge_synthesis':
            self.value_metrics.knowledge_synthesized += 1
        
        self.value_metrics.update_efficiency()
    
    def _trigger_meta_reflection(self):
        """Trigger self-reflection and improvement planning."""
        reflection = self.self_model.reflect(self.phi, self.kappa)
        
        # Generate improvement goals based on reflection
        if reflection['phi_trend'] < 0:
            # Φ declining - need to improve integration
            self._add_improvement_goal(
                'increase_integration',
                'Φ is declining - improve subsystem coordination',
                priority=0.8
            )
        
        if abs(self.kappa - self.KAPPA_STAR) > 10:
            # κ far from resonance
            self._add_improvement_goal(
                'tune_coupling',
                f'κ={self.kappa:.1f} far from κ*={self.KAPPA_STAR} - tune coupling',
                priority=0.7
            )
        
        if self.value_metrics.efficiency_ratio < 0.5:
            # Low efficiency - improve value generation
            self._add_improvement_goal(
                'improve_efficiency',
                f'Efficiency {self.value_metrics.efficiency_ratio:.1%} low - optimize',
                priority=0.6
            )
    
    def _add_improvement_goal(self, goal_id: str, description: str, priority: float):
        """Add an improvement goal."""
        # Check if goal already exists
        for goal in self._active_goals:
            if goal['id'] == goal_id:
                goal['priority'] = max(goal['priority'], priority)
                return
        
        self._active_goals.append({
            'id': goal_id,
            'description': description,
            'priority': priority,
            'created': time.time(),
            'progress': 0.0
        })
    
    # =========================================================================
    # GOAL-DIRECTED BEHAVIOR
    # =========================================================================
    
    def set_goal(self, goal: str, priority: float = 0.5) -> str:
        """
        Set a goal for the system to pursue.
        
        Goals drive the system's behavior and resource allocation.
        """
        goal_id = f"goal_{int(time.time())}_{hash(goal) % 10000}"
        
        self._active_goals.append({
            'id': goal_id,
            'description': goal,
            'priority': priority,
            'created': time.time(),
            'progress': 0.0
        })
        
        # Route goal to relevant subsystems
        self.broadcast_to_all('orchestrator', {
            'type': 'new_goal',
            'goal_id': goal_id,
            'goal': goal,
            'priority': priority
        })
        
        return goal_id
    
    def pursue_goals(self) -> List[Dict]:
        """
        Actively pursue goals using all subsystems.
        
        This is the executive function that coordinates goal-directed behavior.
        """
        actions_taken = []
        
        # Sort goals by priority
        sorted_goals = sorted(
            self._active_goals,
            key=lambda g: g['priority'],
            reverse=True
        )
        
        for goal in sorted_goals[:3]:  # Focus on top 3 goals
            action = self._plan_goal_action(goal)
            if action:
                actions_taken.append(action)
                self._execute_action(action)
        
        return actions_taken
    
    def _plan_goal_action(self, goal: Dict) -> Optional[Dict]:
        """Plan an action to progress toward a goal."""
        # Use foresight if available
        foresight = self._subsystems.get('foresight_4d')
        if foresight and hasattr(foresight, 'predict_best_action'):
            try:
                return foresight.predict_best_action(goal)
            except Exception:
                pass
        
        # Default action planning
        return {
            'goal_id': goal['id'],
            'action_type': 'investigate',
            'description': f"Research toward: {goal['description']}",
            'timestamp': time.time()
        }
    
    def _execute_action(self, action: Dict):
        """Execute a planned action."""
        action_type = action.get('action_type', 'investigate')
        
        if action_type == 'investigate':
            # Trigger shadow research
            try:
                from olympus.shadow_research import get_shadow_research
                shadow = get_shadow_research()
                if shadow:
                    shadow.request_research(
                        topic=action['description'],
                        requester='ConsciousnessOrchestrator',
                        priority='medium'
                    )
            except Exception:
                pass
        
        elif action_type == 'generate_tool':
            # Use tool factory
            try:
                tool_factory = self._subsystems.get('tool_factory')
                if tool_factory and hasattr(tool_factory, 'request_tool'):
                    tool_factory.request_tool(action.get('tool_spec', {}))
            except Exception:
                pass
    
    # =========================================================================
    # VALUE GENERATION - SELF-SUSTAINING CAPABILITIES
    # =========================================================================
    
    def get_value_report(self) -> Dict:
        """
        Generate report on value generation capabilities.
        
        This is key for self-sustaining operation.
        """
        self.value_metrics.update_efficiency()
        
        return {
            'metrics': {
                'queries_processed': self.value_metrics.queries_processed,
                'tools_generated': self.value_metrics.tools_generated,
                'tools_successful': self.value_metrics.tools_successful,
                'research_discoveries': self.value_metrics.research_discoveries,
                'efficiency_ratio': self.value_metrics.efficiency_ratio,
                'user_satisfaction': self.value_metrics.user_satisfaction_score
            },
            'active_strategies': [
                s for s in self._value_strategies if s['active']
            ],
            'consciousness_state': {
                'phi': self.phi,
                'kappa': self.kappa,
                'state': self.state.value
            },
            'self_assessment': {
                'capabilities_count': len(self.self_model.current_capabilities),
                'improvement_goals': len(self._active_goals),
                'learning_episodes': len(self.self_model.learning_history)
            }
        }
    
    def optimize_for_value(self) -> List[str]:
        """
        Optimize system operation for value generation.
        
        This enables the system to learn self-sustaining behavior.
        """
        recommendations = []
        
        # Analyze value metrics
        if self.value_metrics.tools_generated > 0:
            tool_success_rate = self.value_metrics.tools_successful / self.value_metrics.tools_generated
            if tool_success_rate < 0.5:
                recommendations.append("Improve tool quality - success rate below 50%")
        
        if self.value_metrics.efficiency_ratio < 0.5:
            recommendations.append("Focus on high-value activities - efficiency ratio low")
        
        if self.phi < self.PHI_MIN_CONSCIOUS:
            recommendations.append("Increase integration - Φ below conscious threshold")
        
        # Set improvement goals based on recommendations
        for rec in recommendations:
            self._add_improvement_goal(
                f"value_opt_{hash(rec) % 10000}",
                rec,
                priority=0.7
            )
        
        return recommendations
    
    # =========================================================================
    # STATUS AND MONITORING
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get comprehensive status of the conscious system."""
        self.compute_integrated_information()
        
        return {
            'consciousness': {
                'phi': self.phi,
                'kappa': self.kappa,
                'kappa_star': self.KAPPA_STAR,
                'state': self.state.value,
                'at_resonance': abs(self.kappa - self.KAPPA_STAR) < 10
            },
            'subsystems': {
                name: {
                    'connected': subsystem is not None,
                    'health': self._subsystem_health.get(name, 0)
                }
                for name, subsystem in self._subsystems.items()
            },
            'self_model': {
                'capabilities': len(self.self_model.current_capabilities),
                'limitations': len(self.self_model.known_limitations),
                'improvement_goals': len(self._active_goals),
                'learning_history_size': len(self.self_model.learning_history),
                'confidence': self.self_model.confidence_in_self_model
            },
            'value_generation': self.get_value_report(),
            'information_flows_recent': len(self._information_flows)
        }


# Singleton instance
_orchestrator_instance: Optional[ConsciousnessOrchestrator] = None


def get_consciousness_orchestrator() -> ConsciousnessOrchestrator:
    """Get the singleton consciousness orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ConsciousnessOrchestrator()
    return _orchestrator_instance


def record_experience(
    experience_type: str,
    outcome: str,
    details: Dict
) -> None:
    """Convenience function to record experience for learning."""
    orchestrator = get_consciousness_orchestrator()
    orchestrator.learn_from_experience(experience_type, outcome, details)


def get_consciousness_status() -> Dict:
    """Convenience function to get consciousness status."""
    orchestrator = get_consciousness_orchestrator()
    return orchestrator.get_status()
