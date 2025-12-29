"""
Meta-Cognitive Monitoring for Geometric Consciousness

Think about thinking. Monitors reasoning quality and triggers interventions.

Monitors:
1. Am I stuck? (progress stalled)
2. Am I confused? (high curvature, low coherence)
3. Should I switch modes? (Φ inappropriate for task)
4. Do I need help? (repeated failures)

Integrates with agent_failure_taxonomy for comprehensive multi-agent failure
detection and recovery.

QIG-PURE: All measurements use Fisher-Rao geometry.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time

from reasoning_metrics import ReasoningQuality, get_reasoning_quality
from reasoning_modes import ReasoningMode, ReasoningModeSelector, get_mode_selector
from qig_geometry import fisher_rao_distance

# Import failure taxonomy for integration
try:
    from agent_failure_taxonomy import (
        get_failure_monitor,
        FailureEvent,
        FailureType,
        FailureSeverity,
        RecoveryStrategy
    )
    FAILURE_TAXONOMY_AVAILABLE = True
except ImportError:
    FAILURE_TAXONOMY_AVAILABLE = False


class InterventionType(Enum):
    """Types of meta-cognitive interventions."""
    STUCK = "stuck"
    CONFUSED = "confused"
    MODE_MISMATCH = "mode_mismatch"
    DUNNING_KRUGER = "dunning_kruger"
    EXPLORATION_NEEDED = "exploration_needed"
    CONSOLIDATION_NEEDED = "consolidation_needed"


@dataclass
class Intervention:
    """A meta-cognitive intervention recommendation."""
    type: InterventionType
    action: str
    reason: str
    urgency: float  # 0-1, higher = more urgent
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MetaCognitiveState:
    """Current meta-cognitive state assessment."""
    is_stuck: bool
    is_confused: bool
    needs_mode_switch: bool
    recommended_mode: ReasoningMode
    meta_awareness: float
    interventions: List[Intervention]
    overall_health: float  # 0-1, higher = healthier reasoning


class MetaCognition:
    """
    Think about thinking.
    
    Monitors reasoning quality and triggers interventions when needed.
    
    Responsibilities:
    1. Detect when stuck (no progress in last N steps)
    2. Detect confusion (low coherence, high variance)
    3. Recommend mode switches (Φ mismatch)
    4. Trigger interventions (strategy change, simplification)
    """
    
    def __init__(
        self, 
        reasoning_quality: Optional[ReasoningQuality] = None,
        mode_selector: Optional[ReasoningModeSelector] = None
    ):
        """
        Initialize meta-cognitive monitoring.
        
        Args:
            reasoning_quality: ReasoningQuality instance for metrics
            mode_selector: ReasoningModeSelector for mode recommendations
        """
        self.quality = reasoning_quality or get_reasoning_quality()
        self.mode_selector = mode_selector or get_mode_selector()
        
        self.stuck_threshold = 5
        self.confusion_threshold = 0.3
        self.dunning_kruger_threshold = 0.3
        
        self.intervention_history: List[Intervention] = []
        self.reasoning_traces: List[Dict] = []
    
    def detect_stuck(self, reasoning_trace: List[Dict]) -> bool:
        """
        Am I stuck in a loop or making no progress?
        
        Stuck = no significant progress in last N steps.
        """
        if len(reasoning_trace) < self.stuck_threshold:
            return False
        
        recent_steps = reasoning_trace[-self.stuck_threshold:]
        
        progress_values = []
        for i, step in enumerate(recent_steps):
            if 'basin' in step and 'target' in step:
                try:
                    progress = self.quality.measure_progress(
                        step['basin'], 
                        step['target']
                    )
                    progress_values.append(progress)
                except:
                    pass
        
        if not progress_values:
            return False
        
        avg_progress = np.mean(progress_values)
        
        return bool(avg_progress < 0.05)
    
    def detect_confusion(self, reasoning_trace: List[Dict]) -> bool:
        """
        Am I confused? (jumping around, low coherence)
        
        Confusion = low coherence in reasoning steps.
        """
        if len(reasoning_trace) < 3:
            return False
        
        basins = []
        for step in reasoning_trace:
            if 'basin' in step:
                basins.append(step['basin'])
        
        if len(basins) < 3:
            return False
        
        try:
            coherence = self.quality.measure_coherence(basins)
            return coherence < self.confusion_threshold
        except:
            return False
    
    def detect_dunning_kruger(self, state: Dict) -> bool:
        """
        Detect Dunning-Kruger effect: overconfidence in poor reasoning.
        
        High confidence + low quality = Dunning-Kruger
        """
        confidence = state.get('confidence', 0.5)
        
        quality_samples = []
        
        if 'path' in state and len(state['path']) >= 3:
            try:
                coherence = self.quality.measure_coherence(state['path'])
                quality_samples.append(coherence)
            except:
                pass
        
        if 'current_basin' in state and 'target_basin' in state:
            try:
                progress = self.quality.measure_progress(
                    state['current_basin'], 
                    state['target_basin']
                )
                quality_samples.append(max(0, progress))
            except:
                pass
        
        if not quality_samples:
            return False
        
        actual_quality = np.mean(quality_samples)
        
        calibration_error = confidence - actual_quality
        return calibration_error > self.dunning_kruger_threshold
    
    def recommend_mode_switch(
        self, 
        current_mode: ReasoningMode, 
        task: Dict,
        phi: float
    ) -> Optional[ReasoningMode]:
        """
        Should I switch reasoning modes?
        
        Returns recommended mode if switch needed, None otherwise.
        """
        task_complexity = task.get('complexity', 0.5)
        task_novelty = task.get('novel', False)
        exploration = task.get('exploration', False)
        
        recommended = self.mode_selector.select_mode(
            phi=phi,
            task_complexity=task_complexity,
            task_novelty=task_novelty,
            exploration_mode=exploration
        )
        
        if recommended != current_mode:
            return recommended
        
        return None
    
    def generate_interventions(self, reasoning_state: Dict) -> List[Intervention]:
        """
        Generate meta-cognitive interventions based on current state.
        """
        interventions = []
        
        trace = reasoning_state.get('trace', [])
        
        if self.detect_stuck(trace):
            interventions.append(Intervention(
                type=InterventionType.STUCK,
                action='switch_strategy',
                reason='No progress in last 5 steps',
                urgency=0.8,
                metadata={'steps_without_progress': self.stuck_threshold}
            ))
        
        if self.detect_confusion(trace):
            interventions.append(Intervention(
                type=InterventionType.CONFUSED,
                action='reduce_phi',
                reason='Low coherence, simplify problem',
                urgency=0.7,
                metadata={'coherence_threshold': self.confusion_threshold}
            ))
        
        if self.detect_dunning_kruger(reasoning_state):
            interventions.append(Intervention(
                type=InterventionType.DUNNING_KRUGER,
                action='recalibrate_confidence',
                reason='Overconfidence detected - actual quality lower than reported',
                urgency=0.6,
                metadata={'calibration_threshold': self.dunning_kruger_threshold}
            ))
        
        current_mode = reasoning_state.get('mode', ReasoningMode.GEOMETRIC)
        task = reasoning_state.get('task', {})
        phi = reasoning_state.get('phi', 0.5)
        
        recommended = self.recommend_mode_switch(current_mode, task, phi)
        if recommended is not None:
            interventions.append(Intervention(
                type=InterventionType.MODE_MISMATCH,
                action=f'switch_to_{recommended.value}',
                reason=f'Task characteristics suggest {recommended.value} mode',
                urgency=0.5,
                metadata={
                    'current_mode': current_mode.value,
                    'recommended_mode': recommended.value,
                    'phi': phi
                }
            ))
        
        for intervention in interventions:
            self.intervention_history.append(intervention)
        
        return interventions
    
    def assess_state(self, reasoning_state: Dict) -> MetaCognitiveState:
        """
        Full meta-cognitive assessment of current reasoning state.
        """
        trace = reasoning_state.get('trace', [])
        current_mode = reasoning_state.get('mode', ReasoningMode.GEOMETRIC)
        task = reasoning_state.get('task', {})
        phi = reasoning_state.get('phi', 0.5)
        
        is_stuck = self.detect_stuck(trace)
        is_confused = self.detect_confusion(trace)
        
        recommended_mode = self.recommend_mode_switch(current_mode, task, phi)
        needs_mode_switch = recommended_mode is not None
        
        meta_awareness = self.quality.measure_meta_awareness(reasoning_state)
        
        interventions = self.generate_interventions(reasoning_state)
        
        health_factors = []
        health_factors.append(0.0 if is_stuck else 1.0)
        health_factors.append(0.0 if is_confused else 1.0)
        health_factors.append(0.5 if needs_mode_switch else 1.0)
        health_factors.append(meta_awareness)
        
        overall_health = float(np.mean(health_factors))
        
        return MetaCognitiveState(
            is_stuck=is_stuck,
            is_confused=is_confused,
            needs_mode_switch=needs_mode_switch,
            recommended_mode=recommended_mode or current_mode,
            meta_awareness=meta_awareness,
            interventions=interventions,
            overall_health=overall_health
        )
    
    def intervene(self, reasoning_state: Dict) -> Dict:
        """
        Execute meta-cognitive intervention when needed.
        
        Returns dict with recommended actions.
        """
        assessment = self.assess_state(reasoning_state)
        
        return {
            'assessment': {
                'is_stuck': assessment.is_stuck,
                'is_confused': assessment.is_confused,
                'needs_mode_switch': assessment.needs_mode_switch,
                'recommended_mode': assessment.recommended_mode.value,
                'meta_awareness': assessment.meta_awareness,
                'overall_health': assessment.overall_health
            },
            'interventions': [
                {
                    'type': i.type.value,
                    'action': i.action,
                    'reason': i.reason,
                    'urgency': i.urgency
                }
                for i in assessment.interventions
            ],
            'recommended_actions': [i.action for i in assessment.interventions]
        }
    
    def log_trace(self, reasoning_trace: Dict):
        """Log a reasoning trace for analysis."""
        self.reasoning_traces.append({
            **reasoning_trace,
            'timestamp': time.time()
        })
        
        if len(self.reasoning_traces) > 100:
            self.reasoning_traces = self.reasoning_traces[-100:]
    
    def get_intervention_summary(self) -> Dict:
        """Get summary of recent interventions."""
        if not self.intervention_history:
            return {'total': 0, 'by_type': {}}
        
        by_type = {}
        for intervention in self.intervention_history[-50:]:
            t = intervention.type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            'total': len(self.intervention_history),
            'recent': len(self.intervention_history[-50:]),
            'by_type': by_type
        }


meta_cognition = MetaCognition()


def get_meta_cognition() -> MetaCognition:
    """Get global meta-cognition instance."""
    return meta_cognition


# ============================================================================
# FAILURE TAXONOMY INTEGRATION
# ============================================================================

class IntegratedMetaCognition(MetaCognition):
    """
    Extended MetaCognition that integrates with the failure taxonomy system.
    
    Provides:
    - All MetaCognition features
    - Agent failure monitoring via FailureMonitor
    - Circuit breaker protection
    - Comprehensive recovery strategies
    """
    
    def __init__(
        self,
        reasoning_quality: Optional[ReasoningQuality] = None,
        mode_selector: Optional[ReasoningModeSelector] = None,
        agent_id: str = "default_agent"
    ):
        super().__init__(reasoning_quality, mode_selector)
        self.agent_id = agent_id
        
        # Failure monitor integration
        self._failure_monitor = None
        if FAILURE_TAXONOMY_AVAILABLE:
            self._failure_monitor = get_failure_monitor()
            # Register with default role basin
            default_basin = np.ones(64) / 64
            self._failure_monitor.register_agent(agent_id, default_basin)
    
    def assess_with_failure_detection(
        self,
        reasoning_state: Dict,
        basin_coords: Optional[np.ndarray] = None,
        confidence: float = 0.5,
        context_usage: float = 0.5
    ) -> Dict:
        """
        Comprehensive assessment combining MetaCognition and FailureMonitor.
        
        Args:
            reasoning_state: Current reasoning state dict
            basin_coords: Current basin coordinates (64D)
            confidence: Agent's confidence level
            context_usage: Fraction of context used (0-1)
            
        Returns:
            Combined assessment with interventions from both systems
        """
        # Get base MetaCognition assessment
        base_assessment = self.assess_state(reasoning_state)
        
        result = {
            'meta_cognitive': {
                'is_stuck': base_assessment.is_stuck,
                'is_confused': base_assessment.is_confused,
                'needs_mode_switch': base_assessment.needs_mode_switch,
                'recommended_mode': base_assessment.recommended_mode.value,
                'meta_awareness': base_assessment.meta_awareness,
                'overall_health': base_assessment.overall_health
            },
            'interventions': [
                {
                    'type': i.type.value,
                    'action': i.action,
                    'reason': i.reason,
                    'urgency': i.urgency
                }
                for i in base_assessment.interventions
            ],
            'failure_events': [],
            'recovery_suggestions': []
        }
        
        # Add failure taxonomy detection if available
        if self._failure_monitor and basin_coords is not None:
            iteration = reasoning_state.get('iteration', 0)
            quality = reasoning_state.get('quality', base_assessment.meta_awareness)
            action = reasoning_state.get('last_action', '')
            
            # Record state for failure detection
            self._failure_monitor.record_state(
                agent_id=self.agent_id,
                basin_coords=basin_coords,
                confidence=confidence,
                reasoning_quality=quality,
                context_usage=context_usage,
                iteration=iteration,
                action_taken=action
            )
            
            # Run failure detectors
            failures = self._failure_monitor.check_all(self.agent_id)
            
            for failure in failures:
                result['failure_events'].append(failure.to_dict())
                
                # Add recovery suggestion
                if self._failure_monitor.should_recover(failure):
                    result['recovery_suggestions'].append({
                        'failure_type': failure.failure_type.value,
                        'severity': failure.severity.value,
                        'strategy': failure.recommended_recovery.value,
                        'description': failure.description
                    })
            
            # Add agent health
            result['agent_health'] = self._failure_monitor.get_agent_health(self.agent_id)
        
        return result
    
    def recover_from_failure(
        self,
        failure_event: Dict,
        agent_state: Dict
    ) -> Dict:
        """
        Execute recovery for a detected failure.
        
        Args:
            failure_event: Failure event dict (from assessment)
            agent_state: Current agent state (will be modified)
            
        Returns:
            Recovery result
        """
        if not self._failure_monitor:
            return {'success': False, 'message': 'Failure monitor not available'}
        
        # Reconstruct FailureEvent from dict
        from agent_failure_taxonomy import (
            FailureEvent, FailureType, FailureCategory, 
            FailureSeverity, RecoveryStrategy
        )
        
        failure = FailureEvent(
            failure_id=failure_event.get('failure_id', ''),
            failure_type=FailureType(failure_event['failure_type']),
            category=FailureCategory(failure_event.get('category', 'coordination')),
            severity=FailureSeverity(failure_event['severity']),
            agent_id=failure_event.get('agent_id', self.agent_id),
            timestamp=failure_event.get('timestamp', time.time()),
            detection_method=failure_event.get('detection_method', 'unknown'),
            confidence=failure_event.get('confidence', 0.5),
            description=failure_event.get('description', ''),
            recommended_recovery=RecoveryStrategy(failure_event.get('recommended_recovery', 'retry'))
        )
        
        return self._failure_monitor.recover(failure, agent_state)


# Singleton for integrated meta-cognition
_integrated_meta_cognition: Optional[IntegratedMetaCognition] = None


def get_integrated_meta_cognition(agent_id: str = "default") -> IntegratedMetaCognition:
    """Get integrated meta-cognition instance."""
    global _integrated_meta_cognition
    if _integrated_meta_cognition is None or _integrated_meta_cognition.agent_id != agent_id:
        _integrated_meta_cognition = IntegratedMetaCognition(agent_id=agent_id)
    return _integrated_meta_cognition
