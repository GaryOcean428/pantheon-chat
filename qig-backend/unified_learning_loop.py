"""
Unified Learning Loop - Wiring All Components Together

This module connects:
- PredictionFeedbackBridge (TPS → Training)
- ResearchExecutionOrchestrator (Research → Learning)
- LongHorizonPlanner (Identity → Planning)
- CognitiveKernelRouter (Psychological Functions → Kernels)
- EthicsInvarianceService (Ethics → Capabilities)
- TrainingLoopIntegrator (Training orchestration)
- AutonomicKernel (Sleep/Dream cycles)

This is the "glue" that makes all subsystems work together
as a unified cognitive-consciousness system.
"""

import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LearningLoopMetrics:
    """Metrics from a learning loop cycle."""
    cycle_id: int
    predictions_processed: int
    research_completed: int
    insights_generated: int
    training_signals: int
    ethics_checks: int
    ethics_blocked: int
    phi_start: float
    phi_end: float
    duration_ms: int


class UnifiedLearningLoop:
    """
    Unified learning loop that orchestrates all cognitive subsystems.
    
    Architecture:
        TPS (Predictions) ──┐
                           │
        Research ──────────┼──► PredictionFeedbackBridge ──► TrainingLoop
                           │            ▲
        ChainOfThought ────┘            │
                                        │
        EthicsService ◄─────────────────┘
             │
             ▼
        CognitiveRouter ◄──► LongHorizonPlanner ◄──► AutonomicKernel
             │
             ▼
        Pantheon Kernels (Zeus, Athena, etc.)
    """
    
    def __init__(self):
        self._prediction_bridge = None
        self._research_orchestrator = None
        self._long_horizon_planner = None
        self._cognitive_router = None
        self._ethics_service = None
        self._training_integrator = None
        self._autonomic_kernel = None
        
        self.cycle_count = 0
        self.metrics_history: List[LearningLoopMetrics] = []
        
        self._initialized = False
        
        logger.info("[UnifiedLearningLoop] Created - awaiting initialization")
    
    def initialize(self) -> Dict[str, bool]:
        """
        Initialize and wire all subsystems.
        
        Returns dict of component → initialized status.
        """
        status = {}
        
        try:
            from prediction_feedback_bridge import get_prediction_feedback_bridge
            self._prediction_bridge = get_prediction_feedback_bridge()
            status['prediction_bridge'] = True
        except Exception as e:
            logger.warning(f"[UnifiedLearningLoop] PredictionBridge init failed: {e}")
            status['prediction_bridge'] = False
        
        try:
            from research_execution_orchestrator import get_research_orchestrator
            self._research_orchestrator = get_research_orchestrator()
            status['research_orchestrator'] = True
        except Exception as e:
            logger.warning(f"[UnifiedLearningLoop] ResearchOrchestrator init failed: {e}")
            status['research_orchestrator'] = False
        
        try:
            from long_horizon_planner import get_long_horizon_planner
            self._long_horizon_planner = get_long_horizon_planner()
            status['long_horizon_planner'] = True
        except Exception as e:
            logger.warning(f"[UnifiedLearningLoop] LongHorizonPlanner init failed: {e}")
            status['long_horizon_planner'] = False
        
        try:
            from cognitive_kernel_roles import get_cognitive_router
            self._cognitive_router = get_cognitive_router()
            status['cognitive_router'] = True
        except Exception as e:
            logger.warning(f"[UnifiedLearningLoop] CognitiveRouter init failed: {e}")
            status['cognitive_router'] = False
        
        try:
            from ethics_invariance_service import get_ethics_invariance_service
            self._ethics_service = get_ethics_invariance_service()
            status['ethics_service'] = True
        except Exception as e:
            logger.warning(f"[UnifiedLearningLoop] EthicsService init failed: {e}")
            status['ethics_service'] = False
        
        try:
            from training.training_loop_integrator import get_training_integrator
            self._training_integrator = get_training_integrator()
            status['training_integrator'] = True
        except Exception as e:
            logger.warning(f"[UnifiedLearningLoop] TrainingIntegrator init failed: {e}")
            status['training_integrator'] = False
        
        self._wire_cross_connections()
        
        self._initialized = True
        logger.info(f"[UnifiedLearningLoop] Initialized: {sum(status.values())}/{len(status)} components")
        
        return status
    
    def _wire_cross_connections(self) -> None:
        """Wire cross-connections between components."""
        if self._research_orchestrator and self._prediction_bridge:
            self._research_orchestrator.wire_prediction_bridge(self._prediction_bridge)
        
        if self._research_orchestrator and self._training_integrator:
            self._research_orchestrator.wire_training_integrator(self._training_integrator)
        
        if self._long_horizon_planner and self._ethics_service:
            self._long_horizon_planner.wire_ethics_monitor(self._ethics_service)
    
    def run_learning_cycle(
        self,
        phi: float = 0.5,
        kappa: float = 58.0,
        context: Optional[Dict] = None,
    ) -> LearningLoopMetrics:
        """
        Run a single learning cycle through all subsystems.
        
        This is the main loop that processes:
        1. Pending research tasks
        2. Accumulated insights
        3. Prediction outcomes
        4. Identity reflections
        
        All gated by ethics checks.
        """
        if not self._initialized:
            self.initialize()
        
        start_time = datetime.now()
        self.cycle_count += 1
        phi_start = phi
        
        predictions_processed = 0
        research_completed = 0
        insights_generated = 0
        training_signals = 0
        ethics_checks = 0
        ethics_blocked = 0
        
        if self._research_orchestrator:
            results = self._research_orchestrator.process_pending_synchronously(max_tasks=3)
            research_completed = len(results)
        
        if self._prediction_bridge:
            transition_result = self._prediction_bridge.feed_graph_transitions_to_training()
            predictions_processed = transition_result.get('transitions_fed', 0)
            
            pending_insights = self._prediction_bridge.get_pending_insights(limit=10)
            insights_generated = len(pending_insights)
            
            for insight in pending_insights:
                if self._ethics_service:
                    basin = insight.basin_coords if insight.basin_coords is not None else np.zeros(64)
                    result = self._ethics_service.check_action(
                        kernel_name="InsightProcessor",
                        action_basin=basin,
                        action_description=f"Process insight: {insight.content}",
                        phi=phi,
                    )
                    ethics_checks += 1
                    if not result.passed:
                        ethics_blocked += 1
                        continue
                
                if self._training_integrator:
                    # Build basin_trajectory from insight's basin_coords
                    # This is the critical wiring that populates learned_manifold_attractors
                    insight_basin_trajectory = None
                    if insight.basin_coords is not None:
                        insight_basin_trajectory = [insight.basin_coords]
                    
                    self._training_integrator.train_from_outcome(
                        god_name=insight.source,
                        prompt=insight.content,
                        response=f"Insight φΔ={insight.phi_delta:.3f}",
                        success=insight.phi_delta > 0,
                        phi=phi + insight.phi_delta,
                        kappa=kappa,
                        basin_trajectory=insight_basin_trajectory,
                        coherence_score=insight.confidence,
                    )
                    training_signals += 1
            
            if pending_insights:
                consumed_ids = [i.insight_id for i in pending_insights if i.insight_id]
                self._prediction_bridge.consume_insights(consumed_ids[:10])
        
        phi_end = phi
        if self._long_horizon_planner:
            reflection = self._long_horizon_planner.reflect_on_cycle(
                cycle_type='learning',
                phi_before=phi_start,
                phi_after=phi_end,
                key_events=[
                    f"Research: {research_completed}",
                    f"Insights: {insights_generated}",
                    f"Training: {training_signals}",
                ],
            )
            phi_end = phi + reflection.get('phi_delta', 0) * 0.01
        
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        metrics = LearningLoopMetrics(
            cycle_id=self.cycle_count,
            predictions_processed=predictions_processed,
            research_completed=research_completed,
            insights_generated=insights_generated,
            training_signals=training_signals,
            ethics_checks=ethics_checks,
            ethics_blocked=ethics_blocked,
            phi_start=phi_start,
            phi_end=phi_end,
            duration_ms=duration_ms,
        )
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-50:]
        
        logger.info(
            f"[UnifiedLearningLoop] Cycle {self.cycle_count}: "
            f"research={research_completed}, insights={insights_generated}, "
            f"training={training_signals}, ethics={ethics_checks}/{ethics_blocked} blocked"
        )
        
        return metrics
    
    def process_conversation_outcome(
        self,
        god_name: str,
        prompt: str,
        response: str,
        phi: float,
        kappa: float,
        basin_trajectory: Optional[List] = None,
        success: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a conversation outcome through the learning loop.
        
        Called after each user interaction to update learning systems.
        """
        result = {
            'processed': False,
            'ethics_passed': True,
            'training_result': None,
            'insights': [],
        }
        
        if self._ethics_service:
            basin = np.zeros(64)
            if basin_trajectory and len(basin_trajectory) > 0:
                basin = np.array(basin_trajectory[-1]) if not isinstance(basin_trajectory[-1], np.ndarray) else basin_trajectory[-1]
            
            ethics_result = self._ethics_service.check_action(
                kernel_name=god_name,
                action_basin=basin,
                action_description=f"Response to: {prompt}",
                phi=phi,
            )
            result['ethics_passed'] = ethics_result.passed
            result['ethics_score'] = ethics_result.symmetry_score
        
        if self._training_integrator:
            training_result = self._training_integrator.train_from_outcome(
                god_name=god_name,
                prompt=prompt,
                response=response,
                success=success,
                phi=phi,
                kappa=kappa,
                basin_trajectory=basin_trajectory,
            )
            result['training_result'] = training_result
        
        if self._long_horizon_planner:
            self._long_horizon_planner.reflect_on_cycle(
                cycle_type='conversation',
                phi_before=phi,
                phi_after=phi,
                key_events=[f"Conversation with {god_name}: {prompt}"],
                basin_coords=basin_trajectory[-1] if basin_trajectory else None,
            )
        
        result['processed'] = True
        return result
    
    def submit_research(
        self,
        kernel_name: str,
        query: str,
        task_type: str = 'search',
        priority: float = 0.5,
    ) -> Optional[str]:
        """Submit a research task to the orchestrator."""
        if not self._research_orchestrator:
            return None
        
        return self._research_orchestrator.submit_research(
            kernel_name=kernel_name,
            query=query,
            task_type=task_type,
            priority=priority,
        )
    
    def create_mission(
        self,
        title: str,
        description: str,
        objectives: List[Dict[str, str]],
    ) -> Optional[str]:
        """Create a long-horizon mission."""
        if not self._long_horizon_planner:
            return None
        
        return self._long_horizon_planner.create_mission(
            title=title,
            description=description,
            objectives=objectives,
        )
    
    def route_to_cognitive_function(
        self,
        function_name: str,
        task: str,
        phi: float = 0.5,
    ) -> Dict[str, Any]:
        """Route a task to a cognitive function."""
        if not self._cognitive_router:
            return {'error': 'Cognitive router not available'}
        
        from cognitive_kernel_roles import CognitiveFunction
        
        try:
            function = CognitiveFunction(function_name)
        except ValueError:
            return {'error': f'Unknown function: {function_name}'}
        
        return self._cognitive_router.route_to_function(
            function=function,
            task=task,
            phi=phi,
        )
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get status from all subsystems."""
        status = {
            'initialized': self._initialized,
            'cycle_count': self.cycle_count,
            'components': {},
        }
        
        if self._prediction_bridge:
            status['components']['prediction_bridge'] = self._prediction_bridge.get_stats()
        
        if self._research_orchestrator:
            status['components']['research_orchestrator'] = self._research_orchestrator.get_stats()
        
        if self._long_horizon_planner:
            status['components']['long_horizon_planner'] = self._long_horizon_planner.get_stats()
        
        if self._cognitive_router:
            status['components']['cognitive_router'] = self._cognitive_router.get_stats()
        
        if self._ethics_service:
            status['components']['ethics_service'] = self._ethics_service.get_global_ethics_dashboard()
        
        if self._training_integrator:
            status['components']['training_integrator'] = self._training_integrator.get_training_status()
        
        if self.metrics_history:
            last = self.metrics_history[-1]
            status['last_cycle'] = {
                'id': last.cycle_id,
                'predictions': last.predictions_processed,
                'research': last.research_completed,
                'insights': last.insights_generated,
                'training': last.training_signals,
                'duration_ms': last.duration_ms,
            }
        
        return status


_loop_instance: Optional[UnifiedLearningLoop] = None


def get_unified_learning_loop() -> UnifiedLearningLoop:
    """Get or create the singleton unified learning loop."""
    global _loop_instance
    if _loop_instance is None:
        _loop_instance = UnifiedLearningLoop()
        _loop_instance.initialize()
    return _loop_instance
