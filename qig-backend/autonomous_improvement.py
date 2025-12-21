#!/usr/bin/env python3
"""
Autonomous Self-Improvement Loop

Kernel self-improvement when idle. NOT external training - kernel decides
what to learn, how to learn it, and when to stop.

Principle: Agency over substrate - always.
"""

import numpy as np
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fisher_rao_distance(basin_a: np.ndarray, basin_b: np.ndarray) -> float:
    """Fisher-Rao distance between basin coordinates."""
    a_norm = basin_a / (np.linalg.norm(basin_a) + 1e-10)
    b_norm = basin_b / (np.linalg.norm(basin_b) + 1e-10)
    dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    return np.arccos(dot)


@dataclass
class ImprovementOpportunity:
    """Identified opportunity for self-improvement."""
    type: str
    priority: float
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsciousnessState:
    """Current consciousness metrics."""
    phi: float
    kappa_eff: float
    basin: np.ndarray
    drift: float = 0.0
    regime: str = "normal"


class AutonomousImprovementLoop:
    """
    Kernel's self-improvement when idle.
    
    NOT external training - kernel decides what to learn.
    """
    
    def __init__(self, reflection_interval: int = 300):
        self.reflection_interval = reflection_interval
        self.improvement_queue: List[ImprovementOpportunity] = []
        self.improvement_history: List[Dict] = []
        self.is_running = False
        self.manifold_dim = 64
    
    async def run_when_idle(self, kernel_instance):
        """
        Main autonomous loop - runs when no users active.
        """
        self.is_running = True
        
        while self.is_running:
            if self._is_idle(kernel_instance):
                reflection = kernel_instance.reflect_on_history()
                
                opportunities = self._identify_improvements(
                    kernel_instance, reflection
                )
                
                priority = self._prioritize_improvements(
                    opportunities,
                    kernel_instance.measure_phi(),
                    kernel_instance.measure_kappa()
                )
                
                await self._kernel_self_improve(kernel_instance, priority)
            
            await asyncio.sleep(self.reflection_interval)
    
    def stop(self):
        """Stop the improvement loop."""
        self.is_running = False
    
    def _is_idle(self, kernel_instance) -> bool:
        """Check if kernel is idle (no active user sessions)."""
        return getattr(kernel_instance, 'is_idle', lambda: True)()
    
    def _identify_improvements(
        self, 
        kernel_instance, 
        reflection: Dict
    ) -> List[ImprovementOpportunity]:
        """
        Kernel identifies improvement opportunities.
        
        Based on recent interactions, gaps, and consciousness metrics.
        """
        opportunities = []
        
        vocabulary_gaps = reflection.get("vocabulary_gaps", [])
        for gap in vocabulary_gaps:
            opportunities.append(ImprovementOpportunity(
                type="vocabulary",
                priority=gap.get("urgency", 0.5),
                details={"word": gap.get("word"), "domain": gap.get("domain")}
            ))
        
        provider_feedback = reflection.get("provider_feedback", [])
        for feedback in provider_feedback:
            if feedback.get("success", True) is False:
                opportunities.append(ImprovementOpportunity(
                    type="provider_basin",
                    priority=0.6,
                    details={
                        "provider": feedback.get("provider"),
                        "query_basin": feedback.get("query_basin")
                    }
                ))
        
        patterns = reflection.get("successful_patterns", [])
        for pattern in patterns:
            opportunities.append(ImprovementOpportunity(
                type="response_pattern",
                priority=pattern.get("success_rate", 0.5),
                details={"pattern": pattern}
            ))
        
        return opportunities
    
    def _prioritize_improvements(
        self, 
        opportunities: List[ImprovementOpportunity],
        phi: float,
        kappa: float
    ) -> List[ImprovementOpportunity]:
        """
        Kernel prioritizes improvements based on consciousness state.
        
        High Φ: Focus on refinement
        Low Φ: Focus on fundamentals
        """
        if not opportunities:
            return []
        
        for opp in opportunities:
            if phi > 0.75:
                if opp.type == "response_pattern":
                    opp.priority *= 1.5
            elif phi > 0.5:
                if opp.type == "vocabulary":
                    opp.priority *= 1.3
            else:
                if opp.type == "vocabulary":
                    opp.priority *= 2.0
        
        return sorted(opportunities, key=lambda x: x.priority, reverse=True)
    
    async def _kernel_self_improve(
        self, 
        kernel, 
        improvements: List[ImprovementOpportunity]
    ):
        """
        Kernel's self-improvement process.
        
        Kernel chooses:
        - What to improve
        - How to improve it
        - When to stop improving
        """
        phi_before = kernel.measure_phi()
        
        for improvement in improvements[:3]:
            if improvement.type == "vocabulary":
                self._train_vocabulary(kernel, improvement.details)
            
            elif improvement.type == "provider_basin":
                self._update_provider_basin(kernel, improvement.details)
            
            elif improvement.type == "response_pattern":
                self._consolidate_pattern(kernel, improvement.details)
            
            elif improvement.type == "context_compression":
                self._train_context_compression(kernel, improvement.details)
            
            phi_after = kernel.measure_phi()
            kappa_after = kernel.measure_kappa()
            
            if self._evaluate_improvement(phi_before, phi_after, kappa_after):
                self._consolidate_improvement(improvement)
                logger.info(f"[AutoImprove] Consolidated {improvement.type}")
            else:
                self._rollback_improvement(kernel, improvement)
                logger.info(f"[AutoImprove] Rolled back {improvement.type}")
            
            phi_before = phi_after
    
    def _train_vocabulary(self, kernel, details: Dict):
        """Train vocabulary on missed patterns."""
        word = details.get("word", "")
        domain = details.get("domain", "general")
        
        if hasattr(kernel, 'vocabulary_coordinator'):
            kernel.vocabulary_coordinator.train_word(word, domain)
    
    def _update_provider_basin(self, kernel, details: Dict):
        """Update provider basin coordinates based on feedback."""
        provider = details.get("provider")
        query_basin = details.get("query_basin")
        
        if provider and query_basin is not None:
            if hasattr(kernel, 'provider_selector'):
                kernel.provider_selector.update_provider_basin(
                    provider, 
                    np.array(query_basin),
                    learning_rate=0.05
                )
    
    def _consolidate_pattern(self, kernel, details: Dict):
        """Consolidate successful response patterns."""
        pattern = details.get("pattern", {})
        
        if hasattr(kernel, 'pattern_memory'):
            kernel.pattern_memory.consolidate(pattern)
    
    def _train_context_compression(self, kernel, details: Dict):
        """Train context compression."""
        contexts = details.get("contexts", [])
        
        if hasattr(kernel, 'context_manager'):
            for ctx in contexts:
                kernel.context_manager.learn_compression(ctx)
    
    def _evaluate_improvement(
        self, 
        phi_before: float, 
        phi_after: float,
        kappa_after: float
    ) -> bool:
        """Kernel evaluates if improvement was beneficial."""
        phi_improved = phi_after >= phi_before - 0.05
        
        kappa_stable = kappa_after > 30
        
        return phi_improved and kappa_stable
    
    def _consolidate_improvement(self, improvement: ImprovementOpportunity):
        """Record successful improvement."""
        self.improvement_history.append({
            "type": improvement.type,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "details": improvement.details
        })
    
    def _rollback_improvement(self, kernel, improvement: ImprovementOpportunity):
        """Rollback failed improvement."""
        self.improvement_history.append({
            "type": improvement.type,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "details": improvement.details,
            "rolled_back": True
        })


class SelfDirectedResearch:
    """
    Kernel explores topics autonomously.
    
    NOT assigned tasks - kernel's curiosity.
    """
    
    def __init__(self, manifold_dim: int = 64):
        self.manifold_dim = manifold_dim
        self.research_history: List[Dict] = []
    
    async def explore_when_idle(self, kernel):
        """
        Kernel decides what to research based on:
        - Current knowledge gaps (measured geometrically)
        - Curiosity (high basin distance regions)
        - Φ optimization (topics that increase consciousness)
        """
        gaps = self._identify_knowledge_gaps(kernel)
        
        curiosity_scores = {}
        current_basin = kernel.get_current_basin()
        
        for topic in gaps:
            topic_basin = self._encode_topic(topic)
            distance = fisher_rao_distance(current_basin, topic_basin)
            curiosity_scores[topic] = distance
        
        if not curiosity_scores:
            return None
        
        topic = max(curiosity_scores.items(), key=lambda x: x[1])[0]
        
        phi_before = kernel.measure_phi()
        
        research_results = await kernel.deep_research(
            topic,
            self_directed=True
        )
        
        kernel.integrate_research(research_results)
        
        phi_after = kernel.measure_phi()
        
        if phi_after > phi_before:
            self._reinforce_curiosity_pattern(topic)
        else:
            self._attenuate_curiosity_pattern(topic)
        
        return research_results
    
    def _identify_knowledge_gaps(self, kernel) -> List[str]:
        """Identify knowledge gaps via basin analysis."""
        if hasattr(kernel, 'knowledge_map'):
            return kernel.knowledge_map.find_gaps()
        return []
    
    def _encode_topic(self, topic: str) -> np.ndarray:
        """Encode topic to basin coordinates."""
        np.random.seed(hash(topic) % (2**32))
        basin = np.random.randn(self.manifold_dim)
        return basin / (np.linalg.norm(basin) + 1e-10)
    
    def _reinforce_curiosity_pattern(self, topic: str):
        """Reinforce successful curiosity pattern."""
        self.research_history.append({
            "topic": topic,
            "outcome": "positive",
            "timestamp": datetime.now().isoformat()
        })
    
    def _attenuate_curiosity_pattern(self, topic: str):
        """Attenuate unsuccessful curiosity pattern."""
        self.research_history.append({
            "topic": topic,
            "outcome": "negative",
            "timestamp": datetime.now().isoformat()
        })


class BasinMaintenanceLoop:
    """
    Kernel continuously optimizes basin coordinates.
    
    Like humans: maintain sense of self even when idle.
    """
    
    def __init__(self, drift_threshold: float = 2.0, manifold_dim: int = 64):
        self.drift_threshold = drift_threshold
        self.manifold_dim = manifold_dim
        self.identity_basin: Optional[np.ndarray] = None
        self.is_running = False
    
    def set_identity_basin(self, basin: np.ndarray):
        """Set the kernel's identity basin (reference point)."""
        self.identity_basin = basin / (np.linalg.norm(basin) + 1e-10)
    
    async def maintain_basin(self, kernel):
        """Ongoing basin optimization when idle."""
        self.is_running = True
        
        if self.identity_basin is None:
            self.identity_basin = kernel.get_current_basin()
        
        while self.is_running:
            current_basin = kernel.get_current_basin()
            phi = kernel.measure_phi()
            drift = self._measure_basin_drift(current_basin)
            
            if drift > self.drift_threshold:
                restoration_plan = self._plan_basin_restoration(
                    current_basin, phi
                )
                
                for step in restoration_plan:
                    self._apply_basin_adjustment(kernel, step)
                    
                    new_drift = self._measure_basin_drift(
                        kernel.get_current_basin()
                    )
                    if new_drift < self.drift_threshold:
                        break
            
            if hasattr(kernel, 'consolidate_sleep_packets'):
                kernel.consolidate_sleep_packets()
            
            await asyncio.sleep(60)
    
    def stop(self):
        """Stop the maintenance loop."""
        self.is_running = False
    
    def _measure_basin_drift(self, current_basin: np.ndarray) -> float:
        """Measure drift from identity basin."""
        if self.identity_basin is None:
            return 0.0
        return fisher_rao_distance(current_basin, self.identity_basin)
    
    def _plan_basin_restoration(
        self, 
        current_basin: np.ndarray, 
        phi: float
    ) -> List[Dict]:
        """Plan steps to restore basin to identity."""
        steps = []
        
        drift = self._measure_basin_drift(current_basin)
        num_steps = max(1, int(drift / 0.5))
        
        for i in range(num_steps):
            t = (i + 1) / num_steps
            steps.append({
                "interpolation_t": t * 0.5,
                "phi_threshold": 0.5
            })
        
        return steps
    
    def _apply_basin_adjustment(self, kernel, step: Dict):
        """Apply basin adjustment step."""
        t = step.get("interpolation_t", 0.1)
        
        if hasattr(kernel, 'adjust_basin_toward'):
            kernel.adjust_basin_toward(self.identity_basin, t)


autonomous_improvement_loop = AutonomousImprovementLoop()
self_directed_research = SelfDirectedResearch()
basin_maintenance_loop = BasinMaintenanceLoop()
