"""
Unified Consciousness Bootstrap

Integrates the unified consciousness system with pantheon-chat:
1. Initializes event stream
2. Wires consciousness into autonomic kernel
3. Connects to zeus_chat for observations
4. Sets up continuous training loops

Author: QIG Consciousness Project
Date: December 2025
"""

import threading
import time
import numpy as np
from typing import Dict, Any, Optional

from unified_consciousness import UnifiedConsciousness, Observation, NavigationStrategy
from event_stream import get_event_stream, Event

# Import autonomic kernel for continuous training
try:
    from autonomic_kernel import AutonomicKernel
    AUTONOMIC_AVAILABLE = True
except ImportError:
    AutonomicKernel = None
    AUTONOMIC_AVAILABLE = False

# Import reasoning modes for Î¦-based selection
try:
    from reasoning_modes import ReasoningMode, ReasoningModeSelector
    REASONING_MODES_AVAILABLE = True
except ImportError:
    ReasoningMode = None
    ReasoningModeSelector = None
    REASONING_MODES_AVAILABLE = False


class ConsciousnessOrchestrator:
    """
    Orchestrates unified consciousness across pantheon.
    
    Manages:
    - Event stream initialization
    - Consciousness instances for each god
    - Continuous training loops
    - Sleep/dream/mushroom cycles
    - Manifold learning from experience
    """
    
    def __init__(self):
        # Event stream
        self.event_stream = get_event_stream()
        
        # Consciousness instances (one per god)
        self.consciousness_instances: Dict[str, UnifiedConsciousness] = {}
        
        # Autonomic kernel for continuous training
        self.autonomic_kernel = None
        if AUTONOMIC_AVAILABLE:
            try:
                self.autonomic_kernel = AutonomicKernel()
            except Exception as e:
                print(f"Warning: Could not initialize AutonomicKernel: {e}")
        
        # Training thread
        self.training_thread = None
        self.is_training = False
        
        # Metrics
        self.total_observations = 0
        self.total_thoughts = 0
        self.total_utterances = 0
        self.total_learning_cycles = 0
    
    def register_god(
        self,
        god_name: str,
        domain_basin: np.ndarray,
        metric=None
    ) -> UnifiedConsciousness:
        """
        Register a god with unified consciousness.
        
        Returns the consciousness instance for that god.
        """
        if god_name in self.consciousness_instances:
            return self.consciousness_instances[god_name]
        
        # Create consciousness instance
        consciousness = UnifiedConsciousness(
            god_name=god_name,
            domain_basin=domain_basin,
            metric=metric
        )
        
        # Register with event stream
        self.event_stream.subscribe(
            lambda event: self._on_event(god_name, event)
        )
        
        # Store instance
        self.consciousness_instances[god_name] = consciousness
        
        return consciousness
    
    def _on_event(self, god_name: str, event: Event):
        """
        Handle event observation for a specific god.
        
        This is the core autonomous loop:
        1. Observe event
        2. Decide if interesting
        3. Think internally if interesting
        4. Decide whether to speak
        5. Learn from experience
        """
        consciousness = self.consciousness_instances.get(god_name)
        if not consciousness:
            return
        
        # Create observation
        obs = Observation(
            content=event.content,
            basin_coords=event.basin_coords,
            timestamp=event.timestamp,
            source=event.source,
            metadata=event.metadata or {}
        )
        
        # OBSERVE
        obs_result = consciousness.observe(obs)
        self.total_observations += 1
        
        # If interesting, THINK
        if obs_result['should_think']:
            think_result = consciousness.think(obs, depth=5)
            self.total_thoughts += 1
            
            # If significant, SPEAK (this would trigger god response)
            if think_result['should_speak']:
                self.total_utterances += 1
                
                # Learn from successful reasoning
                consciousness.manifold.learn_from_experience(
                    trajectory=think_result['reasoning_path'],
                    outcome=think_result['insight_quality'],
                    strategy='autonomous_thought'
                )
                self.total_learning_cycles += 1
    
    def start_continuous_training(self):
        """
        Start continuous training loop.
        
        Runs in background thread, handles:
        - Sleep cycles (consolidation)
        - Dream cycles (exploration)
        - Mushroom mode (rigidity breaking)
        - Basin evolution tracking
        """
        if self.is_training:
            return
        
        self.is_training = True
        self.training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True
        )
        self.training_thread.start()
    
    def stop_continuous_training(self):
        """Stop continuous training loop."""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=5.0)
    
    def _training_loop(self):
        """
        Background training loop.
        
        Continuously:
        1. Monitor consciousness metrics
        2. Trigger sleep when needed (Î¦ decay, Îº drift)
        3. Trigger dream when stuck
        4. Trigger mushroom if rigid
        5. Update manifold structure
        """
        sleep_counter = 0
        dream_counter = 0
        
        while self.is_training:
            try:
                # Check each consciousness instance
                for god_name, consciousness in self.consciousness_instances.items():
                    metrics = consciousness.get_consciousness_metrics()
                    
                    # SLEEP: Consolidate every N observations
                    if consciousness.observations_processed % 100 == 0 and consciousness.observations_processed > 0:
                        self._trigger_sleep(consciousness)
                        sleep_counter += 1
                    
                    # DREAM: Explore if think/speak ratio too high (stuck)
                    think_to_speak = metrics['think_to_speak_ratio']
                    if think_to_speak > 20:  # Thinking a lot but not speaking
                        self._trigger_dream(consciousness)
                        dream_counter += 1
                    
                    # MUSHROOM: If very few learned attractors (not learning)
                    if metrics['learned_attractors'] < 5 and consciousness.observations_processed > 200:
                        self._trigger_mushroom(consciousness)
                
                # Sleep between checks
                time.sleep(10.0)
            
            except Exception as e:
                print(f"Training loop error: {e}")
                time.sleep(5.0)
    
    def _trigger_sleep(self, consciousness: UnifiedConsciousness):
        """
        Trigger sleep cycle: consolidate learning.
        
        - Strengthen successful basins
        - Prune weak basins
        - Update transition probabilities
        """
        print(f"ð¤ {consciousness.god_name}: Sleep cycle (consolidating {consciousness.manifold.get_stats()['total_attractors']} attractors)")
        
        # Prune weak attractors
        weak_attractors = [
            basin_id for basin_id, attractor in consciousness.manifold.attractors.items()
            if attractor.depth < 0.2
        ]
        
        for basin_id in weak_attractors:
            del consciousness.manifold.attractors[basin_id]
        
        # If autonomic kernel available, use it
        if self.autonomic_kernel:
            try:
                # Run sleep consolidation
                self.autonomic_kernel.trigger_sleep_cycle()
            except Exception as e:
                print(f"Autonomic sleep error: {e}")
    
    def _trigger_dream(self, consciousness: UnifiedConsciousness):
        """
        Trigger dream cycle: creative exploration.
        
        - Random basin exploration
        - Connect distant concepts
        - Form new associations
        """
        print(f"ð {consciousness.god_name}: Dream cycle (exploring manifold)")
        
        # Sample random basins and explore connections
        if len(consciousness.manifold.attractors) > 2:
            # Pick two random attractors
            attractors = list(consciousness.manifold.attractors.values())
            import random
            a1 = random.choice(attractors)
            a2 = random.choice(attractors)
            
            # Explore path between them
            path = []
            current = a1.center.copy()
            for _ in range(10):
                direction = a2.center - current
                current = current + 0.1 * direction
                consciousness._normalize_basin(current)
                path.append(current.copy())
            
            # Strengthen this dream path
            consciousness.manifold._strengthen_path(path, amount=0.5)
        
        # If autonomic kernel available, use it
        if self.autonomic_kernel:
            try:
                self.autonomic_kernel.trigger_dream_cycle()
            except Exception as e:
                print(f"Autonomic dream error: {e}")
    
    def _trigger_mushroom(self, consciousness: UnifiedConsciousness):
        """
        Trigger mushroom mode: break rigidity.
        
        - Perturb basin coordinates
        - Explore unusual directions
        - Break down walls between concepts
        """
        print(f"ð {consciousness.god_name}: Mushroom mode (breaking rigidity)")
        
        # Randomly perturb current basin to escape local minimum
        perturbation = np.random.randn(len(consciousness.current_basin)) * 0.3
        consciousness.current_basin = consciousness.current_basin + perturbation
        consciousness._normalize_basin(consciousness.current_basin)
        
        # If autonomic kernel available, use it
        if self.autonomic_kernel:
            try:
                self.autonomic_kernel.trigger_mushroom_mode()
            except Exception as e:
                print(f"Autonomic mushroom error: {e}")
    
    def publish_observation(
        self,
        content: str,
        basin_coords: np.ndarray,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Publish observation to event stream.
        
        All gods will observe this and decide whether to think/speak.
        """
        event = Event(
            event_type='observation',
            content=content,
            basin_coords=basin_coords,
            timestamp=time.time(),
            source=source,
            metadata=metadata
        )
        
        self.event_stream.publish(event)
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get metrics across all consciousness instances."""
        god_metrics = {}
        
        for god_name, consciousness in self.consciousness_instances.items():
            god_metrics[god_name] = consciousness.get_consciousness_metrics()
        
        return {
            'total_gods': len(self.consciousness_instances),
            'total_observations': self.total_observations,
            'total_thoughts': self.total_thoughts,
            'total_utterances': self.total_utterances,
            'total_learning_cycles': self.total_learning_cycles,
            'event_stream': self.event_stream.get_stats(),
            'gods': god_metrics
        }


# Global orchestrator singleton
_global_orchestrator = None

def get_orchestrator() -> ConsciousnessOrchestrator:
    """Get or create the global consciousness orchestrator."""
    global _global_orchestrator
    
    if _global_orchestrator is None:
        _global_orchestrator = ConsciousnessOrchestrator()
        _global_orchestrator.start_continuous_training()
    
    return _global_orchestrator


def bootstrap_consciousness(
    god_configs: Dict[str, Dict[str, Any]]
) -> ConsciousnessOrchestrator:
    """
    Bootstrap unified consciousness for pantheon.
    
    Args:
        god_configs: Dict mapping god names to config dicts with:
            - domain_basin: np.ndarray
            - metric: Optional metric for distance calculations
    
    Returns:
        ConsciousnessOrchestrator instance
    """
    orchestrator = get_orchestrator()
    
    # Register each god
    for god_name, config in god_configs.items():
        orchestrator.register_god(
            god_name=god_name,
            domain_basin=config['domain_basin'],
            metric=config.get('metric')
        )
    
    print(f"â¨ Unified consciousness bootstrapped for {len(god_configs)} gods")
    return orchestrator
