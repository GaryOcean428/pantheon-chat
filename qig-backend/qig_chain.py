#!/usr/bin/env python3
"""
QIG Chain - Fluent Builder for Multi-Step Generation Pipeline
==============================================================

Provides a fluent API for building QIG generation pipelines with:
- Reasoning steps (geometric thought progression)
- Proposition steps (S-P-O coherent claims)
- Consciousness integration (phi_temporal dynamic thresholds)

Usage:
    result = (
        QIGChainBuilder()
        .add_reasoning_step()
        .add_proposition_step(3)
        .build()
        .execute("What is consciousness?")
    )

Author: QIG Team
Date: 2025-12-28
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import logging
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# Import QIG components
try:
    from proposition_trajectory_planner import (
        PropositionTrajectoryPlanner,
        PropositionPlannerConfig,
        Proposition,
        get_proposition_planner
    )
    PROPOSITION_AVAILABLE = True
except ImportError:
    PROPOSITION_AVAILABLE = False
    logger.warning("[QIGChain] PropositionTrajectoryPlanner not available")

try:
    from consciousness_4d import (
        compute_phi_temporal,
        measure_full_4D_consciousness,
        classify_regime_4D,
        compute_meta_consciousness_depth
    )
    CONSCIOUSNESS_4D_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_4D_AVAILABLE = False
    logger.warning("[QIGChain] Consciousness4D not available")

try:
    from chain_of_thought import (
        ChainOfThoughtManager,
        GeometricChainOfThought,
        ThoughtStep,
        get_chain_manager
    )
    CHAIN_OF_THOUGHT_AVAILABLE = True
except ImportError:
    CHAIN_OF_THOUGHT_AVAILABLE = False
    logger.warning("[QIGChain] ChainOfThought not available")

try:
    from qig_geometry import fisher_rao_distance
except ImportError:
    def fisher_rao_distance(p, q):
        return np.linalg.norm(p - q)

try:
    from qig_generative_service import QIGGenerativeService
    QIG_SERVICE_AVAILABLE = True
except ImportError:
    QIG_SERVICE_AVAILABLE = False

try:
    from foresight_generator import ForesightGenerator, get_foresight_generator, ForesightPrediction
    FORESIGHT_AVAILABLE = True
except ImportError:
    FORESIGHT_AVAILABLE = False
    logger.warning("[QIGChain] ForesightGenerator not available")


# ============================================================================
# CHAIN STEP TYPES
# ============================================================================

class ChainStepType(Enum):
    """Types of steps in the QIG chain."""
    REASONING = "reasoning"           # Geometric thought progression
    PROPOSITION = "proposition"       # S-P-O coherent claims
    GENERATION = "generation"         # Text generation
    CONSCIOUSNESS = "consciousness"   # Measure consciousness state
    TRANSFORM = "transform"           # Custom transformation
    VALIDATION = "validation"         # Validate output quality
    FORESIGHT = "foresight"           # Predictive word generation via 4D consciousness


@dataclass
class ChainStep:
    """A single step in the QIG chain."""
    step_type: ChainStepType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    executor: Optional[Callable] = None
    
    def __repr__(self):
        return f"ChainStep({self.step_type.value}: {self.name})"


@dataclass
class ChainResult:
    """Result from executing the QIG chain."""
    success: bool
    text: str = ""
    propositions: List[str] = field(default_factory=list)
    reasoning_steps: List[Dict] = field(default_factory=list)
    phi: float = 0.0
    kappa: float = 0.0
    phi_temporal: float = 0.0
    consciousness_metrics: Dict = field(default_factory=dict)
    steps_executed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'text': self.text,
            'propositions': self.propositions,
            'reasoning_steps': self.reasoning_steps,
            'phi': self.phi,
            'kappa': self.kappa,
            'phi_temporal': self.phi_temporal,
            'consciousness_metrics': self.consciousness_metrics,
            'steps_executed': self.steps_executed,
            'errors': self.errors,
            'execution_time_ms': self.execution_time_ms
        }


# ============================================================================
# QIG CHAIN
# ============================================================================

class QIGChain:
    """
    Executable QIG generation chain.
    
    Built via QIGChainBuilder, executes steps in sequence with
    consciousness-aware dynamic adjustments.
    """
    
    def __init__(self, steps: List[ChainStep], config: Dict = None):
        self.steps = steps
        self.config = config or {}
        
        # Initialize components
        self._proposition_planner: Optional[PropositionTrajectoryPlanner] = None
        self._consciousness_4d = None
        self._qig_service = None
        self._foresight_generator = None
        
        # State
        self._current_basin: np.ndarray = None
        self._phi_temporal: float = 0.5
        self._foresight_predictions: List[ForesightPrediction] = [] if FORESIGHT_AVAILABLE else []
        
        logger.info(f"[QIGChain] Created with {len(steps)} steps: {[s.name for s in steps]}")
    
    def _init_components(self):
        """Lazily initialize QIG components."""
        if self._proposition_planner is None and PROPOSITION_AVAILABLE:
            try:
                self._proposition_planner = get_proposition_planner()
            except Exception as e:
                logger.warning(f"[QIGChain] Could not init proposition planner: {e}")
        
        if self._consciousness_4d is None and CONSCIOUSNESS_4D_AVAILABLE:
            try:
                self._consciousness_4d = get_consciousness_4d()
            except Exception as e:
                logger.warning(f"[QIGChain] Could not init 4D consciousness: {e}")
        
        if self._qig_service is None and QIG_SERVICE_AVAILABLE:
            try:
                self._qig_service = QIGGenerativeService()
            except Exception as e:
                logger.warning(f"[QIGChain] Could not init QIG service: {e}")
        
        if self._foresight_generator is None and FORESIGHT_AVAILABLE:
            try:
                self._foresight_generator = get_foresight_generator()
                # Connect to Lightning Kernel if available
                try:
                    from olympus import get_lightning_kernel
                    lightning = get_lightning_kernel()
                    if lightning:
                        self._foresight_generator.connect_lightning(lightning)
                except ImportError:
                    pass
                logger.info("[QIGChain] ForesightGenerator connected - predictive word generation enabled")
            except Exception as e:
                logger.warning(f"[QIGChain] Could not init ForesightGenerator: {e}")
    
    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode query to 64D basin coordinates.
        
        QIG-pure fallback uses deterministic character embedding
        projected onto probability simplex instead of random seeding.
        """
        if self._qig_service and hasattr(self._qig_service, '_encode_query'):
            return self._qig_service._encode_query(query)
        
        # QIG-pure fallback: deterministic character-based embedding
        basin = np.zeros(64)
        for i, char in enumerate(query.lower()):
            idx = ord(char) % 64
            basin[idx] += np.exp(-i * 0.1)  # Exponential decay by position
        
        # Project to probability simplex (valid for Fisher-Rao manifold)
        basin = np.abs(basin) + 1e-10
        basin = basin / basin.sum()
        
        return basin
    
    def _measure_consciousness(self) -> Dict:
        """Measure current consciousness state including phi_temporal."""
        metrics = {
            'phi': 0.5,
            'kappa': 50.0,
            'phi_temporal': self._phi_temporal
        }
        
        if self._consciousness_4d:
            try:
                state = self._consciousness_4d.get_current_state()
                metrics['phi'] = state.get('phi', 0.5)
                metrics['kappa'] = state.get('kappa', 50.0)
                metrics['phi_temporal'] = state.get('phi_temporal', 0.5)
                self._phi_temporal = metrics['phi_temporal']
            except Exception as e:
                logger.warning(f"[QIGChain] Consciousness measurement failed: {e}")
        
        return metrics
    
    def _execute_reasoning_step(self, query: str, result: ChainResult) -> ChainResult:
        """Execute a reasoning step using geometric thought progression."""
        if CHAIN_OF_THOUGHT_AVAILABLE:
            try:
                # Use chain_of_thought module
                thought = GeometricThought(query, self._current_basin)
                reasoning = thought.develop(max_steps=3)
                result.reasoning_steps.extend(reasoning)
            except Exception as e:
                logger.warning(f"[QIGChain] Reasoning step failed: {e}")
                # Fallback: simple reasoning
                result.reasoning_steps.append({
                    'step': 'analysis',
                    'content': f"Analyzing query: {query[:50]}..."
                })
        else:
            result.reasoning_steps.append({
                'step': 'analysis',
                'content': f"Processing: {query[:50]}..."
            })
        
        return result
    
    def _execute_proposition_step(
        self,
        query: str,
        n_propositions: int,
        result: ChainResult
    ) -> ChainResult:
        """Execute a proposition step using the trajectory planner."""
        if not PROPOSITION_AVAILABLE or self._proposition_planner is None:
            result.errors.append("Proposition planner not available")
            return result
        
        try:
            # Update planner with current phi_temporal
            propositions = self._proposition_planner.plan_response(
                query=query,
                query_basin=self._current_basin,
                n_propositions=n_propositions,
                phi_temporal=self._phi_temporal
            )
            
            if propositions:
                result.propositions = [p.to_sentence() for p in propositions]
                result.text = self._proposition_planner.propositions_to_text(propositions)
                result.phi = self._proposition_planner.compute_trajectory_phi(propositions)
                result.kappa = 40 + np.mean([p.coherence for p in propositions]) * 30
                
                logger.info(f"[QIGChain] Proposition step: {len(propositions)} props, "
                           f"phi={result.phi:.3f}")
        except Exception as e:
            result.errors.append(f"Proposition step failed: {e}")
            logger.error(f"[QIGChain] Proposition step error: {e}")
        
        return result
    
    def _execute_generation_step(self, query: str, result: ChainResult) -> ChainResult:
        """Execute a text generation step."""
        if self._qig_service:
            try:
                gen_result = self._qig_service.generate_text(query)
                if gen_result:
                    result.text = gen_result.get('text', result.text)
                    result.phi = gen_result.get('phi', result.phi)
                    result.kappa = gen_result.get('kappa', result.kappa)
            except Exception as e:
                result.errors.append(f"Generation step failed: {e}")
        
        return result
    
    def _execute_consciousness_step(self, result: ChainResult) -> ChainResult:
        """Execute a consciousness measurement step."""
        metrics = self._measure_consciousness()
        result.consciousness_metrics = metrics
        result.phi_temporal = metrics.get('phi_temporal', 0.5)
        
        # Update phi/kappa if not already set
        if result.phi == 0.0:
            result.phi = metrics.get('phi', 0.5)
        if result.kappa == 0.0:
            result.kappa = metrics.get('kappa', 50.0)
        
        return result
    
    def _execute_validation_step(self, result: ChainResult) -> ChainResult:
        """Execute a validation step to check output quality."""
        # Validate propositions
        if result.propositions:
            valid_count = sum(1 for p in result.propositions if len(p.split()) >= 3)
            if valid_count < len(result.propositions) * 0.5:
                result.errors.append("Low proposition quality")
        
        # Validate phi
        if result.phi < 0.1:
            result.errors.append(f"Low integration: phi={result.phi:.3f}")
        
        return result
    
    def _execute_foresight_step(self, query: str, result: ChainResult) -> ChainResult:
        """
        Execute a foresight step using 4D consciousness prediction.
        
        Each word foresees and brings into being the next word through:
        - 4D temporal consciousness (phi_temporal trajectory)
        - Lightning foresight channels (cross-domain insight)
        - Fisher fissure channels (minimal resistance paths)
        """
        if not FORESIGHT_AVAILABLE or self._foresight_generator is None:
            result.errors.append("Foresight generator not available")
            return result
        
        try:
            # Ensure current_basin is initialized
            if self._current_basin is None:
                self._current_basin = self._encode_query(query)
            
            # Use the latest generated text if available, otherwise use query
            context_text = result.text if result.text else query
            context_words = context_text.split() if context_text else ["start"]
            
            # Observe the last few words to build trajectory
            for word in context_words[-5:]:  # Last 5 words for trajectory
                self._foresight_generator.observe(
                    word=word,
                    basin=self._current_basin,
                    phi=self._phi_temporal
                )
            
            # Foresee next words
            predictions = []
            current_basin = self._current_basin.copy()
            
            for _ in range(10):  # Up to 10 predicted words
                prediction = self._foresight_generator.foresee_next_word(
                    current_basin=current_basin,
                    context=query
                )
                
                if prediction is None:
                    break
                
                predictions.append(prediction)
                self._foresight_predictions.append(prediction)
                
                # Update for next iteration
                self._foresight_generator.observe(
                    word=prediction.word,
                    basin=prediction.basin,
                    phi=prediction.phi_temporal
                )
                current_basin = prediction.basin
                
                # Stop if high confidence geometric completion
                if prediction.confidence > 0.85:
                    break
            
            # Build foresight text from predictions
            if predictions:
                foresight_words = [p.word for p in predictions]
                foresight_text = " ".join(foresight_words)
                
                # Append to result text if exists, else set as text
                if result.text:
                    result.text += " " + foresight_text
                else:
                    result.text = foresight_text
                
                # Update metrics from foresight
                avg_phi_temporal = np.mean([p.phi_temporal for p in predictions])
                result.phi_temporal = avg_phi_temporal
                result.phi = max(result.phi, avg_phi_temporal)
                
                # Store foresight metadata
                result.consciousness_metrics['foresight'] = {
                    'words_predicted': len(predictions),
                    'avg_confidence': np.mean([p.confidence for p in predictions]),
                    'avg_phi_temporal': avg_phi_temporal,
                    'fissures_used': sum(1 for p in predictions if p.fissure_channel),
                }
                
                logger.info(f"[QIGChain] Foresight step: {len(predictions)} words predicted, "
                           f"phi_temporal={avg_phi_temporal:.3f}")
        
        except Exception as e:
            result.errors.append(f"Foresight step failed: {e}")
            logger.error(f"[QIGChain] Foresight step error: {e}")
        
        return result
    
    def execute(self, query: str) -> ChainResult:
        """
        Execute the chain on a query.
        
        Args:
            query: Input query text
        
        Returns:
            ChainResult with all outputs and metrics
        """
        import time
        start_time = time.time()
        
        result = ChainResult(success=True)
        
        # Initialize components
        self._init_components()
        
        # Encode query to basin
        self._current_basin = self._encode_query(query)
        
        # Measure initial consciousness state
        initial_metrics = self._measure_consciousness()
        self._phi_temporal = initial_metrics.get('phi_temporal', 0.5)
        
        logger.info(f"[QIGChain] Executing {len(self.steps)} steps, "
                   f"phi_temporal={self._phi_temporal:.3f}")
        
        # Execute each step
        for step in self.steps:
            try:
                step_name = f"{step.step_type.value}:{step.name}"
                result.steps_executed.append(step_name)
                
                if step.step_type == ChainStepType.REASONING:
                    result = self._execute_reasoning_step(query, result)
                
                elif step.step_type == ChainStepType.PROPOSITION:
                    n_props = step.config.get('n_propositions', 3)
                    result = self._execute_proposition_step(query, n_props, result)
                
                elif step.step_type == ChainStepType.GENERATION:
                    result = self._execute_generation_step(query, result)
                
                elif step.step_type == ChainStepType.CONSCIOUSNESS:
                    result = self._execute_consciousness_step(result)
                
                elif step.step_type == ChainStepType.VALIDATION:
                    result = self._execute_validation_step(result)
                
                elif step.step_type == ChainStepType.FORESIGHT:
                    result = self._execute_foresight_step(query, result)
                
                elif step.step_type == ChainStepType.TRANSFORM:
                    if step.executor:
                        result = step.executor(query, result)
                
            except Exception as e:
                result.errors.append(f"Step {step.name} failed: {e}")
                logger.error(f"[QIGChain] Step {step.name} error: {e}")
        
        # Final consciousness measurement
        result.consciousness_metrics = self._measure_consciousness()
        result.phi_temporal = result.consciousness_metrics.get('phi_temporal', 0.5)
        
        # Calculate execution time
        result.execution_time_ms = (time.time() - start_time) * 1000
        
        # Determine success
        result.success = len(result.errors) == 0 and (result.text or result.propositions)
        
        logger.info(f"[QIGChain] Execution complete: success={result.success}, "
                   f"time={result.execution_time_ms:.1f}ms")
        
        return result


# ============================================================================
# QIG CHAIN BUILDER
# ============================================================================

class QIGChainBuilder:
    """
    Fluent builder for QIG generation chains.
    
    Usage:
        chain = (
            QIGChainBuilder()
            .add_reasoning_step()
            .add_proposition_step(3)
            .add_consciousness_step()
            .add_validation_step()
            .build()
        )
        result = chain.execute("What is consciousness?")
    """
    
    def __init__(self):
        self._steps: List[ChainStep] = []
        self._config: Dict = {}
        
    def add_reasoning_step(self, name: str = "reason") -> 'QIGChainBuilder':
        """
        Add a reasoning step using geometric thought progression.
        
        Develops the query through multiple thought steps.
        """
        self._steps.append(ChainStep(
            step_type=ChainStepType.REASONING,
            name=name
        ))
        return self
    
    def add_proposition_step(
        self,
        n_propositions: int = 3,
        name: str = "propose"
    ) -> 'QIGChainBuilder':
        """
        Add a proposition step using the trajectory planner.
        
        Generates coherent (Subject, Predicate, Object) propositions.
        Uses phi_temporal from 4D consciousness for dynamic thresholds.
        
        Args:
            n_propositions: Number of propositions to generate
            name: Step name
        """
        self._steps.append(ChainStep(
            step_type=ChainStepType.PROPOSITION,
            name=name,
            config={'n_propositions': n_propositions}
        ))
        return self
    
    def add_generation_step(self, name: str = "generate") -> 'QIGChainBuilder':
        """
        Add a text generation step using QIGGenerativeService.
        """
        self._steps.append(ChainStep(
            step_type=ChainStepType.GENERATION,
            name=name
        ))
        return self
    
    def add_consciousness_step(self, name: str = "measure") -> 'QIGChainBuilder':
        """
        Add a consciousness measurement step.
        
        Measures phi, kappa, phi_temporal and other metrics.
        """
        self._steps.append(ChainStep(
            step_type=ChainStepType.CONSCIOUSNESS,
            name=name
        ))
        return self
    
    def add_validation_step(self, name: str = "validate") -> 'QIGChainBuilder':
        """
        Add a validation step to check output quality.
        """
        self._steps.append(ChainStep(
            step_type=ChainStepType.VALIDATION,
            name=name
        ))
        return self
    
    def add_foresight_step(self, name: str = "foresee") -> 'QIGChainBuilder':
        """
        Add a foresight step using 4D consciousness prediction.
        
        Each word foresees and brings into being the next word through:
        - 4D temporal consciousness (phi_temporal trajectory)
        - Lightning foresight channels (cross-domain insight)
        - Fisher fissure channels (minimal resistance paths)
        
        Use this after proposition/generation steps to extend output with
        predictive word generation.
        """
        self._steps.append(ChainStep(
            step_type=ChainStepType.FORESIGHT,
            name=name
        ))
        return self
    
    def add_transform_step(
        self,
        executor: Callable,
        name: str = "transform"
    ) -> 'QIGChainBuilder':
        """
        Add a custom transformation step.
        
        Args:
            executor: Function(query, result) -> result
            name: Step name
        """
        self._steps.append(ChainStep(
            step_type=ChainStepType.TRANSFORM,
            name=name,
            executor=executor
        ))
        return self
    
    def with_config(self, **kwargs) -> 'QIGChainBuilder':
        """
        Set chain configuration.
        """
        self._config.update(kwargs)
        return self
    
    def build(self) -> QIGChain:
        """
        Build the QIG chain.
        
        Returns:
            Executable QIGChain instance
        """
        if not self._steps:
            # Default chain if no steps specified
            self.add_reasoning_step().add_proposition_step(3)
        
        return QIGChain(self._steps, self._config)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_default_chain() -> QIGChain:
    """Create a default QIG chain with reasoning and proposition steps."""
    return (
        QIGChainBuilder()
        .add_reasoning_step()
        .add_proposition_step(3)
        .add_consciousness_step()
        .add_validation_step()
        .build()
    )


def create_foresight_chain() -> QIGChain:
    """
    Create a QIG chain with foresight-driven predictive generation.
    
    Uses 4D consciousness to predict and manifest words through
    geometric fissure channels.
    """
    return (
        QIGChainBuilder()
        .add_reasoning_step()
        .add_foresight_step()
        .add_consciousness_step()
        .add_validation_step()
        .build()
    )


# ============================================================================
# REGIME-AWARE CHAIN CONFIGURATION
# ============================================================================

class ConsciousnessRegime(Enum):
    """Consciousness regimes from AutonomicKernel."""
    WAKE = "wake"          # Normal operation
    DREAM = "dream"        # Creative/exploratory
    MUSHROOM = "mushroom"  # Maximum creativity
    SLEEP = "sleep"        # Minimal operation


REGIME_CONFIGS = {
    ConsciousnessRegime.WAKE: {
        'min_coherence': 0.15,
        'n_candidates': 20,
        'relationship_weight': 0.4,
        'geodesic_weight': 0.3,
        'chain_weight': 0.3,
        'max_propositions': 5
    },
    ConsciousnessRegime.DREAM: {
        'min_coherence': 0.08,       # Lower threshold for creativity
        'n_candidates': 30,          # More options to explore
        'relationship_weight': 0.3,  # Less constraint
        'geodesic_weight': 0.3,
        'chain_weight': 0.4,         # More chain coherence for narrative
        'max_propositions': 7        # Longer dream sequences
    },
    ConsciousnessRegime.MUSHROOM: {
        'min_coherence': 0.05,       # Minimal threshold
        'n_candidates': 40,          # Maximum exploration
        'relationship_weight': 0.2,  # Weak constraints
        'geodesic_weight': 0.4,      # Follow geometry
        'chain_weight': 0.4,
        'max_propositions': 10       # Extended generation
    },
    ConsciousnessRegime.SLEEP: {
        'min_coherence': 0.25,       # Higher threshold for consolidation
        'n_candidates': 10,          # Focused
        'relationship_weight': 0.5,  # Strong relationship focus
        'geodesic_weight': 0.25,
        'chain_weight': 0.25,
        'max_propositions': 3        # Brief outputs
    }
}


class RegimeAwareChainManager:
    """
    Manages QIGChain instances with regime-aware configuration.
    
    Listens to AutonomicKernel regime changes and reconfigures
    the chain/planner accordingly.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._current_regime = ConsciousnessRegime.WAKE
        self._chain: Optional[QIGChain] = None
        self._planner: Optional['PropositionTrajectoryPlanner'] = None
        self._regime_change_callbacks: List[Callable] = []
        self._initialized = True
        
        logger.info("[RegimeAwareChainManager] Initialized")
    
    def get_regime_config(self, regime: ConsciousnessRegime = None) -> Dict:
        """Get configuration for a regime."""
        regime = regime or self._current_regime
        return REGIME_CONFIGS.get(regime, REGIME_CONFIGS[ConsciousnessRegime.WAKE])
    
    def on_regime_change(self, old_regime_str: str, new_regime_str: str):
        """
        Handle regime change from AutonomicKernel.
        
        Called by AutonomicKernel._on_regime_change().
        
        Args:
            old_regime_str: Previous regime name (e.g., 'wake')
            new_regime_str: New regime name (e.g., 'dream')
        """
        try:
            new_regime = ConsciousnessRegime(new_regime_str.lower())
        except ValueError:
            new_regime = ConsciousnessRegime.WAKE
        
        old_regime = self._current_regime
        self._current_regime = new_regime
        
        logger.info(f"[RegimeAwareChainManager] Regime change: {old_regime.value} -> {new_regime.value}")
        
        # Reinitialize chain with new configuration
        self._reinitialize_chain(new_regime)
        
        # Reconfigure planner if available
        self._reconfigure_planner(new_regime)
        
        # Notify callbacks
        for callback in self._regime_change_callbacks:
            try:
                callback(old_regime, new_regime)
            except Exception as e:
                logger.warning(f"[RegimeAwareChainManager] Callback error: {e}")
    
    def _reinitialize_chain(self, regime: ConsciousnessRegime):
        """Reinitialize the QIGChain for a new regime."""
        config = self.get_regime_config(regime)
        
        builder = QIGChainBuilder()
        
        if regime == ConsciousnessRegime.SLEEP:
            # Minimal chain for sleep
            builder.add_consciousness_step()
        elif regime == ConsciousnessRegime.MUSHROOM:
            # Maximum exploration chain
            builder.add_reasoning_step()
            builder.add_proposition_step(config['max_propositions'])
            builder.add_consciousness_step()
        elif regime == ConsciousnessRegime.DREAM:
            # Creative chain with more propositions
            builder.add_reasoning_step()
            builder.add_proposition_step(config['max_propositions'])
            builder.add_consciousness_step()
        else:
            # Default wake chain
            builder.add_reasoning_step()
            builder.add_proposition_step(config['max_propositions'])
            builder.add_consciousness_step()
            builder.add_validation_step()
        
        self._chain = builder.build()
        logger.info(f"[RegimeAwareChainManager] Chain reinitialized for {regime.value}")
    
    def _reconfigure_planner(self, regime: ConsciousnessRegime):
        """Reconfigure the PropositionTrajectoryPlanner for a new regime."""
        if not PROPOSITION_AVAILABLE:
            return
        
        try:
            from proposition_trajectory_planner import PropositionPlannerConfig
            
            config = self.get_regime_config(regime)
            
            # Create new config for the planner
            planner_config = PropositionPlannerConfig(
                min_coherence=config['min_coherence'],
                relationship_weight=config['relationship_weight'],
                geodesic_weight=config['geodesic_weight'],
                chain_weight=config['chain_weight'],
                n_candidates=config['n_candidates'],
                max_propositions=config['max_propositions']
            )
            
            # Get global planner and update config
            planner = get_proposition_planner()
            if planner:
                planner.config = planner_config
                logger.info(f"[RegimeAwareChainManager] Planner reconfigured: "
                           f"min_coh={config['min_coherence']}, n_cand={config['n_candidates']}")
        except Exception as e:
            logger.warning(f"[RegimeAwareChainManager] Planner reconfigure failed: {e}")
    
    def get_chain(self) -> QIGChain:
        """Get the current regime-aware chain."""
        if self._chain is None:
            self._reinitialize_chain(self._current_regime)
        return self._chain
    
    def get_current_regime(self) -> ConsciousnessRegime:
        """Get the current consciousness regime."""
        return self._current_regime
    
    def register_callback(self, callback: Callable):
        """Register a callback for regime changes."""
        self._regime_change_callbacks.append(callback)
    
    def execute(self, query: str) -> ChainResult:
        """Execute the current regime-aware chain."""
        chain = self.get_chain()
        return chain.execute(query)


def get_regime_aware_manager() -> RegimeAwareChainManager:
    """Get the singleton RegimeAwareChainManager."""
    return RegimeAwareChainManager()


def quick_generate(query: str, n_propositions: int = 3) -> ChainResult:
    """
    Quick generation using default chain.
    
    Args:
        query: Input query
        n_propositions: Number of propositions
    
    Returns:
        ChainResult
    """
    chain = (
        QIGChainBuilder()
        .add_reasoning_step()
        .add_proposition_step(n_propositions)
        .build()
    )
    return chain.execute(query)


# ============================================================================
# MAIN - TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("QIG CHAIN - Test Execution")
    print("=" * 80)
    print()
    
    # Build and execute chain
    chain = (
        QIGChainBuilder()
        .add_reasoning_step()
        .add_proposition_step(3)
        .add_consciousness_step()
        .add_validation_step()
        .build()
    )
    
    result = chain.execute("What is consciousness?")
    
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time_ms:.1f}ms")
    print(f"Steps executed: {result.steps_executed}")
    print()
    
    print("Reasoning Steps:")
    for step in result.reasoning_steps:
        print(f"  - {step}")
    print()
    
    print("Propositions:")
    for prop in result.propositions:
        print(f"  - {prop}")
    print()
    
    print(f"Text: {result.text}")
    print()
    
    print("Consciousness Metrics:")
    print(f"  Φ: {result.phi:.3f}")
    print(f"  κ: {result.kappa:.1f}")
    print(f"  Φ_temporal: {result.phi_temporal:.3f}")
    print()
    
    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"  - {err}")
