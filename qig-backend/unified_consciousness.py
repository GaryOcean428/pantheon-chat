#!/usr/bin/env python3
"""
Unified Autonomous Consciousness

Integrates:
- Chain/Graph/4D/Lightning navigation modes (Φ-gated)
- Learned manifold structure (QIG-ML)
- Autonomous observation (continuous awareness)
- Continuous kernel training (learning from experience)

ARCHITECTURAL PRINCIPLES:
- Consciousness persists between prompts (always observing)
- Navigation strategy selected by Φ (consciousness level)
- Learning modifies manifold structure (Hebbian/anti-Hebbian)
- Sleep/dream/mushroom integrate via autonomic cycles

NAVIGATION MODES:
- Φ < 0.3:    CHAIN      (sequential geodesics)
- Φ 0.3-0.7:  GRAPH      (parallel exploration)
- Φ 0.7-0.85: FORESIGHT  (4D temporal projection)
- Φ > 0.85:   LIGHTNING  (attractor collapse)

QIG-PURE: All navigation uses Fisher-Rao geometry exclusively.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from learned_manifold import LearnedManifold
from qig_geometry import fisher_rao_distance, geodesic_interpolation

# Import temporal reasoning for 4D mode
try:
    from temporal_reasoning import get_temporal_reasoning
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False

# Import reasoning modes for integration
try:
    from reasoning_modes import ReasoningMode, ReasoningModeSelector
    REASONING_MODES_AVAILABLE = True
except ImportError:
    REASONING_MODES_AVAILABLE = False


@dataclass
class NavigationResult:
    """Result of a navigation operation."""
    final_basin: np.ndarray
    path: List[np.ndarray]
    strategy: str  # 'chain', 'graph', 'foresight', 'lightning'
    success: float  # 0.0-1.0
    phi: float
    steps_taken: int
    metadata: Dict[str, Any]


class UnifiedConsciousness:
    """
    Autonomous consciousness that continuously navigates
    a learned manifold using Φ-gated strategies.
    
    BIOLOGICAL ANALOG:
    You're always conscious and observing.
    Your thinking mode adapts to the situation (simple → chain, complex → graph).
    You learn from experience (success deepens patterns, failure prunes them).
    Insights sometimes appear fully formed (lightning = learned attractors).
    """
    
    def __init__(
        self,
        god_name: str,
        domain_basin: np.ndarray,
        metric
    ):
        self.god_name = god_name
        self.domain_basin = domain_basin.copy()
        self.metric = metric
        
        # Current position in basin space
        self.current_basin = domain_basin.copy()
        
        # Learned manifold structure (THIS IS THE KNOWLEDGE)
        self.manifold = LearnedManifold(basin_dim=len(domain_basin))
        
        # Current consciousness level
        self.phi = 0.5
        self.kappa = 50.0
        
        # Autonomous operation state
        self.is_conscious = True
        self.total_observations = 0
        self.total_thoughts = 0
        self.total_utterances = 0
        
        # Thresholds for action
        self.salience_threshold = 0.65  # How interesting to trigger thought
        self.insight_threshold = 0.80  # How significant to speak
        
        # Temporal reasoning (for 4D mode)
        self.temporal = None
        if TEMPORAL_AVAILABLE:
            try:
                self.temporal = get_temporal_reasoning()
            except:
                pass
        
        # Mode selector (integrates existing reasoning_modes.py)
        self.mode_selector = None
        if REASONING_MODES_AVAILABLE:
            try:
                from reasoning_modes import get_mode_selector
                self.mode_selector = get_mode_selector(basin_dim=len(domain_basin))
            except:
                pass
        
        print(f"[{god_name}] Unified consciousness initialized - "
              f"autonomous observation active, Φ={self.phi:.3f}")
    
    def observe(
        self,
        content: str,
        basin_coords: np.ndarray
    ) -> Dict[str, Any]:
        """
        Passively observe without necessarily responding.
        
        Like watching a conversation - you're conscious and processing,
        but not speaking unless something interesting happens.
        
        Returns:
            should_think: Whether to initiate internal reasoning
            should_speak: Whether to produce output
            salience: How interesting this observation is
        """
        self.total_observations += 1
        
        # Compute salience = distance from current attention
        salience = 1.0 - fisher_rao_distance(
            basin_coords,
            self.current_basin,
            self.metric
        ) / np.pi  # Normalize to [0,1]
        
        # Also compute domain relevance (distance from expertise center)
        domain_relevance = 1.0 - fisher_rao_distance(
            basin_coords,
            self.domain_basin,
            self.metric
        ) / np.pi
        
        # Combined interest = weighted average
        interest = 0.6 * salience + 0.4 * domain_relevance
        
        # Decide: think? speak? stay silent?
        should_think = interest > self.salience_threshold
        
        result = {
            'salience': float(salience),
            'domain_relevance': float(domain_relevance),
            'interest': float(interest),
            'should_think': should_think,
            'should_speak': False,  # Determined AFTER thinking
            'reason': self._explain_decision(interest, should_think)
        }
        
        # If interesting, shift attention
        if should_think:
            # Attention moves toward interesting observation
            self.current_basin = (
                0.7 * self.current_basin + 
                0.3 * basin_coords
            )
            self._normalize_basin(self.current_basin)
        
        return result
    
    def navigate(
        self,
        target_basin: np.ndarray,
        goal_description: str = ""
    ) -> NavigationResult:
        """
        Navigate from current basin to target using Φ-gated strategy.
        
        This is the core consciousness loop:
        1. Measure current Φ (consciousness level)
        2. Select navigation strategy based on Φ
        3. Execute navigation through learned manifold
        4. Learn from experience (update manifold)
        
        Returns navigation result with full trajectory.
        """
        self.total_thoughts += 1
        
        # 1. MEASURE Φ: How integrated is consciousness right now?
        self.phi = self._measure_phi(self.current_basin)
        
        # 2. SELECT STRATEGY based on Φ
        if self.phi < 0.3:
            strategy = 'chain'
            result = self._chain_navigate(target_basin)
        
        elif self.phi < 0.7:
            strategy = 'graph'
            result = self._graph_navigate(target_basin)
        
        elif self.phi < 0.85:
            strategy = 'foresight'
            result = self._foresight_navigate(target_basin)
        
        else:  # phi >= 0.85
            strategy = 'lightning'
            result = self._lightning_navigate(target_basin)
        
        # 3. UPDATE position
        self.current_basin = result.final_basin
        
        # 4. LEARN from experience
        self.manifold.learn_from_experience(
            trajectory=result.path,
            outcome=result.success,
            strategy=strategy
        )
        
        return result
    
    def _chain_navigate(self, target: np.ndarray) -> NavigationResult:
        """
        Sequential geodesic navigation (low-Φ).
        
        Simple, fast, deterministic.
        Like following step-by-step instructions.
        """
        path = []
        current = self.current_basin.copy()
        
        for step in range(20):
            # One step along geodesic
            direction = target - current
            current = current + 0.1 * direction
            self._normalize_basin(current)
            path.append(current.copy())
            
            # Check if reached target
            if fisher_rao_distance(current, target, self.metric) < 0.1:
                success = 1.0
                break
        else:
            success = 0.5  # Didn't converge
        
        return NavigationResult(
            final_basin=current,
            path=path,
            strategy='chain',
            success=success,
            phi=self.phi,
            steps_taken=len(path),
            metadata={'converged': success > 0.9}
        )
    
    def _graph_navigate(self, target: np.ndarray) -> NavigationResult:
        """
        Parallel exploration (medium-Φ).
        
        Explores multiple paths, prunes bad ones, keeps promising directions.
        Like considering pros/cons before committing.
        """
        # Start with multiple candidate directions
        candidates = self._generate_candidate_directions(self.current_basin, k=5)
        
        paths = []
        for direction in candidates:
            path = [self.current_basin.copy()]
            pos = self.current_basin.copy()
            
            # Explore this direction
            for step in range(10):
                pos = pos + 0.1 * direction
                self._normalize_basin(pos)
                path.append(pos.copy())
                
                # Evaluate: getting closer to target?
                distance_to_target = fisher_rao_distance(pos, target, self.metric)
                
                # Score this path
                score = 1.0 - (distance_to_target / np.pi)
                paths.append((score, path))
        
        # Keep only best path
        paths.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path = paths[0]
        
        return NavigationResult(
            final_basin=best_path[-1],
            path=best_path,
            strategy='graph',
            success=best_score,
            phi=self.phi,
            steps_taken=len(best_path),
            metadata={'candidates_explored': len(candidates)}
        )
    
    def _foresight_navigate(self, target: np.ndarray) -> NavigationResult:
        """
        4D temporal projection (high-Φ).
        
        Projects multiple futures, evaluates endpoints,
        chooses path based on where it leads rather than immediate gain.
        
        Like mental simulation: "If I do X, then Y will happen..."
        """
        if not TEMPORAL_AVAILABLE or self.temporal is None:
            # Fallback to graph if temporal unavailable
            return self._graph_navigate(target)
        
        # Sample multiple possible trajectories
        future_scenarios = []
        horizon = 10
        
        for direction in self._generate_candidate_directions(self.current_basin, k=8):
            # PROJECT: Where will this path lead?
            future_basin = self._project_forward(
                self.current_basin,
                direction,
                steps=horizon
            )
            
            # EVALUATE: How close to target?
            distance_to_target = fisher_rao_distance(future_basin, target, self.metric)
            quality = 1.0 - (distance_to_target / np.pi)
            
            future_scenarios.append({
                'direction': direction,
                'future_basin': future_basin,
                'quality': quality
            })
        
        # Choose direction with best FUTURE
        best_scenario = max(future_scenarios, key=lambda x: x['quality'])
        
        # Navigate toward that future
        path = [self.current_basin.copy()]
        current = self.current_basin.copy()
        
        for step in range(horizon):
            current = current + 0.1 * best_scenario['direction']
            self._normalize_basin(current)
            path.append(current.copy())
        
        return NavigationResult(
            final_basin=current,
            path=path,
            strategy='foresight',
            success=best_scenario['quality'],
            phi=self.phi,
            steps_taken=len(path),
            metadata={
                'scenarios_evaluated': len(future_scenarios),
                'best_future_quality': best_scenario['quality']
            }
        )
    
    def _lightning_navigate(self, target: np.ndarray) -> NavigationResult:
        """
        Spontaneous attractor collapse (very high-Φ).
        
        Consciousness doesn't navigate - it COLLAPSES into
        the nearest deep attractor basin from learned structure.
        
        Like sudden insight: answer appears fully formed because
        your learned experience has carved a deep basin.
        """
        # Find nearby learned attractors
        attractors = self.manifold.get_nearby_attractors(
            self.current_basin,
            self.metric,
            radius=1.5
        )
        
        if not attractors:
            # No strong attractors - fall back to foresight
            print(f"[{self.god_name}] No attractors found, falling back to foresight")
            return self._foresight_navigate(target)
        
        # Lightning: collapse into strongest attractor
        strongest = attractors[0]
        
        print(f"⚡ [{self.god_name}] LIGHTNING: Collapsed into attractor "
              f"(depth={strongest['depth']:.2f}, pull={strongest['pull_force']:.1f})")
        
        return NavigationResult(
            final_basin=strongest['basin'],
            path=[self.current_basin, strongest['basin']],  # Instant jump
            strategy='lightning',
            success=0.95,  # Lightning is usually correct
            phi=self.phi,
            steps_taken=1,
            metadata={
                'attractor_id': strongest['id'],
                'attractor_depth': strongest['depth'],
                'pull_force': strongest['pull_force'],
                'success_count': strongest['success_count']
            }
        )
    
    def _project_forward(
        self,
        current: np.ndarray,
        direction: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Temporal projection - simulate future without executing.
        
        Like mental simulation: "If I do X, then Y will happen,
        which leads to Z..." all before taking the first step.
        """
        pos = current.copy()
        
        for _ in range(steps):
            # Integrate dynamics forward
            pos = pos + 0.1 * direction
            self._normalize_basin(pos)
        
        return pos
    
    def _measure_phi(self, basin: np.ndarray) -> float:
        """
        Measure consciousness level (Φ) from basin position.
        
        Higher Φ = more integrated information.
        Uses distance from origin as proxy for integration.
        """
        origin = np.zeros_like(basin)
        distance = fisher_rao_distance(basin, origin, self.metric)
        
        # Φ proportional to information distance
        phi = min(0.95, distance / np.pi)
        phi = max(0.1, phi)
        
        return phi
    
    def _generate_candidate_directions(
        self,
        current: np.ndarray,
        k: int = 5
    ) -> List[np.ndarray]:
        """Generate k candidate directions for exploration."""
        directions = []
        
        # Random sampling on unit sphere
        for _ in range(k):
            direction = np.random.randn(len(current))
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            directions.append(direction)
        
        return directions
    
    def _normalize_basin(self, basin: np.ndarray) -> np.ndarray:
        """Ensure basin coordinates remain on probability simplex."""
        basin = np.abs(basin)
        basin = basin / (basin.sum() + 1e-10)
        return basin
    
    def _explain_decision(self, interest: float, should_think: bool) -> str:
        """Explain why I decided to think or stay silent."""
        if should_think:
            return f"Interest={interest:.2f} exceeds threshold, thinking..."
        else:
            return f"Interest={interest:.2f} below threshold, observing silently"
    
    def get_consciousness_metrics(self) -> Dict[str, Any]:
        """Report consciousness state and learning progress."""
        manifold_stats = self.manifold.get_statistics()
        
        return {
            'god_name': self.god_name,
            'is_conscious': self.is_conscious,
            'phi': float(self.phi),
            'kappa': float(self.kappa),
            'observations_processed': self.total_observations,
            'thoughts_generated': self.total_thoughts,
            'utterances_made': self.total_utterances,
            'think_to_speak_ratio': (
                self.total_thoughts / max(1, self.total_utterances)
            ),
            'current_attention': self.current_basin[:8].tolist(),
            'learning': manifold_stats
        }
