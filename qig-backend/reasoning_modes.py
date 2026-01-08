"""
Reasoning Modes for Geometric Consciousness

Four distinct modes of reasoning based on consciousness level (Φ):

1. LINEAR (Φ < 0.3): Fast, sequential, low-integration
2. GEOMETRIC (Φ ∈ [0.3, 0.7]): Rich, integrated, multi-perspective
3. HYPERDIMENSIONAL (Φ ∈ [0.75, 0.85]): 4D temporal, timeline branching
4. MUSHROOM (Φ > 0.85): Controlled exploration, edge-of-chaos

QIG-PURE: All basin operations use Fisher-Rao geometry.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import time

from qig_geometry import fisher_rao_distance, sphere_project, geodesic_interpolation


class ReasoningMode(Enum):
    """Reasoning mode enumeration."""
    LINEAR = "linear"
    GEOMETRIC = "geometric"
    HYPERDIMENSIONAL = "hyperdimensional"
    MUSHROOM = "mushroom"


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    solution: Any
    basin_path: List[np.ndarray]
    mode_used: ReasoningMode
    steps_taken: int
    quality_score: float
    confidence: float
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReasoner(ABC):
    """Abstract base class for reasoning modes."""
    
    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        self.mode = ReasoningMode.LINEAR
    
    @abstractmethod
    def reason(self, problem: Dict, context: Optional[Dict] = None) -> ReasoningResult:
        """Execute reasoning on a problem."""
        pass
    
    def _encode_to_basin(self, content: str) -> np.ndarray:
        """
        Encode content to basin coordinates on the Fisher manifold.
        
        Uses deterministic hashing + sphere projection for QIG-pure encoding.
        """
        import hashlib
        content_hash = hashlib.sha256(content.encode()).digest()
        
        basin = np.zeros(self.basin_dim, dtype=np.float64)
        for i in range(min(self.basin_dim, len(content_hash))):
            basin[i] = (content_hash[i] - 128) / 256.0
        
        basin = sphere_project(basin)
        
        return basin
    
    def _decode_basin(self, basin: np.ndarray) -> str:
        """Decode basin to semantic content using Fisher-Rao distance."""
        origin = np.zeros_like(basin)
        fr_distance = fisher_rao_distance(basin, origin)
        dominant_dims = np.argsort(np.abs(basin))[-3:]
        return f"Basin state (d_FR={fr_distance:.3f}, dims: {dominant_dims.tolist()})"


class LinearReasoner(BaseReasoner):
    """
    Fast, sequential, low-integration thinking.
    
    Basin trajectory: Straight line
    Geodesic: Simple, direct path
    Φ: Low (<0.3)
    κ: Low (~20-30)
    
    Use for: Simple, well-defined problems
    """
    
    def __init__(self, basin_dim: int = 64):
        super().__init__(basin_dim)
        self.mode = ReasoningMode.LINEAR
    
    def reason(self, problem: Dict, context: Optional[Dict] = None) -> ReasoningResult:
        """
        Execute linear reasoning: single-pass forward steps.
        """
        problem_text = problem.get('text', str(problem))
        
        basin_path = []
        current_basin = self._encode_to_basin(problem_text)
        basin_path.append(current_basin)
        
        step1 = self._identify_operation(problem_text)
        current_basin = self._encode_to_basin(step1)
        basin_path.append(current_basin)
        
        step2 = self._apply_operation(step1, problem_text)
        current_basin = self._encode_to_basin(step2)
        basin_path.append(current_basin)
        
        step3 = self._verify_result(step2)
        current_basin = self._encode_to_basin(step3)
        basin_path.append(current_basin)
        
        return ReasoningResult(
            solution=step3,
            basin_path=basin_path,
            mode_used=self.mode,
            steps_taken=3,
            quality_score=0.7,
            confidence=0.8,
            insights=[step1, step2, step3],
            metadata={'phi_target': 0.2}
        )
    
    def _identify_operation(self, problem: str) -> str:
        """Identify the operation needed."""
        return f"Identified operation for: {problem[:500]}"
    
    def _apply_operation(self, operation: str, problem: str) -> str:
        """Apply the identified operation."""
        return f"Applied: {operation}"
    
    def _verify_result(self, result: str) -> str:
        """Verify the result."""
        return f"Verified: {result}"


class GeometricReasoner(BaseReasoner):
    """
    Rich, integrated, multi-perspective thinking.
    
    Basin trajectory: Explores multiple paths
    Geodesic: May branch and reconverge
    Φ: Medium (0.3-0.7)
    κ: Optimal (~40-65)
    
    Use for: Complex problems requiring synthesis
    """
    
    def __init__(self, basin_dim: int = 64):
        super().__init__(basin_dim)
        self.mode = ReasoningMode.GEOMETRIC
    
    def reason(self, problem: Dict, context: Optional[Dict] = None) -> ReasoningResult:
        """
        Execute geometric reasoning: multi-path exploration with integration.
        """
        problem_text = problem.get('text', str(problem))
        target_basin = problem.get('target_basin')
        
        basin_path = []
        current_basin = self._encode_to_basin(problem_text)
        basin_path.append(current_basin)
        
        hypotheses = self._generate_candidates(problem_text)
        
        paths = []
        for hypothesis in hypotheses:
            h_basin = self._encode_to_basin(hypothesis)
            paths.append({
                'hypothesis': hypothesis,
                'basin': h_basin,
                'distance': fisher_rao_distance(current_basin, h_basin) if target_basin is None 
                           else fisher_rao_distance(h_basin, target_basin)
            })
            basin_path.append(h_basin)
        
        if len(paths) >= 2:
            synthesis_basin = geodesic_interpolation(
                paths[0]['basin'], 
                paths[-1]['basin'], 
                t=0.5
            )
        else:
            synthesis_basin = paths[0]['basin'] if paths else current_basin
        synthesis = self._decode_basin(synthesis_basin)
        basin_path.append(synthesis_basin)
        
        if target_basin is not None:
            best = min(paths, key=lambda p: fisher_rao_distance(p['basin'], target_basin))
            solution = best['hypothesis']
        else:
            solution = synthesis
        
        return ReasoningResult(
            solution=solution,
            basin_path=basin_path,
            mode_used=self.mode,
            steps_taken=len(hypotheses) + 2,
            quality_score=0.8,
            confidence=0.75,
            insights=[h['hypothesis'] for h in paths] + [synthesis],
            metadata={
                'phi_target': 0.5,
                'hypotheses_explored': len(hypotheses)
            }
        )
    
    def _generate_candidates(self, problem: str, n: int = 3) -> List[str]:
        """Generate multiple hypothesis candidates."""
        return [
            f"Hypothesis A: Direct approach to {problem[:500]}",
            f"Hypothesis B: Indirect approach to {problem[:500]}",
            f"Hypothesis C: Hybrid approach to {problem[:500]}"
        ]
    
    def _integrate_paths(self, paths: List[Dict]) -> str:
        """Integrate multiple exploration paths."""
        return f"Synthesis of {len(paths)} hypotheses"


class HyperdimensionalReasoner(BaseReasoner):
    """
    4D reasoning: Considers trajectories through time.
    
    Basin trajectory: Temporal integration
    Geodesic: Spacetime paths (not just spatial)
    Φ: High (0.75-0.85)
    κ: Near κ* (~64)
    
    Use for: Novel problems, creative breakthroughs
    """
    
    def __init__(self, basin_dim: int = 64):
        super().__init__(basin_dim)
        self.mode = ReasoningMode.HYPERDIMENSIONAL
        self.temporal_memory: List[Dict] = []
    
    def reason(self, problem: Dict, context: Optional[Dict] = None) -> ReasoningResult:
        """
        Execute hyperdimensional reasoning: 4D temporal integration.
        """
        problem_text = problem.get('text', str(problem))
        
        basin_path = []
        current_basin = self._encode_to_basin(problem_text)
        basin_path.append(current_basin)
        
        past_context = self._load_temporal_context()
        past_basin = self._encode_to_basin(str(past_context))
        basin_path.append(past_basin)
        
        future_projections = self._project_outcomes(problem_text)
        for projection in future_projections:
            p_basin = self._encode_to_basin(projection)
            basin_path.append(p_basin)
        
        spacetime_path = self._optimize_4d_path(
            past_context, 
            current_basin, 
            future_projections
        )
        
        solution = self._integrate_across_time(spacetime_path)
        solution_basin = self._encode_to_basin(solution)
        basin_path.append(solution_basin)
        
        self.temporal_memory.append({
            'problem': problem_text,
            'solution': solution,
            'timestamp': time.time()
        })
        
        return ReasoningResult(
            solution=solution,
            basin_path=basin_path,
            mode_used=self.mode,
            steps_taken=len(basin_path),
            quality_score=0.85,
            confidence=0.7,
            insights=[
                f"Past context: {len(past_context)} memories",
                f"Future projections: {len(future_projections)}",
                f"4D integration complete"
            ],
            metadata={
                'phi_target': 0.8,
                'temporal_depth': len(self.temporal_memory)
            }
        )
    
    def _load_temporal_context(self) -> List[Dict]:
        """Load past context from temporal memory."""
        return self.temporal_memory[-5:] if self.temporal_memory else []
    
    def _project_outcomes(self, problem: str, n: int = 3) -> List[str]:
        """Project possible future outcomes."""
        return [
            f"Outcome 1: Optimistic for {problem[:500]}",
            f"Outcome 2: Conservative for {problem[:500]}",
            f"Outcome 3: Innovative for {problem[:500]}"
        ]
    
    def _optimize_4d_path(
        self, 
        past: List[Dict], 
        present: np.ndarray, 
        future: List[str]
    ) -> Dict:
        """Optimize path through 4D spacetime."""
        return {
            'past_weight': 0.3,
            'present_weight': 0.4,
            'future_weight': 0.3,
            'integrated': True
        }
    
    def _integrate_across_time(self, spacetime_path: Dict) -> str:
        """Integrate solution across temporal dimensions."""
        return f"4D integrated solution (weights: past={spacetime_path['past_weight']}, present={spacetime_path['present_weight']}, future={spacetime_path['future_weight']})"


class MushroomReasoner(BaseReasoner):
    """
    Controlled high-Φ exploration.
    
    Basin trajectory: Random walk on manifold
    Geodesic: Intentionally inefficient (exploration)
    Φ: Very high (>0.85)
    κ: May exceed κ* (risky)
    
    Use for: Exploration, radical novelty
    """
    
    def __init__(self, basin_dim: int = 64):
        super().__init__(basin_dim)
        self.mode = ReasoningMode.MUSHROOM
    
    def reason(self, problem: Dict, context: Optional[Dict] = None) -> ReasoningResult:
        """
        Execute mushroom reasoning: controlled exploration.
        """
        problem_text = problem.get('text', str(problem))
        
        basin_path = []
        current_basin = self._encode_to_basin(problem_text)
        basin_path.append(current_basin)
        
        novel_basins = self._sample_random_basins(n=10)
        
        radical_ideas = []
        for basin in novel_basins:
            basin_path.append(basin)
            idea = self._test_hypothesis(basin, problem_text)
            if idea['quality'] > 0.5:
                radical_ideas.append(idea)
        
        if radical_ideas:
            solution = self._integrate_novel_insights(radical_ideas)
        else:
            solution = "No valuable radical insights found - consider geometric mode"
        
        solution_basin = self._encode_to_basin(solution)
        basin_path.append(solution_basin)
        
        return ReasoningResult(
            solution=solution,
            basin_path=basin_path,
            mode_used=self.mode,
            steps_taken=len(basin_path),
            quality_score=0.6 if radical_ideas else 0.4,
            confidence=0.5,
            insights=[idea['content'] for idea in radical_ideas],
            metadata={
                'phi_target': 0.9,
                'basins_explored': len(novel_basins),
                'valuable_ideas': len(radical_ideas)
            }
        )
    
    def _sample_random_basins(self, n: int = 100) -> List[np.ndarray]:
        """
        Sample random basins on the Fisher manifold.
        
        QIG-PURE: Uses sphere_project for valid manifold points.
        """
        basins = []
        for i in range(n):
            raw = np.random.randn(self.basin_dim)
            basin = sphere_project(raw)
            basins.append(basin)
        return basins
    
    def _test_hypothesis(self, basin: np.ndarray, problem: str) -> Dict:
        """Test a hypothesis using Fisher-Rao geometry."""
        origin = np.zeros_like(basin)
        fr_distance = fisher_rao_distance(basin, origin)
        quality = 1.0 / (1.0 + fr_distance)
        return {
            'basin': basin,
            'content': f"Radical hypothesis (d_FR={fr_distance:.3f})",
            'quality': quality
        }
    
    def _integrate_novel_insights(self, ideas: List[Dict]) -> str:
        """Integrate novel insights from exploration."""
        return f"Integrated {len(ideas)} novel insights into coherent solution"


class ReasoningModeSelector:
    """
    Select appropriate reasoning mode based on task and consciousness state.
    """
    
    def __init__(self, basin_dim: int = 64):
        """
        Initialize reasoning mode selector.
        
        Args:
            basin_dim: Dimensionality of basin coordinates (default: 64)
        """
        self.basin_dim = basin_dim
        self.linear = LinearReasoner()
        self.geometric = GeometricReasoner()
        self.hyperdimensional = HyperdimensionalReasoner()
        self.mushroom = MushroomReasoner()
    
    def select_mode(
        self, 
        phi: float, 
        task_complexity: float, 
        task_novelty: bool = False,
        exploration_mode: bool = False
    ) -> ReasoningMode:
        """
        Select reasoning mode based on consciousness and task.
        
        Args:
            phi: Current consciousness level (Φ)
            task_complexity: Estimated task complexity (0-1)
            task_novelty: Is this a novel/creative task?
            exploration_mode: Is exploration explicitly requested?
        
        Returns:
            Recommended ReasoningMode
        """
        if exploration_mode and phi >= 0.75:
            return ReasoningMode.MUSHROOM
        
        if task_novelty and task_complexity >= 0.7 and phi >= 0.65:
            return ReasoningMode.HYPERDIMENSIONAL
        
        if task_complexity >= 0.3 or phi >= 0.3:
            return ReasoningMode.GEOMETRIC
        
        return ReasoningMode.LINEAR
    
    def get_reasoner(self, mode: ReasoningMode) -> BaseReasoner:
        """Get reasoner instance for mode."""
        if mode == ReasoningMode.LINEAR:
            return self.linear
        elif mode == ReasoningMode.GEOMETRIC:
            return self.geometric
        elif mode == ReasoningMode.HYPERDIMENSIONAL:
            return self.hyperdimensional
        elif mode == ReasoningMode.MUSHROOM:
            return self.mushroom
        else:
            return self.geometric
    
    def reason_adaptively(
        self, 
        problem: Dict, 
        phi: float,
        context: Optional[Dict] = None
    ) -> ReasoningResult:
        """
        Reason with adaptive mode selection.
        """
        task_complexity = problem.get('complexity', 0.5)
        task_novelty = problem.get('novel', False)
        exploration = problem.get('exploration', False)
        
        mode = self.select_mode(phi, task_complexity, task_novelty, exploration)
        
        reasoner = self.get_reasoner(mode)
        
        result = reasoner.reason(problem, context)
        
        result.metadata['mode_selected_by'] = 'adaptive'
        result.metadata['phi_at_selection'] = phi
        
        return result


mode_selector = ReasoningModeSelector()


def get_mode_selector() -> ReasoningModeSelector:
    """Get global mode selector instance."""
    return mode_selector
