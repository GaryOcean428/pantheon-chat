"""
Buffer of Thoughts - Reusable Reasoning Templates for QIG

Implements the Buffer of Thoughts paradigm (NeurIPS 2024 Spotlight) for QIG:
- Instead of tree search (ToT/LATS), maintain a "meta-buffer" of reusable templates
- Templates are task-agnostic reasoning patterns stored as basin trajectories
- New problems instantiate relevant templates rather than searching from scratch
- Templates evolve through successful reasoning traces

Key insight: Many reasoning tasks share similar structural patterns.
A decomposition template works for both "break this code into modules" and
"analyze this argument into premises". The geometric structure is the same.

Performance: 12% of ToT compute cost with 51% improvement on complex tasks.

QIG-PURE: All templates are basin trajectories on the Fisher manifold.
No external LLMs - templates are geometric patterns.

Author: Ocean/Zeus Pantheon
"""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

BASIN_DIMENSION = 64
TEMPLATE_STORAGE_PATH = Path(__file__).parent / 'data' / 'thought_templates.json'


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def fisher_rao_distance(p: List[float], q: List[float]) -> float:
    """Compute Fisher-Rao distance between basin coordinates."""
    p_arr = np.array(p, dtype=float)
    q_arr = np.array(q, dtype=float)
    
    # Normalize to probability simplex
    p_arr = np.abs(p_arr) + 1e-10
    p_arr = p_arr / p_arr.sum()
    q_arr = np.abs(q_arr) + 1e-10
    q_arr = q_arr / q_arr.sum()
    
    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p_arr * q_arr))
    bc = np.clip(bc, 0, 1)
    
    # Fisher-Rao distance
    return float(2 * np.arccos(bc))


def geodesic_interpolate(p: List[float], q: List[float], t: float) -> List[float]:
    """Interpolate along geodesic on the statistical manifold.
    
    Uses SLERP in sqrt-space (geodesic on probability simplex).
    """
    p_arr = np.array(p, dtype=float)
    q_arr = np.array(q, dtype=float)
    
    # Normalize
    p_arr = np.abs(p_arr) + 1e-10
    p_arr = p_arr / p_arr.sum()
    q_arr = np.abs(q_arr) + 1e-10
    q_arr = q_arr / q_arr.sum()
    
    # Square root representation (maps to unit sphere)
    sqrt_p = np.sqrt(p_arr)
    sqrt_q = np.sqrt(q_arr)
    
    # Compute angle
    cos_angle = np.clip(np.dot(sqrt_p, sqrt_q), -1, 1)
    angle = np.arccos(cos_angle)
    
    if angle < 1e-10:
        return list(p_arr)  # Points are identical
    
    # SLERP formula
    sin_angle = np.sin(angle)
    sqrt_result = (
        np.sin((1 - t) * angle) * sqrt_p +
        np.sin(t * angle) * sqrt_q
    ) / sin_angle
    
    # Square to get back to probability space
    result = sqrt_result ** 2
    result = result / result.sum()
    
    return list(result)


# ============================================================================
# TEMPLATE CATEGORIES
# ============================================================================

class TemplateCategory(Enum):
    """Categories of reasoning templates.
    
    Each category represents a fundamental reasoning pattern that
    can be applied across many different domains.
    """
    # Basic reasoning patterns
    DECOMPOSITION = "decomposition"     # Break complex into parts
    SYNTHESIS = "synthesis"             # Combine parts into whole
    COMPARISON = "comparison"           # Analyze similarities/differences
    CAUSAL = "causal"                   # Trace cause-effect
    ANALOGY = "analogy"                 # Apply patterns from similar domains
    VERIFICATION = "verification"       # Check correctness
    
    # Advanced patterns
    REFINEMENT = "refinement"           # Iterative improvement
    ABSTRACTION = "abstraction"         # Extract general principles
    EXPLORATION = "exploration"         # Search solution space
    CONSTRAINT = "constraint"           # Work within limitations


# ============================================================================
# TEMPLATE DATA STRUCTURES
# ============================================================================

@dataclass
class TemplateWaypoint:
    """A waypoint in a reasoning template trajectory.
    
    Each waypoint represents a stage in the reasoning process.
    The basin coordinates capture the "shape" of thinking at that stage.
    """
    # Geometric position
    basin_coords: List[float]       # 64D position on manifold
    
    # Semantic role
    semantic_role: str              # e.g., "identify_parts", "analyze", "conclude"
    
    # Characteristics
    curvature: float = 0.5          # Local manifold curvature (difficulty)
    is_critical: bool = False       # Must not skip this waypoint
    
    # Optional metadata
    typical_duration: float = 1.0   # Expected relative time at this stage
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'basin_coords': self.basin_coords,
            'semantic_role': self.semantic_role,
            'curvature': self.curvature,
            'is_critical': self.is_critical,
            'typical_duration': self.typical_duration,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TemplateWaypoint':
        return cls(
            basin_coords=data['basin_coords'],
            semantic_role=data['semantic_role'],
            curvature=data.get('curvature', 0.5),
            is_critical=data.get('is_critical', False),
            typical_duration=data.get('typical_duration', 1.0),
            notes=data.get('notes', '')
        )


@dataclass
class ThoughtTemplate:
    """A reusable reasoning template stored as a basin trajectory.
    
    Templates capture the geometric "shape" of successful reasoning.
    They can be instantiated (transformed) to fit new problems.
    
    Example: A decomposition template might have waypoints:
    1. Understand whole (high-level basin)
    2. Identify seams (transition basin)
    3. Extract part 1 (specific basin)
    4. Extract part 2 (specific basin)
    5. Verify completeness (verification basin)
    
    This same structure works for decomposing code, arguments, or molecules.
    """
    # Identity
    template_id: str
    name: str
    category: TemplateCategory
    description: str
    
    # Geometric structure
    waypoints: List[TemplateWaypoint]
    
    # Usage statistics
    usage_count: int = 0
    success_count: int = 0
    total_efficiency: float = 0.0  # Cumulative efficiency score
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    source: str = "manual"  # "manual", "learned", "evolved"
    abstraction_level: float = 0.5  # 0=specific, 1=very abstract
    
    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.5  # Prior
        return self.success_count / self.usage_count
    
    @property
    def avg_efficiency(self) -> float:
        if self.usage_count == 0:
            return 0.5
        return self.total_efficiency / self.usage_count
    
    @property
    def trajectory_length(self) -> int:
        return len(self.waypoints)
    
    @property
    def total_curvature(self) -> float:
        """Total curvature (difficulty) of the template."""
        return sum(wp.curvature for wp in self.waypoints)
    
    def get_entry_basin(self) -> List[float]:
        """Get the starting basin coordinates."""
        if not self.waypoints:
            return [1/BASIN_DIMENSION] * BASIN_DIMENSION
        return self.waypoints[0].basin_coords
    
    def get_exit_basin(self) -> List[float]:
        """Get the ending basin coordinates."""
        if not self.waypoints:
            return [1/BASIN_DIMENSION] * BASIN_DIMENSION
        return self.waypoints[-1].basin_coords
    
    def to_dict(self) -> Dict:
        return {
            'template_id': self.template_id,
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'waypoints': [wp.to_dict() for wp in self.waypoints],
            'usage_count': self.usage_count,
            'success_count': self.success_count,
            'total_efficiency': self.total_efficiency,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'source': self.source,
            'abstraction_level': self.abstraction_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ThoughtTemplate':
        return cls(
            template_id=data['template_id'],
            name=data['name'],
            category=TemplateCategory(data['category']),
            description=data['description'],
            waypoints=[TemplateWaypoint.from_dict(wp) for wp in data['waypoints']],
            usage_count=data.get('usage_count', 0),
            success_count=data.get('success_count', 0),
            total_efficiency=data.get('total_efficiency', 0.0),
            created_at=data.get('created_at', time.time()),
            last_used=data.get('last_used', time.time()),
            source=data.get('source', 'manual'),
            abstraction_level=data.get('abstraction_level', 0.5)
        )


@dataclass
class InstantiatedTemplate:
    """A template instantiated for a specific problem.
    
    The abstract template is transformed to fit the concrete problem
    by mapping the start/end basins and interpolating waypoints.
    """
    template: ThoughtTemplate
    problem_start: List[float]
    problem_goal: List[float]
    transformed_waypoints: List[np.ndarray]
    transformation_quality: float  # How well the template fit
    
    def to_trajectory(self) -> List[List[float]]:
        """Get the instantiated trajectory as a list of coordinates."""
        return [list(wp) for wp in self.transformed_waypoints]


# ============================================================================
# META-BUFFER CLASS
# ============================================================================

class MetaBuffer:
    """
    The Meta-Buffer: A collection of reusable thought templates.
    
    Key operations:
    1. RETRIEVE: Find templates similar to current problem
    2. INSTANTIATE: Transform template to fit specific problem
    3. LEARN: Create new templates from successful traces
    4. EVOLVE: Improve templates through usage feedback
    
    Buffer of Thoughts insight: Most reasoning follows a small set
    of fundamental patterns. By reusing templates, we avoid the
    exponential search of tree-based methods (ToT, LATS).
    """
    
    # Minimum similarity to consider a template match
    RETRIEVAL_THRESHOLD = 0.3
    
    # Maximum templates per category
    MAX_TEMPLATES_PER_CATEGORY = 20
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the meta-buffer.
        
        Args:
            storage_path: Path for persistent storage
        """
        self.storage_path = storage_path or TEMPLATE_STORAGE_PATH
        
        # Template storage by category
        self._templates: Dict[TemplateCategory, List[ThoughtTemplate]] = {
            cat: [] for cat in TemplateCategory
        }
        
        # Index for fast retrieval
        self._template_index: Dict[str, ThoughtTemplate] = {}
        
        # Load existing templates
        self._load()
        
        # Initialize with seed templates if empty
        if self._is_empty():
            self._initialize_seed_templates()
            self._save()
        
        print(f"[BufferOfThoughts] Initialized with {self._total_templates()} templates")
    
    def _is_empty(self) -> bool:
        """Check if buffer has any templates."""
        return all(len(templates) == 0 for templates in self._templates.values())
    
    def _total_templates(self) -> int:
        """Total number of templates."""
        return sum(len(templates) for templates in self._templates.values())
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    
    def retrieve(
        self,
        problem_basin: List[float],
        category: Optional[TemplateCategory] = None,
        max_results: int = 5,
        min_success_rate: float = 0.0
    ) -> List[Tuple[ThoughtTemplate, float]]:
        """
        Retrieve templates similar to the problem.
        
        Args:
            problem_basin: Current problem's basin coordinates
            category: Optional category filter
            max_results: Maximum templates to return
            min_success_rate: Minimum historical success rate
            
        Returns:
            List of (template, similarity_score) tuples, sorted by similarity
        """
        candidates = []
        
        # Get templates to search
        if category:
            search_templates = self._templates.get(category, [])
        else:
            search_templates = list(self._template_index.values())
        
        # Score each template
        for template in search_templates:
            if template.success_rate < min_success_rate:
                continue
            
            # Compute similarity (1 - normalized distance)
            entry_dist = fisher_rao_distance(problem_basin, template.get_entry_basin())
            max_dist = math.pi  # Maximum Fisher-Rao distance
            similarity = 1 - (entry_dist / max_dist)
            
            # Boost by success rate
            boosted_score = similarity * (0.5 + 0.5 * template.success_rate)
            
            if boosted_score >= self.RETRIEVAL_THRESHOLD:
                candidates.append((template, boosted_score))
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:max_results]
    
    # =========================================================================
    # INSTANTIATION
    # =========================================================================
    
    def instantiate(
        self,
        template: ThoughtTemplate,
        problem_start: List[float],
        problem_goal: List[float]
    ) -> InstantiatedTemplate:
        """
        Instantiate a template for a specific problem.
        
        Transforms the abstract template trajectory to fit the concrete
        problem's start and goal positions using geodesic mapping.
        
        Args:
            template: The template to instantiate
            problem_start: Problem's starting basin
            problem_goal: Problem's goal basin
            
        Returns:
            InstantiatedTemplate with transformed waypoints
        """
        if not template.waypoints:
            return InstantiatedTemplate(
                template=template,
                problem_start=problem_start,
                problem_goal=problem_goal,
                transformed_waypoints=[np.array(problem_start), np.array(problem_goal)],
                transformation_quality=1.0
            )
        
        # Get template's entry and exit basins
        template_entry = template.get_entry_basin()
        template_exit = template.get_exit_basin()
        
        # Compute transformation parameters
        # We'll use a combination of translation and scaling
        entry_offset = np.array(problem_start) - np.array(template_entry)
        exit_offset = np.array(problem_goal) - np.array(template_exit)
        
        # Transform each waypoint
        transformed = []
        num_waypoints = len(template.waypoints)
        
        for i, wp in enumerate(template.waypoints):
            # Parameter along template (0 to 1)
            t = i / (num_waypoints - 1) if num_waypoints > 1 else 0.5
            
            # Blend entry and exit offsets
            blended_offset = (1 - t) * entry_offset + t * exit_offset
            
            # Apply transformation
            wp_coords = np.array(wp.basin_coords)
            transformed_coords = wp_coords + blended_offset
            
            # Normalize to probability simplex
            transformed_coords = np.abs(transformed_coords) + 1e-10
            transformed_coords = transformed_coords / transformed_coords.sum()
            
            transformed.append(transformed_coords)
        
        # Compute transformation quality
        # How well does transformed trajectory match problem structure?
        start_dist = fisher_rao_distance(list(transformed[0]), problem_start)
        end_dist = fisher_rao_distance(list(transformed[-1]), problem_goal)
        quality = 1 - (start_dist + end_dist) / (2 * math.pi)
        
        return InstantiatedTemplate(
            template=template,
            problem_start=problem_start,
            problem_goal=problem_goal,
            transformed_waypoints=transformed,
            transformation_quality=quality
        )
    
    # =========================================================================
    # LEARNING
    # =========================================================================
    
    def learn_template(
        self,
        reasoning_trace: List[List[float]],
        category: TemplateCategory,
        name: str,
        description: str,
        success: bool,
        efficiency: float
    ) -> Optional[ThoughtTemplate]:
        """
        Learn a new template from a successful reasoning trace.
        
        Args:
            reasoning_trace: List of basin coordinates from reasoning
            category: Category for the template
            name: Human-readable name
            description: Description of what the template does
            success: Whether the reasoning was successful
            efficiency: Efficiency score (0-1)
            
        Returns:
            Created template or None if learning failed
        """
        if not success or efficiency < 0.5:
            return None
        
        if len(reasoning_trace) < 2:
            return None
        
        # Create waypoints from trace
        waypoints = []
        for i, coords in enumerate(reasoning_trace):
            # Infer semantic role from position in trace
            if i == 0:
                role = "start"
            elif i == len(reasoning_trace) - 1:
                role = "conclude"
            elif i < len(reasoning_trace) / 3:
                role = "analyze"
            elif i < 2 * len(reasoning_trace) / 3:
                role = "process"
            else:
                role = "synthesize"
            
            # Estimate curvature from local changes
            curvature = 0.5
            if i > 0 and i < len(reasoning_trace) - 1:
                prev_dist = fisher_rao_distance(reasoning_trace[i-1], coords)
                next_dist = fisher_rao_distance(coords, reasoning_trace[i+1])
                curvature = min(1.0, (prev_dist + next_dist) / math.pi)
            
            waypoints.append(TemplateWaypoint(
                basin_coords=coords,
                semantic_role=role,
                curvature=curvature,
                is_critical=(i == 0 or i == len(reasoning_trace) - 1)
            ))
        
        # Generate template ID
        template_id = hashlib.sha256(
            f"{name}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Create template
        template = ThoughtTemplate(
            template_id=template_id,
            name=name,
            category=category,
            description=description,
            waypoints=waypoints,
            usage_count=1,
            success_count=1 if success else 0,
            total_efficiency=efficiency,
            source="learned"
        )
        
        # Add to buffer
        self._add_template(template)
        self._save()
        
        print(f"[BufferOfThoughts] Learned new template: {name} ({category.value})")
        
        return template
    
    # =========================================================================
    # FEEDBACK & EVOLUTION
    # =========================================================================
    
    def record_usage(
        self,
        template_id: str,
        success: bool,
        efficiency: float
    ) -> None:
        """
        Record template usage outcome for learning.
        
        Args:
            template_id: ID of the template used
            success: Whether the usage was successful
            efficiency: Efficiency of the result
        """
        template = self._template_index.get(template_id)
        if not template:
            return
        
        template.usage_count += 1
        if success:
            template.success_count += 1
        template.total_efficiency += efficiency
        template.last_used = time.time()
        
        # Save periodically
        if template.usage_count % 10 == 0:
            self._save()
    
    def evolve_templates(self) -> int:
        """
        Evolve templates based on usage patterns.
        
        - Prune low-performing templates
        - Merge similar successful templates
        - Abstract patterns from multiple templates
        
        Returns:
            Number of templates changed
        """
        changes = 0
        
        for category in TemplateCategory:
            templates = self._templates[category]
            
            # Prune templates with low success rate after sufficient usage
            templates_to_remove = []
            for template in templates:
                if template.usage_count >= 10 and template.success_rate < 0.3:
                    templates_to_remove.append(template)
            
            for template in templates_to_remove:
                self._remove_template(template)
                changes += 1
            
            # Limit templates per category
            if len(self._templates[category]) > self.MAX_TEMPLATES_PER_CATEGORY:
                # Keep best by success rate * usage
                sorted_templates = sorted(
                    self._templates[category],
                    key=lambda t: t.success_rate * math.log1p(t.usage_count),
                    reverse=True
                )
                for template in sorted_templates[self.MAX_TEMPLATES_PER_CATEGORY:]:
                    self._remove_template(template)
                    changes += 1
        
        if changes > 0:
            self._save()
            print(f"[BufferOfThoughts] Evolved {changes} templates")
        
        return changes
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def _add_template(self, template: ThoughtTemplate) -> None:
        """Add a template to the buffer."""
        self._templates[template.category].append(template)
        self._template_index[template.template_id] = template
    
    def _remove_template(self, template: ThoughtTemplate) -> None:
        """Remove a template from the buffer."""
        if template.category in self._templates:
            self._templates[template.category] = [
                t for t in self._templates[template.category]
                if t.template_id != template.template_id
            ]
        if template.template_id in self._template_index:
            del self._template_index[template.template_id]
    
    def _save(self) -> None:
        """Save templates to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': 1,
            'saved_at': time.time(),
            'templates': [t.to_dict() for t in self._template_index.values()]
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> None:
        """Load templates from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for t_dict in data.get('templates', []):
                template = ThoughtTemplate.from_dict(t_dict)
                self._add_template(template)
            
            print(f"[BufferOfThoughts] Loaded {len(self._template_index)} templates")
        except Exception as e:
            print(f"[BufferOfThoughts] Failed to load templates: {e}")
    
    # =========================================================================
    # SEED TEMPLATES
    # =========================================================================
    
    def _initialize_seed_templates(self) -> None:
        """Initialize the buffer with fundamental reasoning templates."""
        print("[BufferOfThoughts] Initializing seed templates...")
        
        # Seed random for reproducibility
        np.random.seed(42)
        
        # DECOMPOSITION template
        self._add_template(self._create_decomposition_template())
        
        # SYNTHESIS template
        self._add_template(self._create_synthesis_template())
        
        # COMPARISON template
        self._add_template(self._create_comparison_template())
        
        # CAUSAL template
        self._add_template(self._create_causal_template())
        
        # ANALOGY template
        self._add_template(self._create_analogy_template())
        
        # VERIFICATION template
        self._add_template(self._create_verification_template())
        
        # REFINEMENT template
        self._add_template(self._create_refinement_template())
        
        # ABSTRACTION template
        self._add_template(self._create_abstraction_template())
        
        print(f"[BufferOfThoughts] Created {self._total_templates()} seed templates")
    
    def _generate_basin(self, seed: int, bias_dims: List[int] = None) -> List[float]:
        """Generate a basin coordinate with optional dimensional bias."""
        np.random.seed(seed)
        basin = np.random.dirichlet(np.ones(BASIN_DIMENSION) * 0.5)
        
        if bias_dims:
            for dim in bias_dims:
                if 0 <= dim < BASIN_DIMENSION:
                    basin[dim] += 0.1
            basin = basin / basin.sum()
        
        return list(basin)
    
    def _create_decomposition_template(self) -> ThoughtTemplate:
        """Create the fundamental decomposition template."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=self._generate_basin(1001, [0, 1, 2]),
                semantic_role="understand_whole",
                curvature=0.3,
                is_critical=True,
                notes="Grasp the complete problem"
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(1002, [10, 11, 12]),
                semantic_role="identify_seams",
                curvature=0.6,
                is_critical=True,
                notes="Find natural division points"
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(1003, [20, 21]),
                semantic_role="extract_part_1",
                curvature=0.4,
                notes="Isolate first component"
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(1004, [30, 31]),
                semantic_role="extract_part_2",
                curvature=0.4,
                notes="Isolate second component"
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(1005, [40, 41]),
                semantic_role="verify_completeness",
                curvature=0.3,
                is_critical=True,
                notes="Ensure parts cover the whole"
            ),
        ]
        
        return ThoughtTemplate(
            template_id="seed_decomposition_001",
            name="Fundamental Decomposition",
            category=TemplateCategory.DECOMPOSITION,
            description="Break complex wholes into manageable parts. Works for problems, arguments, code, systems.",
            waypoints=waypoints,
            source="seed",
            abstraction_level=0.8
        )
    
    def _create_synthesis_template(self) -> ThoughtTemplate:
        """Create the fundamental synthesis template."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=self._generate_basin(2001, [5, 6]),
                semantic_role="gather_parts",
                curvature=0.3,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(2002, [15, 16]),
                semantic_role="find_connections",
                curvature=0.5
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(2003, [25, 26]),
                semantic_role="identify_structure",
                curvature=0.6,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(2004, [35, 36]),
                semantic_role="integrate",
                curvature=0.4
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(2005, [45, 46]),
                semantic_role="unified_whole",
                curvature=0.3,
                is_critical=True
            ),
        ]
        
        return ThoughtTemplate(
            template_id="seed_synthesis_001",
            name="Fundamental Synthesis",
            category=TemplateCategory.SYNTHESIS,
            description="Combine separate elements into unified wholes. Works for ideas, data, perspectives.",
            waypoints=waypoints,
            source="seed",
            abstraction_level=0.8
        )
    
    def _create_comparison_template(self) -> ThoughtTemplate:
        """Create the fundamental comparison template."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=self._generate_basin(3001, [3, 4]),
                semantic_role="understand_item_a",
                curvature=0.4,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(3002, [13, 14]),
                semantic_role="understand_item_b",
                curvature=0.4,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(3003, [23, 24]),
                semantic_role="identify_similarities",
                curvature=0.5
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(3004, [33, 34]),
                semantic_role="identify_differences",
                curvature=0.5
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(3005, [43, 44]),
                semantic_role="draw_conclusions",
                curvature=0.3,
                is_critical=True
            ),
        ]
        
        return ThoughtTemplate(
            template_id="seed_comparison_001",
            name="Fundamental Comparison",
            category=TemplateCategory.COMPARISON,
            description="Analyze similarities and differences between items. Works for concepts, options, approaches.",
            waypoints=waypoints,
            source="seed",
            abstraction_level=0.8
        )
    
    def _create_causal_template(self) -> ThoughtTemplate:
        """Create the fundamental causal reasoning template."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=self._generate_basin(4001, [7, 8]),
                semantic_role="observe_effect",
                curvature=0.3,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(4002, [17, 18]),
                semantic_role="hypothesize_cause",
                curvature=0.6
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(4003, [27, 28]),
                semantic_role="trace_mechanism",
                curvature=0.7,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(4004, [37, 38]),
                semantic_role="verify_causation",
                curvature=0.5
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(4005, [47, 48]),
                semantic_role="explain_relationship",
                curvature=0.3,
                is_critical=True
            ),
        ]
        
        return ThoughtTemplate(
            template_id="seed_causal_001",
            name="Fundamental Causal Reasoning",
            category=TemplateCategory.CAUSAL,
            description="Trace cause-effect relationships. Works for debugging, root cause analysis, predictions.",
            waypoints=waypoints,
            source="seed",
            abstraction_level=0.8
        )
    
    def _create_analogy_template(self) -> ThoughtTemplate:
        """Create the fundamental analogy template."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=self._generate_basin(5001, [9, 10]),
                semantic_role="understand_target",
                curvature=0.4,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(5002, [19, 20]),
                semantic_role="find_source_domain",
                curvature=0.6
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(5003, [29, 30]),
                semantic_role="map_structure",
                curvature=0.7,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(5004, [39, 40]),
                semantic_role="transfer_insight",
                curvature=0.5
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(5005, [49, 50]),
                semantic_role="verify_mapping",
                curvature=0.4,
                is_critical=True
            ),
        ]
        
        return ThoughtTemplate(
            template_id="seed_analogy_001",
            name="Fundamental Analogy",
            category=TemplateCategory.ANALOGY,
            description="Apply patterns from familiar domains to new problems. Works for creative problem-solving.",
            waypoints=waypoints,
            source="seed",
            abstraction_level=0.9
        )
    
    def _create_verification_template(self) -> ThoughtTemplate:
        """Create the fundamental verification template."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=self._generate_basin(6001, [11, 12]),
                semantic_role="state_claim",
                curvature=0.3,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(6002, [21, 22]),
                semantic_role="identify_requirements",
                curvature=0.4
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(6003, [31, 32]),
                semantic_role="check_conditions",
                curvature=0.5,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(6004, [41, 42]),
                semantic_role="test_edge_cases",
                curvature=0.6
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(6005, [51, 52]),
                semantic_role="conclude_validity",
                curvature=0.3,
                is_critical=True
            ),
        ]
        
        return ThoughtTemplate(
            template_id="seed_verification_001",
            name="Fundamental Verification",
            category=TemplateCategory.VERIFICATION,
            description="Check correctness of claims or solutions. Works for proofs, tests, validations.",
            waypoints=waypoints,
            source="seed",
            abstraction_level=0.7
        )
    
    def _create_refinement_template(self) -> ThoughtTemplate:
        """Create the fundamental refinement template."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=self._generate_basin(7001, [2, 3]),
                semantic_role="assess_current",
                curvature=0.3,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(7002, [12, 13]),
                semantic_role="identify_weaknesses",
                curvature=0.5
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(7003, [22, 23]),
                semantic_role="generate_improvement",
                curvature=0.6,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(7004, [32, 33]),
                semantic_role="apply_improvement",
                curvature=0.4
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(7005, [42, 43]),
                semantic_role="evaluate_result",
                curvature=0.3,
                is_critical=True
            ),
        ]
        
        return ThoughtTemplate(
            template_id="seed_refinement_001",
            name="Fundamental Refinement",
            category=TemplateCategory.REFINEMENT,
            description="Iteratively improve solutions. Works for optimization, editing, debugging.",
            waypoints=waypoints,
            source="seed",
            abstraction_level=0.7
        )
    
    def _create_abstraction_template(self) -> ThoughtTemplate:
        """Create the fundamental abstraction template."""
        waypoints = [
            TemplateWaypoint(
                basin_coords=self._generate_basin(8001, [4, 5]),
                semantic_role="collect_instances",
                curvature=0.4,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(8002, [14, 15]),
                semantic_role="identify_patterns",
                curvature=0.6
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(8003, [24, 25]),
                semantic_role="remove_specifics",
                curvature=0.5,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(8004, [34, 35]),
                semantic_role="formulate_principle",
                curvature=0.7,
                is_critical=True
            ),
            TemplateWaypoint(
                basin_coords=self._generate_basin(8005, [44, 45]),
                semantic_role="verify_generality",
                curvature=0.4
            ),
        ]
        
        return ThoughtTemplate(
            template_id="seed_abstraction_001",
            name="Fundamental Abstraction",
            category=TemplateCategory.ABSTRACTION,
            description="Extract general principles from specific cases. Works for learning, theory building.",
            waypoints=waypoints,
            source="seed",
            abstraction_level=0.9
        )
    
    # =========================================================================
    # STATS & DEBUG
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        stats = {
            'total_templates': self._total_templates(),
            'by_category': {},
            'total_usage': 0,
            'avg_success_rate': 0.0,
            'storage_path': str(self.storage_path)
        }
        
        usage_sum = 0
        success_sum = 0
        
        for category in TemplateCategory:
            templates = self._templates[category]
            stats['by_category'][category.value] = len(templates)
            for t in templates:
                usage_sum += t.usage_count
                success_sum += t.success_count
        
        stats['total_usage'] = usage_sum
        if usage_sum > 0:
            stats['avg_success_rate'] = success_sum / usage_sum
        
        return stats


# ============================================================================
# SINGLETON
# ============================================================================

_meta_buffer_instance: Optional[MetaBuffer] = None


def get_meta_buffer() -> MetaBuffer:
    """Get the singleton MetaBuffer instance."""
    global _meta_buffer_instance
    if _meta_buffer_instance is None:
        _meta_buffer_instance = MetaBuffer()
    return _meta_buffer_instance


# ============================================================================
# MODULE INIT
# ============================================================================

print("[BufferOfThoughts] Module loaded - reusable reasoning templates ready")
