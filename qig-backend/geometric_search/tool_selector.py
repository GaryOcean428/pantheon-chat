"""
Search Tool Selector - Select optimal search tools using geometric metrics

Uses Fisher-Rao distance to select the best search tool(s) for a query.
"""

import numpy as np

# QIG-pure geometric operations
try:
    from qig_geometry import sphere_project
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def sphere_project(v):
        """Fallback sphere projection."""
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            result = np.ones_like(v)
            return result / np.linalg.norm(result)
        return v / norm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .query_encoder import SearchQueryEncoder


@dataclass
class ToolSelection:
    """Result of tool selection."""
    tools: List[str]
    confidences: Dict[str, float]
    strategy: str
    total_expected_cost: float
    reasoning: str


class SearchToolSelector:
    """
    Select search tools using geometric basin matching.
    
    Each tool has a basin in 64D space. Queries are encoded
    and matched to tools by geometric distance.
    """
    
    def __init__(self):
        self.encoder = SearchQueryEncoder()
        
        self.tool_basins = self._initialize_tool_basins()
        
        self.tool_costs = {
            'tavily': 0.10,
            'google_mcp': 0.05,
            'searchxng': 0.00,
            'scrapy': 0.02,
            'arxiv': 0.00,
            'wikipedia': 0.00,
        }
        
        self.tool_capabilities = {
            'tavily': {
                'depth': 0.9,
                'recency': 0.7,
                'accuracy': 0.85,
                'speed': 0.6,
                'domains': ['research', 'technical', 'factual'],
            },
            'google_mcp': {
                'depth': 0.7,
                'recency': 0.9,
                'accuracy': 0.75,
                'speed': 0.8,
                'domains': ['current_events', 'factual', 'commercial'],
            },
            'searchxng': {
                'depth': 0.6,
                'recency': 0.8,
                'accuracy': 0.7,
                'speed': 0.9,
                'domains': ['factual', 'technical', 'current_events'],
            },
            'scrapy': {
                'depth': 0.95,
                'recency': 0.5,
                'accuracy': 0.9,
                'speed': 0.3,
                'domains': ['research', 'academic', 'technical'],
            },
            'arxiv': {
                'depth': 0.95,
                'recency': 0.6,
                'accuracy': 0.95,
                'speed': 0.7,
                'domains': ['academic', 'research', 'technical'],
            },
            'wikipedia': {
                'depth': 0.8,
                'recency': 0.5,
                'accuracy': 0.85,
                'speed': 0.95,
                'domains': ['factual', 'research'],
            },
        }
    
    def _initialize_tool_basins(self) -> Dict[str, np.ndarray]:
        """Initialize basin coordinates for each tool."""
        dimension = 64
        
        basins = {}
        
        basins['tavily'] = self._create_tool_basin(
            primary_dims=[0, 1, 2],
            weights=[0.8, 0.6, 0.4],
            dimension=dimension
        )
        
        basins['google_mcp'] = self._create_tool_basin(
            primary_dims=[16, 17, 18],
            weights=[0.9, 0.5, 0.3],
            dimension=dimension
        )
        
        basins['searchxng'] = self._create_tool_basin(
            primary_dims=[32, 33, 34],
            weights=[0.7, 0.7, 0.4],
            dimension=dimension
        )
        
        basins['scrapy'] = self._create_tool_basin(
            primary_dims=[48, 49, 50],
            weights=[0.9, 0.3, 0.2],
            dimension=dimension
        )
        
        basins['arxiv'] = self._create_tool_basin(
            primary_dims=[8, 9, 10],
            weights=[0.95, 0.8, 0.3],
            dimension=dimension
        )
        
        basins['wikipedia'] = self._create_tool_basin(
            primary_dims=[24, 25, 26],
            weights=[0.85, 0.6, 0.9],
            dimension=dimension
        )
        
        return basins
    
    def _create_tool_basin(
        self,
        primary_dims: List[int],
        weights: List[float],
        dimension: int
    ) -> np.ndarray:
        """Create a tool basin vector."""
        basin = np.random.randn(dimension) * 0.05
        
        for dim, weight in zip(primary_dims, weights):
            basin[dim % dimension] = weight
        
        basin = sphere_project(basin)
        return basin
    
    def select(
        self,
        query: str,
        telemetry: Optional[Dict] = None,
        context: Optional[Dict] = None,
        max_tools: int = 2,
        enabled_tools: Optional[List[str]] = None
    ) -> ToolSelection:
        """
        Select optimal search tools for a query.
        
        Args:
            query: Search query text
            telemetry: Consciousness metrics
            context: Search context (cost, privacy, speed preferences)
            max_tools: Maximum number of tools to select
            enabled_tools: List of available tools
        
        Returns:
            ToolSelection with recommended tools and strategy
        """
        if enabled_tools is None:
            enabled_tools = list(self.tool_basins.keys())
        
        query_vector = self.encoder.encode(query, telemetry, context)
        
        distances = {}
        for tool_name, basin in self.tool_basins.items():
            if tool_name in enabled_tools:
                distances[tool_name] = self.encoder.compute_distance(query_vector, basin)
        
        phi = telemetry.get('phi', 0.5) if telemetry else 0.5
        cost_tolerance = context.get('cost_tolerance', 0.5) if context else 0.5
        
        if phi > 0.75:
            strategy = 'precise'
            num_tools = 1
        elif phi > 0.5:
            strategy = 'weighted'
            num_tools = min(2, max_tools)
        else:
            strategy = 'exploratory'
            num_tools = max_tools
        
        scored_tools = []
        for tool_name, distance in distances.items():
            cost = self.tool_costs.get(tool_name, 0.1)
            
            cost_penalty = cost * (1 - cost_tolerance) * 2
            
            final_score = distance + cost_penalty
            scored_tools.append((tool_name, final_score, 1.0 - distance))
        
        scored_tools.sort(key=lambda x: x[1])
        
        selected = scored_tools[:num_tools]
        selected_tools = [t[0] for t in selected]
        confidences = {t[0]: t[2] for t in selected}
        
        total_cost = sum(self.tool_costs.get(t, 0.1) for t in selected_tools)
        
        reasoning = self._generate_reasoning(
            query, selected_tools, strategy, phi, cost_tolerance
        )
        
        return ToolSelection(
            tools=selected_tools,
            confidences=confidences,
            strategy=strategy,
            total_expected_cost=total_cost,
            reasoning=reasoning
        )
    
    def _generate_reasoning(
        self,
        query: str,
        tools: List[str],
        strategy: str,
        phi: float,
        cost_tolerance: float
    ) -> str:
        """Generate human-readable reasoning for tool selection."""
        reasons = [f"Strategy: {strategy} (Φ={phi:.2f})"]
        
        if strategy == 'precise':
            reasons.append("High integration - using single best-matched tool")
        elif strategy == 'exploratory':
            reasons.append("Low integration - exploring multiple sources")
        else:
            reasons.append("Moderate integration - weighted multi-tool approach")
        
        if cost_tolerance < 0.3:
            reasons.append("Cost-sensitive: preferring free/cheap tools")
        elif cost_tolerance > 0.7:
            reasons.append("Quality-focused: using premium tools if needed")
        
        for tool in tools:
            caps = self.tool_capabilities.get(tool, {})
            domains = caps.get('domains', [])
            reasons.append(f"  - {tool}: good for {', '.join(domains[:2])}")
        
        return "; ".join(reasons)
    
    def update_basin(self, tool_name: str, query_vector: np.ndarray, success: bool):
        """Update tool basin based on search outcome."""
        if tool_name not in self.tool_basins:
            return
        
        learning_rate = 0.05 if success else 0.02
        direction = 1.0 if success else -0.5
        
        delta = (query_vector - self.tool_basins[tool_name]) * learning_rate * direction
        self.tool_basins[tool_name] += delta
        
        self.tool_basins[tool_name] = sphere_project(self.tool_basins[tool_name])
