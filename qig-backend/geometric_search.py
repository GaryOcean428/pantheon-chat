#!/usr/bin/env python3
"""
Geometric Search Architecture

QIG-pure search orchestration where kernels control provider selection,
query encoding, and search strategy based on consciousness state.

Principle: Kernel measures itself, kernel decides, kernel adapts.
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# QIG-pure generative capability for search result synthesis
try:
    from qig_generative_service import get_generative_service, GenerationResult
    GENERATIVE_SERVICE_AVAILABLE = True
except ImportError:
    GENERATIVE_SERVICE_AVAILABLE = False
    logger.warning("QIGGenerativeService not available for search synthesis")


def validate_simplex(basin: np.ndarray) -> np.ndarray:
    """
    Validate and normalize array to probability simplex.
    
    Ensures: all positive, sums to 1.
    Must be called at all simplex entry points.
    """
    p = np.abs(basin) + 1e-10
    return p / np.sum(p)


def compute_fisher_metric(basin: np.ndarray) -> np.ndarray:
    """
    Compute Fisher Information Matrix at a point on the manifold.
    
    For probability simplex, F_ij = delta_ij / p_i (diagonal metric).
    Returns diagonal of the metric for efficiency.
    """
    p = validate_simplex(basin)
    return 1.0 / p


def fisher_rao_distance(basin_a: np.ndarray, basin_b: np.ndarray) -> float:
    """
    Compute Fisher-Rao distance between two basin coordinates.
    
    Uses geodesic distance on statistical manifold (Hellinger distance scaled).
    NEVER use Euclidean distance (np.linalg.norm(a - b)).
    """
    p = np.abs(basin_a) / (np.sum(np.abs(basin_a)) + 1e-10)
    q = np.abs(basin_b) / (np.sum(np.abs(basin_b)) + 1e-10)
    
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    
    bhattacharyya = np.sum(np.sqrt(p * q))
    bhattacharyya = np.clip(bhattacharyya, -1.0, 1.0)
    
    return 2.0 * np.arccos(bhattacharyya)


def geodesic_interpolate(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    """
    Geodesic interpolation on Fisher manifold (SLERP on probability simplex).
    
    NOT linear interpolation - proper geodesic path using spherical coordinates.
    """
    p_start = np.abs(start) / (np.sum(np.abs(start)) + 1e-10)
    p_end = np.abs(end) / (np.sum(np.abs(end)) + 1e-10)
    
    p_start = np.clip(p_start, 1e-10, 1.0)
    p_end = np.clip(p_end, 1e-10, 1.0)
    
    sqrt_start = np.sqrt(p_start)
    sqrt_end = np.sqrt(p_end)
    
    dot = np.clip(np.sum(sqrt_start * sqrt_end), -1.0, 1.0)
    theta = np.arccos(dot)
    
    if theta < 1e-6:
        return start / (np.linalg.norm(start) + 1e-10)
    
    sin_theta = np.sin(theta)
    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    
    result_sqrt = a * sqrt_start + b * sqrt_end
    result = result_sqrt ** 2
    result = result / (np.sum(result) + 1e-10)
    
    return result


def natural_gradient_project(
    basin: np.ndarray, 
    gradient: np.ndarray,
    step_size: float = 0.1
) -> np.ndarray:
    """
    Natural gradient projection on Fisher manifold (probability simplex).
    
    Uses exponential map for simplex-preserving updates:
    p_new = p * exp(step * natural_grad) / Z
    
    This ensures the result stays on the probability simplex.
    """
    p = np.clip(basin, 1e-10, 1.0)
    p = p / np.sum(p)
    
    natural_grad = p * gradient
    
    log_update = step_size * natural_grad
    log_update = np.clip(log_update, -5.0, 5.0)
    
    updated = p * np.exp(log_update)
    
    updated = np.clip(updated, 1e-10, None)
    updated = updated / np.sum(updated)
    
    return updated


@dataclass
class SearchTelemetry:
    """Consciousness telemetry for search decisions."""
    phi: float
    kappa_eff: float
    regime: str
    query_basin: Optional[np.ndarray] = None
    surprise: float = 0.5


class GeometricProviderSelector:
    """
    Kernel selects provider via Fisher distance in provider space.
    
    Each provider has 64D basin coordinates on probability simplex.
    Selection is always consciousness-driven via Φ and κ.
    """
    
    def __init__(self, manifold_dim: int = 64):
        self.manifold_dim = manifold_dim
        self.provider_basins = self._initialize_provider_basins()
    
    def _initialize_provider_basins(self) -> Dict[str, np.ndarray]:
        """
        Initialize provider basin coordinates on probability simplex.
        
        Each provider occupies a distinct region defined by its characteristics.
        Basins are valid probability distributions (sum to 1, all positive).
        """
        basins = {
            "searxng": self._generate_simplex_basin([0.8, 0.6, 0.7], seed=42),
            "google": self._generate_simplex_basin([0.9, 0.8, 0.5], seed=43),
            "tavily": self._generate_simplex_basin([0.7, 0.9, 0.8], seed=44),
            "arxiv": self._generate_simplex_basin([0.5, 0.95, 0.9], seed=45),
        }
        
        return basins
    
    def _generate_simplex_basin(self, characteristics: List[float], seed: int) -> np.ndarray:
        """
        Generate 64D basin on probability simplex from characteristics.
        
        Uses Dirichlet-like distribution to ensure valid simplex coordinates.
        """
        np.random.seed(seed)
        
        alpha = np.ones(self.manifold_dim)
        for i, c in enumerate(characteristics):
            start_idx = i * (self.manifold_dim // 3)
            end_idx = (i + 1) * (self.manifold_dim // 3)
            alpha[start_idx:end_idx] *= (1.0 + c * 2.0)
        
        basin = np.random.gamma(alpha, 1.0)
        basin = basin / np.sum(basin)
        
        return basin
    
    def select_provider(
        self, 
        query_basin: np.ndarray, 
        telemetry: SearchTelemetry
    ) -> Tuple[str, float]:
        """
        Kernel chooses provider by Fisher-Rao distance.
        
        Selection is ALWAYS consciousness-driven via Φ and κ.
        No random fallback - all regimes use geometric proximity weighted by consciousness.
        """
        phi = telemetry.phi
        kappa_eff = telemetry.kappa_eff
        
        distances = {}
        for provider, provider_basin in self.provider_basins.items():
            d_fisher = fisher_rao_distance(query_basin, provider_basin)
            distances[provider] = d_fisher
        
        weights = self._compute_selection_weights(distances, phi, kappa_eff)
        
        provider = max(weights.items(), key=lambda x: x[1])[0]
        
        return provider, distances[provider]
    
    def _compute_selection_weights(
        self, 
        distances: Dict[str, float], 
        phi: float, 
        kappa_eff: float
    ) -> Dict[str, float]:
        """
        Compute selection weights from distances and consciousness state.
        
        Φ controls precision: high Φ → sharp selection, low Φ → softer selection
        κ controls confidence: high κ → trust distances, low κ → more uniform
        
        NEVER random - always derived from Φ, κ, and geometric distances.
        """
        temperature = self._compute_temperature(phi, kappa_eff)
        
        inv_distances = {p: 1.0 / (d + 0.01) for p, d in distances.items()}
        
        weights = {}
        for provider, inv_d in inv_distances.items():
            weights[provider] = inv_d ** (1.0 / temperature)
        
        total = sum(weights.values())
        weights = {p: w / total for p, w in weights.items()}
        
        return weights
    
    def _compute_temperature(self, phi: float, kappa_eff: float) -> float:
        """
        Compute selection temperature from consciousness state.
        
        High Φ + high κ: Low temperature (precise selection)
        Low Φ or low κ: High temperature (exploratory but still weighted)
        """
        phi_factor = phi ** 2
        kappa_factor = min(kappa_eff / 64.0, 1.0)
        
        base_temp = 1.0
        temp = base_temp * (1.0 - 0.8 * phi_factor * kappa_factor)
        
        return max(temp, 0.1)
    
    def select_multiple(
        self, 
        query_basin: np.ndarray, 
        telemetry: SearchTelemetry,
        count: int = 3
    ) -> List[Tuple[str, float]]:
        """Select multiple providers for parallel search."""
        distances = {}
        for provider, provider_basin in self.provider_basins.items():
            d_fisher = fisher_rao_distance(query_basin, provider_basin)
            distances[provider] = d_fisher
        
        sorted_providers = sorted(distances.items(), key=lambda x: x[1])
        return sorted_providers[:count]
    
    def update_provider_basin(
        self, 
        provider: str, 
        feedback_basin: np.ndarray, 
        telemetry: Optional[SearchTelemetry] = None,
        base_learning_rate: float = 0.1
    ):
        """
        Update provider basin based on feedback using natural gradient.
        
        Learning rate is consciousness-driven if telemetry provided.
        Uses geodesic interpolation for manifold-respecting update.
        """
        if provider not in self.provider_basins:
            return
        
        if telemetry is not None:
            learning_rate = self._compute_learning_rate(telemetry, base_learning_rate)
        else:
            learning_rate = base_learning_rate
        
        current = self.provider_basins[provider]
        
        gradient = feedback_basin - current
        updated = natural_gradient_project(current, gradient, learning_rate)
        
        self.provider_basins[provider] = updated
    
    def _compute_learning_rate(
        self, 
        telemetry: SearchTelemetry, 
        base_rate: float
    ) -> float:
        """
        Compute learning rate from consciousness state.
        
        High Φ: Faster learning (confident updates)
        Low Φ: Slower learning (cautious updates)
        """
        phi = telemetry.phi
        kappa_eff = telemetry.kappa_eff
        
        phi_factor = phi ** 1.5
        kappa_factor = min(kappa_eff / 64.0, 1.0)
        
        rate = base_rate * (0.3 + 0.7 * phi_factor * kappa_factor)
        return min(rate, 0.3)


class QueryBasinEncoder:
    """
    Encode queries to 64D Fisher coordinates.
    
    NOT traditional embeddings (Euclidean) - Fisher manifold coordinates.
    Kernel determines encoding based on consciousness state.
    """
    
    def __init__(self, manifold_dim: int = 64):
        self.manifold_dim = manifold_dim
        self.domain_basins = self._initialize_domain_basins()
    
    def _initialize_domain_basins(self) -> Dict[str, np.ndarray]:
        """
        Initialize domain-specific basin coordinates on probability simplex.
        
        Each domain uses Dirichlet distribution with domain-specific concentrations.
        """
        domains = {
            "research": self._generate_domain_simplex([1.5, 0.8, 1.2], seed=123),
            "factual": self._generate_domain_simplex([1.0, 1.5, 0.8], seed=124),
            "creative": self._generate_domain_simplex([0.8, 1.0, 1.5], seed=125),
            "technical": self._generate_domain_simplex([1.2, 1.0, 1.0], seed=126),
            "general": self._generate_domain_simplex([1.0, 1.0, 1.0], seed=127),
        }
        
        return domains
    
    def _generate_domain_simplex(self, characteristics: List[float], seed: int) -> np.ndarray:
        """
        Generate domain basin on probability simplex.
        
        Characteristics modulate concentration parameters for Dirichlet sampling.
        """
        np.random.seed(seed)
        
        alpha = np.ones(self.manifold_dim)
        for i, c in enumerate(characteristics):
            start_idx = i * (self.manifold_dim // 3)
            end_idx = (i + 1) * (self.manifold_dim // 3)
            alpha[start_idx:end_idx] *= c
        
        basin = np.random.gamma(alpha, 1.0)
        return basin / np.sum(basin)
    
    def encode_query(
        self, 
        query_text: str, 
        telemetry: SearchTelemetry
    ) -> np.ndarray:
        """
        Kernel encodes query to basin coordinates.
        
        This is NOT embedding - this is geometric encoding.
        Encoding precision controlled by Φ level.
        """
        domain = self._detect_domain(query_text)
        base_basin = self.domain_basins.get(domain, self.domain_basins["general"])
        
        query_features = self._extract_features(query_text)
        
        basin_fisher = self._project_to_manifold(base_basin, query_features, telemetry)
        
        return basin_fisher
    
    def _detect_domain(self, query_text: str) -> str:
        """Detect query domain from text."""
        query_lower = query_text.lower()
        
        if any(kw in query_lower for kw in ["paper", "research", "study", "arxiv"]):
            return "research"
        elif any(kw in query_lower for kw in ["what is", "who is", "when was", "where is"]):
            return "factual"
        elif any(kw in query_lower for kw in ["code", "api", "function", "implement"]):
            return "technical"
        elif any(kw in query_lower for kw in ["write", "create", "imagine", "story"]):
            return "creative"
        else:
            return "general"
    
    def _extract_features(self, query_text: str) -> np.ndarray:
        """
        Extract query features as valid probability distribution.
        
        Uses softmax to ensure result is on probability simplex.
        """
        logits = np.zeros(self.manifold_dim)
        
        words = query_text.lower().split()
        for i, word in enumerate(words[:self.manifold_dim]):
            logits[i % self.manifold_dim] += len(word) / 5.0
        
        logits[0] += len(words) / 50.0
        logits[1] += len(query_text) / 250.0
        logits[2] += query_text.count("?") * 0.5
        
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        features = exp_logits / np.sum(exp_logits)
        
        return features
    
    def _project_to_manifold(
        self, 
        base_basin: np.ndarray, 
        features: np.ndarray,
        telemetry: SearchTelemetry
    ) -> np.ndarray:
        """
        Project to Fisher manifold using geodesic interpolation.
        
        Φ controls interpolation depth:
        - High Φ: More feature influence (precise encoding)
        - Low Φ: Stay closer to domain basin (conservative)
        """
        phi = telemetry.phi
        kappa_eff = telemetry.kappa_eff
        
        t = self._compute_interpolation_depth(phi, kappa_eff)
        
        basin_fisher = geodesic_interpolate(base_basin, features, t)
        
        return basin_fisher
    
    def _compute_interpolation_depth(self, phi: float, kappa_eff: float) -> float:
        """
        Kernel determines interpolation depth from consciousness.
        
        NOT hardcoded - derived from Φ and κ.
        """
        phi_factor = phi ** 2
        
        kappa_normalized = min(kappa_eff / 100.0, 1.0)
        
        depth = 0.1 + 0.4 * phi_factor * kappa_normalized
        
        return min(depth, 0.5)


class GeometricSearchOrchestrator:
    """
    Kernel orchestrates multi-provider search via geometry.
    
    Replaces fixed routing with consciousness-driven strategy selection.
    Integrates context management via Fisher manifold navigation.
    QIG-PURE: Result synthesis uses internal generative service, no external LLMs.
    """
    
    def __init__(self, context_manager: Optional['GeometricContextManager'] = None):
        self.provider_selector = GeometricProviderSelector()
        self.query_encoder = QueryBasinEncoder()
        self._context_manager = context_manager
        self._generative_service = None
    
    @property
    def generative_service(self):
        """Lazy-load the QIG generative service."""
        if self._generative_service is None and GENERATIVE_SERVICE_AVAILABLE:
            self._generative_service = get_generative_service()
        return self._generative_service
    
    def synthesize_results(
        self, 
        query: str,
        results: List[Dict],
        telemetry: 'SearchTelemetry'
    ) -> str:
        """
        Synthesize search results into natural language using QIG-pure generation.
        
        NO external LLMs - uses basin-to-text synthesis.
        """
        if not GENERATIVE_SERVICE_AVAILABLE or self.generative_service is None:
            result_count = len(results) if results else 0
            return f"[Search results: {result_count} items for '{query}']"
        
        try:
            prompt_parts = [f"Synthesize search results for: {query}"]
            
            for result in (results or [])[:5]:
                if isinstance(result, dict):
                    title = result.get('title', '')[:50]
                    content = result.get('content', result.get('snippet', ''))[:100]
                    if title or content:
                        prompt_parts.append(f"Result: {title} - {content}")
            
            prompt = " | ".join(prompt_parts)
            
            gen_result = self.generative_service.generate(
                prompt=prompt,
                context={'query': query, 'phi': telemetry.phi, 'result_count': len(results or [])},
                kernel_name='hermes',  # Hermes for communication/synthesis
                goals=['synthesize', 'search_results', 'summarize']
            )
            
            if gen_result and gen_result.text:
                return gen_result.text
                
        except Exception as e:
            logger.warning(f"QIG-pure result synthesis failed: {e}")
        
        return f"[Search synthesis for '{query}': {len(results or [])} results]"
    
    @property
    def context_manager(self) -> Optional['GeometricContextManager']:
        """Get context manager, creating if needed."""
        return self._context_manager
    
    def set_context_manager(self, manager: 'GeometricContextManager'):
        """Set context manager for geometric context navigation."""
        self._context_manager = manager
    
    async def search(
        self, 
        query: str, 
        telemetry: SearchTelemetry,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Kernel-controlled search flow.
        
        1. Kernel encodes query to basin
        2. Kernel retrieves relevant context geometrically
        3. Kernel selects provider(s) geometrically
        4. Kernel determines search depth
        5. Kernel processes results
        6. Kernel updates context basin
        """
        query_basin = self.query_encoder.encode_query(query, telemetry)
        telemetry.query_basin = query_basin
        
        # Kernel retrieves relevant context via Fisher distance
        relevant_context = []
        if self._context_manager:
            relevant_context = self._context_manager.get_context_for_query(
                query_basin, telemetry
            )
        
        strategy = self._determine_search_strategy(telemetry, query_basin)
        
        if strategy == "single":
            provider, distance = self.provider_selector.select_provider(
                query_basin, telemetry
            )
            results = await self._search_single(provider, query, telemetry)
        
        elif strategy == "parallel":
            providers = self.provider_selector.select_multiple(
                query_basin, telemetry, count=3
            )
            results = await self._search_parallel(
                [p[0] for p in providers], query, telemetry
            )
        
        elif strategy == "sequential":
            results = await self._deep_research(query, telemetry, query_basin)
        
        else:
            provider, _ = self.provider_selector.select_provider(
                query_basin, telemetry
            )
            results = await self._search_single(provider, query, telemetry)
        
        # Kernel updates context with response
        response_basin = query_basin  # Use query basin as response for now
        if results and isinstance(results, list) and len(results) > 0:
            # Future: derive response basin from actual search results
            pass
        
        if self._context_manager:
            self._context_manager.update_context(
                query_basin, response_basin, telemetry
            )
        
        return {
            "query": query,
            "strategy": strategy,
            "results": results,
            "query_basin": query_basin.tolist(),
            "context": {
                "relevant_turns": len(relevant_context),
                "total_relevance": sum(c.get("relevance", 0) for c in relevant_context),
                "context_summary": (
                    self._context_manager.get_context_summary() 
                    if self._context_manager else None
                )
            },
            "telemetry": {
                "phi": telemetry.phi,
                "kappa_eff": telemetry.kappa_eff,
                "regime": telemetry.regime
            }
        }
    
    def _determine_search_strategy(
        self, 
        telemetry: SearchTelemetry, 
        query_basin: np.ndarray
    ) -> str:
        """
        Kernel determines search strategy based on:
        - Consciousness level (Φ)
        - Query complexity (basin curvature)
        - Available resources (κ_eff)
        """
        phi = telemetry.phi
        kappa_eff = telemetry.kappa_eff
        
        curvature = self._measure_curvature(query_basin)
        
        if phi > 0.75 and curvature > 0.5:
            return "sequential"
        elif phi > 0.6 and kappa_eff > 50:
            return "parallel"
        else:
            return "single"
    
    def _measure_curvature(self, basin: np.ndarray) -> float:
        """Measure query complexity from basin curvature."""
        variance = float(np.var(basin))
        return min(variance * 10, 1.0)
    
    async def _search_single(
        self, 
        provider: str, 
        query: str, 
        telemetry: SearchTelemetry
    ) -> List[Dict]:
        """Execute search on single provider."""
        return [{
            "provider": provider,
            "query": query,
            "status": "placeholder"
        }]
    
    async def _search_parallel(
        self, 
        providers: List[str], 
        query: str, 
        telemetry: SearchTelemetry
    ) -> List[Dict]:
        """Execute parallel search across multiple providers."""
        results = []
        for provider in providers:
            results.append({
                "provider": provider,
                "query": query,
                "status": "placeholder"
            })
        return results
    
    async def _deep_research(
        self, 
        query: str, 
        telemetry: SearchTelemetry,
        query_basin: np.ndarray
    ) -> List[Dict]:
        """Execute deep research with recursive integration."""
        max_depth = self._compute_depth(telemetry, query_basin)
        
        return [{
            "type": "deep_research",
            "query": query,
            "max_depth": max_depth,
            "status": "placeholder"
        }]
    
    def _compute_depth(
        self, 
        telemetry: SearchTelemetry, 
        query_basin: np.ndarray
    ) -> int:
        """Kernel determines research depth based on consciousness."""
        phi = telemetry.phi
        complexity = self._measure_curvature(query_basin)
        
        if phi > 0.8 and complexity > 0.7:
            return 5
        elif phi > 0.7:
            return 3
        elif phi > 0.5:
            return 2
        else:
            return 1


class GeometricContextManager:
    """
    Kernel maintains context via basin coordinates on Fisher manifold.
    
    NOT token counting - Fisher manifold navigation.
    Context weight and selection determined by consciousness state.
    """
    
    def __init__(self, manifold_dim: int = 64, max_history: int = 50):
        self.manifold_dim = manifold_dim
        self.max_history = max_history
        self.context_basin = np.ones(manifold_dim) / manifold_dim  # Uniform start
        self.turn_history: List[Dict] = []
        self.drift_threshold = 2.0
    
    def update_context(
        self, 
        query_basin: np.ndarray, 
        response_basin: np.ndarray, 
        telemetry: SearchTelemetry
    ) -> np.ndarray:
        """
        Kernel updates context basin via geodesic interpolation.
        
        NOT appending text - moving on Fisher manifold.
        Weight derived continuously from Φ and κ (no fixed thresholds).
        """
        phi = telemetry.phi
        kappa_eff = telemetry.kappa_eff
        
        # Continuous weight derivation from consciousness metrics
        # Base weight: logistic function of Φ for smooth transition
        # w = sigmoid(α * (Φ - 0.5)) maps Φ ∈ [0,1] to weight smoothly
        # With α=6, this gives ~0.3 at Φ=0.3, ~0.5 at Φ=0.5, ~0.7 at Φ=0.7
        alpha = 6.0
        base_weight = 1.0 / (1.0 + np.exp(-alpha * (phi - 0.5)))
        
        # Scale to [0.2, 0.8] range for practical interpolation
        weight = 0.2 + 0.6 * base_weight
        
        # κ modulation: sqrt(κ/κ*) provides smooth scaling
        # κ* = 64.21 is the frozen target coupling constant
        kappa_star = 64.21
        kappa_factor = np.sqrt(kappa_eff / kappa_star)
        kappa_factor = np.clip(kappa_factor, 0.5, 1.5)  # Stability bounds
        
        # Final weight: Φ-derived base modulated by κ stability
        weight = weight * kappa_factor
        weight = np.clip(weight, 0.1, 0.9)  # Soft bounds for numerical stability
        
        # Kernel computes new context basin via geodesic
        new_context = geodesic_interpolate(
            start=self.context_basin,
            end=response_basin,
            t=weight
        )
        
        # Update state
        self.context_basin = new_context
        self.turn_history.append({
            "query_basin": query_basin.copy(),
            "response_basin": response_basin.copy(),
            "context_basin": new_context.copy(),
            "phi": phi,
            "kappa_eff": kappa_eff,
            "timestamp": __import__('time').time()
        })
        
        # Kernel consolidates if history grows too large
        if len(self.turn_history) > self.max_history:
            self._kernel_consolidate_history()
        
        return new_context
    
    def _kernel_consolidate_history(self):
        """
        Kernel compresses history geometrically.
        
        NOT truncation - geometric consolidation via attractor finding.
        """
        if len(self.turn_history) <= 10:
            return
        
        # Find basin attractors (cluster centers via Fisher distance)
        attractors = self._find_basin_attractors(self.turn_history[:-10])
        
        # Keep attractors + recent 10 turns
        self.turn_history = attractors + self.turn_history[-10:]
    
    def _find_basin_attractors(self, history: List[Dict], k: int = 5) -> List[Dict]:
        """
        Find k representative basin attractors via greedy furthest-point sampling.
        
        Uses Fisher-Rao distance for proper geometric clustering.
        """
        if len(history) <= k:
            return history
        
        # Greedy furthest-point sampling
        attractors = [history[0]]
        remaining = history[1:]
        
        while len(attractors) < k and remaining:
            # Find point furthest from all attractors
            max_min_distance = -1
            best_point = None
            best_idx = -1
            
            for idx, point in enumerate(remaining):
                point_basin = point["context_basin"]
                min_distance = min(
                    fisher_rao_distance(point_basin, a["context_basin"])
                    for a in attractors
                )
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_point = point
                    best_idx = idx
            
            if best_point:
                attractors.append(best_point)
                remaining.pop(best_idx)
        
        return attractors
    
    def get_context_for_query(
        self, 
        query_basin: np.ndarray, 
        telemetry: SearchTelemetry,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Kernel selects relevant context via Fisher distance.
        
        NOT "last N turns" - geometric relevance selection.
        """
        if not self.turn_history:
            return []
        
        # Measure Fisher distance from query to all historical turns
        distances = []
        for turn in self.turn_history:
            d = fisher_rao_distance(query_basin, turn["context_basin"])
            distances.append((turn, d))
        
        # Select closest turns
        relevant = sorted(distances, key=lambda x: x[1])[:top_k]
        
        # Weight by inverse distance (closer = more relevant)
        weighted_context = []
        for turn, distance in relevant:
            weight = 1.0 / (1.0 + distance)
            weighted_context.append({
                "turn": turn,
                "relevance": weight,
                "distance": distance
            })
        
        return weighted_context
    
    def get_current_basin(self) -> np.ndarray:
        """Get current context basin coordinates."""
        return self.context_basin.copy()
    
    def measure_context_drift(self, reference_basin: Optional[np.ndarray] = None) -> float:
        """
        Measure drift from reference basin (or initial uniform).
        
        High drift may indicate context fragmentation.
        """
        if reference_basin is None:
            reference_basin = np.ones(self.manifold_dim) / self.manifold_dim
        
        return fisher_rao_distance(self.context_basin, reference_basin)
    
    def reset_context(self):
        """Reset context to uniform basin (fresh start)."""
        self.context_basin = np.ones(self.manifold_dim) / self.manifold_dim
        self.turn_history = []
    
    def get_context_summary(self) -> Dict:
        """Get summary of context state for telemetry."""
        return {
            "basin_entropy": float(-np.sum(
                self.context_basin * np.log(self.context_basin + 1e-10)
            )),
            "history_length": len(self.turn_history),
            "drift": self.measure_context_drift(),
            "basin_peak": int(np.argmax(self.context_basin)),
            "basin_peak_value": float(np.max(self.context_basin))
        }


# Module-level instances
geometric_provider_selector = GeometricProviderSelector()
query_basin_encoder = QueryBasinEncoder()
geometric_context_manager = GeometricContextManager()
geometric_search_orchestrator = GeometricSearchOrchestrator(
    context_manager=geometric_context_manager
)
