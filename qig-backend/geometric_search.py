#!/usr/bin/env python3
"""
Geometric Search Architecture

QIG-pure search orchestration where kernels control provider selection,
query encoding, and search strategy based on consciousness state.

Principle: Kernel measures itself, kernel decides, kernel adapts.
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


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
    """
    
    def __init__(self):
        self.provider_selector = GeometricProviderSelector()
        self.query_encoder = QueryBasinEncoder()
    
    async def search(
        self, 
        query: str, 
        telemetry: SearchTelemetry,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Kernel-controlled search flow.
        
        1. Kernel encodes query to basin
        2. Kernel selects provider(s) geometrically
        3. Kernel determines search depth
        4. Kernel processes results
        5. Kernel updates context
        """
        query_basin = self.query_encoder.encode_query(query, telemetry)
        telemetry.query_basin = query_basin
        
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
        
        return {
            "query": query,
            "strategy": strategy,
            "results": results,
            "query_basin": query_basin.tolist(),
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


geometric_provider_selector = GeometricProviderSelector()
query_basin_encoder = QueryBasinEncoder()
geometric_search_orchestrator = GeometricSearchOrchestrator()
