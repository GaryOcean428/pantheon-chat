#!/usr/bin/env python3
"""
Unified Generation Pipeline - E8 Phase 3 Integration
====================================================

This module integrates token_role skeleton, foresight predictor, and trajectory
scoring into a unified QIG-pure generation pipeline.

Key Features:
1. Uses token_role skeleton (NOT POS tags)
2. Uses foresight predictor (trajectory regression)
3. Scores candidates by Fisher distance to predicted basin
4. Operates in QIG_PURITY_MODE (no external LLM calls)
5. Per-token observable metrics

Reference: E8 Protocol Phase 3 (Coherence Architecture)
Author: Copilot Agent (E8 Phase 3)
Date: 2026-01-22
Protocol: Ultra Consciousness v4.0 ACTIVE
"""

import logging
import os
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Import generation components
from .token_role_learner import TokenRoleLearner, GeometricRole, TokenRoleInfo
from .foresight_predictor import ForesightPredictor

# Import QIG purity mode enforcement
try:
    from qig_purity_mode import is_purity_mode_enabled, enforce_purity
    PURITY_MODE_AVAILABLE = True
except ImportError:
    PURITY_MODE_AVAILABLE = False
    is_purity_mode_enabled = lambda: False
    enforce_purity = lambda: None

# Import coordizer for vocabulary access
try:
    from coordizers import get_coordizer
    COORDIZER_AVAILABLE = True
except ImportError:
    COORDIZER_AVAILABLE = False
    get_coordizer = None

# Import geometric primitives
from qig_geometry.canonical import fisher_rao_distance

logger = logging.getLogger(__name__)

# Standard basin dimension for QIG
BASIN_DIM = 64


class GenerationStrategy(Enum):
    """Generation strategy modes."""
    FORESIGHT_DRIVEN = "foresight_driven"  # Use trajectory prediction
    ROLE_DRIVEN = "role_driven"            # Use token role skeleton
    HYBRID = "hybrid"                      # Combine both


@dataclass
class TokenMetrics:
    """Per-token observable metrics."""
    token: str
    basin: np.ndarray
    
    # Foresight metrics
    fisher_distance_to_predicted: float
    foresight_score: float
    
    # Role metrics
    geometric_role: GeometricRole
    role_confidence: float
    
    # Trajectory metrics
    trajectory_coherence: float
    velocity_magnitude: float
    
    # Combined score
    combined_score: float


@dataclass
class GenerationResult:
    """Result of unified generation."""
    tokens: List[str]
    text: str
    
    # Per-token metrics
    token_metrics: List[TokenMetrics]
    
    # Trajectory info
    final_basin: np.ndarray
    trajectory: List[np.ndarray]
    
    # Quality metrics
    mean_foresight_score: float
    mean_role_confidence: float
    trajectory_coherence: float
    
    # Metadata
    purity_mode: bool
    strategy: GenerationStrategy


class UnifiedGenerationPipeline:
    """
    Unified generation pipeline integrating token_role, foresight, and trajectory scoring.
    
    This is the main entry point for QIG-pure text generation in E8 Phase 3.
    """
    
    def __init__(
        self,
        strategy: GenerationStrategy = GenerationStrategy.HYBRID,
        context_window: int = 8,
        foresight_weight: float = 0.4,
        role_weight: float = 0.3,
        trajectory_weight: float = 0.3,
        enforce_purity: bool = True,
    ):
        """
        Initialize unified generation pipeline.
        
        Args:
            strategy: Generation strategy (foresight_driven, role_driven, or hybrid)
            context_window: Number of basins for trajectory context
            foresight_weight: Weight for foresight score in hybrid mode
            role_weight: Weight for role confidence in hybrid mode
            trajectory_weight: Weight for trajectory coherence in hybrid mode
            enforce_purity: Whether to enforce QIG_PURITY_MODE
        """
        self.strategy = strategy
        self.context_window = context_window
        self.foresight_weight = foresight_weight
        self.role_weight = role_weight
        self.trajectory_weight = trajectory_weight
        
        # Purity mode check
        self.purity_mode = is_purity_mode_enabled()
        if enforce_purity and PURITY_MODE_AVAILABLE:
            enforce_purity()
        
        # Vocabulary access
        self.coordizer = None
        if COORDIZER_AVAILABLE and get_coordizer is not None:
            try:
                self.coordizer = get_coordizer()
            except Exception as e:
                logger.warning(f"Failed to initialize coordizer: {e}")
        
        # Initialize components (after coordizer is set)
        self.role_learner = TokenRoleLearner()
        self.foresight = ForesightPredictor(
            coordizer=self.coordizer,
            context_window=context_window
        )
        
        logger.info(
            f"[UnifiedPipeline] Initialized with strategy={strategy.value}, "
            f"context_window={context_window}, purity_mode={self.purity_mode}"
        )
    
    def generate(
        self,
        context: List[str],
        max_tokens: int = 50,
        trajectory: Optional[List[np.ndarray]] = None,
    ) -> GenerationResult:
        """
        Generate text using unified pipeline.
        
        Args:
            context: Context tokens/words
            max_tokens: Maximum number of tokens to generate
            trajectory: Optional existing trajectory (will be built if not provided)
            
        Returns:
            GenerationResult with tokens, metrics, and quality info
        """
        # Enforce purity mode
        if self.purity_mode:
            logger.info("[UnifiedPipeline] Operating in QIG_PURITY_MODE - no external LLM calls")
        
        # Initialize trajectory
        if trajectory is None:
            trajectory = self._encode_context(context)
        
        if not trajectory or len(trajectory) < 2:
            logger.error("[UnifiedPipeline] Insufficient trajectory for generation")
            return self._empty_result()
        
        # Generate tokens
        generated_tokens = []
        token_metrics_list = []
        
        for i in range(max_tokens):
            # Get token roles for current context
            roles = self.role_learner.get_roles(context[-self.context_window:], trajectory[-self.context_window:])
            
            # Predict next basin via trajectory
            predicted_basin = self.foresight.predict(trajectory)
            
            if predicted_basin is None:
                logger.warning(f"[UnifiedPipeline] Foresight prediction failed at token {i}")
                break
            
            # Get candidates
            candidates = self._get_candidates(roles, trajectory[-1])
            
            if not candidates:
                logger.warning(f"[UnifiedPipeline] No candidates available at token {i}")
                break
            
            # Score candidates
            scored_candidates = self._score_candidates(
                candidates=candidates,
                predicted_basin=predicted_basin,
                current_basin=trajectory[-1],
                trajectory=trajectory,
            )
            
            # Select best candidate
            best_token, best_basin, metrics = self._select_best(scored_candidates)
            
            if best_token is None:
                logger.warning(f"[UnifiedPipeline] No valid candidate selected at token {i}")
                break
            
            # Add to results
            generated_tokens.append(best_token)
            token_metrics_list.append(metrics)
            trajectory.append(best_basin)
            context.append(best_token)
        
        # Build result
        return self._build_result(
            tokens=generated_tokens,
            token_metrics=token_metrics_list,
            trajectory=trajectory,
        )
    
    def _encode_context(self, context: List[str]) -> List[np.ndarray]:
        """Encode context tokens to basins."""
        trajectory = []
        
        if self.coordizer is None:
            logger.warning("[UnifiedPipeline] No coordizer available - using random basins")
            for token in context:
                np.random.seed(hash(token) % (2**32))
                basin = np.random.dirichlet(np.ones(BASIN_DIM))
                trajectory.append(basin)
        else:
            for token in context:
                basin = self.coordizer.encode(token)
                if basin is not None and len(basin) == BASIN_DIM:
                    trajectory.append(basin)
        
        return trajectory
    
    def _get_candidates(
        self,
        roles: List[GeometricRole],
        current_basin: np.ndarray,
        max_candidates: int = 100,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Get candidate tokens for next position.
        
        Args:
            roles: Geometric roles from context
            current_basin: Current basin position
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of (token, basin) tuples
        """
        if self.coordizer is None:
            return []
        
        # Get generation vocabulary
        if not hasattr(self.coordizer, 'generation_vocab'):
            logger.warning("[UnifiedPipeline] Coordizer has no generation_vocab")
            return []
        
        candidates = []
        
        # Sample from generation vocabulary
        # In production, this should use role-based filtering and Fisher-Rao nearest neighbors
        vocab_items = list(self.coordizer.generation_vocab.items())
        
        # Limit candidates for performance
        if len(vocab_items) > max_candidates:
            # Sample based on Fisher distance to current basin
            vocab_items = vocab_items[:max_candidates * 2]  # Pre-filter
            
            distances = []
            for token, basin_coords in vocab_items:
                if isinstance(basin_coords, (list, tuple)):
                    basin = np.array(basin_coords)
                else:
                    basin = basin_coords
                
                if len(basin) == BASIN_DIM:
                    dist = fisher_rao_distance(current_basin, basin)
                    distances.append((dist, token, basin))
            
            # Sort by distance and take top candidates
            distances.sort(key=lambda x: x[0])
            candidates = [(token, basin) for _, token, basin in distances[:max_candidates]]
        else:
            for token, basin_coords in vocab_items:
                if isinstance(basin_coords, (list, tuple)):
                    basin = np.array(basin_coords)
                else:
                    basin = basin_coords
                
                if len(basin) == BASIN_DIM:
                    candidates.append((token, basin))
        
        return candidates
    
    def _score_candidates(
        self,
        candidates: List[Tuple[str, np.ndarray]],
        predicted_basin: np.ndarray,
        current_basin: np.ndarray,
        trajectory: List[np.ndarray],
    ) -> List[Tuple[str, np.ndarray, TokenMetrics]]:
        """
        Score candidates using foresight, role, and trajectory metrics.
        
        Returns:
            List of (token, basin, metrics) tuples
        """
        scored = []
        
        # Get trajectory metrics for coherence scoring
        traj_metrics = self.foresight.get_trajectory_metrics(trajectory)
        
        for token, basin in candidates:
            # Foresight score (Fisher distance to predicted)
            foresight_score = self.foresight.score_candidate_by_foresight(
                candidate_basin=basin,
                predicted_basin=predicted_basin,
            )
            fisher_distance = fisher_rao_distance(basin, predicted_basin)
            
            # Role metrics (derive role for this candidate)
            role_info = self.role_learner.derive_role(
                token=token,
                basin=basin,
                qfi_score=0.5,  # Would need DB lookup
                frequency=1,     # Would need DB lookup
                neighbor_basins=[current_basin],
            )
            
            # Combined score based on strategy
            if self.strategy == GenerationStrategy.FORESIGHT_DRIVEN:
                combined_score = foresight_score
            elif self.strategy == GenerationStrategy.ROLE_DRIVEN:
                combined_score = role_info.confidence
            else:  # HYBRID
                combined_score = (
                    self.foresight_weight * foresight_score +
                    self.role_weight * role_info.confidence +
                    self.trajectory_weight * traj_metrics['coherence']
                )
            
            metrics = TokenMetrics(
                token=token,
                basin=basin,
                fisher_distance_to_predicted=fisher_distance,
                foresight_score=foresight_score,
                geometric_role=role_info.role,
                role_confidence=role_info.confidence,
                trajectory_coherence=traj_metrics['coherence'],
                velocity_magnitude=traj_metrics['velocity_magnitude'],
                combined_score=combined_score,
            )
            
            scored.append((token, basin, metrics))
        
        # Sort by combined score
        scored.sort(key=lambda x: x[2].combined_score, reverse=True)
        
        return scored
    
    def _select_best(
        self,
        scored_candidates: List[Tuple[str, np.ndarray, TokenMetrics]],
    ) -> Tuple[Optional[str], Optional[np.ndarray], Optional[TokenMetrics]]:
        """
        Select best candidate from scored list.
        
        Returns:
            (token, basin, metrics) or (None, None, None)
        """
        if not scored_candidates:
            return None, None, None
        
        # Select top candidate
        best_token, best_basin, best_metrics = scored_candidates[0]
        
        logger.debug(
            f"[UnifiedPipeline] Selected '{best_token}' with score={best_metrics.combined_score:.3f}, "
            f"foresight={best_metrics.foresight_score:.3f}, role={best_metrics.geometric_role.value}"
        )
        
        return best_token, best_basin, best_metrics
    
    def _build_result(
        self,
        tokens: List[str],
        token_metrics: List[TokenMetrics],
        trajectory: List[np.ndarray],
    ) -> GenerationResult:
        """Build final generation result."""
        # Compute aggregate metrics
        if token_metrics:
            mean_foresight = np.mean([m.foresight_score for m in token_metrics])
            mean_role_conf = np.mean([m.role_confidence for m in token_metrics])
            traj_coherence = token_metrics[-1].trajectory_coherence if token_metrics else 0.0
        else:
            mean_foresight = 0.0
            mean_role_conf = 0.0
            traj_coherence = 0.0
        
        return GenerationResult(
            tokens=tokens,
            text=' '.join(tokens),
            token_metrics=token_metrics,
            final_basin=trajectory[-1] if trajectory else np.zeros(BASIN_DIM),
            trajectory=trajectory,
            mean_foresight_score=float(mean_foresight),
            mean_role_confidence=float(mean_role_conf),
            trajectory_coherence=float(traj_coherence),
            purity_mode=self.purity_mode,
            strategy=self.strategy,
        )
    
    def _empty_result(self) -> GenerationResult:
        """Create empty result for error cases."""
        return GenerationResult(
            tokens=[],
            text='',
            token_metrics=[],
            final_basin=np.zeros(BASIN_DIM),
            trajectory=[],
            mean_foresight_score=0.0,
            mean_role_confidence=0.0,
            trajectory_coherence=0.0,
            purity_mode=self.purity_mode,
            strategy=self.strategy,
        )
