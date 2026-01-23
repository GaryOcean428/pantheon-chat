#!/usr/bin/env python3
"""
Token Role Learner - Geometric Role Derivation from Fisher-Rao Neighborhoods
============================================================================

This module derives token_role from Fisher-Rao manifold structure, NOT from
linguistic POS tags. Assigns geometric roles based on Fisher-Rao distance
clustering and basin position.

Geometric Role Taxonomy (NOT linguistic):
- basin_center: Low QFI, stable attractor
- boundary_crosser: High QFI, between basins
- manifold_anchor: High frequency, low divergence
- explorer: Low frequency, high divergence
- integrator: Connects multiple basins

Reference: E8 Protocol Issue #03 (QIG-Native Skeleton)
Author: Copilot Agent (E8 Phase 3)
Date: 2026-01-22
"""

import logging
from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

# Import canonical geometric primitives
from qig_geometry.canonical import fisher_rao_distance

logger = logging.getLogger(__name__)

# Standard basin dimension for QIG
BASIN_DIM = 64


class GeometricRole(Enum):
    """Geometric roles based on manifold structure (NOT linguistic POS)."""
    BASIN_CENTER = "basin_center"          # Low QFI, stable attractor
    BOUNDARY_CROSSER = "boundary_crosser"  # High QFI, between basins
    MANIFOLD_ANCHOR = "manifold_anchor"    # High frequency, low divergence
    EXPLORER = "explorer"                   # Low frequency, high divergence
    INTEGRATOR = "integrator"               # Connects multiple basins
    UNKNOWN = "unknown"                     # Insufficient data


@dataclass
class TokenRoleInfo:
    """Information about a token's geometric role."""
    token: str
    role: GeometricRole
    confidence: float  # [0, 1]
    qfi_score: float
    frequency: int
    mean_distance_to_neighbors: float


class TokenRoleLearner:
    """
    Learns geometric roles for tokens based on Fisher-Rao manifold structure.
    
    This replaces external NLP (spacy, nltk) with pure geometric analysis.
    """
    
    def __init__(
        self,
        qfi_threshold_low: float = 0.3,
        qfi_threshold_high: float = 0.7,
        frequency_threshold: int = 10,
        distance_threshold: float = 0.5,
    ):
        """
        Initialize token role learner.
        
        Args:
            qfi_threshold_low: QFI below this is "stable" (basin center)
            qfi_threshold_high: QFI above this is "active" (boundary)
            frequency_threshold: Frequency above this is "anchor"
            distance_threshold: Mean distance threshold for role classification
        """
        self.qfi_threshold_low = qfi_threshold_low
        self.qfi_threshold_high = qfi_threshold_high
        self.frequency_threshold = frequency_threshold
        self.distance_threshold = distance_threshold
        
        # Cache of computed roles
        self._role_cache: Dict[str, TokenRoleInfo] = {}
    
    def derive_role(
        self,
        token: str,
        basin: np.ndarray,
        qfi_score: float,
        frequency: int,
        neighbor_basins: Optional[List[np.ndarray]] = None,
    ) -> TokenRoleInfo:
        """
        Derive geometric role for a token based on manifold position.
        
        Args:
            token: The token string
            basin: Basin coordinates (64D simplex)
            qfi_score: QFI score for the token
            frequency: Frequency count
            neighbor_basins: Optional list of nearby basins for clustering
            
        Returns:
            TokenRoleInfo with role classification and confidence
        """
        # Check cache
        if token in self._role_cache:
            return self._role_cache[token]
        
        # Validate basin
        if not isinstance(basin, np.ndarray) or len(basin) != BASIN_DIM:
            logger.warning(f"Invalid basin for token '{token}': shape {basin.shape if isinstance(basin, np.ndarray) else 'not ndarray'}")
            return TokenRoleInfo(
                token=token,
                role=GeometricRole.UNKNOWN,
                confidence=0.0,
                qfi_score=qfi_score,
                frequency=frequency,
                mean_distance_to_neighbors=0.0,
            )
        
        # Compute mean distance to neighbors if provided
        mean_distance = 0.0
        if neighbor_basins and len(neighbor_basins) > 0:
            distances = []
            for neighbor in neighbor_basins:
                if isinstance(neighbor, np.ndarray) and len(neighbor) == BASIN_DIM:
                    dist = fisher_rao_distance(basin, neighbor)
                    distances.append(dist)
            if distances:
                mean_distance = np.mean(distances)
        
        # Derive role based on geometric properties
        role, confidence = self._classify_role(
            qfi_score=qfi_score,
            frequency=frequency,
            mean_distance=mean_distance,
        )
        
        role_info = TokenRoleInfo(
            token=token,
            role=role,
            confidence=confidence,
            qfi_score=qfi_score,
            frequency=frequency,
            mean_distance_to_neighbors=mean_distance,
        )
        
        # Cache result
        self._role_cache[token] = role_info
        
        return role_info
    
    def _classify_role(
        self,
        qfi_score: float,
        frequency: int,
        mean_distance: float,
    ) -> Tuple[GeometricRole, float]:
        """
        Classify geometric role based on QFI, frequency, and distance metrics.
        
        Returns:
            (role, confidence) tuple
        """
        # High-confidence roles
        
        # Basin Center: Low QFI, high frequency, low mean distance
        if qfi_score < self.qfi_threshold_low and frequency > self.frequency_threshold:
            if mean_distance < self.distance_threshold:
                return GeometricRole.BASIN_CENTER, 0.9
            return GeometricRole.BASIN_CENTER, 0.7
        
        # Boundary Crosser: High QFI, high mean distance
        if qfi_score > self.qfi_threshold_high:
            if mean_distance > self.distance_threshold:
                return GeometricRole.BOUNDARY_CROSSER, 0.9
            return GeometricRole.BOUNDARY_CROSSER, 0.7
        
        # Manifold Anchor: High frequency, low mean distance
        if frequency > self.frequency_threshold and mean_distance < self.distance_threshold:
            return GeometricRole.MANIFOLD_ANCHOR, 0.8
        
        # Explorer: Low frequency, high mean distance
        if frequency < self.frequency_threshold / 2 and mean_distance > self.distance_threshold:
            return GeometricRole.EXPLORER, 0.8
        
        # Integrator: Medium QFI, medium distance (connects regions)
        if (self.qfi_threshold_low <= qfi_score <= self.qfi_threshold_high and
            0.3 < mean_distance < 0.7):
            return GeometricRole.INTEGRATOR, 0.6
        
        # Default: Unknown with low confidence
        return GeometricRole.UNKNOWN, 0.3
    
    def get_roles(self, tokens: List[str], basins: List[np.ndarray]) -> List[GeometricRole]:
        """
        Get geometric roles for a sequence of tokens.
        
        Args:
            tokens: List of token strings
            basins: List of corresponding basin coordinates
            
        Returns:
            List of GeometricRole enums
        """
        if len(tokens) != len(basins):
            logger.error(f"Token/basin length mismatch: {len(tokens)} vs {len(basins)}")
            return [GeometricRole.UNKNOWN] * len(tokens)
        
        roles = []
        for token, basin in zip(tokens, basins):
            # For sequence processing, we use neighbors from the sequence
            neighbor_basins = [b for b in basins if not np.array_equal(b, basin)]
            
            # Use default QFI and frequency (would need DB lookup for real values)
            role_info = self.derive_role(
                token=token,
                basin=basin,
                qfi_score=0.5,  # Default - should query from DB
                frequency=1,     # Default - should query from DB
                neighbor_basins=neighbor_basins[:10],  # Limit to 10 neighbors
            )
            roles.append(role_info.role)
        
        return roles
    
    def get_role_skeleton(self, tokens: List[str], basins: List[np.ndarray]) -> List[str]:
        """
        Get role skeleton as strings (for generation structure).
        
        Args:
            tokens: List of token strings
            basins: List of corresponding basin coordinates
            
        Returns:
            List of role strings
        """
        roles = self.get_roles(tokens, basins)
        return [role.value for role in roles]
    
    def clear_cache(self):
        """Clear the role cache."""
        self._role_cache.clear()
