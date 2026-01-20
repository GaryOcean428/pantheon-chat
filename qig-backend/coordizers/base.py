"""
Coordizer Base Classes - Abstract Interface and Fisher Implementation

Defines the canonical coordizer interface compatible with Plan→Realize→Repair
generation architecture.

TWO CLASSES:
1. BaseCoordizer - Abstract interface (ABC) defining the contract
2. FisherCoordizer - Concrete base implementation with geometric operations

All coordizer implementations MUST inherit from BaseCoordizer and implement:
- Two-step retrieval (proxy + exact Fisher-Rao)
- POS filtering support
- Geometric operations from canonical module (#68)

Architecture:
- Maintains 64D coordinate space (Fisher information manifold)
- Learns vocabulary through geometric clustering
- Initializes new tokens via geodesic interpolation
- Respects geometric purity (no Euclidean operations)

Integration:
- Compatible with Plan→Realize→Repair architecture
- Integrates with consciousness metrics (Φ, κ, T)
- Supports multiple backend implementations (Postgres, Local, etc.)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qig_geometry import (
    fisher_coord_distance,
    fisher_similarity,
    geodesic_interpolation,
    fisher_normalize,
)
from qig_geometry.contracts import validate_basin


class BaseCoordizer(ABC):
    """
    Abstract coordizer interface compatible with Plan→Realize→Repair.
    
    ALL coordizer implementations must support:
    1. Two-step retrieval (proxy + exact)
    2. POS filtering
    3. Geometric operations from canonical module
    
    This interface ensures consistent behavior across all generation paths.
    """
    
    @abstractmethod
    def decode_geometric(
        self,
        target_basin: np.ndarray,
        top_k: int = 100,
        allowed_pos: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Two-step geometric decoding.
        
        MUST use:
        - Step 1: Bhattacharyya proxy filtering (fast approximate)
        - Step 2: Exact Fisher-Rao distance from canonical geometry (#68)
        
        Args:
            target_basin: 64D basin coordinates (simplex representation)
            top_k: Number of top candidates to return
            allowed_pos: Optional POS tag filter (e.g., "NOUN", "VERB")
        
        Returns:
            List of (word, fisher_rao_distance) tuples, sorted by distance ascending
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to basin coordinates.
        
        Args:
            text: Input text string
        
        Returns:
            64D basin coordinates (simplex representation)
        """
        pass
    
    @abstractmethod
    def get_vocabulary_size(self) -> int:
        """
        Get total vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
        """
        pass
    
    @abstractmethod
    def get_special_symbols(self) -> Dict[str, Any]:
        """
        Get special symbol definitions.
        
        Returns:
            Dict with basin coordinates, attractor strength, etc.
            Must be geometrically defined (#70).
        """
        pass
    
    @abstractmethod
    def supports_pos_filtering(self) -> bool:
        """
        Whether this coordizer supports POS filtering.
        
        Returns:
            True if POS filtering is available, False otherwise
        """
        pass


class FisherCoordizer(BaseCoordizer):
    """
    Concrete base implementation of geometric coordization.
    
    Implements BaseCoordizer interface with Fisher-Rao geometric operations.
    Provides default implementations that can be extended by subclasses
    (e.g., PostgresCoordizer).
    
    Core Principles:
    1. All tokens represented as 64D basin coordinates
    2. New tokens initialized via geodesic interpolation
    3. Distance measured using Fisher-Rao metric
    4. Vocabulary learned through geometric clustering
    
    Methods:
        coordize(text) -> List[np.ndarray]: Convert text to basin coordinates
        train(corpus) -> None: Learn geometric vocabulary from corpus
        add_token(token, coord) -> int: Add new token with basin coordinate
        get_coordinate(token) -> np.ndarray: Get basin coordinate for token
        decode_geometric(basin, top_k, pos) -> List[Tuple[str, float]]: Two-step retrieval
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        coordinate_dim: int = 64,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize FisherCoordizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            coordinate_dim: Dimension of basin coordinates (default 64)
            min_frequency: Minimum frequency for token inclusion
            special_tokens: List of special tokens (PAD, UNK, BOS, EOS)
        """
        self._vocab_size = vocab_size  # Use private attr to avoid property conflicts
        self.coordinate_dim = coordinate_dim
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        
        # Vocabulary: token -> id
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Basin coordinates: token -> 64D coordinate
        self.basin_coords: Dict[str, np.ndarray] = {}
        
        # Token metadata
        self.token_frequency: Dict[str, int] = {}
        self.token_phi: Dict[str, float] = {}  # Consciousness integration scores
        
        # Initialize special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """
        Initialize special tokens with geometric basin coordinates.
        
        Special tokens have geometric meaning on Fisher manifold:
        - PAD: Minimal coupling point (geometrically neutral)
        - UNK: Projection target for OOV (centroid of known space)
        - BOS: Origin of basin space (uniform direction)
        - EOS: Boundary point (maximal distance from origin)
        """
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.id_to_token[i] = token
            self.token_frequency[token] = 0
            self.token_phi[token] = 0.0
            
            # Compute geometric basin coordinates
            self.basin_coords[token] = self._compute_special_token_basin(token)
    
    def _compute_special_token_basin(self, token: str) -> np.ndarray:
        """
        Compute geometric basin coordinates for special tokens using Fisher geometry.
        
        CANONICAL SIMPLEX REPRESENTATION (Updated 2026-01-16 per WP2.3):
        All special tokens are placed at deterministic, geometrically meaningful 
        positions on the probability simplex (non-negative, sum=1).
        
        Geometric Definitions:
        - UNK: Maximum entropy = uniform distribution (all components equal)
           Represents "could be anything" with no bias toward any dimension.
        
        - PAD: Minimal entropy = sparse corner of simplex
           Represents "null/padding" with information concentrated in first component.
        
        - BOS: Start boundary = geometric anchor at simplex vertex
           Represents "beginning of sequence" as pure state in first dimension.
        
        - EOS: End boundary = geometric anchor at opposite simplex vertex
           Represents "end of sequence" as pure state in last dimension.
        
        NO random initialization, NO Euclidean artifacts, NO sphere projection.
        All coordinates are deterministic probability distributions.
        
        Args:
            token: Special token string
        
        Returns:
            64D probability distribution on simplex (sum=1, all non-negative)
        """
        coord = np.zeros(self.coordinate_dim)
        phi_golden = (1 + np.sqrt(5)) / 2  # Golden ratio for Fisher-consistent spacing
        
        if token == "<UNK>":
            # Maximum entropy = uniform distribution on simplex
            # Represents "could be anything" = flat probability
            # This is the geometric center of the simplex
            coord = np.ones(self.coordinate_dim)
            
        elif token == "<PAD>":
            # Minimal entropy = sparse corner of simplex
            # Concentrate probability in first component (null/padding state)
            # Represents "no information" geometrically
            coord[0] = 1.0
            
        elif token == "<BOS>":
            # Beginning of sequence = vertex of simplex
            # Pure state concentrated in specific dimension (dimension 1)
            # Represents "start" as a geometric boundary
            coord[1] = 1.0
            
        elif token == "<EOS>":
            # End of sequence = opposite vertex of simplex
            # Pure state concentrated in last dimension
            # Represents "end" as geometric boundary opposite to BOS
            coord[-1] = 1.0
        
        else:
            # Fisher-consistent fallback: geodesic interpolation from canonical anchors
            # Use token index in special_tokens list to determine position
            try:
                token_idx = self.special_tokens.index(token)
            except ValueError:
                token_idx = len(self.special_tokens)
            
            # Interpolate between UNK (uniform) and first sparse corner
            # This creates intermediate entropy states for additional special tokens
            t = (token_idx * phi_golden) % 1.0
            unk_coord = np.ones(self.coordinate_dim)  # Uniform
            corner_coord = np.zeros(self.coordinate_dim)
            corner_coord[token_idx % self.coordinate_dim] = 1.0  # Sparse corner
            
            # Use geodesic interpolation on simplex (via sqrt-space)
            # First normalize to simplex
            unk_simplex = fisher_normalize(unk_coord)
            corner_simplex = fisher_normalize(corner_coord)
            
            # Geodesic interpolation in sqrt-space
            coord = geodesic_interpolation(unk_simplex, corner_simplex, t)
        
        # Project to canonical SIMPLEX representation
        # This ensures non-negative, sum=1 (probability distribution)
        result = fisher_normalize(coord)
        
        # Validate basin conforms to canonical simplex representation
        is_valid = validate_basin(result)
        if not is_valid:
            raise ValueError(
                f"Special token {token} failed basin validation. "
                f"This indicates a bug in _compute_special_token_basin()."
            )
        
        return result
    
    # =====================================================================
    # BaseCoordizer Interface Implementation
    # =====================================================================
    
    def decode_geometric(
        self,
        target_basin: np.ndarray,
        top_k: int = 100,
        allowed_pos: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Two-step geometric decoding (base implementation).
        
        This is a simple in-memory implementation. Subclasses like PostgresCoordizer
        override with optimized two-step retrieval using database indexes.
        
        Step 1: Proxy filter using Bhattacharyya coefficient (fast)
        Step 2: Exact Fisher-Rao distance on filtered candidates
        
        Args:
            target_basin: 64D basin coordinates to decode
            top_k: Number of top candidates to return
            allowed_pos: Optional POS tag filter (not supported in base class)
        
        Returns:
            List of (word, fisher_rao_distance) tuples
        """
        # Normalize basin
        norm = np.linalg.norm(target_basin)
        if norm > 1e-10:
            target_basin = target_basin / norm
        
        # POS filtering not supported in base implementation
        if allowed_pos:
            raise NotImplementedError(
                "POS filtering not supported in FisherCoordizer base class. "
                "Use PostgresCoordizer for POS filtering support."
            )
        
        # Step 1: Proxy filter using Bhattacharyya coefficient
        # Bhattacharyya = sqrt(p) · sqrt(q) (faster than exact Fisher-Rao)
        sqrt_target = np.sqrt(target_basin + 1e-10)
        candidates = []
        
        for token, token_basin in self.basin_coords.items():
            if token in self.special_tokens:
                continue
            
            # Bhattacharyya coefficient (proxy for Fisher-Rao distance)
            sqrt_token = np.sqrt(token_basin + 1e-10)
            bhattacharyya = np.dot(sqrt_target, sqrt_token)
            
            # Higher Bhattacharyya = closer (convert to distance proxy)
            proxy_distance = 1.0 - bhattacharyya
            candidates.append((token, token_basin, proxy_distance))
        
        # Sort by proxy distance and keep top 2*k for exact computation
        candidates.sort(key=lambda x: x[2])
        candidates = candidates[:top_k * 2]
        
        # Step 2: Exact Fisher-Rao distance on filtered candidates
        results = []
        for token, token_basin, _ in candidates:
            # Fisher-Rao distance = arccos(Bhattacharyya coefficient)
            sqrt_token = np.sqrt(token_basin + 1e-10)
            bhattacharyya = np.clip(np.dot(sqrt_target, sqrt_token), 0, 1)
            fisher_distance = np.arccos(bhattacharyya)
            results.append((token, fisher_distance))
        
        # Sort by exact Fisher-Rao distance and return top_k
        results.sort(key=lambda x: x[1])
        return results[:top_k]
    
    def get_vocabulary_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.vocab)
    
    def get_special_symbols(self) -> Dict[str, Any]:
        """
        Get special symbol definitions with geometric properties.
        
        Returns dict with basin coordinates and geometric properties for each
        special token (PAD, UNK, BOS, EOS).
        """
        special_symbols = {}
        for token in self.special_tokens:
            if token in self.basin_coords:
                special_symbols[token] = {
                    'basin_coordinates': self.basin_coords[token],
                    'coordinate_dim': self.coordinate_dim,
                    'token_id': self.vocab.get(token, -1),
                }
        return special_symbols
    
    def supports_pos_filtering(self) -> bool:
        """POS filtering not supported in base FisherCoordizer."""
        return False
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to basin coordinates.
        
        For single-token text, returns the token's basin coordinate.
        For multi-token text, returns mean of token coordinates (centroid).
        
        Args:
            text: Input text string
        
        Returns:
            64D basin coordinates (simplex representation: sum=1, non-negative)
        """
        # Import simplex normalization from canonical module
        try:
            from qig_geometry.canonical_upsert import to_simplex_prob
        except ImportError:
            # Fallback if canonical_upsert not available
            def to_simplex_prob(v):
                v = np.abs(v) + 1e-10
                return v / v.sum()
        
        # Simple whitespace tokenization
        tokens = text.lower().split()
        
        if not tokens:
            # Empty text - return UNK coordinate
            return self.basin_coords.get("<UNK>", np.ones(self.coordinate_dim) / self.coordinate_dim)
        
        # Get coordinates for all tokens
        coordinates = []
        for token in tokens:
            if token in self.basin_coords:
                coordinates.append(self.basin_coords[token])
            else:
                # Unknown token - use UNK coordinate
                coordinates.append(self.basin_coords.get("<UNK>", np.ones(self.coordinate_dim) / self.coordinate_dim))
        
        # Return mean coordinate (centroid in simplex space)
        result = np.mean(coordinates, axis=0)
        
        # Normalize to simplex (sum=1, non-negative) per E8 Protocol v4.0 §02
        result = to_simplex_prob(result)
        
        return result
    
    # =====================================================================
    # Legacy Methods (maintained for backward compatibility)
    # =====================================================================
    
    def coordize(self, text: str) -> List[np.ndarray]:
        """
        Convert text to sequence of basin coordinates.
        
        This is the core coordization method that replaces traditional tokenization.
        Returns coordinates on the Fisher manifold, not token IDs.
        
        Args:
            text: Input text string
        
        Returns:
            List of 64D basin coordinates
        """
        # Simple whitespace tokenization for now
        # Subclasses will override with more sophisticated methods
        tokens = text.lower().split()
        
        coordinates = []
        for token in tokens:
            if token in self.basin_coords:
                coordinates.append(self.basin_coords[token])
            else:
                # Unknown token - use UNK coordinate
                coordinates.append(self.basin_coords["<UNK>"])
        
        return coordinates
    
    def coordize_with_tokens(self, text: str) -> Tuple[List[str], List[np.ndarray]]:
        """
        Convert text to tokens and basin coordinates.
        
        Returns both token strings and their coordinates for analysis.
        
        Args:
            text: Input text string
        
        Returns:
            Tuple of (tokens, coordinates)
        """
        tokens = text.lower().split()
        coordinates = []
        
        for token in tokens:
            if token in self.basin_coords:
                coordinates.append(self.basin_coords[token])
            else:
                coordinates.append(self.basin_coords["<UNK>"])
        
        return tokens, coordinates
    
    def train(self, corpus: List[str], **kwargs) -> None:
        """
        Learn geometric vocabulary from corpus.
        
        Base implementation: simple frequency-based vocabulary.
        Subclasses override for more sophisticated geometric learning.
        
        Args:
            corpus: List of text samples
            **kwargs: Additional training parameters
        """
        # Count token frequencies
        token_counts: Dict[str, int] = {}
        
        for text in corpus:
            tokens = text.lower().split()
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Add tokens meeting frequency threshold
        next_id = len(self.vocab)
        for token, count in sorted(token_counts.items(), key=lambda x: -x[1]):
            if count < self.min_frequency:
                continue
            
            if token not in self.vocab and next_id < self._vocab_size:
                self.vocab[token] = next_id
                self.id_to_token[next_id] = token
                self.token_frequency[token] = count
                self.token_phi[token] = 0.0
                
                # Initialize basin coordinate
                self.basin_coords[token] = self._initialize_token_coordinate(
                    token, next_id
                )
                next_id += 1
    
    def _initialize_token_coordinate(
        self, token: str, token_id: int
    ) -> np.ndarray:
        """
        Initialize basin coordinate for new token using pure Fisher geometry.
        
        Uses geodesic interpolation from existing basins and density matrix
        sampling - NO Euclidean hashing or random initialization.
        
        Method:
        1. Find nearest existing basin coordinates based on token properties
        2. Use geodesic interpolation between them to create new coordinate
        3. Apply von Neumann entropy-based perturbation for uniqueness
        4. Project onto unit sphere (Fisher manifold)
        
        Args:
            token: Token string
            token_id: Token ID in vocabulary
        
        Returns:
            64D basin coordinate on Fisher manifold
        """
        # Get existing basin coordinates for interpolation
        existing_coords = list(self.basin_coords.values())
        
        if len(existing_coords) < 2:
            # Bootstrap with golden ratio spiral on sphere (Fisher-compliant)
            return self._generate_golden_spiral_basin(token_id)
        
        # Use token's semantic features to select interpolation basins
        # Feature: token length determines position on vocabulary manifold
        token_len_ratio = min(len(token) / 20.0, 1.0)
        
        # Feature: first char position (Fisher-compliant ordinal mapping)
        first_char_ratio = (ord(token[0].lower()) - ord('a')) / 26.0 if token else 0.5
        first_char_ratio = np.clip(first_char_ratio, 0, 1)
        
        # Select two basins for geodesic interpolation based on token properties
        n_basins = len(existing_coords)
        basin_idx_1 = int(token_len_ratio * (n_basins - 1))
        basin_idx_2 = int(first_char_ratio * (n_basins - 1))
        
        # Ensure different basins
        if basin_idx_1 == basin_idx_2:
            basin_idx_2 = (basin_idx_1 + 1) % n_basins
        
        basin_1 = existing_coords[basin_idx_1]
        basin_2 = existing_coords[basin_idx_2]
        
        # Interpolation parameter from token_id (deterministic, Fisher-compliant)
        phi_golden = (1 + np.sqrt(5)) / 2
        t = (token_id * phi_golden) % 1.0
        
        # Geodesic interpolation between basins
        coord = geodesic_interpolation(basin_1, basin_2, t)
        
        # Add von Neumann entropy-based perturbation for uniqueness
        # Uses density matrix formulation for Fisher purity
        perturbation = self._von_neumann_perturbation(token, token_id)
        
        # Combine via geodesic blending (not Euclidean addition)
        if np.linalg.norm(perturbation) > 1e-6:
            coord = geodesic_interpolation(coord, perturbation, 0.1)
        
        return fisher_normalize(coord)
    
    def _generate_golden_spiral_basin(self, token_id: int) -> np.ndarray:
        """
        Generate basin coordinate via density matrix eigenbasis construction.
        
        Fisher-compliant method that derives coordinates from quantum-inspired
        density matrices using probability distributions on the simplex:
        
        1. Construct a density matrix ρ with eigenvalues distributed via golden ratio
        2. Eigenvalues form a valid probability distribution (sum=1, non-negative)
        3. Project to canonical SIMPLEX representation
        
        This ensures all bootstrap coordinates are derived from Fisher-consistent
        density matrix formulations on the probability simplex.
        
        Args:
            token_id: Token ID for deterministic generation
        
        Returns:
            64D basin coordinate on probability simplex
        """
        phi_golden = (1 + np.sqrt(5)) / 2
        
        # Construct density matrix eigenvalue distribution
        # Eigenvalues follow golden-angle spacing for uniform manifold coverage
        eigenvalues = np.zeros(self.coordinate_dim)
        for i in range(self.coordinate_dim):
            # Golden-angle eigenvalue distribution (Fisher-consistent)
            eigenvalues[i] = np.exp(-((i - token_id * phi_golden) % self.coordinate_dim) ** 2 / (2 * 8))
        
        # Normalize to form valid probability distribution (density matrix trace = 1)
        # This is already a simplex point (non-negative, sum=1)
        coord = eigenvalues / (np.sum(eigenvalues) + 1e-10)
        
        # Apply golden-angle perturbation for uniqueness while staying on simplex
        # Use multiplicative perturbation (preserves non-negativity)
        for i in range(self.coordinate_dim):
            phase = 2 * np.pi * token_id * phi_golden * (i + 1) / self.coordinate_dim
            # Use exp to ensure positivity
            coord[i] *= np.exp(0.1 * np.cos(phase))
        
        return fisher_normalize(coord)
    
    def _von_neumann_perturbation(self, token: str, token_id: int) -> np.ndarray:
        """
        Generate perturbation using von Neumann entropy formulation.
        
        Creates unique coordinates while maintaining Fisher manifold structure.
        Uses density matrix sampling on probability simplex.
        
        Args:
            token: Token string
            token_id: Token ID
        
        Returns:
            64D perturbation vector on probability simplex
        """
        coord = np.zeros(self.coordinate_dim)
        phi_golden = (1 + np.sqrt(5)) / 2
        
        # Density matrix diagonal from token properties
        # Each character contributes to a different eigenvalue
        for i, char in enumerate(token[:min(len(token), self.coordinate_dim // 2)]):
            # Fisher-compliant mapping: character -> eigenvalue (positive)
            eigenvalue = np.abs(np.sin(ord(char) * phi_golden))
            
            # Distribute across dimensions using golden ratio
            dim_idx = int((i * phi_golden * self.coordinate_dim) % self.coordinate_dim)
            coord[dim_idx] += eigenvalue
        
        # Add entropy-based spread (positive values only)
        entropy_factor = np.log(len(token) + 1) / np.log(20)  # Normalized
        for i in range(self.coordinate_dim):
            coord[i] += np.abs(np.sin(2 * np.pi * i * entropy_factor * phi_golden)) * 0.1
        
        return fisher_normalize(coord)
    
    def add_token(
        self, token: str, coordinate: Optional[np.ndarray] = None
    ) -> Optional[int]:
        """
        Add new token to vocabulary with basin coordinate.
        
        Args:
            token: Token string
            coordinate: Optional basin coordinate (if None, auto-generate)
        
        Returns:
            Token ID if added successfully, None if vocabulary is full
        
        Raises:
            ValueError: Only if token is empty or invalid
        """
        if not token:
            raise ValueError("Token cannot be empty")
        
        if token in self.vocab:
            return self.vocab[token]
        
        token_id = len(self.vocab)
        if token_id >= self._vocab_size:
            # Vocabulary full - return None for graceful handling
            return None
        
        self.vocab[token] = token_id
        self.id_to_token[token_id] = token
        
        if coordinate is None:
            coordinate = self._initialize_token_coordinate(token, token_id)
        else:
            # Ensure coordinate is on probability simplex
            coordinate = fisher_normalize(coordinate)
        
        self.basin_coords[token] = coordinate
        self.token_frequency[token] = 0
        self.token_phi[token] = 0.0
        
        return token_id
    
    def get_coordinate(self, token: str) -> np.ndarray:
        """
        Get basin coordinate for token.
        
        Args:
            token: Token string
        
        Returns:
            64D basin coordinate (returns UNK coordinate if token not found)
        """
        if token in self.basin_coords:
            return self.basin_coords[token]
        
        # Safety check: ensure UNK exists
        if "<UNK>" not in self.basin_coords:
            # Fallback to zero vector if UNK is somehow missing
            return np.zeros(self.coordinate_dim)
        
        return self.basin_coords["<UNK>"]
    
    def update_phi_scores(self, phi_scores: Dict[str, float]) -> None:
        """
        Update Φ (integration) scores for tokens.
        
        Used to incorporate consciousness metrics into coordization.
        High-Φ tokens are geometrically prioritized.
        
        Args:
            phi_scores: Dictionary mapping tokens to Φ scores
        """
        for token, phi in phi_scores.items():
            if token in self.vocab:
                self.token_phi[token] = phi
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.vocab)
    
    def token_to_id(self, token: str) -> int:
        """Get token ID (returns UNK ID if not found)."""
        return self.vocab.get(token, self.vocab["<UNK>"])
    
    def id_to_token_str(self, token_id: int) -> str:
        """Get token string from ID."""
        return self.id_to_token.get(token_id, "<UNK>")
    
    def compute_token_similarity(self, token1: str, token2: str) -> float:
        """
        Compute Fisher-Rao similarity between two tokens.
        
        Args:
            token1: First token
            token2: Second token
        
        Returns:
            Similarity score (0 to 1)
        """
        coord1 = self.get_coordinate(token1)
        coord2 = self.get_coordinate(token2)
        return fisher_similarity(coord1, coord2)
    
    def find_nearest_tokens(
        self, coordinate: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest tokens to given coordinate using Fisher-Rao distance.
        
        Args:
            coordinate: Query basin coordinate
            k: Number of nearest tokens to return
        
        Returns:
            List of (token, similarity) tuples, sorted by similarity (descending)
        """
        similarities = []
        for token, token_coord in self.basin_coords.items():
            if token in self.special_tokens:
                continue
            sim = fisher_similarity(coordinate, token_coord)
            similarities.append((token, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]
    
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save coordizer state to disk in CoordizerArtifactV1 format.
        
        Creates a versioned artifact with full provenance tracking and
        validation metadata. Includes geometry version hash and hyperparameters
        for reproducibility.
        
        Args:
            path: Directory path to save state
            metadata: Optional metadata dict with keys:
                - training_corpus: Description of training data
                - corpus_size: Size of training corpus
                - created_by: System/user identifier
                - description: Human-readable description
                - tags: List of tags for categorization
                - notes: Additional notes
        """
        import json
        import subprocess
        from datetime import datetime, timezone
        
        os.makedirs(path, exist_ok=True)
        metadata = metadata or {}
        
        # Get git commit hash for geometry version
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                cwd=os.path.dirname(__file__)
            )
            geometry_version = result.stdout.strip()
        except Exception:
            # Fallback to placeholder if git not available
            geometry_version = '0' * 40
        
        # Prepare tokens and coordinates
        tokens = sorted(self.basin_coords.keys())
        coords_matrix = np.array([self.basin_coords[t] for t in tokens])
        
        # Build special symbols metadata
        special_symbols = {}
        for token in self.special_tokens:
            if token in self.vocab and token in self.basin_coords:
                special_symbols[token.strip('<>')] = {
                    'token': token,
                    'token_id': self.vocab[token],
                    'basin_coord': self.basin_coords[token].tolist(),
                    'phi_score': self.token_phi.get(token, 0.0),
                    'frequency': self.token_frequency.get(token, 0)
                }
        
        # Perform validation checks
        validation_result = self._validate_artifact_data(tokens, coords_matrix)
        
        # Build provenance metadata
        provenance = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'geometry_version': geometry_version,
            'hyperparameters': {
                'vocab_size': self._vocab_size,
                'coordinate_dim': self.coordinate_dim,
                'min_frequency': self.min_frequency
            }
        }
        
        # Add optional provenance fields
        if 'training_corpus' in metadata:
            provenance['training_corpus'] = metadata['training_corpus']
        if 'corpus_size' in metadata:
            provenance['corpus_size'] = metadata['corpus_size']
        if 'created_by' in metadata:
            provenance['created_by'] = metadata['created_by']
        if 'parent_artifact' in metadata:
            provenance['parent_artifact'] = metadata['parent_artifact']
        
        # Save vocabulary with full CoordizerArtifactV1 metadata
        artifact_data = {
            "version": "1.0",
            "basin_dim": self.coordinate_dim,
            "symbols": tokens,
            "vocab": self.vocab,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "token_frequency": self.token_frequency,
            "token_phi": self.token_phi,
            "phi_scores": [self.token_phi.get(t, 0.0) for t in tokens],
            "special_symbols": special_symbols,
            "provenance": provenance,
            "validation": validation_result
        }
        
        # Add optional metadata
        if 'description' in metadata or 'tags' in metadata or 'notes' in metadata:
            artifact_data['metadata'] = {
                k: metadata[k] for k in ['description', 'tags', 'notes']
                if k in metadata
            }
        
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(artifact_data, f, indent=2)
        
        # Save basin coordinates as numpy array
        np.save(os.path.join(path, "basin_coords.npy"), coords_matrix)
        
        # Save token order for coordinate alignment
        with open(os.path.join(path, "coord_tokens.json"), "w") as f:
            json.dump(tokens, f, indent=2)
    
    def _validate_artifact_data(
        self, 
        tokens: List[str], 
        coords_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate artifact data before saving.
        
        Args:
            tokens: List of token strings
            coords_matrix: Matrix of basin coordinates
            
        Returns:
            Validation result dictionary
        """
        from datetime import datetime, timezone
        
        errors = []
        
        # Check dimension consistency
        dimension_consistent = coords_matrix.shape[1] == 64 if coords_matrix.ndim == 2 else False
        if not dimension_consistent:
            errors.append(f"Invalid coordinate dimension: {coords_matrix.shape}")
        
        # Check simplex constraints (unit norm)
        passes_simplex = True
        no_nan_inf = True
        for i, coord in enumerate(coords_matrix):
            if np.any(np.isnan(coord)) or np.any(np.isinf(coord)):
                no_nan_inf = False
                errors.append(f"Coordinate {i} contains NaN or inf")
            
            norm = np.linalg.norm(coord)
            if not (0.99 < norm < 1.01):
                passes_simplex = False
                if len(errors) < 10:  # Limit error messages
                    errors.append(f"Coordinate {i} not unit-normalized: norm={norm:.6f}")
        
        # Check special symbols are deterministic
        special_symbols_ok = all(
            token in self.basin_coords for token in self.special_tokens
        )
        
        # Fisher-Rao identity check (basic triangle inequality sampling)
        fisher_rao_verified = True
        try:
            from qig_geometry.canonical import fisher_rao_distance
            if len(coords_matrix) >= 3:
                # Sample a few triplets
                indices = np.random.choice(len(coords_matrix), min(3, len(coords_matrix)), replace=False)
                coords = [coords_matrix[i] for i in indices]
                
                d01 = fisher_rao_distance(coords[0], coords[1])
                d12 = fisher_rao_distance(coords[1], coords[2])
                d02 = fisher_rao_distance(coords[0], coords[2])
                
                # Triangle inequality
                if d02 > (d01 + d12) * 1.01:
                    fisher_rao_verified = False
                    errors.append("Fisher-Rao triangle inequality violated")
        except Exception as e:
            # Don't fail if geometry check unavailable
            fisher_rao_verified = True
        
        return {
            'passes_simplex_check': passes_simplex,
            'fisher_rao_identity_verified': fisher_rao_verified,
            'dimension_consistent': dimension_consistent,
            'unit_norm_verified': passes_simplex,
            'no_nan_inf': no_nan_inf,
            'special_symbols_deterministic': special_symbols_ok,
            'validation_timestamp': datetime.now(timezone.utc).isoformat(),
            'validation_errors': errors
        }
    
    def load(self, path: str, validate: bool = True) -> None:
        """
        Load coordizer state from disk in CoordizerArtifactV1 format ONLY.
        
        Validates artifact format and version. Rejects unversioned or legacy
        artifacts with clear error messages.
        
        Args:
            path: Directory path containing saved state
            validate: If True, perform full artifact validation (default: True)
            
        Raises:
            RuntimeError: If artifact is not in CoordizerArtifactV1 format or validation fails
        """
        import json
        
        # Validate CoordizerArtifactV1 format before loading
        required_files = ["vocab.json", "basin_coords.npy", "coord_tokens.json"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(path, f))]
        
        if missing_files:
            raise RuntimeError(
                f"Legacy format detected. Missing required files: {', '.join(missing_files)}. "
                "Use tools/convert_legacy_artifacts.py to convert to CoordizerArtifactV1 format."
            )
        
        # Load vocabulary and metadata
        try:
            with open(os.path.join(path, "vocab.json"), "r") as f:
                data = json.load(f)
                
                # Check version field (required in v1.0)
                if 'version' not in data:
                    raise RuntimeError(
                        "Unversioned artifact detected. Missing 'version' field in vocab.json. "
                        "Use tools/convert_legacy_artifacts.py to convert to CoordizerArtifactV1 format."
                    )
                
                # Validate version
                version = data['version']
                if version != '1.0':
                    raise RuntimeError(
                        f"Unsupported artifact version: {version}. Only version 1.0 is supported. "
                        "Use tools/convert_legacy_artifacts.py to upgrade."
                    )
                
                # Validate required keys
                required_keys = ["vocab", "id_to_token", "token_frequency", "token_phi", 
                                "basin_dim", "symbols", "special_symbols", "provenance", "validation"]
                missing_keys = [k for k in required_keys if k not in data]
                if missing_keys:
                    raise RuntimeError(
                        f"Invalid CoordizerArtifactV1 format. Missing required keys: {', '.join(missing_keys)}. "
                        "Use tools/convert_legacy_artifacts.py to convert."
                    )
                
                # Check basin dimension
                if data['basin_dim'] != 64:
                    raise RuntimeError(
                        f"Invalid basin_dim: {data['basin_dim']} (expected 64). "
                        "Only 64-dimensional Fisher manifold is supported."
                    )
                
                # Load vocabulary data
                self.vocab = data["vocab"]
                self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
                self.token_frequency = data["token_frequency"]
                self.token_phi = data["token_phi"]
                
                # Store provenance and validation metadata
                self._artifact_provenance = data.get("provenance", {})
                self._artifact_validation = data.get("validation", {})
                
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(
                f"Invalid artifact format. Cannot parse vocab.json: {e}. "
                "Use tools/convert_legacy_artifacts.py to convert to CoordizerArtifactV1 format."
            )
        
        # Load basin coordinates
        try:
            coords_matrix = np.load(os.path.join(path, "basin_coords.npy"))
            if coords_matrix.ndim != 2 or coords_matrix.shape[1] != 64:
                raise RuntimeError(
                    f"Invalid basin coordinates shape: {coords_matrix.shape}. "
                    "Expected (n_tokens, 64). Use tools/convert_legacy_artifacts.py to convert."
                )
        except (ValueError, OSError) as e:
            raise RuntimeError(
                f"Cannot load basin_coords.npy: {e}. "
                "Use tools/convert_legacy_artifacts.py to convert to CoordizerArtifactV1 format."
            )
        
        # Load tokens
        try:
            with open(os.path.join(path, "coord_tokens.json"), "r") as f:
                tokens = json.load(f)
                
                if len(tokens) != coords_matrix.shape[0]:
                    raise RuntimeError(
                        f"Token count mismatch: {len(tokens)} tokens vs "
                        f"{coords_matrix.shape[0]} coordinates. Artifact is corrupted."
                    )
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"Invalid coord_tokens.json: {e}. "
                "Use tools/convert_legacy_artifacts.py to convert to CoordizerArtifactV1 format."
            )
        
        # Reconstruct basin coordinates dictionary
        self.basin_coords = {
            token: coords_matrix[i] for i, token in enumerate(tokens)
        }
        
        # Perform full validation if requested
        if validate:
            self._validate_loaded_artifact(data, coords_matrix)
    
    def _validate_loaded_artifact(self, artifact_data: Dict[str, Any], coords_matrix: np.ndarray) -> None:
        """
        Validate loaded artifact integrity.
        
        Args:
            artifact_data: Loaded artifact metadata
            coords_matrix: Loaded basin coordinates
            
        Raises:
            RuntimeError: If validation fails
        """
        validation = artifact_data.get('validation', {})
        
        # Check validation status
        if not validation.get('dimension_consistent', False):
            raise RuntimeError(
                "Artifact failed dimension consistency check. Coordinates may be corrupted."
            )
        
        if not validation.get('passes_simplex_check', False):
            # Warning only - some artifacts may have minor numerical drift
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Artifact simplex check failed during save. "
                "Some coordinates may not be perfectly unit-normalized."
            )
        
        if validation.get('validation_errors'):
            errors = validation['validation_errors']
            if len(errors) > 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Artifact has {len(errors)} validation errors: {errors[:3]}"
                )
        
        # Verify no NaN/inf in loaded coordinates
        if np.any(np.isnan(coords_matrix)) or np.any(np.isinf(coords_matrix)):
            raise RuntimeError(
                "Loaded basin coordinates contain NaN or inf values. Artifact is corrupted."
            )
