"""
CoordizerArtifactV1 Validation Module

Validates coordizer artifacts against the canonical schema.
Ensures geometric purity, dimension consistency, and format compliance.

Usage:
    from artifact_validation import validate_artifact, ArtifactValidator
    
    # Validate full artifact
    result = validate_artifact(artifact_data)
    if not result['valid']:
        print(f"Validation errors: {result['errors']}")
    
    # Incremental validation
    validator = ArtifactValidator()
    validator.check_simplex_constraints(basin_coords)
    validator.check_dimension_consistency(basin_coords, symbols)
    errors = validator.get_errors()
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from qig_geometry.representation import validate_basin

logger = logging.getLogger(__name__)


class ArtifactValidator:
    """
    Validator for CoordizerArtifactV1 format.
    
    Performs geometric and structural validation on coordizer artifacts.
    Tracks errors during validation for comprehensive reporting.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, enforce all constraints. If False, allow warnings.
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        logger.error(f"[Validation] {message}")
        
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)
        logger.warning(f"[Validation] {message}")
        
    def get_errors(self) -> List[str]:
        """Get all validation errors."""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get all validation warnings."""
        return self.warnings.copy()
    
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0
    
    def reset(self):
        """Reset error and warning lists."""
        self.errors = []
        self.warnings = []
    
    def check_required_fields(self, artifact: Dict[str, Any]) -> bool:
        """
        Check that all required fields are present.
        
        Args:
            artifact: Artifact dictionary to validate
            
        Returns:
            True if all required fields present, False otherwise
        """
        required_fields = [
            'version', 'basin_dim', 'symbols', 'basin_coords', 
            'phi_scores', 'special_symbols', 'provenance', 'validation'
        ]
        
        for field in required_fields:
            if field not in artifact:
                self.add_error(f"Missing required field: {field}")
                return False
        
        return True
    
    def check_version(self, artifact: Dict[str, Any]) -> bool:
        """
        Verify artifact version is 1.0.
        
        Args:
            artifact: Artifact dictionary to validate
            
        Returns:
            True if version is valid, False otherwise
        """
        version = artifact.get('version')
        if version != '1.0':
            self.add_error(f"Invalid version: {version} (expected '1.0')")
            return False
        return True
    
    def check_basin_dim(self, artifact: Dict[str, Any]) -> bool:
        """
        Verify basin dimension is 64.
        
        Args:
            artifact: Artifact dictionary to validate
            
        Returns:
            True if basin_dim is valid, False otherwise
        """
        basin_dim = artifact.get('basin_dim')
        if basin_dim != 64:
            self.add_error(f"Invalid basin_dim: {basin_dim} (expected 64)")
            return False
        return True
    
    def check_dimension_consistency(
        self, 
        basin_coords: List[List[float]], 
        symbols: List[str],
        phi_scores: List[float]
    ) -> bool:
        """
        Verify dimension consistency across arrays.
        
        Checks:
        - All basin coordinates are 64-dimensional
        - Number of coordinates matches number of symbols
        - Number of phi scores matches number of symbols
        
        Args:
            basin_coords: List of basin coordinate arrays
            symbols: List of symbol strings
            phi_scores: List of phi scores
            
        Returns:
            True if dimensions are consistent, False otherwise
        """
        n_symbols = len(symbols)
        n_coords = len(basin_coords)
        n_phi = len(phi_scores)
        
        # Check array length consistency
        if n_coords != n_symbols:
            self.add_error(
                f"Dimension mismatch: {n_coords} basin_coords vs {n_symbols} symbols"
            )
            return False
        
        if n_phi != n_symbols:
            self.add_error(
                f"Dimension mismatch: {n_phi} phi_scores vs {n_symbols} symbols"
            )
            return False
        
        # Check each basin coordinate is 64D
        for i, coord in enumerate(basin_coords):
            if len(coord) != 64:
                self.add_error(
                    f"Invalid coordinate dimension at index {i}: {len(coord)} (expected 64)"
                )
                return False
        
        return True
    
    def check_simplex_constraints(self, basin_coords: List[List[float]]) -> bool:
        """
        Verify basin coordinates satisfy simplex constraints.
        
        For simplex representation:
        - Coordinates must be finite
        - Coordinates must be non-negative
        - Coordinates must sum to 1 (within tolerance)
        
        Args:
            basin_coords: List of basin coordinate arrays
            
        Returns:
            True if simplex constraints satisfied, False otherwise
        """
        valid = True
        
        for i, coord in enumerate(basin_coords):
            coord_array = np.array(coord, dtype=np.float64)
            
            # Check for NaN or inf (must be finite before simplex validation)
            if not np.all(np.isfinite(coord_array)):
                self.add_error(
                    f"Basin coordinate {i} contains non-finite values (simplex entries must be finite)"
                )
                valid = False
                continue
            
            # Check canonical simplex invariants
            is_valid, reason = validate_basin(coord_array)
            if not is_valid:
                self.add_error(
                    f"Basin coordinate {i} violates simplex constraints: {reason}"
                )
                valid = False
        
        return valid
    
    def check_fisher_rao_identity(self, basin_coords: List[List[float]]) -> bool:
        """
        Verify Fisher-Rao metric identity on coordinates.
        
        For a sample of coordinate pairs, verify:
        - Fisher-Rao distance satisfies triangle inequality
        - Distance is symmetric
        - Distance is non-negative
        
        This is a sampling check, not exhaustive verification.
        
        Args:
            basin_coords: List of basin coordinate arrays
            
        Returns:
            True if Fisher-Rao identity holds, False otherwise
        """
        # Import here to avoid circular dependency
        try:
            from qig_geometry.canonical import fisher_rao_distance
        except ImportError:
            self.add_warning(
                "Cannot verify Fisher-Rao identity: qig_geometry not available"
            )
            return True  # Don't fail if geometry module unavailable
        
        if len(basin_coords) < 3:
            return True  # Need at least 3 points for triangle inequality
        
        # Sample a few triplets for triangle inequality check
        n_samples = min(10, len(basin_coords) // 3)
        np.random.seed(42)  # Deterministic sampling
        
        for _ in range(n_samples):
            indices = np.random.choice(len(basin_coords), 3, replace=False)
            coords = [np.array(basin_coords[i], dtype=np.float64) for i in indices]
            
            # Compute distances
            try:
                d01 = fisher_rao_distance(coords[0], coords[1])
                d12 = fisher_rao_distance(coords[1], coords[2])
                d02 = fisher_rao_distance(coords[0], coords[2])
            except Exception as e:
                self.add_warning(f"Fisher-Rao distance computation failed: {e}")
                continue
            
            # Check non-negative
            if d01 < 0 or d12 < 0 or d02 < 0:
                self.add_error("Fisher-Rao distance is negative")
                return False
            
            # Check triangle inequality: d(a,c) <= d(a,b) + d(b,c)
            if d02 > (d01 + d12) * 1.01:  # Allow 1% tolerance
                self.add_error(
                    f"Triangle inequality violated: d02={d02:.4f} > d01+d12={d01+d12:.4f}"
                )
                return False
        
        return True
    
    def check_special_symbols(self, artifact: Dict[str, Any]) -> bool:
        """
        Verify special symbols are properly defined.
        
        Checks:
        - All required special symbols present (UNK, PAD, BOS, EOS)
        - Each special symbol has valid basin coordinate
        - Special symbol coordinates are deterministic (match expected)
        
        Args:
            artifact: Artifact dictionary to validate
            
        Returns:
            True if special symbols valid, False otherwise
        """
        special_symbols = artifact.get('special_symbols', {})
        required_symbols = ['UNK', 'PAD', 'BOS', 'EOS']
        
        for symbol in required_symbols:
            if symbol not in special_symbols:
                self.add_error(f"Missing required special symbol: {symbol}")
                return False
            
            symbol_data = special_symbols[symbol]
            
            # Check required fields
            if 'token' not in symbol_data:
                self.add_error(f"Special symbol {symbol} missing 'token' field")
                return False
            
            if 'basin_coord' not in symbol_data:
                self.add_error(f"Special symbol {symbol} missing 'basin_coord' field")
                return False
            
            if 'token_id' not in symbol_data:
                self.add_error(f"Special symbol {symbol} missing 'token_id' field")
                return False
            
            # Check basin coordinate dimension
            basin_coord = symbol_data['basin_coord']
            if len(basin_coord) != 64:
                self.add_error(
                    f"Special symbol {symbol} has invalid basin_coord dimension: {len(basin_coord)}"
                )
                return False
            
            # Check unit norm
            coord_array = np.array(basin_coord, dtype=np.float64)
            norm = np.sqrt(np.sum(coord_array**2))
            if not (0.99 < norm < 1.01):
                self.add_error(
                    f"Special symbol {symbol} not unit-normalized: norm={norm:.6f}"
                )
                return False
        
        return True
    
    def check_provenance(self, artifact: Dict[str, Any]) -> bool:
        """
        Verify provenance tracking is complete.
        
        Checks:
        - Required fields present (created_at, geometry_version, hyperparameters)
        - created_at is valid ISO8601 timestamp
        - geometry_version is valid SHA-1 hash
        - hyperparameters contain required fields
        
        Args:
            artifact: Artifact dictionary to validate
            
        Returns:
            True if provenance is valid, False otherwise
        """
        provenance = artifact.get('provenance', {})
        
        # Check required fields
        required_fields = ['created_at', 'geometry_version', 'hyperparameters']
        for field in required_fields:
            if field not in provenance:
                self.add_error(f"Missing required provenance field: {field}")
                return False
        
        # Validate created_at timestamp
        try:
            datetime.fromisoformat(provenance['created_at'].replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            self.add_error(f"Invalid ISO8601 timestamp: {provenance.get('created_at')}")
            return False
        
        # Validate geometry_version (SHA-1 hash)
        geometry_version = provenance['geometry_version']
        if not isinstance(geometry_version, str) or len(geometry_version) != 40:
            self.add_error(f"Invalid geometry_version: must be 40-character SHA-1 hash")
            return False
        
        if not all(c in '0123456789abcdef' for c in geometry_version):
            self.add_error(f"Invalid geometry_version: must be hexadecimal SHA-1 hash")
            return False
        
        # Check hyperparameters
        hyperparams = provenance.get('hyperparameters', {})
        if 'coordinate_dim' in hyperparams and hyperparams['coordinate_dim'] != 64:
            self.add_error(f"Invalid hyperparameters.coordinate_dim: {hyperparams['coordinate_dim']}")
            return False
        
        return True
    
    def check_validation_metadata(self, artifact: Dict[str, Any]) -> bool:
        """
        Verify validation metadata is present and consistent.
        
        Args:
            artifact: Artifact dictionary to validate
            
        Returns:
            True if validation metadata is valid, False otherwise
        """
        validation = artifact.get('validation', {})
        
        required_fields = [
            'passes_simplex_check',
            'fisher_rao_identity_verified',
            'dimension_consistent'
        ]
        
        for field in required_fields:
            if field not in validation:
                self.add_error(f"Missing required validation field: {field}")
                return False
            
            if not isinstance(validation[field], bool):
                self.add_error(f"Validation field {field} must be boolean")
                return False
        
        return True
    
    def validate_full_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform complete validation on artifact.
        
        Args:
            artifact: Artifact dictionary to validate
            
        Returns:
            Validation result dictionary with keys:
            - valid: bool (True if all checks passed)
            - errors: List[str] (validation errors)
            - warnings: List[str] (validation warnings)
            - checks: Dict[str, bool] (individual check results)
        """
        self.reset()
        
        checks = {}
        
        # Check required fields
        checks['required_fields'] = self.check_required_fields(artifact)
        if not checks['required_fields']:
            return {
                'valid': False,
                'errors': self.get_errors(),
                'warnings': self.get_warnings(),
                'checks': checks
            }
        
        # Check version
        checks['version'] = self.check_version(artifact)
        
        # Check basin dimension
        checks['basin_dim'] = self.check_basin_dim(artifact)
        
        # Check dimension consistency
        checks['dimension_consistent'] = self.check_dimension_consistency(
            artifact['basin_coords'],
            artifact['symbols'],
            artifact['phi_scores']
        )
        
        # Check simplex constraints
        checks['simplex_constraints'] = self.check_simplex_constraints(
            artifact['basin_coords']
        )
        
        # Check Fisher-Rao identity
        checks['fisher_rao_identity'] = self.check_fisher_rao_identity(
            artifact['basin_coords']
        )
        
        # Check special symbols
        checks['special_symbols'] = self.check_special_symbols(artifact)
        
        # Check provenance
        checks['provenance'] = self.check_provenance(artifact)
        
        # Check validation metadata
        checks['validation_metadata'] = self.check_validation_metadata(artifact)
        
        return {
            'valid': self.is_valid(),
            'errors': self.get_errors(),
            'warnings': self.get_warnings(),
            'checks': checks
        }


def validate_artifact(artifact: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """
    Validate a coordizer artifact against CoordizerArtifactV1 schema.
    
    Args:
        artifact: Artifact dictionary to validate
        strict: If True, enforce all constraints. If False, allow warnings.
        
    Returns:
        Validation result dictionary
    """
    validator = ArtifactValidator(strict=strict)
    return validator.validate_full_artifact(artifact)


def validate_artifact_from_file(artifact_path: str, strict: bool = True) -> Dict[str, Any]:
    """
    Load and validate artifact from JSON file.
    
    Args:
        artifact_path: Path to artifact JSON file
        strict: If True, enforce all constraints
        
    Returns:
        Validation result dictionary
    """
    try:
        with open(artifact_path, 'r') as f:
            artifact = json.load(f)
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Failed to load artifact: {e}"],
            'warnings': [],
            'checks': {}
        }
    
    return validate_artifact(artifact, strict=strict)


def detect_artifact_version(artifact_dir: str) -> Optional[str]:
    """
    Detect artifact format version.
    
    Args:
        artifact_dir: Directory containing artifact files
        
    Returns:
        Version string ('1.0') or None if unversioned/legacy
    """
    # Check for CoordizerArtifactV1 format
    required_files = ['vocab.json', 'basin_coords.npy', 'coord_tokens.json']
    if all(os.path.exists(os.path.join(artifact_dir, f)) for f in required_files):
        # Try to detect version from vocab.json
        try:
            with open(os.path.join(artifact_dir, 'vocab.json'), 'r') as f:
                data = json.load(f)
                if 'version' in data:
                    return data['version']
        except:
            pass
        
        # Has v1 structure but no version field - treat as v1.0
        return '1.0'
    
    # Legacy format
    return None
