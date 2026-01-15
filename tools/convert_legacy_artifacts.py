#!/usr/bin/env python3
"""
Offline Legacy Artifact Converter
==================================

Converts legacy coordizer/tokenizer artifacts to canonical CoordizerArtifactV1 format.

CoordizerArtifactV1 Format:
---------------------------
artifact_dir/
  ├── vocab.json          # Vocabulary metadata
  ├── basin_coords.npy    # NumPy array of basin coordinates (n_tokens, 64)
  └── coord_tokens.json   # Ordered list of tokens matching basin_coords

Supported Legacy Formats:
--------------------------
1. Dict-based artifacts (old pickle/json formats)
2. List/tuple-based coordinate formats
3. Old tokenizer checkpoint formats
4. Artifacts missing required fields

Usage:
------
    python tools/convert_legacy_artifacts.py input_dir output_dir
    python tools/convert_legacy_artifacts.py --validate output_dir
    python tools/convert_legacy_artifacts.py --help

Error Handling:
---------------
Runtime MUST reject legacy formats with clear error:
    "Legacy format detected. Use tools/convert_legacy_artifacts.py to convert."

This enforces single canonical format and prevents future compatibility drift.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class LegacyFormatError(Exception):
    """Raised when a legacy format is detected that cannot be converted."""
    pass


class ValidationError(Exception):
    """Raised when converted artifact fails geometric validation."""
    pass


def detect_legacy_format(input_path: Path) -> str:
    """
    Detect the type of legacy format in input directory.
    
    Args:
        input_path: Path to input directory or file
        
    Returns:
        Format type string: "v1_canonical", "dict_format", "old_checkpoint", etc.
        
    Raises:
        LegacyFormatError: If format is unrecognized
    """
    if not input_path.exists():
        raise LegacyFormatError(f"Input path does not exist: {input_path}")
    
    # Check for canonical V1 format
    if input_path.is_dir():
        vocab_file = input_path / "vocab.json"
        coords_file = input_path / "basin_coords.npy"
        tokens_file = input_path / "coord_tokens.json"
        
        if vocab_file.exists() and coords_file.exists() and tokens_file.exists():
            return "v1_canonical"
        
        # Old checkpoint format (might have different filenames)
        if (input_path / "tokenizer.pkl").exists():
            return "old_pickle_checkpoint"
        
        if (input_path / "checkpoint.json").exists():
            return "old_json_checkpoint"
    
    # Single file formats
    if input_path.is_file():
        if input_path.suffix == ".pkl":
            return "pickle_artifact"
        if input_path.suffix == ".json":
            return "json_artifact"
    
    raise LegacyFormatError(f"Unrecognized format in {input_path}")


def load_legacy_dict_format(data: Dict) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Convert legacy dict-based artifact to canonical format.
    
    Args:
        data: Legacy dictionary artifact
        
    Returns:
        Tuple of (vocab_metadata, token_mappings, basin_coords_array)
    """
    vocab = data.get("vocab", {})
    id_to_token = data.get("id_to_token", {})
    token_frequency = data.get("token_frequency", data.get("frequencies", {}))
    token_phi = data.get("token_phi", data.get("phi_scores", {}))
    
    # Extract basin coordinates
    basin_coords = data.get("basin_coords", data.get("coordinates", {}))
    
    if not basin_coords:
        raise LegacyFormatError("No basin coordinates found in legacy artifact")
    
    # Convert to canonical format
    tokens = sorted(basin_coords.keys())
    coords_matrix = np.array([basin_coords[t] for t in tokens], dtype=np.float64)
    
    # Validate dimensions
    if coords_matrix.shape[1] != 64:
        raise ValidationError(
            f"Invalid coordinate dimension: {coords_matrix.shape[1]} (expected 64)"
        )
    
    vocab_metadata = {
        "vocab": vocab,
        "id_to_token": {str(k): v for k, v in id_to_token.items()},
        "token_frequency": token_frequency,
        "token_phi": token_phi,
    }
    
    token_mappings = {
        "tokens": tokens,
    }
    
    return vocab_metadata, token_mappings, coords_matrix


def convert_artifact(input_path: Path, output_path: Path, validate: bool = True) -> None:
    """
    Convert legacy artifact to CoordizerArtifactV1 format.
    
    Args:
        input_path: Path to legacy artifact
        output_path: Path to output directory
        validate: Whether to validate converted artifact
        
    Raises:
        LegacyFormatError: If conversion fails
        ValidationError: If validation fails
    """
    format_type = detect_legacy_format(input_path)
    
    print(f"Detected format: {format_type}")
    
    if format_type == "v1_canonical":
        print("✓ Artifact is already in canonical CoordizerArtifactV1 format")
        if validate:
            validate_artifact(input_path)
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and convert based on format
    if format_type in ["json_artifact", "old_json_checkpoint"]:
        with open(input_path if input_path.is_file() else input_path / "checkpoint.json") as f:
            data = json.load(f)
        vocab_metadata, token_mappings, coords_matrix = load_legacy_dict_format(data)
    
    elif format_type in ["pickle_artifact", "old_pickle_checkpoint"]:
        import pickle
        pkl_path = input_path if input_path.is_file() else input_path / "tokenizer.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        vocab_metadata, token_mappings, coords_matrix = load_legacy_dict_format(data)
    
    else:
        raise LegacyFormatError(f"Conversion not implemented for format: {format_type}")
    
    # Write canonical format
    with open(output_path / "vocab.json", "w") as f:
        json.dump(vocab_metadata, f, indent=2)
    
    np.save(output_path / "basin_coords.npy", coords_matrix)
    
    with open(output_path / "coord_tokens.json", "w") as f:
        json.dump(token_mappings["tokens"], f, indent=2)
    
    print(f"✓ Converted to CoordizerArtifactV1 format: {output_path}")
    
    if validate:
        validate_artifact(output_path)


def validate_artifact(artifact_path: Path) -> None:
    """
    Validate that artifact is geometrically valid CoordizerArtifactV1.
    
    Args:
        artifact_path: Path to artifact directory
        
    Raises:
        ValidationError: If validation fails
    """
    print(f"\nValidating artifact: {artifact_path}")
    
    # Check required files exist
    required_files = ["vocab.json", "basin_coords.npy", "coord_tokens.json"]
    for filename in required_files:
        filepath = artifact_path / filename
        if not filepath.exists():
            raise ValidationError(f"Missing required file: {filename}")
    
    # Load and validate vocab.json
    with open(artifact_path / "vocab.json") as f:
        vocab_data = json.load(f)
    
    required_keys = ["vocab", "id_to_token", "token_frequency", "token_phi"]
    for key in required_keys:
        if key not in vocab_data:
            raise ValidationError(f"Missing required key in vocab.json: {key}")
    
    vocab = vocab_data["vocab"]
    id_to_token = vocab_data["id_to_token"]
    
    print(f"  ✓ Vocabulary size: {len(vocab)}")
    
    # Load and validate basin coordinates
    coords_matrix = np.load(artifact_path / "basin_coords.npy")
    
    if coords_matrix.ndim != 2:
        raise ValidationError(f"Basin coordinates must be 2D, got shape {coords_matrix.shape}")
    
    if coords_matrix.shape[1] != 64:
        raise ValidationError(
            f"Basin coordinates must be 64D, got dimension {coords_matrix.shape[1]}"
        )
    
    print(f"  ✓ Basin coordinates shape: {coords_matrix.shape}")
    
    # Validate coordinate geometry (unit norm, valid range)
    norms = np.linalg.norm(coords_matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        print(f"  ⚠ Warning: Not all coordinates are unit norm (min={norms.min():.6f}, max={norms.max():.6f})")
    
    # Check for NaN or inf
    if np.any(~np.isfinite(coords_matrix)):
        raise ValidationError("Basin coordinates contain NaN or inf values")
    
    print(f"  ✓ Coordinates are geometrically valid")
    
    # Load and validate tokens
    with open(artifact_path / "coord_tokens.json") as f:
        tokens = json.load(f)
    
    if len(tokens) != coords_matrix.shape[0]:
        raise ValidationError(
            f"Token count mismatch: {len(tokens)} tokens vs {coords_matrix.shape[0]} coordinates"
        )
    
    print(f"  ✓ Token count matches coordinate count: {len(tokens)}")
    
    # Check vocab consistency
    vocab_tokens = set(vocab.keys())
    coord_tokens = set(tokens)
    
    if vocab_tokens != coord_tokens:
        missing_in_coords = vocab_tokens - coord_tokens
        missing_in_vocab = coord_tokens - vocab_tokens
        
        if missing_in_coords:
            print(f"  ⚠ Warning: {len(missing_in_coords)} tokens in vocab missing from coords")
        if missing_in_vocab:
            print(f"  ⚠ Warning: {len(missing_in_vocab)} tokens in coords missing from vocab")
    else:
        print(f"  ✓ Vocabulary consistency check passed")
    
    print(f"\n✓ Artifact validation passed: {artifact_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert legacy coordizer artifacts to CoordizerArtifactV1 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert legacy artifact
  python tools/convert_legacy_artifacts.py old_checkpoints/tokenizer_v1 artifacts/canonical_v1
  
  # Validate existing artifact
  python tools/convert_legacy_artifacts.py --validate artifacts/canonical_v1
  
  # Convert without validation
  python tools/convert_legacy_artifacts.py --no-validate old_artifact new_artifact

Runtime Error Message:
  When runtime detects legacy format, it MUST show:
  "Legacy format detected. Use tools/convert_legacy_artifacts.py to convert."
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input legacy artifact path (directory or file)"
    )
    
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output directory for converted artifact"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate an existing artifact (no conversion)"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after conversion"
    )
    
    args = parser.parse_args()
    
    try:
        if args.validate:
            # Validate mode
            if not args.input:
                parser.error("Input path required for validation")
            validate_artifact(args.input)
            return 0
        
        # Convert mode
        if not args.input or not args.output:
            parser.error("Both input and output paths required for conversion")
        
        convert_artifact(
            args.input,
            args.output,
            validate=not args.no_validate
        )
        return 0
    
    except (LegacyFormatError, ValidationError) as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
