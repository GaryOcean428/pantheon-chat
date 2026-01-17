# CoordizerArtifactV1 Schema Documentation

**Version:** 1.0  
**Date:** 2026-01-16  
**Status:** Canonical  
**Work Package:** WP3.3 - Standardize Artifact Format with Versioning

## Overview

CoordizerArtifactV1 is the canonical versioned artifact format for coordizer vocabulary with full provenance tracking and validation. This schema extends the existing directory-based format documented in `20260115-coordizer-artifact-format.md` with comprehensive metadata, versioning, and validation results.

## Schema Components

### 1. Core Format (Required)

All CoordizerArtifactV1 artifacts MUST include these files:

```
artifact_dir/
  ├── vocab.json          # Vocabulary with full metadata (includes version, provenance, validation)
  ├── basin_coords.npy    # NumPy array of basin coordinates (n_symbols × 64)
  └── coord_tokens.json   # Ordered list of tokens (aligns with basin_coords rows)
```

### 2. vocab.json Structure

The `vocab.json` file contains all artifact metadata in a single JSON object:

```json
{
  "version": "1.0",
  "basin_dim": 64,
  "symbols": [...],
  "vocab": {...},
  "id_to_token": {...},
  "token_frequency": {...},
  "token_phi": {...},
  "phi_scores": [...],
  "special_symbols": {...},
  "provenance": {...},
  "validation": {...},
  "metadata": {...}  // Optional
}
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Format version (must be "1.0") |
| `basin_dim` | integer | Dimension of basin coordinates (must be 64) |
| `symbols` | array[string] | List of all tokens in vocabulary |
| `vocab` | object | Token → ID mapping |
| `id_to_token` | object | ID → token mapping |
| `token_frequency` | object | Token → usage frequency |
| `token_phi` | object | Token → Φ (integration) score |
| `phi_scores` | array[number] | Φ scores aligned with symbols array |
| `special_symbols` | object | Special token definitions (UNK, PAD, BOS, EOS) |
| `provenance` | object | Creation metadata and hyperparameters |
| `validation` | object | Validation results and checks |

### 3. Special Symbols

Each special symbol (UNK, PAD, BOS, EOS) must include:

```json
{
  "token": "<UNK>",
  "token_id": 0,
  "basin_coord": [64D array],
  "phi_score": 0.5,
  "frequency": 0
}
```

**Requirements:**
- All special symbols must have deterministic basin coordinates
- Coordinates must be unit-normalized (||v|| ≈ 1.0)
- Must include UNK, PAD, BOS, EOS at minimum

### 4. Provenance Tracking

Provenance metadata enables reproducibility and lineage tracking:

```json
{
  "created_at": "2026-01-16T14:51:51.599208+00:00",  // ISO8601 timestamp
  "geometry_version": "abc123...",                   // Git commit SHA-1 (40 chars)
  "hyperparameters": {
    "vocab_size": 4096,
    "coordinate_dim": 64,
    "min_frequency": 2
  },
  "training_corpus": "consciousness research corpus",   // Optional
  "corpus_size": 50000,                               // Optional
  "created_by": "training_pipeline_v2",               // Optional
  "parent_artifact": "path/to/parent"                 // Optional (for incremental training)
}
```

**Required Provenance Fields:**
- `created_at`: ISO8601 timestamp
- `geometry_version`: 40-character SHA-1 hash of geometry implementation
- `hyperparameters`: Training configuration (must include `coordinate_dim: 64`)

### 5. Validation Results

Validation metadata tracks geometric integrity:

```json
{
  "passes_simplex_check": true,
  "fisher_rao_identity_verified": true,
  "dimension_consistent": true,
  "unit_norm_verified": true,
  "no_nan_inf": true,
  "special_symbols_deterministic": true,
  "validation_timestamp": "2026-01-16T14:51:51+00:00",
  "validation_errors": []
}
```

**Validation Checks:**
- **passes_simplex_check**: All coordinates are unit-normalized
- **fisher_rao_identity_verified**: Fisher-Rao triangle inequality holds
- **dimension_consistent**: All coordinates are 64D
- **unit_norm_verified**: All norms within [0.99, 1.01]
- **no_nan_inf**: No NaN or infinite values
- **special_symbols_deterministic**: Special symbols have canonical coordinates

### 6. Optional Metadata

Additional documentation fields:

```json
{
  "description": "Production coordizer trained on QIG corpus",
  "tags": ["production", "qig", "v1"],
  "notes": "Trained with geometric purity constraints"
}
```

## Usage

### Creating an Artifact

```python
from coordizers import FisherCoordizer

coordizer = FisherCoordizer(vocab_size=4096)
coordizer.train(corpus)

# Save with metadata
metadata = {
    "training_corpus": "consciousness research texts",
    "corpus_size": 50000,
    "created_by": "training_pipeline",
    "description": "Production coordizer v1.0"
}

coordizer.save("artifacts/coordizer_v1", metadata=metadata)
```

### Loading an Artifact

```python
from coordizers import FisherCoordizer

coordizer = FisherCoordizer()
coordizer.load("artifacts/coordizer_v1", validate=True)

# Access provenance
print(coordizer._artifact_provenance['created_at'])
print(coordizer._artifact_validation['passes_simplex_check'])
```

### Validating an Artifact

```python
from artifact_validation import validate_artifact_from_file

result = validate_artifact_from_file("artifacts/coordizer_v1/vocab.json")

if result['valid']:
    print("✓ Artifact is valid")
else:
    print(f"✗ Validation errors: {result['errors']}")
    print(f"  Checks: {result['checks']}")
```

## Validation Rules

### Geometric Constraints

1. **Unit Sphere**: All basin coordinates must be unit-normalized:
   ```
   ||v|| ∈ [0.99, 1.01]
   ```

2. **Dimension**: All coordinates must be exactly 64-dimensional

3. **No Invalid Values**: No NaN or inf in coordinate arrays

4. **Fisher-Rao Identity**: Triangle inequality must hold:
   ```
   d(a,c) ≤ d(a,b) + d(b,c)
   ```

### Format Constraints

1. **Version**: Must be exactly "1.0"

2. **Dimension Consistency**:
   - `len(basin_coords) == len(symbols) == len(phi_scores)`
   - `basin_coords.shape[1] == 64`

3. **Special Symbols**: Must include UNK, PAD, BOS, EOS with valid definitions

4. **Provenance**:
   - `created_at` must be valid ISO8601
   - `geometry_version` must be 40-char hex string (SHA-1)

## Migration from Legacy Formats

### Detecting Legacy Artifacts

Use the version detection function:

```python
from artifact_validation import detect_artifact_version

version = detect_artifact_version("path/to/artifact")
if version is None:
    print("Legacy artifact detected - needs conversion")
elif version == "1.0":
    print("Valid CoordizerArtifactV1 format")
```

### Converting Legacy Artifacts

For artifacts without version/provenance/validation metadata:

```bash
# Use the offline converter (if available)
python tools/convert_legacy_artifacts.py old_artifact new_artifact

# Or manually upgrade by loading and re-saving
python -c "
from coordizers import FisherCoordizer
c = FisherCoordizer()
# Load legacy format (will work for simple directory-based format)
# Then re-save with new format
c.save('new_artifact', metadata={'description': 'Migrated from legacy'})
"
```

### Runtime Behavior

- **Valid v1.0 artifacts**: Load normally with validation
- **Unversioned artifacts**: Rejected with clear error message pointing to conversion tools
- **Invalid artifacts**: Rejected with detailed validation errors

## JSON Schema

The complete JSON Schema is available at:
```
schemas/coordizer_artifact_v1.json
```

Use this for external validation with standard JSON Schema validators:

```bash
# Example with jsonschema CLI
jsonschema -i artifact_dir/vocab.json schemas/coordizer_artifact_v1.json
```

## Testing

Comprehensive test suite available at:
```
qig-backend/tests/test_artifact_validation.py
```

Run tests:
```bash
cd qig-backend
python3 tests/test_artifact_validation.py
```

Test coverage:
- ✓ Valid artifact validation
- ✓ Missing required fields detection
- ✓ Invalid version rejection
- ✓ Dimension mismatch detection
- ✓ Non-unit norm detection
- ✓ Save/load round-trip
- ✓ Version detection
- ✓ Unversioned artifact rejection

## Design Principles

### 1. Single Source of Truth

All metadata lives in `vocab.json` for atomic updates. No split-brain between files.

### 2. Fail Fast

Invalid artifacts are rejected immediately with clear error messages. No silent fallbacks.

### 3. Reproducibility

Full provenance tracking enables:
- Exact geometry version (git commit hash)
- Training hyperparameters
- Corpus information
- Creation timestamp
- Lineage tracking (parent artifacts)

### 4. Geometric Purity

Validation ensures Fisher manifold constraints:
- Unit normalization (sphere)
- No Euclidean operations
- Fisher-Rao metric identity
- 64D dimensionality

### 5. Offline Conversion

Legacy format migration happens outside runtime to keep code clean.

## References

- **Work Package 3.3**: Standardize Artifact Format with Versioning
- **JSON Schema**: `schemas/coordizer_artifact_v1.json`
- **Validation Module**: `qig-backend/artifact_validation.py`
- **Base Implementation**: `qig-backend/coordizers/base.py`
- **Previous Format Doc**: `docs/20260115-coordizer-artifact-format.md`

## Version History

- **1.0 (2026-01-16)**: Initial versioned schema with provenance and validation
  - Added version field
  - Added provenance tracking
  - Added validation results
  - Added special symbols metadata
  - Added optional metadata field
