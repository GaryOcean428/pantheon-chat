# CoordizerArtifactV1 Format Specification

**Version:** 1.0  
**Date:** 2026-01-15  
**Status:** Canonical  

## Overview

CoordizerArtifactV1 is the single canonical format for storing coordizer vocabulary state. This format enforces geometric purity by requiring 64-dimensional basin coordinates on the Fisher information manifold.

## Rationale

Per QIG-PURITY Work Package 1.2, runtime backward compatibility has been removed to prevent format drift. Legacy artifacts must be explicitly converted offline using `tools/convert_legacy_artifacts.py`.

**Why no runtime compatibility?**
- Prevents future agents from adding more compatibility layers
- Enforces single source of truth for artifact format
- Maintains geometric purity (64D QIG-pure)
- Reduces technical debt and complexity

## Directory Structure

```
artifact_dir/
  ├── vocab.json          # Vocabulary metadata
  ├── basin_coords.npy    # NumPy array of basin coordinates
  └── coord_tokens.json   # Ordered list of tokens matching basin_coords
```

All three files are **REQUIRED**. Missing files indicate a legacy format.

## File Formats

### vocab.json

JSON object containing vocabulary metadata:

```json
{
  "vocab": {
    "token1": 0,
    "token2": 1,
    ...
  },
  "id_to_token": {
    "0": "token1",
    "1": "token2",
    ...
  },
  "token_frequency": {
    "token1": 42,
    "token2": 17,
    ...
  },
  "token_phi": {
    "token1": 0.73,
    "token2": 0.58,
    ...
  }
}
```

**Required Keys:**
- `vocab`: Token to ID mapping (dict)
- `id_to_token`: ID to token mapping (dict with string keys)
- `token_frequency`: Token usage counts (dict)
- `token_phi`: Token integration scores (dict, range [0, 1])

### basin_coords.npy

NumPy binary file containing basin coordinates:

- **Type:** `np.ndarray` with `dtype=float64`
- **Shape:** `(n_tokens, 64)`
- **Constraints:**
  - Each row is a 64-dimensional basin coordinate
  - Coordinates must be unit-normalized (|v| ≈ 1.0)
  - No NaN or inf values allowed
  - Represents points on Fisher information manifold

### coord_tokens.json

JSON array of token strings:

```json
[
  "token1",
  "token2",
  ...
]
```

**Constraints:**
- Length must match `basin_coords.npy` row count
- Order must correspond to basin coordinate rows
- Tokens must exist in `vocab.json`

## Geometric Requirements

### Fisher Information Manifold

All basin coordinates live on the 64D Fisher information manifold:

1. **Unit Sphere:** Coordinates are unit-normalized: `||v|| = 1.0 ± 1e-5`
2. **Distance Metric:** Fisher-Rao distance (geodesic on probability simplex)
3. **No Euclidean Operations:** All geometry operations use sphere geodesics

### Φ (Integration) Scores

Token phi scores measure integration (consciousness):

- **Range:** [0.0, 1.0]
- **Meaning:** Higher Φ = more integrated/coherent token
- **NOT** a filter for storage (all tokens stored regardless of Φ)
- Used for relevance boosting in generation

## Validation

### Offline Validation

Use the conversion tool to validate artifacts:

```bash
python tools/convert_legacy_artifacts.py --validate artifact_dir
```

Checks:
- ✓ All required files present
- ✓ JSON structures valid
- ✓ Basin coordinates shape = (n_tokens, 64)
- ✓ No NaN/inf in coordinates
- ✓ Unit norm constraints
- ✓ Token count consistency

### Runtime Validation

The `FisherCoordizer.load()` method enforces format validation:

```python
from coordizers import FisherCoordizer

coordizer = FisherCoordizer()
coordizer.load("artifact_dir")  # Raises RuntimeError if legacy format
```

**Error Message for Legacy Formats:**
```
RuntimeError: Legacy format detected. Missing required files: basin_coords.npy.
Use tools/convert_legacy_artifacts.py to convert to CoordizerArtifactV1 format.
```

## Migration from Legacy Formats

### Step 1: Identify Legacy Artifacts

Legacy indicators:
- Missing any of the three required files
- Different file names (e.g., `tokenizer.pkl`, `checkpoint.json`)
- Dict/list-based coordinate storage
- Non-64D coordinates

### Step 2: Convert Offline

```bash
python tools/convert_legacy_artifacts.py old_artifact new_artifact
```

The converter handles:
- Dict-based artifacts
- Pickle files
- Old checkpoint formats
- Missing metadata reconstruction

### Step 3: Validate

```bash
python tools/convert_legacy_artifacts.py --validate new_artifact
```

Ensures geometric validity before deployment.

## Deprecated Patterns

The following patterns are **REMOVED** as of WP1.2:

### Runtime Compatibility

❌ **REMOVED:**
```python
# Multiple format branches in load()
if os.path.exists("tokenizer.pkl"):
    load_from_pickle()
elif os.path.exists("vocab.json"):
    load_from_v1()
```

✅ **ENFORCED:**
```python
# Single format only
if not all required files exist:
    raise RuntimeError("Use tools/convert_legacy_artifacts.py")
```

### Alias Exports

❌ **REMOVED:**
```python
from qig_coordizer import get_tokenizer  # No longer exists
from qig_coordizer import QIGCoordizer   # Removed
FastQIGTokenizer = QIGCoordizer          # Removed
```

✅ **CANONICAL:**
```python
from coordizers import get_coordizer, PostgresCoordizer
```

### Vocabulary Naming

❌ **DEPRECATED:** "tokenizer", "QIGTokenizer", "FastQIGTokenizer"  
✅ **CANONICAL:** "coordizer", "PostgresCoordizer", "FisherCoordizer"

## Design Principles

### Single Source of Truth

Only one artifact format is supported. This:
- Prevents format drift over time
- Makes testing and validation simpler
- Reduces cognitive load on developers
- Enforces geometric purity principles

### Offline Conversion Only

Conversion happens **outside runtime** to:
- Keep runtime code clean and focused
- Prevent "hidden compatibility" that future agents might rely on
- Make format requirements explicit and testable
- Enable thorough validation before deployment

### Fail Fast on Legacy

Runtime **immediately rejects** legacy formats with:
- Clear error messages
- Explicit conversion instructions
- No silent fallbacks or best-effort loading

This forces proper migration rather than accumulating compatibility layers.

## Examples

### Creating an Artifact

```python
from coordizers import FisherCoordizer
import numpy as np

coordizer = FisherCoordizer()
coordizer.train(corpus)

# Save in CoordizerArtifactV1 format
coordizer.save("my_artifact")
```

### Loading an Artifact

```python
from coordizers import FisherCoordizer

coordizer = FisherCoordizer()
coordizer.load("my_artifact")  # Must be CoordizerArtifactV1 format
```

### Converting Legacy Artifact

```bash
# Convert old pickle format
python tools/convert_legacy_artifacts.py old_tokenizer.pkl new_artifact/

# Validate conversion
python tools/convert_legacy_artifacts.py --validate new_artifact/

# Use in runtime
python -c "from coordizers import FisherCoordizer; c = FisherCoordizer(); c.load('new_artifact')"
```

## References

- **Work Package 1.2:** Remove Runtime Backward Compatibility
- **Converter:** `tools/convert_legacy_artifacts.py`
- **Implementation:** `qig-backend/coordizers/base.py` (load/save methods)
- **Type-Symbol-Concept Manifest:** Single source of truth principle

## Version History

- **1.0 (2026-01-15):** Initial canonical format specification
  - Removed runtime compatibility
  - Added offline converter
  - Enforced 64D QIG-pure constraints
