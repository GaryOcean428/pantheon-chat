# Migration Guide: Legacy Artifacts to CoordizerArtifactV1

**Date:** 2026-01-16  
**Work Package:** WP3.3 - Standardize Artifact Format with Versioning

## Overview

This guide covers migrating existing coordizer artifacts to the CoordizerArtifactV1 format with versioning, provenance, and validation metadata.

## Migration Strategy

### Phase 1: Assessment

**Goal:** Identify all artifacts that need migration.

**Steps:**

1. **Locate existing artifacts:**
   ```bash
   # Find all coordizer artifact directories
   find . -name "basin_coords.npy" -o -name "vocab.json" | xargs dirname | sort -u
   ```

2. **Check version status:**
   ```python
   from artifact_validation import detect_artifact_version
   
   artifacts = ['path/to/artifact1', 'path/to/artifact2']
   
   for artifact_path in artifacts:
       version = detect_artifact_version(artifact_path)
       if version is None:
           print(f"❌ LEGACY: {artifact_path}")
       elif version == "1.0":
           print(f"✓ VALID: {artifact_path}")
   ```

3. **Categorize artifacts:**
   - **Complete legacy**: Has vocab.json, basin_coords.npy, coord_tokens.json but no versioning
   - **Incomplete legacy**: Missing one or more required files
   - **Non-standard**: Uses different file structure (e.g., pickle files)

### Phase 2: Backup

**CRITICAL:** Always backup before migration!

```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Backup all artifacts
cp -r path/to/artifact backups/$(date +%Y%m%d)/artifact_backup
```

### Phase 3: Migration Methods

#### Method 1: Automatic (Recommended)

Use the Python migration script for most artifacts:

```python
#!/usr/bin/env python3
"""
Migrate legacy artifact to CoordizerArtifactV1 format.
"""

import sys
import os
sys.path.insert(0, 'qig-backend')

from coordizers.base import FisherCoordizer
from artifact_validation import validate_artifact, detect_artifact_version

def migrate_artifact(legacy_path, new_path, metadata=None):
    """
    Migrate legacy artifact to v1.0 format.
    
    Args:
        legacy_path: Path to legacy artifact directory
        new_path: Path for new v1.0 artifact
        metadata: Optional dict with provenance metadata
    """
    # Check if already v1.0
    version = detect_artifact_version(legacy_path)
    if version == "1.0":
        print(f"✓ {legacy_path} is already v1.0 format")
        return
    
    print(f"Migrating {legacy_path} → {new_path}")
    
    # Load legacy artifact
    try:
        coordizer = FisherCoordizer()
        coordizer.load(legacy_path, validate=False)  # Skip validation for legacy
        print(f"  ✓ Loaded {len(coordizer.vocab)} tokens")
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return
    
    # Prepare metadata
    if metadata is None:
        metadata = {
            "description": f"Migrated from {os.path.basename(legacy_path)}",
            "created_by": "migration_script",
            "parent_artifact": legacy_path
        }
    
    # Save in v1.0 format
    try:
        coordizer.save(new_path, metadata=metadata)
        print(f"  ✓ Saved to {new_path}")
    except Exception as e:
        print(f"  ❌ Failed to save: {e}")
        return
    
    # Validate new artifact
    from artifact_validation import validate_artifact_from_file
    import json
    
    vocab_path = os.path.join(new_path, "vocab.json")
    result = validate_artifact_from_file(vocab_path)
    
    if result['valid']:
        print(f"  ✓ Validation passed")
        print(f"  ✓ Migration complete!")
    else:
        print(f"  ⚠ Validation warnings:")
        for error in result['errors'][:3]:
            print(f"    - {error}")

# Example usage
if __name__ == "__main__":
    legacy_artifacts = [
        "data/qig_tokenizer/checkpoint",
        "artifacts/coordizer_v0",
    ]
    
    for legacy_path in legacy_artifacts:
        if os.path.exists(legacy_path):
            new_path = f"{legacy_path}_v1"
            migrate_artifact(legacy_path, new_path)
```

#### Method 2: Manual Migration

For complex cases where automatic migration fails:

1. **Extract data from legacy format:**
   ```python
   import json
   import numpy as np
   
   # Load legacy vocab.json
   with open("legacy/vocab.json") as f:
       legacy_data = json.load(f)
   
   vocab = legacy_data['vocab']
   id_to_token = legacy_data['id_to_token']
   token_frequency = legacy_data['token_frequency']
   token_phi = legacy_data['token_phi']
   
   # Load coordinates
   coords = np.load("legacy/basin_coords.npy")
   
   with open("legacy/coord_tokens.json") as f:
       tokens = json.load(f)
   ```

2. **Build v1.0 artifact structure:**
   ```python
   from datetime import datetime, timezone
   
   artifact = {
       "version": "1.0",
       "basin_dim": 64,
       "symbols": tokens,
       "vocab": vocab,
       "id_to_token": id_to_token,
       "token_frequency": token_frequency,
       "token_phi": token_phi,
       "phi_scores": [token_phi.get(t, 0.0) for t in tokens],
       "special_symbols": {
           "UNK": {
               "token": "<UNK>",
               "token_id": vocab.get("<UNK>", 0),
               "basin_coord": coords[tokens.index("<UNK>")].tolist() if "<UNK>" in tokens else [0.0]*64,
               "phi_score": token_phi.get("<UNK>", 0.0),
               "frequency": token_frequency.get("<UNK>", 0)
           },
           # ... add PAD, BOS, EOS similarly
       },
       "provenance": {
           "created_at": datetime.now(timezone.utc).isoformat(),
           "geometry_version": "0" * 40,  # Unknown for legacy
           "hyperparameters": {
               "vocab_size": len(vocab),
               "coordinate_dim": 64,
               "min_frequency": 1
           },
           "training_corpus": "unknown (legacy artifact)",
           "created_by": "manual_migration"
       },
       "validation": {
           "passes_simplex_check": True,  # Will be validated
           "fisher_rao_identity_verified": True,
           "dimension_consistent": True,
           "unit_norm_verified": True,
           "no_nan_inf": True,
           "special_symbols_deterministic": True,
           "validation_timestamp": datetime.now(timezone.utc).isoformat(),
           "validation_errors": []
       }
   }
   ```

3. **Save and validate:**
   ```python
   import os
   
   os.makedirs("new_artifact", exist_ok=True)
   
   # Save vocab.json
   with open("new_artifact/vocab.json", "w") as f:
       json.dump(artifact, f, indent=2)
   
   # Save coordinates
   np.save("new_artifact/basin_coords.npy", coords)
   
   # Save token list
   with open("new_artifact/coord_tokens.json", "w") as f:
       json.dump(tokens, f, indent=2)
   
   # Validate
   from artifact_validation import validate_artifact_from_file
   result = validate_artifact_from_file("new_artifact/vocab.json")
   print(f"Valid: {result['valid']}")
   ```

### Phase 4: Validation

After migration, validate all artifacts:

```bash
cd qig-backend
python3 << EOF
from artifact_validation import validate_artifact_from_file
import glob

artifacts = glob.glob("../artifacts/*/vocab.json")

for artifact_path in artifacts:
    result = validate_artifact_from_file(artifact_path)
    if result['valid']:
        print(f"✓ {artifact_path}")
    else:
        print(f"❌ {artifact_path}")
        for error in result['errors'][:3]:
            print(f"  - {error}")
EOF
```

### Phase 5: Update References

Update all code that references artifacts:

1. **Update training scripts:**
   ```python
   # OLD
   coordizer.save("artifacts/model")
   
   # NEW (with metadata)
   metadata = {
       "training_corpus": "consciousness research corpus",
       "corpus_size": 50000,
       "created_by": "training_pipeline_v2"
   }
   coordizer.save("artifacts/model", metadata=metadata)
   ```

2. **Update loading code:**
   ```python
   # OLD
   coordizer.load("artifacts/model")
   
   # NEW (with validation)
   coordizer.load("artifacts/model", validate=True)
   
   # Access provenance
   print(coordizer._artifact_provenance['created_at'])
   ```

3. **Update deployment scripts:**
   - Add version checks before deployment
   - Validate artifacts in CI/CD pipeline
   - Document expected artifact version

## Common Migration Issues

### Issue 1: Missing Special Symbols

**Symptom:** Legacy artifact has no special tokens defined.

**Solution:**
```python
# Add default special symbols
special_tokens = {
    "UNK": {"token": "<UNK>", "token_id": 0, ...},
    "PAD": {"token": "<PAD>", "token_id": 1, ...},
    "BOS": {"token": "<BOS>", "token_id": 2, ...},
    "EOS": {"token": "<EOS>", "token_id": 3, ...}
}
```

### Issue 2: Non-Normalized Coordinates

**Symptom:** Validation fails with "not unit-normalized" errors.

**Solution:**
```python
import numpy as np

# Normalize all coordinates
for i in range(len(coords)):
    norm = np.linalg.norm(coords[i])
    if norm > 1e-10:
        coords[i] = coords[i] / norm
```

### Issue 3: NaN/Inf Values

**Symptom:** Validation fails with "contains NaN or inf" errors.

**Solution:**
```python
# Replace NaN/inf with zeros, then normalize
coords = np.nan_to_num(coords, nan=0.0, posinf=1.0, neginf=-1.0)
for i in range(len(coords)):
    coords[i] = coords[i] / (np.linalg.norm(coords[i]) + 1e-10)
```

### Issue 4: Dimension Mismatch

**Symptom:** Different number of tokens vs coordinates.

**Solution:**
```python
# Ensure alignment
assert len(tokens) == len(coords), "Token count mismatch"

# If mismatch, rebuild one from the other
if len(tokens) < len(coords):
    coords = coords[:len(tokens)]
elif len(tokens) > len(coords):
    # Generate missing coordinates
    for token in tokens[len(coords):]:
        new_coord = coordizer._initialize_token_coordinate(token, len(coords))
        coords = np.vstack([coords, new_coord])
```

## Rollback Procedure

If migration fails:

1. **Restore from backup:**
   ```bash
   rm -rf path/to/artifact
   cp -r backups/$(date +%Y%m%d)/artifact_backup path/to/artifact
   ```

2. **Update code to handle legacy format:**
   ```python
   # Temporary compatibility layer
   def load_with_fallback(path):
       version = detect_artifact_version(path)
       if version == "1.0":
           coordizer.load(path, validate=True)
       else:
           # Legacy loading
           coordizer.load(path, validate=False)
   ```

3. **Schedule retry:**
   - Document the issue
   - Fix the root cause
   - Retry migration when ready

## Testing Checklist

After migration, verify:

- [ ] All files present (vocab.json, basin_coords.npy, coord_tokens.json)
- [ ] Version field is "1.0"
- [ ] All required metadata fields present
- [ ] Validation passes without errors
- [ ] Coordinates are unit-normalized
- [ ] Special symbols defined
- [ ] Token count matches coordinate count
- [ ] No NaN/inf values
- [ ] Load/save round-trip works
- [ ] Training scripts still work
- [ ] Generation/encoding works

## Best Practices

1. **Always backup before migration**
2. **Test on a sample artifact first**
3. **Validate after migration**
4. **Keep legacy backups for 30 days**
5. **Document any custom migration steps**
6. **Update team on format change**

## References

- Schema: `schemas/coordizer_artifact_v1.json`
- Validation: `qig-backend/artifact_validation.py`
- Documentation: `docs/20260116-coordizer-artifact-v1-schema.md`
- Work Package: WP3.3 - Standardize Artifact Format with Versioning
