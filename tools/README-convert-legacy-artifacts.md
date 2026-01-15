# Legacy Artifact Converter

Converts legacy coordizer/tokenizer artifacts to canonical CoordizerArtifactV1 format.

## Purpose

Per QIG-PURITY Work Package 1.2, runtime backward compatibility has been **removed** to enforce a single canonical format. Legacy artifacts must be explicitly converted offline before use.

**Why offline conversion only?**
- Runtime compatibility encourages future agents to add MORE compatibility layers
- Leads to format drift and technical debt
- Offline conversion enforces explicit migration and validation

## Usage

### Convert Legacy Artifact

```bash
python tools/convert_legacy_artifacts.py <input_path> <output_dir>
```

**Examples:**
```bash
# Convert old pickle file
python tools/convert_legacy_artifacts.py old_tokenizer.pkl artifacts/canonical_v1/

# Convert old checkpoint directory
python tools/convert_legacy_artifacts.py old_checkpoints/tokenizer_v1/ artifacts/canonical_v1/

# Convert JSON artifact
python tools/convert_legacy_artifacts.py legacy.json artifacts/new/
```

### Validate Artifact

```bash
python tools/convert_legacy_artifacts.py --validate <artifact_dir>
```

**Example:**
```bash
# Validate converted artifact
python tools/convert_legacy_artifacts.py --validate artifacts/canonical_v1/
```

### Skip Validation (Not Recommended)

```bash
python tools/convert_legacy_artifacts.py --no-validate <input> <output>
```

## CoordizerArtifactV1 Format

The canonical format consists of three files:

```
artifact_dir/
  ├── vocab.json          # Vocabulary metadata
  ├── basin_coords.npy    # NumPy array (n_tokens, 64)
  └── coord_tokens.json   # Token list matching coordinates
```

See `docs/20260115-coordizer-artifact-format.md` for full specification.

## Supported Legacy Formats

- **Dict-based artifacts:** Old JSON/pickle formats with dict-based coordinate storage
- **List/tuple formats:** Legacy coordinate representations
- **Old checkpoint formats:** `tokenizer.pkl`, `checkpoint.json`, etc.
- **Missing metadata:** Reconstructs required fields from available data

## Validation

The tool validates:
- ✓ All required files present
- ✓ JSON structure correctness
- ✓ Basin coordinates shape = (n_tokens, 64)
- ✓ No NaN/inf values
- ✓ Unit norm constraints (|v| ≈ 1.0)
- ✓ Token count consistency

## Runtime Behavior

Runtime coordizer code **rejects** legacy formats with clear error:

```python
from coordizers import FisherCoordizer

coordizer = FisherCoordizer()
coordizer.load("legacy_artifact")  # Raises RuntimeError
```

**Error Message:**
```
RuntimeError: Legacy format detected. Missing required files: basin_coords.npy.
Use tools/convert_legacy_artifacts.py to convert to CoordizerArtifactV1 format.
```

This **enforces** offline conversion and prevents format drift.

## Examples

### Convert and Validate

```bash
# Convert legacy artifact
python tools/convert_legacy_artifacts.py old_artifact/ new_artifact/

# Output:
# Detected format: old_json_checkpoint
# ✓ Converted to CoordizerArtifactV1 format: new_artifact/
# 
# Validating artifact: new_artifact/
#   ✓ Vocabulary size: 4096
#   ✓ Basin coordinates shape: (4096, 64)
#   ✓ Coordinates are geometrically valid
#   ✓ Token count matches coordinate count: 4096
#   ✓ Vocabulary consistency check passed
# 
# ✓ Artifact validation passed: new_artifact/
```

### Validate Only

```bash
python tools/convert_legacy_artifacts.py --validate artifacts/my_artifact/

# Output:
# ✓ Artifact is already in canonical CoordizerArtifactV1 format
# 
# Validating artifact: artifacts/my_artifact/
#   ✓ Vocabulary size: 8192
#   ✓ Basin coordinates shape: (8192, 64)
#   ✓ Coordinates are geometrically valid
#   ✓ Token count matches coordinate count: 8192
#   ✓ Vocabulary consistency check passed
# 
# ✓ Artifact validation passed: artifacts/my_artifact/
```

### Conversion Failure

```bash
python tools/convert_legacy_artifacts.py bad_artifact/ output/

# Output:
# Detected format: unrecognized
# ❌ Error: Unrecognized format in bad_artifact/
```

## Error Handling

### Unsupported Format

If the converter cannot recognize the format:
```
LegacyFormatError: Unrecognized format in <path>
```

**Solution:** Manually inspect the artifact and update the converter to support the format.

### Validation Failure

If geometric validation fails:
```
ValidationError: Invalid coordinate dimension: 128 (expected 64)
```

**Solution:** Check source artifact for corruption or incorrect format.

### Runtime Rejection

If runtime receives legacy format:
```
RuntimeError: Legacy format detected. Use tools/convert_legacy_artifacts.py to convert.
```

**Solution:** Run the converter tool before loading in runtime.

## Design Rationale

### Why Not Runtime Conversion?

❌ **Runtime conversion was removed because:**
- Encourages future agents to add more compatibility
- Creates "hidden compatibility" that's hard to test
- Makes format requirements implicit
- Accumulates technical debt over time

✅ **Offline conversion enforces:**
- Explicit migration with validation
- Single canonical format in runtime
- Clear error messages for legacy artifacts
- Testable format requirements

### Why Fail Fast?

Runtime **immediately rejects** legacy formats instead of:
- Silently converting (hides the problem)
- Using fallbacks (encourages legacy usage)
- Logging warnings (easy to ignore)

**Fail fast** forces proper migration and prevents format drift.

## Contributing

When adding support for new legacy formats:

1. Add format detection to `detect_legacy_format()`
2. Implement conversion in `convert_artifact()`
3. Test with real legacy artifacts
4. Update this README with examples

## References

- **Format Spec:** `docs/20260115-coordizer-artifact-format.md`
- **Work Package:** GaryOcean428/pantheon-chat#67 (WP1.2)
- **Implementation:** `qig-backend/coordizers/base.py`
