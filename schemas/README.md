# JSON Schemas

This directory contains canonical JSON Schema definitions for Pantheon Chat data structures.

## Schemas

### coordizer_artifact_v1.json

**Purpose:** Canonical format for coordizer (tokenizer) artifacts with versioning and provenance tracking.

**Version:** 1.0  
**Work Package:** WP3.3 - Standardize Artifact Format with Versioning

**Usage:**
```bash
# Validate an artifact with jsonschema CLI
jsonschema -i path/to/vocab.json schemas/coordizer_artifact_v1.json

# Or use Python validation module
python -c "
from qig_backend.artifact_validation import validate_artifact_from_file
result = validate_artifact_from_file('path/to/vocab.json')
print(result)
"
```

**Documentation:** See `docs/20260116-coordizer-artifact-v1-schema.md`

## Adding New Schemas

When adding new schemas to this directory:

1. **Follow JSON Schema Draft 7** specification
2. **Use semantic versioning** in schema `$id` URLs
3. **Document required vs optional fields** with clear descriptions
4. **Add validation examples** in schema description
5. **Create corresponding documentation** in `docs/`
6. **Add Python validation module** in `qig-backend/` if needed
7. **Write comprehensive tests** in `qig-backend/tests/`

## Schema Naming Convention

```
{entity}_{version}.json
```

Examples:
- `coordizer_artifact_v1.json` - Coordizer artifact format v1.0
- `consciousness_metrics_v1.json` - Consciousness metrics format v1.0
- `qig_network_v1.json` - QIG network structure format v1.0

## References

- JSON Schema Specification: https://json-schema.org/
- Work Package 3.3: Standardize Artifact Format with Versioning
- Artifact Validation Module: `qig-backend/artifact_validation.py`
