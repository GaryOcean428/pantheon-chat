# Implementation Guides

This directory contains implementation guides for developing features in pantheon-chat.

## Available Guides

### [Geometric Operations](./20250101-geometric-operations-v1.md)

Canonical patterns for implementing geometric operations:
- Fisher-Rao distance computation
- Geodesic interpolation
- Nearest neighbor search
- Consciousness metric computation
- Two-step retrieval

**Key Principle**: NO Euclidean operations on basin coordinates. Always use `fisher_rao_distance()`.

## Guide Naming Convention

Guides follow ISO-aligned naming:
```
YYYYMMDD-topic-name-version[STATUS].md
```

Statuses:
- `DRAFT` - Work in progress
- `FINAL` - Approved and stable
- `DEPRECATED` - No longer recommended

## Related Documentation

- [CANONICAL_PHYSICS.md](../CANONICAL_PHYSICS.md) - Physics foundations
- [CANONICAL_ARCHITECTURE.md](../CANONICAL_ARCHITECTURE.md) - Architecture patterns
- [FROZEN_FACTS.md](../FROZEN_FACTS.md) - Validated constants
