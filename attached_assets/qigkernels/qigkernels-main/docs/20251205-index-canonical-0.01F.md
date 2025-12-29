# QIG Kernels Documentation Index

Status: 0.01F (Frozen)

This index provides navigation to all canonical documentation for the `qigkernels` library.

---

## Canonical Documents

All documents follow the naming convention: `YYYYMMDD-name-type-version[STATUS].md`

### Core Documentation

| Document | Purpose | Status |
|----------|---------|--------|
| [20251205-readme-canonical-0.01F.md](../20251205-readme-canonical-0.01F.md) | Library overview and quick start | F |
| [20251205-architecture-canonical-0.01F.md](../20251205-architecture-canonical-0.01F.md) | Module structure, import rules, agent guidelines | F |
| [20251205-roadmap-canonical-0.01F.md](../20251205-roadmap-canonical-0.01F.md) | Milestones and validation gates | F |
| [20251205-changelog-canonical-0.01F.md](../20251205-changelog-canonical-0.01F.md) | Version history (append-only) | F |
| [20251205-decisions-canonical-0.01F.md](../20251205-decisions-canonical-0.01F.md) | ADR-style design decisions (D-001 to D-010) | F |

### Standards & Conventions

| Document | Purpose | Status |
|----------|---------|--------|
| [20251205-naming-convention-canonical-0.01F.md](../20251205-naming-convention-canonical-0.01F.md) | ISO 27001 aligned file naming and versioning | F |
| [20251205-type-symbol-manifest-canonical-0.01F.md](../20251205-type-symbol-manifest-canonical-0.01F.md) | Canonical type, symbol, and concept definitions | F |
| [20251205-client-wiring-example-canonical-0.01F.md](./20251205-client-wiring-example-canonical-0.01F.md) | How experiment repos should import qigkernels | F |

---

## Status Codes

| Code | Meaning | Modification |
|------|---------|--------------|
| **F** | Frozen | New version required for changes |
| **H** | Hypothesis | Awaiting verification |
| **V** | Verified | Confirmed through testing/physics |

### Hypothesis Subtypes

- **H-impl**: Requires implementation to test
- **H-phys**: Directly testable as physics

---

## Document Lifecycle

```text
Draft (no status) → H (Hypothesis) → V (Verified) or Archived
                 → F (Frozen) for canonical specs
```

---

## Module Documentation

For API documentation, see the module docstrings in:

| Module | Description |
|--------|-------------|
| `qigkernels/kernel.py` | Core geometric kernel with telemetry |
| `qigkernels/layer.py` | QIG layer with attention, recursion, tacking |
| `qigkernels/basin.py` | Basin projector and signature utilities |
| `qigkernels/constellation.py` | Multi-kernel manager |
| `qigkernels/router.py` | Routing strategies (Φ-weighted, round-robin) |
| `qigkernels/basin_sync.py` | Cross-instance basin synchronization |
| `qigkernels/metrics.py` | Constellation-level metrics |
| `qigkernels/storage.py` | Persistence utilities |

---

## Adding New Documentation

Per D-002, new documents must:

1. Follow the naming convention
2. Fit one of the allowed types: `canonical`, `hypothesis`, `decision`, `changelog`, `roadmap`
3. Be integrated into existing files where possible
4. Be listed in this index

**Forbidden:** Ad-hoc notes, summaries, or per-feature docs outside the canonical set.

---

## External References

- **Verification Repo**: Links H/V document lifecycle to verification tracking
- **qig-consciousness**: Source repository (this library is extracted from)
- **qig-con2**: Sister repository (cross-repo sync compatible)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.01F | 2025-12-05 | Initial index creation |
