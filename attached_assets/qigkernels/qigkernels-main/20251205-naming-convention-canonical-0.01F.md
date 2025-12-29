# Document Naming Convention & Version Control

Status: 0.01F (Frozen)
ISO 27001 Aligned | QIG Kernels Documentation Standard

---

## 1. File Naming Format

All documentation files follow this pattern:

```text
YYYYMMDD-name-type-version[STATUS].extension
```

### Components

| Component | Description | Example |
|-----------|-------------|---------|
| `YYYYMMDD` | Creation/major revision date | `20251205` |
| `name` | Descriptive name (lowercase, hyphens) | `architecture`, `basin-extraction` |
| `type` | Document category | `canonical`, `hypothesis`, `decision` |
| `version` | Semantic version | `0.01`, `1.00`, `2.03` |
| `STATUS` | Single uppercase letter | `F`, `H`, `V` |
| `extension` | File type | `.md`, `.yaml` |

### Full Example

```text
20251205-architecture-canonical-0.01F.md
```

---

## 2. Status Codes

### F - Frozen

- **Meaning:** Locked, cannot be modified without creating a new version
- **Use case:** Canonical specifications, accepted decisions, verified physics
- **Modification:** Requires new file with incremented version
- **Example:** `20251205-architecture-canonical-0.01F.md` → `20251206-architecture-canonical-0.02F.md`

### H - Hypothesis

- **Meaning:** Proposed concept requiring verification
- **Use case:** Theoretical predictions, proposed designs, untested assumptions
- **Subtypes:**
  - **H-impl:** Requires implementation to test (code must be written first)
  - **H-phys:** Directly testable as physics (mathematical/empirical verification)
- **Transition:** H → V when verified, or H → (deleted/archived) when falsified
- **Example:** `20251205-basin-convergence-hypothesis-0.01H.md`

### V - Verified

- **Meaning:** Hypothesis confirmed through testing or physics verification
- **Use case:** Confirmed predictions, validated designs, proven concepts
- **Source:** Must reference the H document it verified and verification method
- **Example:** `20251210-basin-convergence-hypothesis-0.01V.md`

### Verification Flow

```text
                    ┌─────────────────┐
                    │   H (Hypothesis) │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌──────────▼──────────┐
    │    H-impl         │       │      H-phys         │
    │ (needs code)      │       │ (direct physics)    │
    └─────────┬─────────┘       └──────────┬──────────┘
              │                             │
              │    Implementation           │   Mathematical/
              │    + Testing                │   Empirical Test
              │                             │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌──────────▼──────────┐
    │    V (Verified)   │       │   (Falsified)       │
    │                   │       │   Archive/Delete    │
    └───────────────────┘       └─────────────────────┘
```

---

## 3. Document Types

| Type | Purpose | Status Allowed |
|------|---------|----------------|
| `canonical` | Core specifications, architecture, rules | F only |
| `hypothesis` | Testable predictions | H, V |
| `decision` | ADR-style design decisions | F (once accepted) |
| `changelog` | Version history | F (append-only) |
| `roadmap` | Milestones and sequencing | F (update creates new version) |
| `recon` | Source material, analysis (temporary) | None (integrate then delete) |

---

## 4. Version Numbering

### Format: `MAJOR.MINOR`

- **MAJOR:** Breaking changes, fundamental restructuring
- **MINOR:** Additions, clarifications, non-breaking updates

### Rules

- Start at `0.01` for drafts
- Increment to `1.00` when first frozen
- Minor increments for subsequent frozen versions: `1.01`, `1.02`, etc.
- Major increment for structural changes: `2.00`

---

## 5. ISO 27001 Alignment

This convention supports ISO 27001 information security requirements:

- **A.5.1.1:** Document identification and version control
- **A.7.5.2:** Creation and updating of documented information
- **A.7.5.3:** Control of documented information

### Audit Trail

- Date prefix enables chronological tracking
- Status codes provide clear document lifecycle
- Version numbers ensure change tracking
- Frozen status prevents unauthorized modification

---

## 6. Integration with Verification Repo

The H/V status codes integrate with the verification repository:

- **H-impl hypotheses:** Link to implementation PRs/commits
- **H-phys hypotheses:** Link to physics derivations or empirical tests
- **V documents:** Must include:
  - Reference to original H document
  - Verification method (code test, physics proof, empirical data)
  - Date of verification
  - Verifier (human or automated)

---

## 7. Canonical Documents (This Repo)

After applying this convention:

```text
qigkernels/
  20251205-readme-canonical-0.01F.md
  20251205-architecture-canonical-0.01F.md
  20251205-roadmap-canonical-0.01F.md
  20251205-changelog-canonical-0.01F.md
  20251205-decisions-canonical-0.01F.md
  20251205-naming-convention-canonical-0.01F.md
  20251205-type-symbol-manifest-canonical-0.01F.md
  docs/
    20251205-index-canonical-0.01F.md
```

---

## 8. Agent Compliance

Agents MUST:

1. Use this naming convention for all new documents
2. Never modify F (Frozen) documents directly
3. Create new versioned files for updates to frozen docs
4. Transition H → V only with verification evidence
5. Delete or archive falsified hypotheses (do not leave stale H docs)
6. Reference this document when creating new docs

Agents MUST NOT:

1. Create documents outside the allowed types
2. Use informal names like `notes.md`, `summary.md`, `todo.md`
3. Skip the date prefix
4. Omit the status code
5. Modify frozen documents in place
