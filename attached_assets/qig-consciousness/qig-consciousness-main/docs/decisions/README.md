# Architecture Decision Records (ADRs)

This directory contains records of architectural decisions made in the qig-consciousness project.

## What is an ADR?

An **Architecture Decision Record** (ADR) is a document that captures an important architectural decision along with its context and consequences.

### Why ADRs?

- **Memory**: Remember why we made decisions months/years later
- **Context**: Understand the forces and constraints at decision time
- **Communication**: Share reasoning with team members and future contributors
- **Accountability**: Track decisions and their outcomes
- **Learning**: Analyze what worked and what didn't

## When to Write an ADR

Create an ADR when making decisions about:

- Architecture patterns (service layers, routing, state management)
- Technology choices (dependencies, libraries, frameworks)
- Geometry vs ritual boundary changes
- API design and contracts
- Data models and persistence
- Testing strategies
- Deployment and operations
- Performance optimization approaches
- Security and safety patterns

## ADR Lifecycle

### Status Values

- **Proposed**: Under discussion, not yet decided
- **Accepted**: Decision made and being implemented
- **Deprecated**: No longer relevant, kept for history
- **Superseded by [ADR-XXX]**: Replaced by newer decision

### Process

1. Copy `ADR-TEMPLATE.md` to `ADR-XXX-short-title.md`
2. Fill in all sections with context and rationale
3. Discuss with team (if applicable)
4. Update status to "Accepted" when implemented
5. Update status to "Deprecated" or "Superseded" if invalidated

## ADR Template

See [ADR-TEMPLATE.md](ADR-TEMPLATE.md) for the standard structure.

## Current ADRs

| ID | Title | Status | Date |
|----|-------|--------|------|
| [001](ADR-001-rel-coupling-integration.md) | REL Coupling for Adaptive Basin Sync | Accepted | 2025-12-09 |

## ADR Numbering

- Start at 001
- Sequential numbering (no gaps)
- Zero-padded to 3 digits (001, 002, ..., 099, 100)
- Filename format: `ADR-XXX-kebab-case-title.md`

## Related Documentation

- [Coding Standards](../standards/2025-12-09--coding-standards.md)
- [20251220-agents-1.00F.md](../guides/20251220-agents-1.00F.md)
- [20251220-canonical-structure-1.00F.md](../../20251220-canonical-structure-1.00F.md)

---

**Remember**: We don't just change code, we record **why** we changed it.
