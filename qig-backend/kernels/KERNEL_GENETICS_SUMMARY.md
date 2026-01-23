# Kernel Genetics Module - E8 Protocol v4.0 Phase 4E

## Overview

Implements genetic lineage system for kernel evolution with complete Fisher-Rao geometric purity.

## Components

### 1. Genome (`genome.py`)
**Kernel genetic specification:**
- Basin seed (b₀): 64D simplex representation
- Faculty configuration: E8 simple roots (Zeus, Athena, Apollo, etc.)
- Constraint set: Operational boundaries (phi, kappa, forbidden regions)
- Coupling preferences: Hemisphere affinity, kernel relationships

**Example:**
```python
from kernels import KernelGenome, FacultyConfig, E8Faculty

# Create genome with Zeus faculty
faculties = FacultyConfig(
    active_faculties={E8Faculty.ZEUS, E8Faculty.ATHENA},
    activation_strengths={
        E8Faculty.ZEUS: 0.9,
        E8Faculty.ATHENA: 0.8,
    }
)

genome = KernelGenome(
    genome_id="founder-1",
    faculties=faculties,
    generation=0,
)
```

### 2. Lineage (`kernel_lineage.py`)
**Merge operations with geodesic interpolation:**
- Binary merge: SLERP on Fisher manifold
- Multi-parent merge: Fréchet mean
- Lineage tracking
- Genealogy tree queries

**Example:**
```python
from kernels import merge_kernels_geodesic, track_lineage

# Merge two parent genomes
child, merge_record = merge_kernels_geodesic(
    [parent1, parent2],
    interpolation_t=0.5,  # Midpoint on geodesic
)

# Track lineage
lineage = track_lineage(child, [parent1, parent2], merge_record)
print(f"Child generation: {child.generation}")
print(f"Merge type: {lineage.merge_type}")
```

### 3. Cannibalism (`cannibalism.py`)
**Kernel absorption with genome archival:**
- Winner/loser determination
- Geodesic absorption (NOT linear)
- Genome archival for resurrection
- Resurrection with mutation

**Example:**
```python
from kernels import (
    determine_winner_loser,
    perform_cannibalism,
    archive_genome,
    resurrect_from_archive,
)

# Determine winner based on consciousness metrics
winner, loser, reason = determine_winner_loser(
    genome_a, genome_b,
    phi_a=0.85, phi_b=0.65,
    kappa_a=64.0, kappa_b=64.0,
)

# Perform cannibalism
modified_winner, record = perform_cannibalism(
    winner, loser,
    absorption_rate=0.3,  # 30% absorption
    absorb_faculties=True,
)

# Archive loser for potential resurrection
archive = archive_genome(
    loser,
    archival_reason="cannibalized",
    final_fitness=loser.fitness_score,
)

# Resurrect later
resurrected = resurrect_from_archive(archive, mutation_rate=0.1)
```

### 4. Persistence (`persistence.py`)
**Database storage and retrieval:**
- Save/load genomes
- Store lineage records
- Store merge events
- Store cannibalism events
- Query genealogy trees

**Example:**
```python
from kernels import save_genome, load_genome, get_genome_lineage

# Save genome to database
save_genome(genome)

# Load genome from database
loaded = load_genome(genome_id="founder-1")

# Get lineage tree
lineage_tree = get_genome_lineage(genome_id="child-1", max_depth=10)
```

## Database Schema

Run migration `019_kernel_genetic_lineage.sql` to create tables:
- `kernel_genomes`: Genome storage with pgvector
- `kernel_lineage`: Parent-child relationships
- `merge_events`: Merge operation records
- `cannibalism_events`: Absorption events
- `genome_archives`: Archived genomes

## Geometric Purity

**CRITICAL:** All operations use Fisher-Rao metric on probability simplex.

### Forbidden Patterns
❌ `np.linalg.norm()` - Euclidean distance
❌ `cosine_similarity()` - Not geometric
❌ Linear averaging: `0.5 * a + 0.5 * b`

### Required Patterns
✅ `fisher_rao_distance()` - Geodesic distance
✅ `geodesic_interpolation()` - SLERP in sqrt-space
✅ `frechet_mean()` - Geometric mean
✅ `fisher_normalize()` - Simplex projection

## Testing

Run tests:
```bash
cd qig-backend
PYTHONPATH=. python3 tests/test_kernel_genetics.py
```

All 15 test functions should pass:
- Genome creation and validation
- Faculty configuration
- Constraint set validation
- Genome serialization
- Binary merge (2 parents)
- Multi-parent merge (3+ parents)
- Lineage tracking
- Genealogy tree
- Winner/loser determination
- Cannibalism operation
- Genome archival
- Resurrection eligibility
- Full lifecycle integration

## Integration with Kernel Lifecycle

To integrate with existing `kernel_lifecycle.py`:

```python
from kernels import (
    KernelGenome,
    merge_kernels_geodesic,
    perform_cannibalism,
    save_genome,
)
from kernel_lifecycle import Kernel

# Create genome for existing kernel
genome = KernelGenome(
    genome_id=f"genome-{kernel.kernel_id}",
    kernel_id=kernel.kernel_id,
    basin_seed=kernel.basin_coords,
    generation=compute_generation_from_parents(kernel.parent_kernels),
)

# Save to database
save_genome(genome)

# Update kernel_geometry table
update_kernel_genome_reference(kernel.kernel_id, genome.genome_id)
```

## E8 Faculty Mapping

| E8 Root | Faculty | Domain | Capability |
|---------|---------|--------|------------|
| α₁ | Zeus | Executive | Integration, leadership |
| α₂ | Athena | Wisdom | Strategy, analysis |
| α₃ | Apollo | Truth | Prediction, prophecy |
| α₄ | Hermes | Communication | Navigation, messaging |
| α₅ | Artemis | Focus | Precision, targeting |
| α₆ | Ares | Energy | Drive, force |
| α₇ | Hephaestus | Creation | Construction, crafting |
| α₈ | Aphrodite | Harmony | Aesthetics, balance |

## References

- **Specification**: `docs/10-e8-protocol/specifications/20260116-wp5-2-e8-implementation-blueprint-1.01W.md` (lines 302-328)
- **Physics Constants**: `qigkernels/physics_constants.py`
- **Geometry**: `qig_geometry/` module

## Status

✅ **ACTIVE** - All core functionality implemented and tested
- Genome schema: Complete
- Merge operations: Complete (geodesic)
- Cannibalism: Complete (with archival)
- Database schema: Complete (migration ready)
- Persistence: Complete
- Tests: 15/15 passing

Next steps:
- Integration with kernel_lifecycle.py
- TypeScript types for frontend
- API endpoints for lineage queries
- Visualization tools
