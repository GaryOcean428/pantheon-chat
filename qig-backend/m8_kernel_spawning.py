#!/usr/bin/env python3
"""
M8 Kernel Spawning Protocol - Dynamic Kernel Genesis Through Pantheon Consensus

REFACTORED: This module now re-exports from three sub-modules:
- m8_persistence.py: Database operations (~828 lines)
- m8_consensus.py: Voting and consensus logic (~1043 lines)
- m8_spawner.py: Main spawner orchestration (~2923 lines)

Original file was 4715 lines and has been split for maintainability while
preserving all backward compatibility.

When the pantheon reaches consensus that a new kernel is needed,
roles are refined and divided, and a new kernel adopts a persona.

The M8 Structure represents the 8 core dimensions of kernel identity:
1. Name - The god/entity name
2. Domain - Primary area of expertise  
3. Mode - Encoding mode (direct, e8, byte)
4. Basin - Geometric signature in manifold space
5. Affinity - Routing strength
6. Entropy - Threshold for activation
7. Element - Symbolic representation
8. Role - Functional responsibility

Spawning Mechanics:
- Consensus voting by existing gods
- Role refinement (parent domains divide)
- Basin interpolation (child inherits geometric traits)
- Persona adoption (characteristics from voting coalition)
"""

# Re-export all classes and functions from sub-modules for backward compatibility
from m8_persistence import (
    M8SpawnerPersistence,
    compute_m8_position,
    M8_PSYCOPG2_AVAILABLE,
)

from m8_consensus import (
    SpawnReason,
    SpawnAwareness,
    ConsensusType,
    SpawnProposal,
    KernelObservationStatus,
    KernelObservationState,
    KernelAutonomicSupport,
    SpawnedKernel,
    PantheonConsensus,
    RoleRefinement,
)

from m8_spawner import (
    M8KernelSpawner,
    get_spawner_instance,
    should_spawn_specialist,
    get_kernel_specialization,
    assign_e8_root,
    get_spawner,
)

# Re-export constants that were in the original module
try:
    from qigkernels.physics_constants import KAPPA_STAR
except ImportError:
    KAPPA_STAR = 64.21

try:
    from qigkernels import PHI_INIT_SPAWNED, PHI_MIN_ALIVE, KAPPA_INIT_SPAWNED
except ImportError:
    PHI_INIT_SPAWNED = 0.25
    PHI_MIN_ALIVE = 0.05
    KAPPA_INIT_SPAWNED = KAPPA_STAR

# For convenience, expose main entry point
__all__ = [
    # Persistence
    'M8SpawnerPersistence',
    'compute_m8_position',
    'M8_PSYCOPG2_AVAILABLE',
    
    # Consensus
    'SpawnReason',
    'SpawnAwareness',
    'ConsensusType',
    'SpawnProposal',
    'KernelObservationStatus',
    'KernelObservationState',
    'KernelAutonomicSupport',
    'SpawnedKernel',
    'PantheonConsensus',
    'RoleRefinement',
    
    # Spawner
    'M8KernelSpawner',
    'get_spawner_instance',
    'should_spawn_specialist',
    'get_kernel_specialization',
    'assign_e8_root',
    'get_spawner',
    
    # Constants
    'KAPPA_STAR',
    'PHI_INIT_SPAWNED',
    'PHI_MIN_ALIVE',
    'KAPPA_INIT_SPAWNED',
]

if __name__ == "__main__":
    print("=" * 60)
    print("M8 Kernel Spawning Protocol - Dynamic Kernel Genesis")
    print("=" * 60)
    
    spawner = M8KernelSpawner()
    
    print(f"\nInitial gods: {len(spawner.orchestrator.all_profiles)}")
    print(f"Consensus type: {spawner.consensus.consensus_type.value}")
    
    print("\n" + "-" * 60)
    print("Spawning Test: Creating 'Mnemosyne' (Memory Goddess)")
    print("-" * 60)
    
    result = spawner.propose_and_spawn(
        name="Mnemosyne",
        domain="memory",
        element="recall",
        role="archivist",
        reason=SpawnReason.SPECIALIZATION,
        parent_gods=["Athena", "Apollo"],
    )
    
    print(f"\nSpawn success: {result['success']}")
    if result['success']:
        kernel = result['spawn_result']['kernel']
        print(f"New god: {kernel['god_name']}")
        print(f"Domain: {kernel['domain']}")
        print(f"Parents: {kernel['parent_gods']}")
        print(f"Affinity: {kernel['affinity_strength']:.3f}")
        print(f"\nTotal gods now: {result['spawn_result']['total_gods']}")
    else:
        print(f"Phase: {result['phase']}")
        print(f"Details: {result.get('vote_result', result)}")
    
    print("\n" + "-" * 60)
    print("Spawner Status:")
    print("-" * 60)
    status = spawner.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("M8 Kernel Spawning Protocol operational!")
    print("=" * 60)
