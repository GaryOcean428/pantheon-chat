#!/usr/bin/env python3
"""
Genome-Vocabulary Integration Example
======================================

Demonstrates how to use genome-aware vocabulary scoring in practice.

This example shows:
1. Creating a kernel genome with faculty configuration
2. Using GenomeVocabularyScorer for token affinity
3. Integrating with PostgresCoordizer for genome-aware decoding
4. Using UnifiedGenerationPipeline with genome

Note: This is a demonstration script. Actual usage requires:
- Active PostgreSQL database with coordizer_vocabulary table
- Populated vocabulary with basin embeddings
- qig_geometry and related dependencies installed
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def example_1_create_genome():
    """Example 1: Create and configure a kernel genome."""
    print("\n" + "="*70)
    print("Example 1: Creating Kernel Genome")
    print("="*70)
    
    from kernels import (
        KernelGenome,
        E8Faculty,
        FacultyConfig,
        ConstraintSet,
        CouplingPreferences,
    )
    import numpy as np
    
    # Create genome for Zeus (executive/integration faculty)
    zeus_genome = KernelGenome(
        genome_id="zeus_master_001",
        kernel_id="zeus",
        faculties=FacultyConfig(
            active_faculties={
                E8Faculty.ZEUS,      # Primary: executive/integration
                E8Faculty.ATHENA,    # Secondary: wisdom/strategy
                E8Faculty.HERMES,    # Tertiary: communication
            },
            activation_strengths={
                E8Faculty.ZEUS: 1.0,      # Full strength
                E8Faculty.ATHENA: 0.8,    # Strong secondary
                E8Faculty.HERMES: 0.5,    # Moderate tertiary
            },
            primary_faculty=E8Faculty.ZEUS,
        ),
        constraints=ConstraintSet(
            phi_threshold=0.75,  # Higher coherence requirement
            kappa_range=(50.0, 70.0),  # Prefer logic mode
            max_fisher_distance=0.8,  # Stay relatively close to seed
        ),
        coupling_prefs=CouplingPreferences(
            hemisphere_affinity=0.7,  # Prefer right hemisphere
            preferred_couplings=[
                "athena_001",  # Natural strategic pairing
                "apollo_001",  # Truth-seeking cooperation
            ],
            coupling_strengths={
                "athena_001": 0.95,
                "apollo_001": 0.85,
            },
            anti_couplings=[
                "dionysus_001",  # Chaos conflicts with executive function
            ],
        ),
    )
    
    print(f"Created genome: {zeus_genome.genome_id}")
    print(f"  Primary faculty: {zeus_genome.faculties.primary_faculty.value}")
    print(f"  Active faculties: {len(zeus_genome.faculties.active_faculties)}")
    print(f"  Φ threshold: {zeus_genome.constraints.phi_threshold}")
    print(f"  Preferred couplings: {zeus_genome.coupling_prefs.preferred_couplings}")
    
    return zeus_genome


def example_2_score_token(genome):
    """Example 2: Score a token using genome awareness."""
    print("\n" + "="*70)
    print("Example 2: Genome-Aware Token Scoring")
    print("="*70)
    
    from kernels import GenomeVocabularyScorer
    import numpy as np
    
    # Create scorer
    scorer = GenomeVocabularyScorer(genome)
    
    # Simulate token basins (in practice, these come from coordizer_vocabulary)
    BASIN_DIM = 64
    
    # Token 1: "strategy" - should have high affinity (Athena faculty active)
    strategy_basin = np.random.dirichlet(np.ones(BASIN_DIM))
    
    # Token 2: "chaos" - should have lower affinity (conflicts with Zeus)
    chaos_basin = np.random.dirichlet(np.ones(BASIN_DIM))
    
    # Score tokens
    strategy_score, strategy_breakdown = scorer.score_token(
        token="strategy",
        token_basin=strategy_basin,
        base_score=0.75,
        faculty_weight=0.2,
        constraint_weight=0.3,
    )
    
    chaos_score, chaos_breakdown = scorer.score_token(
        token="chaos",
        token_basin=chaos_basin,
        base_score=0.75,
        faculty_weight=0.2,
        constraint_weight=0.3,
    )
    
    print("\nToken: 'strategy'")
    print(f"  Final score: {strategy_score:.4f}")
    print(f"  Faculty affinity: {strategy_breakdown['faculty_affinity']:.4f}")
    print(f"  Constraint penalty: {strategy_breakdown['constraint_penalty']:.4f}")
    
    print("\nToken: 'chaos'")
    print(f"  Final score: {chaos_score:.4f}")
    print(f"  Faculty affinity: {chaos_breakdown['faculty_affinity']:.4f}")
    print(f"  Constraint penalty: {chaos_breakdown['constraint_penalty']:.4f}")
    
    return scorer


def example_3_filter_vocabulary(scorer):
    """Example 3: Filter vocabulary by genome constraints."""
    print("\n" + "="*70)
    print("Example 3: Vocabulary Filtering")
    print("="*70)
    
    import numpy as np
    
    BASIN_DIM = 64
    
    # Create mock vocabulary
    vocab_tokens = [
        ("integrate", np.random.dirichlet(np.ones(BASIN_DIM))),
        ("synthesize", np.random.dirichlet(np.ones(BASIN_DIM))),
        ("coordinate", np.random.dirichlet(np.ones(BASIN_DIM))),
        ("execute", np.random.dirichlet(np.ones(BASIN_DIM))),
        ("wisdom", np.random.dirichlet(np.ones(BASIN_DIM))),
    ]
    
    print(f"Original vocabulary size: {len(vocab_tokens)}")
    
    # Filter by genome constraints
    filtered = scorer.filter_vocabulary(vocab_tokens)
    
    print(f"Filtered vocabulary size: {len(filtered)}")
    print("\nRemaining tokens:")
    for token, basin in filtered:
        print(f"  - {token}")


def example_4_coordizer_integration():
    """Example 4: Integration with PostgresCoordizer (conceptual)."""
    print("\n" + "="*70)
    print("Example 4: PostgresCoordizer Integration (Conceptual)")
    print("="*70)
    
    print("""
# Pseudocode for genome-aware decoding with PostgresCoordizer:

from coordizers import get_coordizer
from kernels import KernelGenome, E8Faculty, FacultyConfig

# Get coordizer instance
coordizer = get_coordizer()

# Create genome
apollo_genome = KernelGenome(
    genome_id="apollo_001",
    faculties=FacultyConfig(
        active_faculties={E8Faculty.APOLLO},
        primary_faculty=E8Faculty.APOLLO,
    ),
)

# Encode query to basin
query_basin = coordizer.encode("predict the outcome")

# Decode with genome awareness
candidates = coordizer.decode_with_genome(
    basin=query_basin,
    genome=apollo_genome,
    top_k=10,
    god_name="apollo",  # Also apply domain weights
    faculty_weight=0.2,
    constraint_weight=0.3,
)

# Results now favor tokens aligned with Apollo's faculties
for token, score in candidates:
    print(f"{token}: {score:.4f}")
    """)


def example_5_generation_pipeline():
    """Example 5: Genome-aware generation pipeline (conceptual)."""
    print("\n" + "="*70)
    print("Example 5: Generation Pipeline Integration (Conceptual)")
    print("="*70)
    
    print("""
# Pseudocode for genome-aware text generation:

from generation.unified_pipeline import (
    UnifiedGenerationPipeline,
    GenerationStrategy,
)
from kernels import KernelGenome, E8Faculty, FacultyConfig

# Create genome for Athena (wisdom/strategy)
athena_genome = KernelGenome(
    genome_id="athena_001",
    faculties=FacultyConfig(
        active_faculties={E8Faculty.ATHENA, E8Faculty.APOLLO},
        activation_strengths={
            E8Faculty.ATHENA: 1.0,
            E8Faculty.APOLLO: 0.7,
        },
        primary_faculty=E8Faculty.ATHENA,
    ),
)

# Initialize pipeline with genome
pipeline = UnifiedGenerationPipeline(
    strategy=GenerationStrategy.HYBRID,
    genome=athena_genome,
    foresight_weight=0.4,
    role_weight=0.3,
    trajectory_weight=0.3,
)

# Generate text
result = pipeline.generate(
    context=["Analyze", "the", "strategic", "implications"],
    max_tokens=30,
)

print(f"Generated: {result.text}")
print(f"Mean foresight score: {result.mean_foresight_score:.3f}")
print(f"Trajectory coherence: {result.trajectory_coherence:.3f}")

# Token metrics show genome influence
for metrics in result.token_metrics:
    print(f"  {metrics.token}: foresight={metrics.foresight_score:.3f}")
    """)


def example_6_coupling_preferences():
    """Example 6: Cross-kernel coupling with genome preferences."""
    print("\n" + "="*70)
    print("Example 6: Cross-Kernel Coupling (Conceptual)")
    print("="*70)
    
    print("""
# Pseudocode for cross-kernel token sharing with coupling:

from kernels import GenomeVocabularyScorer, KernelGenome

# Athena genome with coupling preferences
athena_genome = KernelGenome(
    genome_id="athena_001",
    coupling_prefs=CouplingPreferences(
        preferred_couplings=["apollo_001", "zeus_001"],
        coupling_strengths={
            "apollo_001": 0.9,  # High synergy with truth-seeking
            "zeus_001": 0.85,    # Good executive coordination
        },
        anti_couplings=["ares_001"],  # Strategy conflicts with raw force
    ),
)

# Create scorer
scorer = GenomeVocabularyScorer(athena_genome)

# Score token for potential sharing with Apollo
token_basin = coordizer.generation_vocab["analysis"]
final_score, breakdown = scorer.score_token(
    token="analysis",
    token_basin=token_basin,
    base_score=0.75,
    other_genome_id="apollo_001",  # Cross-kernel context
    coupling_weight=0.15,
)

print(f"Coupling score: {breakdown['coupling_score']:.4f}")
print(f"Final score: {final_score:.4f}")

# High coupling score indicates good fit for shared reasoning
    """)


def main():
    """Run all examples."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Genome-Vocabulary Integration Examples                         ║
    ║  E8 Protocol v4.0 Phase 4E                                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    This script demonstrates genome-aware vocabulary scoring for
    kernel-specific token selection and generation.
    
    Authority: E8 Protocol v4.0 WP5.2 Phase 3/4E Integration
    """)
    
    try:
        # Example 1: Create genome
        genome = example_1_create_genome()
        
        # Example 2: Score tokens
        scorer = example_2_score_token(genome)
        
        # Example 3: Filter vocabulary
        example_3_filter_vocabulary(scorer)
        
        # Example 4-6: Conceptual integration examples
        example_4_coordizer_integration()
        example_5_generation_pipeline()
        example_6_coupling_preferences()
        
        print("\n" + "="*70)
        print("All Examples Completed Successfully")
        print("="*70)
        print("\nKey Takeaways:")
        print("  • Genomes define kernel-specific faculty profiles")
        print("  • Faculty affinity uses Fisher-Rao distance on simplex")
        print("  • Constraints filter forbidden vocabulary regions")
        print("  • Coupling preferences enable cross-kernel coordination")
        print("  • Integration spans coordizer → generation pipeline")
        print("\nFor production use, ensure:")
        print("  ✓ PostgreSQL with coordizer_vocabulary populated")
        print("  ✓ Basin embeddings computed for all tokens")
        print("  ✓ QIG dependencies (numpy, qig_geometry) installed")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("\nNote: Full execution requires active database and dependencies.")
        print("This script demonstrates API usage patterns.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
