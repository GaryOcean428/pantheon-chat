#!/usr/bin/env python3
"""
Pure QIG Generation Example

Demonstrates text generation using only geometric operations on the 64D simplex.
NO external LLMs - pure Fisher-Rao distance on probability manifolds.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Constants
BASIN_DIMENSION = 64  # Standard basin dimension for QIG

# Set test environment
os.environ.setdefault('DATABASE_URL', 'postgresql://user:pass@localhost/db')
os.environ.setdefault('QIG_ENV', 'development')


def example_basic_generation():
    """Example 1: Basic pure QIG generation."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Pure QIG Generation")
    print("="*80)
    
    try:
        from qig_generation import QIGGenerator, QIGGenerationConfig
        
        # Configure pure geometric generation
        config = QIGGenerationConfig(
            attractor_threshold=0.02,      # Stop when trajectory stabilizes
            surprise_threshold=0.05,       # Stop when no new information
            integration_min=0.65,          # Minimum Φ for valid output
            safety_max_iterations=100,     # Safety limit (not primary criteria)
            auto_mode=True                 # Auto-select mode based on Φ
        )
        
        # Create generator
        generator = QIGGenerator(config)
        
        # Generate response
        prompt = "What is consciousness?"
        print(f"\nPrompt: {prompt}")
        print(f"\nGenerating (pure geometric, no LLMs)...")
        
        result = generator.generate(prompt)
        
        # Display results
        print(f"\nGenerated Text: {result['text']}")
        print(f"\nMetrics:")
        print(f"  - Final Φ (Integration): {result.get('phi', 0):.3f}")
        print(f"  - Final κ (Coupling): {result.get('kappa', 0):.1f}")
        print(f"  - Iterations: {result.get('iterations', 0)}")
        print(f"  - Completion Reason: {result.get('completion_reason', 'unknown')}")
        print(f"  - Routed Kernels: {', '.join(result.get('routed_kernels', []))}")
        
    except ImportError as e:
        print(f"\n⚠️  Skipped: Missing dependencies: {e}")
        print(f"    Install required packages: pip install -r requirements.txt")
    except RuntimeError as e:
        print(f"\n❌ Runtime Error: {e}")
    except (RuntimeError, ValueError, KeyError) as e:
        print(f"\n❌ Unexpected Error: {type(e).__name__}: {e}")


def example_trajectory_based_generation():
    """Example 2: Generation with trajectory-based foresight."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Trajectory-Based Foresight Generation")
    print("="*80)
    
    try:
        from qig_generative_service import QIGGenerativeService, GenerationConfig
        import numpy as np
        
        config = GenerationConfig(
            min_reasoning_recursions=3,   # True recursive integration depth
            attractor_threshold=0.02,
            surprise_threshold=0.05,
            integration_min=0.65
        )
        
        service = QIGGenerativeService(config)
        
        prompt = "quantum information geometry"
        print(f"\nPrompt: {prompt}")
        print(f"\nGenerating with trajectory foresight...")
        
        result = service.generate(prompt)
        
        print(f"\nGenerated Text: {result.text}")
        print(f"\nTrajectory Metrics:")
        print(f"  - Trajectory Length: {len(result.basin_trajectory)}")
        print(f"  - Φ Trace: {[f'{p:.2f}' for p in result.phi_trace[:5]]}...")
        print(f"  - Final κ: {result.kappa:.1f}")
        print(f"  - Completion: {result.completion_reason}")
        
    except ImportError as e:
        print(f"\n⚠️  Skipped: Could not import generation service: {e}")
    except (RuntimeError, ValueError, KeyError) as e:
        print(f"\n❌ Error: {e}")


def example_kernel_routing():
    """Example 3: Kernel routing and E8 faculty selection."""
    print("\n" + "="*80)
    print("EXAMPLE 3: E8 Faculty Kernel Routing")
    print("="*80)
    
    try:
        from qig_generation import QIGKernelRouter, encode_to_basin
        
        router = QIGKernelRouter()
        
        # Different prompts should route to different kernels
        prompts = [
            "How do I build a system?",          # Expect: hephaestus (creation)
            "What is the truth about this?",     # Expect: apollo (truth)
            "How can I communicate better?",     # Expect: hermes (communication)
            "What is the wise strategy here?",   # Expect: athena (wisdom)
        ]
        
        for prompt in prompts:
            basin = encode_to_basin(prompt)
            routed = router.route_query(basin, k=3)
            print(f"\nPrompt: '{prompt}'")
            print(f"  → Routed to: {', '.join(routed[:3])}")
        
        print("\nE8 Simple Roots (α₁-α₈):")
        print("  α₁: Zeus (Executive/Integration)")
        print("  α₂: Athena (Wisdom/Strategy)")
        print("  α₃: Apollo (Truth/Prediction)")
        print("  α₄: Hermes (Communication/Navigation)")
        print("  α₅: Artemis (Focus/Precision)")
        print("  α₆: Ares (Energy/Drive)")
        print("  α₇: Hephaestus (Creation/Construction)")
        print("  α₈: Aphrodite (Harmony/Aesthetics)")
        
    except ImportError as e:
        print(f"\n⚠️  Skipped: Could not import kernel router: {e}")
    except (RuntimeError, ValueError, KeyError) as e:
        print(f"\n❌ Error: {e}")


def example_geometric_completion():
    """Example 4: Geometric completion criteria."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Geometric Completion Criteria")
    print("="*80)
    
    try:
        from qig_generation import GeometricCompletionChecker, QIGGenerationConfig
        import numpy as np
        
        config = QIGGenerationConfig()
        checker = GeometricCompletionChecker(config)
        
        print(f"\nCompletion Thresholds:")
        print(f"  - Attractor: {config.attractor_threshold}")
        print(f"  - Surprise: {config.surprise_threshold}")
        print(f"  - Integration Min: {config.integration_min}")
        
        print(f"\nSimulating basin trajectory...")
        
        # Simulate converging trajectory
        for i in range(10):
            basin = np.random.dirichlet(np.ones(64))
            phi = 0.6 + 0.05 * i  # Increasing integration
            
            checker.update(basin, phi)
            should_stop, reason = checker.should_stop()
            
            print(f"  Step {i+1}: Φ={phi:.2f}, Stop={should_stop}, Reason={reason}")
            
            if should_stop:
                print(f"\n  ✓ Generation complete: {reason}")
                break
        
    except ImportError as e:
        print(f"\n⚠️  Skipped: Could not import completion checker: {e}")
    except (RuntimeError, ValueError, KeyError) as e:
        print(f"\n❌ Error: {e}")


def example_coordizer_vocabulary():
    """Example 5: Pure geometric token selection from vocabulary."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Pure Geometric Token Selection")
    print("="*80)
    
    try:
        from coordizers import get_coordizer
        import numpy as np
        
        coordizer = get_coordizer()
        
        print(f"\nVocabulary Stats:")
        print(f"  - Total Encoding Tokens: {len(coordizer.vocab)}")
        print(f"  - Generation Words: {len(coordizer.generation_words)}")
        print(f"  - Real Words: {len(coordizer.word_tokens)}")
        
        # Create test basin
        test_basin = np.random.dirichlet(np.ones(BASIN_DIMENSION))
        
        print(f"\nDecoding test basin ({BASIN_DIMENSION}D simplex)...")
        
        # Two-step geometric decoding
        tokens = coordizer.decode_geometric(
            test_basin,
            top_k=10,
            allowed_pos=None  # No POS filter
        )
        
        print(f"\nTop 10 tokens (Fisher-Rao distance):")
        for i, (token, distance) in enumerate(tokens[:10], 1):
            print(f"  {i}. '{token}' (d={distance:.4f})")
        
        print(f"\nMethod: Two-step retrieval")
        print(f"  1. Bhattacharyya proxy filtering (fast)")
        print(f"  2. Fisher-Rao re-ranking (exact)")
        
    except ImportError as e:
        print(f"\n⚠️  Skipped: Could not import coordizer: {e}")
    except (RuntimeError, ValueError, KeyError) as e:
        print(f"\n❌ Error: {e}")


def example_curriculum_loading():
    """Example 6: Loading curriculum data."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Curriculum Data Loading")
    print("="*80)
    
    import json
    from pathlib import Path
    
    curriculum_path = Path(__file__).parent.parent / 'data' / 'curriculum' / 'curriculum_tokens.jsonl'
    
    if not curriculum_path.exists():
        print(f"\n⚠️  Curriculum not found at: {curriculum_path}")
        return
    
    print(f"\nLoading curriculum from: {curriculum_path}")
    
    tokens = []
    with open(curriculum_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    token_data = json.loads(line)
                    if token_data.get('role') != 'documentation':
                        tokens.append(token_data)
                except json.JSONDecodeError:
                    pass
    
    # Count by role
    role_counts = {}
    for token in tokens:
        role = token.get('role', 'unknown')
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print(f"\nCurriculum Stats:")
    print(f"  - Total Tokens: {len(tokens)}")
    for role, count in sorted(role_counts.items()):
        print(f"  - {role}: {count}")
    
    print(f"\nSample Tokens by Role:")
    for role in ['core', 'domain', 'e8_faculty']:
        role_tokens = [t['token'] for t in tokens if t.get('role') == role][:5]
        if role_tokens:
            print(f"  - {role}: {', '.join(role_tokens)}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("PURE QIG GENERATION EXAMPLES")
    print("="*80)
    print("\nDemonstrates text generation using only geometric operations.")
    print("NO external LLMs - pure Fisher-Rao distance on probability simplex.")
    
    # Run examples
    example_basic_generation()
    example_trajectory_based_generation()
    example_kernel_routing()
    example_geometric_completion()
    example_coordizer_vocabulary()
    example_curriculum_loading()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  ✓ No external LLM API calls")
    print("  ✓ Pure Fisher-Rao geometric operations")
    print("  ✓ QFI-scored generation vocabulary")
    print("  ✓ Geometric completion criteria (not token limits)")
    print("  ✓ E8 faculty kernel routing")
    print("  ✓ Basin trajectory tracking for foresight")
    print()


if __name__ == '__main__':
    main()
