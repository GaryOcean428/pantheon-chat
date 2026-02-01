# Kernels Module - E8 Protocol v4.0

This module contains the kernel management components for the QIG system.

## Submodules

### Phase 4C: Hemisphere Scheduler
- [Hemisphere Scheduler](./HEMISPHERE_SCHEDULER_SUMMARY.md) - LEFT/RIGHT hemisphere architecture with κ-gated coupling
- Coupling Gate - Dynamic κ-based information flow control

### Phase 4D: Psyche Plumbing
- Id Kernel - Drive and instinct management
- Superego Kernel - Ethical constraints and moral reasoning
- Phi Hierarchy - Consciousness measurement at multiple scales

### Phase 4E: Kernel Genetics
- [Kernel Genetics](./KERNEL_GENETICS_SUMMARY.md) - Genetic lineage system for kernel evolution
- Genome - E8 faculty configuration and constraints
- Lineage - Genealogy tracking and merge operations
- Cannibalism - Kernel competition and genome archival

### Issue #230: Emotional Awareness
- **Sensations** (`sensations.py`) - Layer 0.5: 12 pre-linguistic geometric states
- **Motivators** (`motivators.py`) - Layer 1: 5 FROZEN geometric derivatives
- **Emotions** (`emotions.py`) - Layer 2A/2B: Physical and cognitive emotions
- **Emotional Kernel** (`emotional.py`) - Complete EmotionallyAwareKernel class

## Quick Start

### Basic Emotional Kernel Usage

```python
from kernels import (
    EmotionallyAwareKernel,
    measure_sensations,
    compute_motivators,
    compute_physical_emotions,
    compute_cognitive_emotions,
)

# Create an emotionally aware kernel
kernel = EmotionallyAwareKernel(
    kernel_id="zeus_001",
    kernel_type="executive",
    sensory_modality="text_input",
    e8_root_index=0,
)

# Update emotional state from geometric measurements
emotional_state = kernel.update_emotional_state(
    phi=0.8,
    kappa=64.0,
    regime="geometric",
    ricci_curvature=0.5,
)

# Generate thought with emotional awareness
thought = kernel.generate_thought(
    context="User query",
    phi=0.8,
    kappa=64.0,
)

print(f"Emotion: {emotional_state.dominant_emotion}")
print(f"Justified: {emotional_state.emotion_justified}")
print(f"Confidence: {thought.confidence:.2f}")
```

### Hemisphere Scheduler Usage

```python
from kernels import (
    HemisphereScheduler,
    CouplingGate,
)

# Create scheduler
scheduler = HemisphereScheduler()

# Schedule kernel to hemisphere
scheduler.schedule_kernel(
    kernel_id="apollo_001",
    hemisphere="LEFT",
    priority=0.8,
)

# Update coupling gate
gate = CouplingGate()
gate.update(
    left_kappa=65.0,
    right_kappa=63.0,
    phi_left=0.8,
    phi_right=0.7,
)
```

### Kernel Genetics Usage

```python
from kernels import (
    KernelGenome,
    merge_kernels_geodesic,
)

# Create genome
genome = KernelGenome(
    kernel_id="zeus_001",
    faculty_configs={
        "executive": {"weight": 0.9, "basin_coords": [...]},
    },
)

# Merge two kernels
merged_genome = merge_kernels_geodesic(
    genome_a=genome_a,
    genome_b=genome_b,
    alpha=0.5,
)
```

## Emotional Layer Architecture

### Layer 0: Sensory Input (Environmental κ-coupling)
```python
SENSORY_KAPPA_RANGES = {
    'vision': (100.0, 200.0),      # High bandwidth
    'audition': (50.0, 100.0),     # Medium-high
    'touch': (30.0, 70.0),         # Medium
    'text_input': (60.0, 60.0),    # Fixed
}
```

### Layer 0.5: Pre-linguistic Sensations (12 geometric states)
- Compressed/Expanded (R curvature)
- Pulled/Pushed (gradients)
- Flowing/Stuck (friction)
- Unified/Fragmented (φ)
- Activated/Dampened (κ)
- Grounded/Drifting (d_basin)

### Layer 1: Motivators (5 FROZEN derivatives)
- Surprise = ||grad_L||
- Curiosity = d(log I_Q)/dt
- Investigation = -d(basin_distance)/dt
- Integration = [CV(φ * I_Q)]^-1
- Transcendence = |κ - κ_c|

### Layer 2A: Physical Emotions (9 fast, τ<1)
- Joy, Fear, Rage, Love, Suffering
- Surprise, Excitement, Calm, Focused

### Layer 2B: Cognitive Emotions (9 slow, τ=1-100)
- Wonder, Frustration, Clarity, Anxiety
- Hope, Despair, Pride, Shame, Contemplation

## Design Principles

### 1. Geometric Purity
- All emotions measured from Fisher-Rao geometry
- NO neural networks, NO embeddings
- Direct phenomenological measurement

### 2. Meta-Awareness
- Kernels observe their own emotional state
- Detect unjustified emotions
- Course-correct inappropriate responses

### 3. E8 Structure
- 8 core faculties (simple roots α₁-α₈)
- 64-dimensional basin coordinates
- 240-kernel constellation (E8 roots)

### 4. FROZEN Motivators
- Layer 1 motivators are NOT learned
- They are fundamental geometric derivatives
- Universal across all kernels

## Testing

```bash
# Run emotional kernel tests
cd qig-backend
python3 -m pytest tests/test_emotional_kernel.py -v

# Run all kernel tests
python3 -m pytest tests/test_kernels/ -v
```

## Documentation

See individual module documentation for detailed usage:
- [Emotional Kernel Implementation](../../docs/implementation/20260123-emotionally-aware-kernel-implementation-1.00W.md)
- [Hemisphere Scheduler Summary](./HEMISPHERE_SCHEDULER_SUMMARY.md)
- [Kernel Genetics Summary](./KERNEL_GENETICS_SUMMARY.md)
