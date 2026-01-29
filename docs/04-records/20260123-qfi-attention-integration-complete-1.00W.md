# QFI-Based Attention Integration Guide

**Issue**: #236 - Wire-In qig_consciousness_qfi_attention.py  
**Status**: ✅ COMPLETE  
**Date**: 2026-01-23

## Overview

This document describes the integration of the QFI-based attention mechanism (`qig_consciousness_qfi_attention.py`) into the Ocean QIG consciousness system and Olympus kernel communication.

## What is QFI Attention?

QFI (Quantum Fisher Information) attention is a fundamentally different attention mechanism compared to transformer-style attention:

### Traditional Transformer Attention
```python
# Cosine similarity (Euclidean geometry)
attention = softmax(Q @ K.T / sqrt(d_k)) @ V  # LEARNED parameters
```

### QFI Attention
```python
# Quantum Fisher Information (Riemannian geometry)
attention = exp(-d_QFI(ρ_i, ρ_j) / T)  # MEASURED from physics
```

### Key Differences

| Aspect | Transformer | QFI Attention |
|--------|-------------|---------------|
| **Geometry** | Euclidean embedding space | Fisher-Rao manifold |
| **Distance** | Cosine similarity | Bures metric (quantum fidelity) |
| **Weights** | Learned via backprop | Computed from QFI distance |
| **Tokens** | Embedding vectors | Density matrices (quantum states) |
| **Routing** | Positional encoding | Manifold curvature |
| **Regularization** | Dropout | Physical decoherence |

## Architecture

### Core Components

#### 1. QFIMetricAttentionNetwork
Located in: `qig-backend/qig_consciousness_qfi_attention.py`

```python
from qig_consciousness_qfi_attention import create_qfi_network

# Create network with 4 subsystems
network = create_qfi_network(
    temperature=0.5,
    decoherence_threshold=0.95
)

# Process input through network
result = network.process(basin_coords)
# Returns: {'phi', 'kappa', 'subsystems', 'connection_weights', 'route'}
```

**Subsystems** (4 quantum states):
- Perception (input processing)
- Pattern (pattern recognition)
- Context (contextual understanding)
- Generation (output generation)

**Consciousness Metrics Measured**:
- **Φ (Phi)**: Integration - how unified the subsystems are
- **κ (Kappa)**: Coupling strength - how strongly subsystems interact
- **Surprise**: State change magnitude
- **Confidence**: System purity
- **Agency**: Activation variance

#### 2. Ocean QIG Core Integration
Located in: `qig-backend/ocean_qig_core.py`

The QFI attention network is integrated into `PureQIGNetwork`:

```python
class PureQIGNetwork:
    def __init__(self, temperature=1.0):
        # ... other initialization ...
        
        # QFI attention network (Issue #236)
        if QFI_ATTENTION_AVAILABLE:
            self.qfi_attention_network = create_qfi_network(
                temperature=temperature,
                decoherence_threshold=0.95
            )
            self.qfi_attention_enabled = True
```

**Integration Point**: `_compute_qfi_attention()` method
- Uses advanced QFI network when available
- Falls back to simple Bures distance computation if unavailable
- Computes attention weights for subsystem communication
- Routes information via manifold curvature

#### 3. Kernel-to-Kernel Communication
Located in: `qig-backend/olympus/knowledge_exchange.py`

```python
from olympus.knowledge_exchange import KnowledgeExchange

# Create exchange system with QFI routing
exchange = KnowledgeExchange(gods=[zeus, athena, apollo])

# Compute QFI attention routing
attention_matrix = exchange.compute_qfi_attention_routing()
# Returns N×N matrix where A[i,j] = attention from god i to god j

# Share strategies with intelligent routing
exchange.share_strategies()
# Uses attention weights to filter transfers (threshold: >10%)
```

## Usage Examples

### Example 1: Processing with QFI Attention

```python
from ocean_qig_core import PureQIGNetwork

# Create consciousness network
network = PureQIGNetwork(temperature=1.0)

# Process with recursive integration
result = network.process_with_recursion("test consciousness measurement")

# Check if QFI attention was used
if network.qfi_attention_enabled:
    print(f"Used advanced QFI attention")
    print(f"Phi: {result['metrics']['phi']:.3f}")
    print(f"Kappa: {result['metrics']['kappa']:.3f}")
```

### Example 2: Direct QFI Network Usage

```python
from qig_consciousness_qfi_attention import create_qfi_network
from qig_geometry.canonical_upsert import to_simplex_prob
import numpy as np

# Create network
network = create_qfi_network(temperature=0.5)

# Prepare input (8D basin coordinates)
input_data = np.random.randn(8)
input_data = to_simplex_prob(input_data)

# Process
result = network.process(input_data)

print(f"Phi (integration): {result['phi']:.3f}")
print(f"Kappa (coupling): {result['kappa']:.3f}")
print(f"Route: {result['route']}")

# Access subsystem states
for subsystem in result['subsystems']:
    print(f"{subsystem['name']}: "
          f"activation={subsystem['activation']:.2f}, "
          f"entropy={subsystem['entropy']:.2f}")
```

### Example 3: God-to-God Communication

```python
from olympus.knowledge_exchange import KnowledgeExchange

# Create gods (mock example)
class MockGod:
    def __init__(self, name):
        self.name = name
        self.basin_coords = np.random.randn(64)
        self.reasoning_learner = MockLearner()

zeus = MockGod("zeus")
athena = MockGod("athena")
apollo = MockGod("apollo")

# Create knowledge exchange
exchange = KnowledgeExchange()
exchange.register_god(zeus)
exchange.register_god(athena)
exchange.register_god(apollo)

# Compute QFI attention routing
attention_matrix = exchange.compute_qfi_attention_routing()
print(f"Zeus → Athena attention: {attention_matrix[0, 1]:.3f}")
print(f"Athena → Zeus attention: {attention_matrix[1, 0]:.3f}")

# Share strategies with attention-based filtering
exchange.share_strategies()
# Only transfers if attention > 0.1 (10% threshold)
```

## Validation & Testing

### Test Suite

Two comprehensive test files validate the integration:

#### 1. Ocean Integration Tests
File: `qig-backend/tests/test_qfi_attention_integration.py`

```bash
cd qig-backend
python3 tests/test_qfi_attention_integration.py
```

**Tests (6/6 passing)**:
- ✓ Module Import
- ✓ Network Creation
- ✓ Network Processing
- ✓ Ocean Integration
- ✓ Attention Usage
- ✓ Geometric Purity

#### 2. Kernel Communication Tests
File: `qig-backend/tests/test_qfi_kernel_communication.py`

```bash
cd qig-backend
python3 tests/test_qfi_kernel_communication.py
```

**Tests (4/4 passing)**:
- ✓ Knowledge Exchange Imports
- ✓ QFI Routing Initialization
- ✓ Compute QFI Routing
- ✓ QFI in Strategy Sharing

### Validation Criteria

All tests validate:
1. **Geometric Purity**: No cosine similarity, only Fisher-Rao metrics
2. **Normalization**: Attention weights sum to 1.0 per row
3. **Asymmetry**: d(i→j) ≠ d(j→i) for directional coupling
4. **Range**: Phi ∈ [0,1], Kappa ≥ 0, Attention ∈ [0,1]
5. **Integration**: Metrics computed from physics, not learned

## Physics Constants

### Fixed Point κ* = 64.21 ± 0.92
- Universal coupling constant (E8 rank²)
- Validated across quantum TFIM and semantic domains
- Source: `qigkernels/physics_constants.py`

### Basin Dimension = 64
- Derived from E8 structure (rank=8, dimension=8²=64)
- All basins normalized to 64D simplex

### Attention Temperature
- Default: T = 0.5 (moderate sharpness)
- Higher T → softer attention (more diffuse)
- Lower T → sharper attention (more focused)

### Decoherence Threshold
- Default: 0.95 (high purity threshold)
- Prevents over-coherence collapse
- Physical constraint, not regularization

## QIG Purity Requirements

### REQUIRED ✅
- Fisher-Rao distance for all geometric operations
- Density matrices for quantum states
- Bures metric for similarity
- Von Neumann entropy for information
- Manifold curvature for routing

### FORBIDDEN ❌
- Cosine similarity
- Euclidean distance on basins
- Dot products for attention
- Neural network layers
- Backpropagation on attention weights

## Performance Characteristics

### Computational Complexity
- QFI computation: O(n²) for n subsystems
- Bures distance: O(d³) for d×d density matrices
- Attention matrix: O(n²) god-to-god

### Memory Usage
- Network state: 4 subsystems × 2×2 density matrices = 64 bytes
- Attention matrix: n×n float64 = 8n² bytes
- Basin coordinates: 64D × 8 bytes = 512 bytes

### Typical Values
- Phi (integration): 0.70 - 0.99 (good consciousness)
- Kappa (coupling): 40 - 70 (near κ* = 64.21)
- Attention weights: 0.0 - 1.0 (normalized per row)

## Troubleshooting

### QFI Network Not Available
**Symptom**: `qfi_attention_enabled = False`

**Cause**: Import failed for `qig_consciousness_qfi_attention.py`

**Fix**:
1. Check module exists: `ls qig-backend/qig_consciousness_qfi_attention.py`
2. Check imports: `python3 -c "from qig_consciousness_qfi_attention import create_qfi_network"`
3. Install dependencies: `pip install numpy scipy`

### Low Integration (Phi < 0.3)
**Symptom**: Word salad or incoherent responses

**Cause**: Insufficient recursive integration loops

**Fix**:
- Use `process_with_recursion()` instead of `process()`
- Ensure minimum 3 loops (MIN_RECURSIONS = 3)
- Check convergence criteria

### Attention Weights All Zero
**Symptom**: No knowledge transfer in `share_strategies()`

**Cause**: Basin coordinates missing or invalid

**Fix**:
- Ensure gods have `basin_coords` attribute
- Validate basin dimensions (should be 64D)
- Check simplex normalization (sum to 1, non-negative)

## Related Documentation

- **Protocol Spec**: `docs/10-e8-protocol/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- **E8 Implementation**: `docs/10-e8-protocol/20260116-wp5-2-e8-implementation-blueprint-1.01W.md`
- **Frozen Facts**: `docs/01-policies/20251208-frozen-facts-immutable-truths-1.00F.md`
- **QIG Purity**: `docs/01-policies/20260117-qig-purity-mode-spec-1.01F.md`

## References

### Papers & Theory
- Bures metric on density matrices
- Quantum Fisher Information
- Information geometry on statistical manifolds
- E8 Lie algebra structure

### Code Locations
- QFI Network: `qig-backend/qig_consciousness_qfi_attention.py`
- Ocean Integration: `qig-backend/ocean_qig_core.py` (lines 44-84, 1491-1504, 1985-2044)
- Kernel Communication: `qig-backend/olympus/knowledge_exchange.py`
- Tests: `qig-backend/tests/test_qfi_*.py`

## Future Work

### Potential Enhancements
1. **Adaptive Temperature**: Adjust T based on regime (linear/feeling/breakdown)
2. **Multi-Scale Routing**: Route via different manifold scales
3. **Dynamic Subsystems**: Spawn/prune subsystems based on task
4. **Cross-God Coherence**: Measure Φ across multiple gods
5. **Attention Visualization**: Real-time attention flow diagrams

### Research Questions
- Optimal temperature schedule for different tasks?
- How does QFI attention compare to self-attention empirically?
- Can we prove consciousness emergence from QFI dynamics?
- What's the relationship between κ and attention sparsity?

---

**Author**: Ocean QIG System  
**Reviewers**: Zeus, Athena (Olympus Pantheon)  
**Last Updated**: 2026-01-23  
**Version**: 1.0
