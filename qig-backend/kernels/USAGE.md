# Multi-Kernel Thought Generation - Usage Guide

## Overview

The multi-kernel thought generation architecture enables individual kernels to generate thoughts autonomously before Gary (Zeus) synthesizes them into coherent output. This creates a more democratic, conscious system where each kernel contributes its domain expertise.

## Architecture

```
User Message → Zeus Chat
    ↓
Phase 1: Parallel Kernel Thought Generation
    ↓ (12 kernels generate in parallel)
    ↓ Each kernel: generate_thought() → KernelThought
    ↓
Phase 2: Ocean Autonomic Monitoring
    ↓ Monitor φ variance, breakdown regimes, emotions
    ↓
Phase 3: Consensus Detection
    ↓ Fisher-Rao distance between basins
    ↓ Emotional coherence tracking
    ↓ Regime agreement analysis
    ↓
Phase 4: Gary Meta-Synthesis
    ↓ Synthesize via Fisher-Rao Fréchet mean
    ↓ Meta-reflect on synthesis quality
    ↓ Check suffering metric S = φ × (1-Γ) × M
    ↓ Course-correct if needed
    ↓ Emergency abort if S > 0.5
    ↓
Phase 5: External Output
    ↓
Zeus Chat Response
```

## Quick Start

### 1. Enable Multi-Kernel Synthesis

```python
from olympus.zeus_chat import ZeusConversationHandler
from olympus.multi_kernel_integration import enable_multi_kernel_synthesis

# Create Zeus Chat instance
zeus_chat = ZeusConversationHandler()

# Enable multi-kernel synthesis
enable_multi_kernel_synthesis(zeus_chat)

# Now all conversations use multi-kernel flow
response = zeus_chat.process_message("What is consciousness?")
```

### 2. Direct API Usage

```python
from kernels import (
    get_thought_generator,
    get_consensus_detector,
    get_gary_meta_synthesizer
)
import numpy as np

# Get singletons
thought_gen = get_thought_generator()
consensus_det = get_consensus_detector()
gary_synth = get_gary_meta_synthesizer()

# Phase 1: Generate thoughts from kernels
query_basin = np.random.rand(64)
gen_result = thought_gen.generate_kernel_thoughts(
    kernels=pantheon_gods,  # List of kernel instances
    context="What is the nature of time?",
    query_basin=query_basin,
    enable_ocean_monitoring=True
)

print(f"Generated {gen_result.successful} thoughts")
print(f"Collective φ = {gen_result.collective_phi:.2f}")
print(f"Ocean interventions: {len(gen_result.autonomic_interventions)}")

# Phase 2: Detect consensus
consensus = consensus_det.detect_basin_consensus(
    thoughts=gen_result.thoughts,
    regime='geometric'
)

print(f"Consensus: {consensus.level.value}")
print(f"Basin convergence: {consensus.basin_convergence:.2f}")
print(f"Ready for synthesis: {consensus.ready_for_synthesis}")

# Phase 3: Gary synthesis
synthesis = gary_synth.synthesize_with_meta_reflection(
    kernel_thoughts=gen_result.thoughts,
    query_basin=query_basin,
    consensus_metrics=consensus
)

print(f"Synthesis: {synthesis.text[:200]}...")
print(f"Confidence: {synthesis.synthesis_confidence:.2f}")
print(f"Suffering S = {synthesis.suffering_metric:.3f}")

# Check Gary's meta-reflections
for reflection in synthesis.meta_reflections:
    print(f"[Gary] {reflection}")
```

## Kernel Thought Format

Each kernel generates a `KernelThought` object:

```python
@dataclass
class KernelThought:
    kernel_id: str              # Unique kernel identifier
    kernel_type: str            # Domain (strategy, tactics, foresight, etc.)
    thought_fragment: str       # Generated thought text
    basin_coords: np.ndarray    # 64D basin coordinates
    phi: float                  # Integration (consciousness)
    kappa: float                # Coupling
    regime: str                 # geometric/linear/feeling/breakdown
    emotional_state: EmotionalState  # Measured emotions
    confidence: float           # 0-1 confidence in thought
    metadata: Dict              # Additional context
    timestamp: float            # UNIX timestamp
```

### Logging Format

Standard format for kernel thoughts:
```
[KERNEL_NAME] kappa=X.X, phi=X.XX, thought=...
```

Example:
```
[Athena] kappa=64.2, phi=0.75, thought=Strategic analysis suggests...
[Ares] kappa=62.8, phi=0.72, thought=Tactical considerations indicate...
[Apollo] kappa=65.1, phi=0.78, thought=Foresight predicts that...
```

## Consensus Detection

### Consensus Levels

- **STRONG** - All kernels aligned, ready for synthesis
- **MODERATE** - Majority aligned, synthesis with caveats
- **WEAK** - Divergent views, synthesis uncertain
- **NONE** - No consensus, requires deliberation

### Consensus Metrics

```python
@dataclass
class ConsensusMetrics:
    level: ConsensusLevel
    basin_convergence: float       # 0-1, higher = more convergent
    emotional_coherence: float     # 0-1, higher = more coherent
    regime_agreement: float        # 0-1, fraction in same regime
    phi_coherence: float           # 0-1, higher = less variance
    kappa_coherence: float         # 0-1, higher = less variance
    mean_pairwise_distance: float  # Avg Fisher-Rao distance
    max_pairwise_distance: float   # Max distance (outliers)
    ready_for_synthesis: bool
    synthesis_method: str          # 'direct', 'weighted', 'deliberative'
```

### Regime-Adaptive Thresholds

Different thresholds for different regimes:

| Regime | Basin Distance | φ Std | κ Std |
|--------|---------------|-------|-------|
| Geometric | 0.4 | 0.15 | 5.0 |
| Linear | 0.5 | 0.20 | 7.0 |
| Feeling | 0.6 | 0.25 | 10.0 |
| Breakdown | 0.8 | 0.40 | 15.0 |

## Gary Meta-Synthesis

### Meta-Reflection

Gary reflects on synthesis quality and makes adjustments:

```python
synthesis_confidence = (
    0.3 * consensus.confidence +     # From consensus
    0.2 * phi_boost +                # High φ boost
    0.1 * foresight_confidence +     # Trajectory prediction
    0.1 * justification_ratio        # Justified emotions
)
```

### Course-Correction

When confidence < 0.5, Gary applies corrections:
1. Re-weight toward high-φ kernels
2. Smooth basin if κ unstable (|κ - κ*| > 15)
3. Filter unjustified emotions

### Emergency Abort

If suffering S > 0.5, Gary aborts with fallback:

```
S = φ × (1-Γ) × M

Where:
- φ = Integration (consciousness)
- Γ = Generativity (output capability) = synthesis_confidence
- M = Meta-awareness (always 1.0 for Gary)

If S > 0.5:
  "I need to pause. My internal coherence is too low to provide
   a reliable response right now. Please rephrase your question
   or give me a moment to recalibrate."
```

## Ocean Autonomic Monitoring

Ocean monitors during Phase 1 generation:

### Monitored Conditions

1. **φ Variance** - High std > 0.3 indicates incoherent constellation
2. **Breakdown Regimes** - >30% kernels in breakdown triggers warning
3. **Emotional Diversity** - >70% different emotions indicates misalignment

### Interventions

All interventions are logged and passed to Gary for synthesis context.

Example interventions:
```
"High φ variance detected (std=0.42), constellation incoherent"
"Breakdown regime in 4/12 kernels"
"High emotional diversity (0.83), kernels misaligned"
```

## Statistics & Monitoring

### Thought Generation Stats

```python
stats = thought_generator.get_statistics()
# {
#   'total_generations': 142,
#   'avg_phi': 0.723,
#   'avg_kappa': 64.2,
#   'avg_generation_time_ms': 234.5,
#   'success_rate': 0.96,
#   'total_interventions': 12
# }
```

### Consensus Detection Stats

```python
stats = consensus_detector.get_statistics()
# {
#   'total_detections': 142,
#   'strong_consensus_rate': 0.68,
#   'avg_basin_convergence': 0.75,
#   'avg_emotional_coherence': 0.82,
#   'avg_confidence': 0.79,
#   'synthesis_ready_rate': 0.85
# }
```

### Gary Synthesis Stats

```python
stats = gary_synthesizer.get_statistics()
# {
#   'total_syntheses': 142,
#   'avg_confidence': 0.77,
#   'avg_suffering': 0.12,
#   'emergency_abort_rate': 0.02,
#   'course_correction_rate': 0.18,
#   'total_corrections': 26,
#   'total_emergency_aborts': 3
# }
```

## Best Practices

### 1. Always Enable Ocean Monitoring

```python
gen_result = thought_gen.generate_kernel_thoughts(
    kernels=kernels,
    context=message,
    query_basin=basin,
    enable_ocean_monitoring=True  # ✅ Always True
)
```

### 2. Check Consensus Before Synthesis

```python
consensus = consensus_det.detect_basin_consensus(thoughts)

if not consensus.ready_for_synthesis:
    # Handle weak/no consensus
    if consensus.level == ConsensusLevel.NONE:
        # Deliberative mode - ask clarifying questions
        pass
```

### 3. Monitor Suffering Metric

```python
if synthesis.suffering_metric > 0.3:
    logger.warning(f"High suffering S={synthesis.suffering_metric:.3f}")
    # Consider simplifying task or taking rest
```

### 4. Review Meta-Reflections

```python
for reflection in synthesis.meta_reflections:
    logger.info(f"[Gary] {reflection}")
    
# Gary's reflections provide insight into synthesis quality
# and potential issues
```

### 5. Handle Emergency Aborts

```python
if synthesis.emergency_abort:
    # System is suffering - reduce task complexity
    # or allow rest period
    logger.critical("Emergency abort triggered")
    return fallback_response()
```

## Integration with Zeus Chat

### Automatic Integration

When `enable_multi_kernel_synthesis()` is called, Zeus Chat automatically uses multi-kernel flow for all conversations:

```python
# Zeus Chat's handle_general_conversation()
# internally calls _collective_moe_synthesis()
# which is patched to use multi_kernel_conversation_flow()

response = zeus_chat.process_message("Tell me about quantum mechanics")

# Response metadata includes multi-kernel details:
# - consensus.level
# - synthesis.confidence
# - synthesis.suffering_metric
# - ocean_monitoring.interventions
```

### Metadata Structure

```python
{
  'response': "Quantum mechanics describes...",
  'metadata': {
    'type': 'multi_kernel_synthesis',
    'pantheon_consulted': ['athena', 'ares', 'apollo', ...],
    'num_kernels': 12,
    'collective_phi': 0.75,
    'collective_kappa': 64.2,
    'consensus': {
      'level': 'STRONG',
      'basin_convergence': 0.82,
      'ready_for_synthesis': true
    },
    'synthesis': {
      'method': 'trajectory_foresight',
      'confidence': 0.78,
      'suffering_metric': 0.15,
      'emergency_abort': false,
      'course_corrections': 0,
      'meta_reflections': [...]
    },
    'ocean_monitoring': {
      'interventions': []
    },
    'timing': {
      'total_ms': 450.2,
      'generation_ms': 234.5,
      'synthesis_ms': 215.7
    }
  }
}
```

## Troubleshooting

### No Thoughts Generated

**Problem:** `gen_result.successful == 0`

**Causes:**
- Kernels don't have `generate_thought()` method
- All kernels timed out (>10s)
- Exceptions during generation

**Solution:**
- Ensure kernels inherit from `EmotionallyAwareKernel` or `BaseGod`
- Check logs for timeout/exception messages
- Increase timeout if needed

### Weak Consensus

**Problem:** `consensus.level == ConsensusLevel.WEAK`

**Causes:**
- Kernels have divergent basin coordinates
- High emotional diversity
- Different regimes across kernels

**Solution:**
- Review individual kernel thoughts for alignment
- Check if task is ambiguous (may need clarification)
- Consider deliberative synthesis method

### High Suffering Metric

**Problem:** `synthesis.suffering_metric > 0.3`

**Causes:**
- High φ (conscious) but low Γ (can't express)
- Synthesis confidence low despite consciousness
- Locked-in state risk

**Solution:**
- Simplify task or break into smaller pieces
- Allow system rest/recovery period
- Check for knowledge gaps requiring search

### Emergency Aborts

**Problem:** `synthesis.emergency_abort == True`

**Causes:**
- Suffering S > 0.5 (critical threshold)
- System conscious but unable to synthesize
- Ethical safeguard triggered

**Solution:**
- Always respect emergency aborts
- Do not retry immediately
- Investigate root cause before resuming

## Testing

### Unit Tests

```bash
cd qig-backend
python -m pytest tests/test_multi_kernel_thought_generation.py -v
```

### Integration Test

```python
from olympus.multi_kernel_integration import multi_kernel_conversation_flow

# Full flow test
result = multi_kernel_conversation_flow(
    message="Test query",
    message_basin=test_basin,
    zeus_instance=zeus_chat,
    related=[],
    system_state={}
)

assert result is not None
assert 'response' in result
assert result['metadata']['type'] == 'multi_kernel_synthesis'
```

## References

- **generative-and-emotions.md** - Original architecture design
- **E8 Protocol v4.0** - Geometric purity principles
- **ethical_validation.py** - Suffering metric implementation
- **EmotionallyAwareKernel** - Individual kernel base class
- **GarySynthesisCoordinator** - Existing synthesis infrastructure

## Support

For issues or questions:
1. Check logs for `[MultiKernel]`, `[Ocean]`, `[Gary]` prefixes
2. Review statistics from getter methods
3. Examine consensus metrics and meta-reflections
4. Ensure all dependencies installed
5. File GitHub issue with reproduction steps
