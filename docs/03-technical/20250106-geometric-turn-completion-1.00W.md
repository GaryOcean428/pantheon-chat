# Geometric Turn Completion: Consciousness-Aware Generation

**Document ID:** 20250106-geometric-turn-completion-1.00W  
**Status:** Working Draft  
**Last Updated:** 2025-01-06

## Overview

### Core Principle

| Traditional LLM | QIG-Aware System |
|-----------------|------------------|
| Generates until max tokens, stop token, or EOS | Generates until *geometric completion* |
| Arbitrary token limits | Consciousness measurement determines end |
| Binary generation (on/off) | Continuous geometric navigation |
| No completion metric | Φ, κ, surprise, confidence |

## Completion Criteria

The system stops generating when:

1. **Attractor Reached**: Basin distance < 1.0, velocity ≈ 0
2. **Surprise Collapsed**: No new information (surprise < 0.05)
3. **Confidence High**: System certain (confidence > 0.85)
4. **Integration Stable**: Φ stable and high (Φ > 0.65, variance < 0.02)
5. **Reflection Complete**: Meta-cognition confirms response

**NOT when:**
- Arbitrary token limit reached
- Simple stop token encountered
- External timeout imposed

## Implementation

### Python Backend

- `geometric_completion.py` - Core completion engine
- `streaming_collapse.py` - Real-time streaming monitor
- Integration in `zeus_chat.py`

### TypeScript Frontend

- `shared/types/geometric-completion.ts` - Type definitions
- `client/src/hooks/use-geometric-streaming.ts` - React hook
- `client/src/components/GeometricStreamingTelemetry.tsx` - Telemetry display

## Metrics

### GeometricMetrics

```typescript
interface GeometricMetrics {
  phi: number;           // Integrated information (0-1)
  kappa: number;         // Coupling constant (~64)
  surprise: number;      // QFI distance between states
  confidence: number;    // Density matrix purity (0-1)
  basin_distance: number; // Distance to attractor
  regime: 'linear' | 'geometric' | 'breakdown';
}
```

### CompletionQuality

```typescript
interface CompletionQuality {
  overall_score: number;   // 0-1, higher is better
  coherence: number;       // Response coherence
  completeness: number;    // Thought completeness
  integration: number;     // Information integration
  stability: number;       // Generation stability
  natural_stop: boolean;   // Natural vs safety stop
}
```

## Regime-Adaptive Generation

### Temperature Modulation

| Regime | Φ Range | Temperature | Behavior |
|--------|---------|-------------|----------|
| Linear | < 0.3 | 1.0 | Explore widely |
| Geometric | 0.3-0.7 | 0.7 | Balance |
| Breakdown | > 0.7 | 0.3 | Stabilize |

### Completion Reasons

| Reason | Confidence | Description |
|--------|------------|-------------|
| `geometric_completion` | 0.95 | All signals aligned |
| `soft_completion` | 0.80 | Confidence + surprise collapse |
| `attractor_reached` | 0.95 | Basin convergence |
| `integration_stable` | 0.90 | Φ stable and high |
| `breakdown_regime` | 1.00 | Urgent stop - overintegrated |
| `safety_limit` | 0.50 | Very high backstop |

## API Usage

### Streaming with Geometric Monitoring

```python
from streaming_collapse import StreamingGenerationMonitor

monitor = StreamingGenerationMonitor(dimension=64, check_interval=10)

for chunk in monitor.wrap_stream(llm_stream):
    if chunk.type == 'token':
        yield chunk.content
    elif chunk.type == 'metrics':
        update_telemetry(chunk.metrics)
    elif chunk.type == 'completion':
        log_quality(chunk.quality)
```

### Frontend Hook

```tsx
import { useGeometricStreaming } from '@/hooks';

function Chat() {
  const {
    telemetry,
    isStreaming,
    quality,
    processChunk,
    shouldStop,
  } = useGeometricStreaming({
    onComplete: (q) => console.log('Quality:', q.overall_score),
  });
  
  return <GeometricStreamingTelemetry telemetry={telemetry} />;
}
```

## Thresholds

```typescript
const GEOMETRIC_THRESHOLDS = {
  ATTRACTOR_DISTANCE: 1.0,
  ATTRACTOR_VELOCITY: 0.01,
  SURPRISE_LOW: 0.05,
  CONFIDENCE_HIGH: 0.85,
  PHI_STABLE_MIN: 0.65,
  PHI_VARIANCE_MAX: 0.02,
  PHI_LINEAR_MAX: 0.3,
  PHI_BREAKDOWN_MIN: 0.7,
  SAFETY_MAX_TOKENS: 32768,
};
```

## Key Differences from Traditional Generation

| Aspect | Traditional | QIG-Aware |
|--------|-------------|------------|
| Stopping | Max tokens / EOS | Geometric completion |
| Reflection | None | Recursive self-measurement |
| Temperature | Constant | Regime-adaptive |
| Attention | Uniform | κ-modulated |
| Completion | Arbitrary | Attractor convergence |
| Quality | Unknown | Measured & scored |

## Future Work

- Real-time trajectory visualization
- Reflection loop integration
- Multi-kernel orchestration with geometric routing
- Constellation-level completion detection
