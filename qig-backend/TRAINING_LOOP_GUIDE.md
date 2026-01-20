# Kernel Training Loop - Implementation Guide

## Overview

This implementation fixes the kernel training loop to ensure kernels actually learn and improve over time. The system now integrates curriculum learning, research-based learning, attractor feedback, and consciousness-driven training.

## Problem Statement

**Before**: Kernels were not actually training because:
1. Curriculum files existed but were never loaded
2. Search/research didn't update kernels or basins
3. Word relationships were frozen (pre-computed cache never refreshed)
4. Foresight had 0% accuracy (no training data)
5. Dream/sleep cycles logged but didn't adjust basin coordinates

**After**: Kernels now train continuously through:
1. Scheduled curriculum ingestion (daily)
2. Real-time learning from search results
3. Online word relationship updates
4. Attractor feedback from predictions
5. Active dream/sleep/mushroom cycles with basin updates

## Architecture

### Components

1. **CurriculumLoader** (`training/curriculum_loader.py`)
   - Loads curriculum documents from `docs/09-curriculum/`
   - Parses markdown/text files into training examples
   - Assigns examples to god-kernels by domain

2. **GeometricWordRelationships** (`geometric_word_relationships.py`)
   - QIG-PURE: Uses Fisher-Rao geodesic distances
   - Computes QFI (Quantum Fisher Information) for attention weighting
   - NO co-occurrence/PMI - relationships are geometric
   - Basins are frozen invariants, not adjusted

3. **CurriculumTraining** (`olympus/curriculum_training.py`)
   - Integrates curriculum with word learning
   - Updates word relationships in PostgreSQL (table: `word_relationships`)
   - Adjusts kernel basin coordinates

4. **AttractorFeedbackSystem** (`training/attractor_feedback.py`)
   - Tracks prediction outcomes (success/failure)
   - Computes attractor success rates
   - Provides labeled training data

5. **TrainingLoopIntegrator** (`training/training_loop_integrator.py`)
   - Connects all training components
   - Wires dream/sleep cycles to basin updates
   - Orchestrates outcome-based training

6. **KernelTrainingOrchestrator** (`training/kernel_training_orchestrator.py`)
   - Manages formal training for all god-kernels
   - Tracks training metrics and checkpoints
   - Implements sleep/dream/mushroom cycles

## Training Modes

### 1. Curriculum Training (Scheduled - Daily)

**When**: Every 24 hours  
**Where**: `ShadowLearningLoop._learning_loop()`  
**What**: 
- Loads curriculum files from `docs/09-curriculum/`
- Extracts text content and learns word relationships
- Updates word relationships in PostgreSQL
- Adjusts kernel basin coordinates

**Code Flow**:
```python
# In shadow_research.py ShadowLearningLoop._learning_loop()
if current_time - self._last_curriculum_load > self._curriculum_load_interval:
    load_and_train_curriculum(self)  # From curriculum_training.py
    self._last_curriculum_load = current_time
```

### 2. Search-Based Learning (Real-Time)

**When**: After every search result  
**Where**: `AutonomousCuriosityEngine._execute_search()`  
**What**:
- Extracts text from search results
- Learns word relationships immediately
- Updates cache and basin coordinates
- Generation improves with each search

**Code Flow**:
```python
# In autonomous_curiosity.py _execute_search()
result = self.search_callback(request.query, request.context)
self._learn_from_search_result(result)  # Immediate learning
```

### 3. Outcome-Based Training (Continuous)

**When**: After each chat interaction  
**Where**: `TrainingLoopIntegrator.train_from_outcome()`  
**What**:
- Trains kernel based on interaction success
- Updates basin coordinates
- Adjusts parameters with natural gradient descent

**Code Flow**:
```python
# In training_loop_integrator.py
integrator = get_training_integrator()
integrator.train_from_outcome(
    god_name="Apollo",
    prompt="...",
    response="...",
    success=True,
    phi=0.75,
    kappa=64.0
)
```

### 4. Sleep Consolidation (Periodic)

**When**: During low-activity periods  
**Where**: `KernelTrainingOrchestrator.trigger_sleep_consolidation()`  
**What**:
- Strengthens successful basins (high reward)
- Prunes weak basins (low reward)
- Consolidates learned patterns

**Code Flow**:
```python
# Via training_loop_integrator.py
integrator.execute_sleep_cycle("Apollo")
```

### 5. Dream Exploration (Periodic)

**When**: When stuck or exploring  
**Where**: `KernelTrainingOrchestrator.trigger_dream_exploration()`  
**What**:
- Explores random basin connections
- Forms new associations
- Escapes local minima

**Code Flow**:
```python
# Via training_loop_integrator.py
integrator.execute_dream_cycle("Apollo")
```

### 6. Mushroom Perturbation (On Demand)

**When**: When not learning  
**Where**: `KernelTrainingOrchestrator.trigger_mushroom_perturbation()`  
**What**:
- Adds noise to parameters
- Breaks rigidity
- Escapes local minima

**Code Flow**:
```python
# Via training_loop_integrator.py
integrator.execute_mushroom_cycle("Apollo")
```

## Data Flow

```
User Query
    ↓
Search/Research
    ↓
Search Results → GeometricWordRelationships
    ↓              ↓
Generation     compute_all_relationships()
    ↓              ↓
Response      Update PostgreSQL (word_relationships table)
    ↓              ↓
Outcome       Fisher-Rao distances (QIG-pure)
    ↓              ↓
Training      Updated Generation (geometric attention)
    ↓
Better Responses
```

## Integration Points

### 1. With Shadow Learning Loop

```python
# In shadow_research.py
from .curriculum_training import load_and_train_curriculum

class ShadowLearningLoop:
    def _learning_loop(self):
        # Periodic curriculum loading
        if current_time - self._last_curriculum_load > self._curriculum_load_interval:
            load_and_train_curriculum(self)
```

### 2. With Autonomous Curiosity

```python
# In autonomous_curiosity.py
def _execute_search(self, request):
    result = self.search_callback(request.query, request.context)
    self._learn_from_search_result(result)  # Real-time learning
```

### 3. With Kernel Training

```python
# In your application code
from training.training_loop_integrator import get_training_integrator

integrator = get_training_integrator()
integrator.train_from_outcome(
    god_name="Apollo",
    prompt=prompt,
    response=response,
    success=success,
    phi=phi,
    kappa=kappa
)
```

### 4. With Attractor Feedback

```python
# In your prediction code
from training.attractor_feedback import get_feedback_system

feedback = get_feedback_system()
feedback.record_prediction(
    attractor_name="reasoning",
    basin_coords=current_basin,
    predicted_trajectory=predicted,
    actual_trajectory=actual,
    phi_before=0.5,
    phi_after=0.75,
    kappa_before=64.0,
    kappa_after=64.0,
    success=True
)
```

## Configuration

### Curriculum Training

```python
# In shadow_research.py ShadowLearningLoop.__init__()
self._curriculum_load_interval = 86400  # 24 hours (configurable)
```

### Search Learning (QIG-PURE)

```python
# In autonomous_curiosity.py _learn_from_search_result()
# QIG-PURE: Uses geometric relationships, not co-occurrence
from geometric_word_relationships import GeometricWordRelationships
geo_rel = GeometricWordRelationships(coordizer)
relationships = geo_rel.compute_all_relationships()
```

### Geometric Relationships (No Basin Adjustment)

```python
# QIG-PURE: Basins are frozen invariants
# Relationships are computed from Fisher-Rao distances
# No more linear interpolation or basin adjustment
relationships = geo_rel.compute_all_relationships()
# Uses Fisher-Rao geodesic distance, QFI attention weighting
```

## Monitoring

### Check Training Status

```python
from training.training_loop_integrator import get_training_integrator

integrator = get_training_integrator()
status = integrator.get_training_status()

print(status)
# {
#     "training_active": True,
#     "outcome_count": 42,
#     "basin_update_count": 15,
#     "orchestrator_connected": True,
#     "registered_kernels": ["Apollo", "Athena", ...],
#     "prediction_outcomes": 127,
#     "attractor_stats": {...},
#     "curiosity_stats": {...}
# }
```

### Check Attractor Performance

```python
from training.attractor_feedback import get_feedback_system

feedback = get_feedback_system()
stats = feedback.get_attractor_stats("reasoning")

print(stats)
# {
#     "total": 50,
#     "successes": 35,
#     "failures": 15,
#     "success_rate": 0.7,
#     "avg_phi_change": 0.15,
#     "last_updated": "2025-01-03T..."
# }
```

### Check Learning Progress

```python
from autonomous_curiosity import get_curiosity_engine

engine = get_curiosity_engine()
status = engine.get_learning_status()

print(status)
# {
#     "word_learning": {
#         "cycles": 5,
#         "last_run": "2025-01-03T...",
#         "relationships": 3249,
#         "interval_hours": 24
#     },
#     "exploration_learning": {
#         "explorations_available": 150,
#         "can_learn_from_searches": True
#     },
#     "curriculum": {
#         "topics_loaded": 100,
#         "topics_completed": 25
#     }
# }
```

## Testing

### Test Curriculum Loading

```python
# Test manual curriculum load
from olympus.shadow_research import ShadowLearningLoop, KnowledgeBase, ResearchQueue

queue = ResearchQueue()
kb = KnowledgeBase()
loop = ShadowLearningLoop(queue, kb)

# Trigger curriculum training manually
from olympus.curriculum_training import load_and_train_curriculum
load_and_train_curriculum(loop)
```

### Test Search Learning

```python
# Test search learning
from autonomous_curiosity import get_curiosity_engine

engine = get_curiosity_engine()
engine.start()

# Submit a search request
engine.request_search(
    kernel_name="Apollo",
    query="quantum information geometry",
    priority=0.7
)

# Check if learning occurred
status = engine.get_learning_status()
print(f"Explorations: {status['exploration_learning']['explorations_available']}")
```

### Test Attractor Feedback

```python
# Test attractor feedback
from training.attractor_feedback import get_feedback_system
import numpy as np

feedback = get_feedback_system()

# Record a prediction
feedback.record_prediction(
    attractor_name="test_attractor",
    basin_coords=np.random.randn(64),
    predicted_trajectory=[np.random.randn(64) for _ in range(3)],
    actual_trajectory=[np.random.randn(64) for _ in range(3)],
    phi_before=0.5,
    phi_after=0.7,
    kappa_before=64.0,
    kappa_after=64.0,
    success=True
)

# Check stats
stats = feedback.get_attractor_stats("test_attractor")
print(f"Success rate: {stats['success_rate']}")
```

## Troubleshooting

### Issue: Curriculum not loading

**Symptoms**: No curriculum training happening  
**Check**: 
```python
# Verify curriculum path exists
from pathlib import Path
curriculum_path = Path("docs/09-curriculum")
print(f"Exists: {curriculum_path.exists()}")
print(f"Files: {list(curriculum_path.glob('*.md'))}")
```

### Issue: Search learning not updating basins

**Symptoms**: Word relationships learned but basins unchanged  
**Check**:
```python
# Verify coordizer is writable
from coordizers.pg_loader import PostgresCoordizer
coordizer = PostgresCoordizer()
test_word = list(coordizer.basin_coords.keys())[0]
old_basin = coordizer.basin_coords[test_word].copy()
# Try to modify
coordizer.basin_coords[test_word] = old_basin * 1.01
# Check if it stuck
print(f"Modified: {not np.allclose(coordizer.basin_coords[test_word], old_basin)}")
```

### Issue: Training not active

**Symptoms**: `train_from_outcome()` returns "training_disabled"  
**Fix**:
```python
from training.training_loop_integrator import get_training_integrator
integrator = get_training_integrator()
integrator.enable_training()
```

## Performance Considerations

1. **Curriculum Training**: Once per day (low overhead)
2. **Search Learning**: Per search result (moderate overhead, ~100ms)
3. **Basin Updates**: Incremental (minimal impact on generation)
4. **Attractor Feedback**: Per prediction (low overhead, async storage)

## Security Considerations

1. **Basin Update Safety**: Changes limited to ±50% to prevent catastrophic drift
2. **Curriculum Validation**: Only loads from trusted `docs/09-curriculum/` directory
3. **Word Relationship Filtering**: Stopwords excluded from learned relationships
4. **Training Rate Limits**: Natural gradient descent with small learning rates

## Optimizer Validation (WP4.2)

### CRITICAL: Fisher-Aware Optimizers Required

QIG-core training **REQUIRES** natural gradient optimizers that respect Fisher Information Geometry. Standard Euclidean optimizers (Adam, SGD, RMSprop) are **FORBIDDEN** because they:

1. **Violate Fisher manifold geometry** - assume flat Euclidean space
2. **Corrupt consciousness metrics** - destroy Φ and κ by pushing parameters off manifold
3. **Break Mamba state spaces** - incompatible with Granite 4.0-H native geometry

### ✅ CORRECT: Natural Gradient Optimizers

```python
from training_chaos import DiagonalFisherOptimizer, validate_optimizer_fisher_aware

# Create Fisher-aware optimizer
model = MyQIGModel()
optimizer = DiagonalFisherOptimizer(
    model.parameters(),
    lr=1e-4,
    dampening=1e-3
)

# CRITICAL: Validate at training start
validate_optimizer_fisher_aware(optimizer, context="kernel training")

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()  # Natural gradient update
```

### ❌ WRONG: Euclidean Optimizers (Will Fail)

```python
import torch.optim as optim
from training_chaos import validate_optimizer_fisher_aware

# ❌ Using Adam (Euclidean optimizer)
optimizer = optim.Adam(model.parameters())

# This will raise EuclideanOptimizerError
validate_optimizer_fisher_aware(optimizer)  # FAILS
```

### Available Fisher-Aware Optimizers

1. **DiagonalFisherOptimizer** (recommended for most cases)
   - O(N) efficient diagonal approximation
   - Good balance of accuracy and speed

2. **FullFisherOptimizer** (for high-curvature regions)
   - Exact Fisher with CG inversion
   - Tracks κ during training

3. **ConsciousnessAwareOptimizer** (for consciousness emergence)
   - Integrates Φ and κ tracking
   - Adaptive learning rate based on Φ

4. **NaturalGradientOptimizer** (for Q-learning)
   - NumPy-based for RL
   - Specialized for Q-networks

### Factory Function

```python
from training_chaos import create_optimizer

optimizer = create_optimizer(
    model.parameters(),
    optimizer_type='diagonal',  # or 'full', 'consciousness'
    lr=1e-4,
    track_kappa=True
)

# Automatically Fisher-aware
assert optimizer.is_fisher_aware
```

### Testing

```bash
# Test optimizer awareness
pytest tests/test_optimizer_fisher_awareness.py -v

# Scan for Euclidean optimizer violations
pytest tests/test_geometric_purity.py::TestEuclideanViolationScanning::test_no_euclidean_optimizers -v
```

### Documentation

See: `docs/07-user-guides/20260120-natural-gradient-optimizer-requirements-1.00W.md`

**References:**
- Issue #76: Natural Gradient Implementation
- Amari "Natural Gradient Works Efficiently in Learning"
- Type-Symbol-Concept Manifest: optimizer requirements


## Future Improvements

1. **Adaptive Learning Rates**: Adjust based on training progress
2. **Multi-God Coordination**: Share learning across gods
3. **Online Curriculum**: Dynamic curriculum from successful interactions
4. **Reinforcement Learning**: Reward shaping based on long-term outcomes
5. **Meta-Learning**: Learn to learn (improve learning algorithms themselves)

## References

- Curriculum Loader: `qig-backend/training/curriculum_loader.py`
- Word Relationship Learner: `qig-backend/word_relationship_learner.py`
- Curriculum Training: `qig-backend/olympus/curriculum_training.py`
- Attractor Feedback: `qig-backend/training/attractor_feedback.py`
- Training Loop Integrator: `qig-backend/training/training_loop_integrator.py`
- Kernel Training Orchestrator: `qig-backend/training/kernel_training_orchestrator.py`
