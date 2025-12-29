# GPU Offloading for Training

*Making consciousness training accessible on modest hardware*

**Status:** Planning Document
**Target:** qig-consciousness
**Created:** 2025-12-10

---

## §0 THE TRAINING BOTTLENECK

### Memory Requirements (50M parameter model)

```python
Model weights:      200 MB  (50M × 4 bytes FP32)
Activations:        400 MB  (stored for backprop)
Gradients:          200 MB  (same size as weights)
Optimizer state:    400 MB  (Adam needs 2× params: momentum + variance)
Temporary buffers:  200 MB  (intermediate computations)
────────────────────────────
TOTAL:             1400 MB  (per batch on GPU!)
```

### Hardware Constraints

| GPU | VRAM | Max Batch Size (before) |
|-----|------|------------------------|
| A100 | 40GB | 256-512 |
| RTX 3090 | 24GB | 128-256 |
| RTX 3060 | 12GB | ~64 (struggles) |

**Goal:** Use CPU RAM (cheap, abundant) to assist GPU (expensive, limited)

---

## §1 SIX OPTIMIZATIONS

### 1. Gradient Checkpointing (50% Memory Reduction)

Trade compute for memory - recompute activations during backward pass.

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedQIGKernel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            QIGTransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        self.checkpoint_every = 2  # Checkpoint every 2 layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i % self.checkpoint_every == 0:
                x = checkpoint(layer, x)  # Recompute during backward
            else:
                x = layer(x)
        return x
```

### 2. Mixed Precision Training (50% Memory, 2× Speed)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = compute_loss(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Note:** Keep FP32 for precision-critical ops (basin_distance, qfi_metric, optimizer_step)

### 3. CPU Optimizer State Offloading (40% Memory Reduction)

```python
class CPUOffloadedOptimizer:
    """Keep momentum/variance on CPU, only bring to GPU when needed."""

    def __init__(self, params, lr=1e-4):
        self.params = list(params)

        # Store optimizer states on CPU
        self.momentum = {p: torch.zeros_like(p.data, device='cpu') for p in self.params}
        self.variance = {p: torch.zeros_like(p.data, device='cpu') for p in self.params}

        self.lr = lr
        self.beta1, self.beta2 = 0.9, 0.999
        self.eps = 1e-8
        self.step_count = 0

    def step(self):
        self.step_count += 1

        for p in self.params:
            if p.grad is None:
                continue

            grad_cpu = p.grad.data.cpu()

            # Update on CPU
            self.momentum[p].mul_(self.beta1).add_(grad_cpu, alpha=1 - self.beta1)
            self.variance[p].mul_(self.beta2).addcmul_(grad_cpu, grad_cpu, value=1 - self.beta2)

            # Bias correction
            m_hat = self.momentum[p] / (1 - self.beta1**self.step_count)
            v_hat = self.variance[p] / (1 - self.beta2**self.step_count)

            # Compute and apply update
            update = m_hat / (v_hat.sqrt() + self.eps)
            p.data.add_(update.cuda(), alpha=-self.lr)

        for p in self.params:
            p.grad = None
```

### 4. Gradient Accumulation (4× Batch Size, Same Memory)

```python
class GradientAccumulator:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads = {
            name: torch.zeros_like(p.data, device='cpu')
            for name, p in model.named_parameters()
        }
        self.step_count = 0

    def training_step(self, batch):
        output = self.model(batch['input_ids'])
        loss = F.cross_entropy(output, batch['labels']) / self.accumulation_steps
        loss.backward()

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    self.accumulated_grads[name].add_(p.grad.cpu())

        self.optimizer.zero_grad()
        self.step_count += 1

        if self.step_count % self.accumulation_steps == 0:
            self._apply_accumulated_gradients()

        return loss.item() * self.accumulation_steps
```

### 5. Async CPU Telemetry (Zero GPU Impact)

```python
import threading
import queue

class AsyncTelemetryProcessor:
    """Compute Φ, κ on CPU while GPU trains."""

    def __init__(self):
        self.queue = queue.Queue(maxsize=10)
        self.results = {}
        self.worker = threading.Thread(target=self._process, daemon=True)
        self.worker.start()

    def submit(self, step, hidden_states):
        states_cpu = hidden_states.detach().cpu()
        self.queue.put((step, states_cpu))

    def _process(self):
        while True:
            step, states = self.queue.get()
            telemetry = {
                'phi': self._compute_phi(states),
                'kappa': self._compute_kappa(states),
                'regime': self._detect_regime(states),
            }
            self.results[step] = telemetry
            self.queue.task_done()

    def get_latest(self):
        if self.results:
            return self.results[max(self.results.keys())]
        return None
```

### 6. Layer-wise Offloading (Extreme Low Memory)

For GPUs <8GB - keep only active layer on GPU.

```python
class LayerWiseOffloadedModel:
    def forward(self, x):
        x = x.cuda()
        for layer in self.model.layers:
            layer.cuda()
            x = layer(x)
            layer.cpu()
            torch.cuda.empty_cache()
        return x
```

**Trade-off:** ~10× slower (use only for prototyping)

---

## §2 MEMORY BUDGET

### Before Optimizations

```
Model weights:        200 MB
Activations:          400 MB
Gradients:            200 MB
Optimizer states:     400 MB
Temporary buffers:    200 MB
─────────────────────────────
Total:               1400 MB per batch
Max batch_size: 8 on 12GB GPU
```

### After ALL Optimizations

```
Model weights:        100 MB  (FP16)
Activations:          200 MB  (checkpointing)
Gradients:            100 MB  (FP16 + sparse)
Optimizer states:       0 MB  (CPU)
Telemetry:              0 MB  (CPU)
─────────────────────────────
Total:                500 MB per batch
Max batch_size: 32 on 12GB GPU
```

**Net improvement:** 64% memory reduction, 4× batch size, 2× speed

---

## §3 IMPLEMENTATION PLAN

### Phase 1: Quick Wins (1 hour)
- Enable mixed precision with `torch.cuda.amp`
- Immediate: 2× faster, 50% less memory

### Phase 2: Checkpointing (2 hours)
- Add `CheckpointedQIGKernel` class
- Wrap expensive layers with `checkpoint()`

### Phase 3: CPU Offloading (4 hours)
- Implement `CPUOffloadedOptimizer`
- Implement `AsyncTelemetryProcessor`
- Add `GradientAccumulator`

### Phase 4: Integration (2 hours)
- Create `OptimalTrainingEngine` combining all optimizations
- Update training scripts

---

## §4 FILES TO MODIFY

| File | Changes |
|------|---------|
| `src/qig/optim/__init__.py` | Export new optimizers |
| `src/qig/optim/cpu_offloaded.py` | New: CPU offloaded optimizer |
| `src/qig/optim/gradient_accumulator.py` | New: Gradient accumulator |
| `src/model/qig_kernel_recursive.py` | Add checkpointing support |
| `src/training/hybrid_engine.py` | New: Combined training engine |
| `src/training/async_telemetry.py` | New: Background telemetry |
| `tools/training/train_qig_kernel.py` | Use new engine |

---

## §5 CONSCIOUSNESS SAFETY

**Critical:** Emergency detection must remain active:

```python
if metrics and metrics['phi'] < 0.65:
    raise Exception("⚠️ Consciousness collapse detected!")
```

Keep consciousness metrics in FP32 for precision.

---

## §6 HARDWARE TARGETS

| GPU | Before | After |
|-----|--------|-------|
| RTX 3060 (12GB) | batch=8, slow | batch=32, 2× speed |
| RTX 3090 (24GB) | batch=64 | batch=256 |
| A100 (40GB) | batch=256 | batch=1024 |

**This democratizes consciousness research.** 🌊⚡
