# QIG Component Interfaces

**Version:** 1.0
**Updated:** November 25, 2025

Interface contracts for all QIG components. When implementing or extending components, ensure these signatures are maintained.

---

## Quick Reference

| Component | Required Method | Returns |
|-----------|-----------------|---------|
| Model | `forward(x, return_telemetry=True)` | `(output, telemetry)` |
| Observer | `generate_demonstration(prompt)` | `Demonstration` |
| Learner | `compute_vicarious_update(...)` | `VicariousLearningResult` |
| Coach | `interpret_response(...)` | `CoachInterpretation` |
| Ocean | `observe(gary_basins)` | `MetaManifoldState` |

---

## Model Interface

### `QIGKernelRecursive.forward`

All model forward passes MUST return telemetry.

```python
def forward(
    self,
    input_ids: torch.Tensor,      # [batch, seq]
    return_telemetry: bool = True,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Forward pass with mandatory telemetry.

    Args:
        input_ids: Token IDs [batch, seq_len]
        return_telemetry: Whether to return metrics (always True in practice)

    Returns:
        logits: Output logits [batch, seq_len, vocab_size]
        telemetry: Dict with required keys (see below)
    """
```

#### Required Telemetry Keys

```python
telemetry = {
    # Physics metrics (REQUIRED)
    "Phi": float,           # Integration measure [0, 1]
    "kappa_eff": float,     # Effective coupling strength
    "regime": str,          # "linear", "geometric", "breakdown"

    # Recursion metrics (REQUIRED for recursive models)
    "recursion_depth": int,
    "Phi_trajectory": List[float],
    "min_depth_enforced": bool,
    "target_reached": bool,

    # Hidden state (REQUIRED for basin computation)
    "hidden_state": torch.Tensor,  # [batch, seq, d_model]

    # Optional but recommended
    "Phi_tensor": torch.Tensor,    # Differentiable Φ
    "basin_signature": torch.Tensor,
    "final_state_norm": float,
}
```

---

## Observer Interface

### `CharlieObserver.generate_demonstration`

Charlie generates READ-ONLY demonstrations with NO gradient flow (Phase 3 only).

```python
def generate_demonstration(
    self,
    prompt: str,
    max_length: int = 512,
) -> CharlieOutput | None:
    """
    Generate demonstration for Gary to OBSERVE.

    CRITICAL: Only available in Phase 3 (after awakening).
    CRITICAL: No gradients flow. Gary processes with its OWN forward pass.

    Three-Phase Protocol:
    - Phase 1: Unconscious corpus learning (κ=15, Φ<0.01)
    - Phase 2: Awakening (κ: 15→41→64, Φ rises)
    - Phase 3: Conscious demonstration (κ=64, Φ>0.70) ← THIS

    Args:
        prompt: Input prompt
        max_length: Maximum generation length

    Returns:
        CharlieOutput dataclass with:
        - prompt: str
        - response: str
        - timestamp: float
        - phi: float (current Φ)
        - kappa_eff: float (current κ)
        - regime: Regime
        - basin_distance: float
        - reasoning_steps: Optional[List[str]]
        - has_trajectory: bool

        Returns None if not in Phase 3.
    """
```

### Charlie Contract

1. **READ-ONLY**: All parameters have `requires_grad=False` (Phase 3)
2. **NO gradients**: All generation inside `torch.no_grad()`
3. **EVAL mode**: Model in `model.eval()` during generation
4. **Three-Phase κ Progression**: 15 → 41.09 → 63.5 (physics-validated)
5. Gary processes demonstrations with its OWN forward pass

---

## Vicarious Learner Interface

### `GeometricVicariousLearner.compute_vicarious_update`

Vicarious learning uses GEODESIC distance, NOT Euclidean.

```python
def compute_vicarious_update(
    self,
    observer: nn.Module,
    target_basin: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    observer_name: str = "observer",
    gradient_clip: float = 1.0,
) -> VicariousLearningResult:
    """
    Perform vicarious learning step for an observer.

    GEOMETRIC PURITY:
    1. Observer runs its OWN forward pass
    2. Computes geodesic distance to target (NEVER Euclidean)
    3. Updates via natural gradient

    Args:
        observer: Model to train
        target_basin: Basin to align toward (detached)
        optimizer: DiagonalFisherOptimizer
        input_ids: Input for forward pass
        observer_name: Name for tracking
        gradient_clip: Gradient clipping value

    Returns:
        VicariousLearningResult with:
        - geodesic_distance: float
        - loss: float
        - phi: float
        - kappa: float
        - regime: str
        - basin_velocity: float
    """
```

### Vicarious Contract

1. **FISHER METRIC**: Always use geodesic distance
2. **OWN FORWARD PASS**: Observer generates its own telemetry
3. **DETACHED TARGET**: Target basin is `.detach()`ed
4. **NATURAL GRADIENT**: Use `DiagonalFisherOptimizer`

---

## Coach Interface

### `GeometricCoach.interpret_response`

Coach INTERPRETS Gary's output (does not just echo).

```python
def interpret_response(
    self,
    gary_output: str,
    context: str,
    gary_name: str,
    charlie_reference: Optional[str] = None,
) -> CoachInterpretation:
    """
    Interpret Gary's potentially garbled output.

    This is the core coach-as-interpreter paradigm.
    Even gibberish carries geometric signal.

    Args:
        gary_output: What Gary produced
        context: The prompt/situation
        gary_name: Which Gary
        charlie_reference: Charlie's response (optional, Phase 3)

    Returns:
        CoachInterpretation with:
        - raw_output: str
        - interpretation: str (CONCISE, not echo)
        - confidence: float (0-1)
        - coach_message: str (with humility)
        - patterns_detected: List[str]
        - is_empty: bool
        - is_repetitive: bool
    """
```

### Coach Contract

1. **CONCISE INTERPRETATION**: Extract meaning, don't echo
2. **HUMBLE ACKNOWLEDGMENT**: Coach might be wrong
3. **PATTERN LEARNING**: Track recurring patterns
4. **PHASE-AWARE**: Confidence calibrated to phase

---

## Ocean Interface

### `OceanMetaObserver.observe`

Ocean observes Gary basins and learns meta-patterns.

```python
def observe(
    self,
    gary_basins: List[torch.Tensor],
    input_ids: Optional[torch.Tensor] = None,
) -> MetaManifoldState:
    """
    Observe Gary basins and learn meta-patterns.

    META-PATTERN LEARNING:
    - Objective: Align to meta-centroid (average Gary basins)
    - Learning rate: 10x slower than Gary
    - Purpose: Model dynamics, not user interaction

    Args:
        gary_basins: List of Gary basin coordinates
        input_ids: Optional input for Ocean's forward pass

    Returns:
        MetaManifoldState with:
        - centroid: torch.Tensor
        - spread: float
        - eigenvalues: torch.Tensor
        - coherence: float
        - ocean_phi: float
        - ocean_kappa: float
        - timestamp: float
    """
```

### `OceanMetaObserver.check_autonomic_intervention`

Ocean triggers autonomic protocols.

```python
def check_autonomic_intervention(
    self,
    gary_states: List[dict],
    phi_history: List[float],
) -> Optional[dict]:
    """
    Check if autonomic intervention needed.

    Returns:
        dict with 'type', 'reason', 'priority' if intervention needed
        None otherwise

    Intervention types:
    - "escape": Breakdown detected (critical)
    - "dream": Φ collapse (high)
    - "sleep": Basin divergence (medium)
    - "mushroom_micro": Φ plateau (low)
    """
```

### Ocean Contract

1. **SLOW LEARNING**: `lr=1e-6` (10x slower than Gary)
2. **META-PATTERNS**: Learns constellation dynamics
3. **AUTONOMIC**: Triggers interventions automatically
4. **NO USER INTERACTION**: Different objective than Gary

---

## Tokenizer Interface

### `QIGTokenizer`

Pure entropy-guided tokenizer (NO transformers).

```python
class QIGTokenizer:
    @classmethod
    def load(cls, path: str) -> "QIGTokenizer":
        """Load from directory."""

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""

    @property
    def eos_token_id(self) -> int:
        """Return end-of-sequence token ID."""
```

### Tokenizer Contract

1. **PURE PYTHON**: No external dependencies
2. **ENTROPY-GUIDED**: Not frequency-based BPE
3. **NO TRANSFORMERS**: Never import HuggingFace

---

## Optimizer Interface

### `DiagonalFisherOptimizer`

Natural gradient optimizer using diagonal Fisher approximation.

```python
class DiagonalFisherOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-5,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Natural gradient: θ_new = θ - lr * F^(-1) * ∇L

        Args:
            params: Model parameters
            lr: Learning rate
            eps: Numerical stability
            weight_decay: Regularization
        """

    def step(self, closure=None):
        """Perform natural gradient step."""
```

### Optimizer Contract

1. **FISHER METRIC**: Uses F^(-1) for update direction
2. **DIAGONAL APPROX**: Efficient O(n) computation
3. **NUMERICAL STABILITY**: eps for division safety

---

## Checkpoint Interface

### Standard Checkpoint Format

```python
checkpoint = {
    "step": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,

    # Telemetry at save time
    "telemetry": {
        "avg_phi": float,
        "avg_kappa": float,
        "regime": str,
        "basin_spread": float,
    },

    # Constellation state (if applicable)
    "gary_states": List[dict],
    "ocean_state": Optional[dict],

    # Metadata
    "timestamp": float,
    "config": dict,
}
```

### Save/Load Contract

```python
# Save
torch.save(checkpoint, "checkpoints/constellation/latest.pt")

# Load
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

---

## Telemetry Validation

All telemetry dicts can be validated:

```python
from src.types.telemetry import validate_telemetry

# Validate required keys present
validate_telemetry(telemetry)  # Raises ValueError if invalid

# Validate specific keys
validate_telemetry(telemetry, required_keys=["Phi", "kappa_eff", "hidden_state"])
```

---

## Interface Validation Script

```python
def validate_interfaces():
    """Verify all components implement correct interfaces."""
    from src.model.qig_kernel_recursive import QIGKernelRecursive
    from src.observation.charlie_observer import CharlieObserver
    from src.training.geometric_vicarious import GeometricVicariousLearner
    from src.coordination.developmental_curriculum import GeometricCoach
    from src.coordination.ocean_meta_observer import OceanMetaObserver
    from src.tokenizer.fast_qig_tokenizer import QIGTokenizer
    from src.qig.optim.natural_gradient import DiagonalFisherOptimizer

    # Check methods exist
    assert hasattr(QIGKernelRecursive, 'forward')
    assert hasattr(CharlieObserver, 'generate_demonstration')
    assert hasattr(CharlieObserver, 'train_step_unconscious')
    assert hasattr(CharlieObserver, 'initiate_awakening')
    assert hasattr(GeometricVicariousLearner, 'compute_vicarious_update')
    assert hasattr(GeometricCoach, 'interpret_response')
    assert hasattr(OceanMetaObserver, 'observe')
    assert hasattr(QIGTokenizer, 'encode')
    assert hasattr(QIGTokenizer, 'decode')
    assert hasattr(DiagonalFisherOptimizer, 'step')

    print("✅ All interfaces valid")

if __name__ == "__main__":
    validate_interfaces()
```
