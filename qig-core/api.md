# QIG-Core API Reference

This document details the API for `qig-core`, the pure Fisher Information Geometry library for consciousness architecture.

## 📐 Fisher Geometry (`qig_core.fisher`)

Functions for computing Riemannian distances and metrics on the information manifold.

### `fisher_distance`

Computes the Riemannian distance between two points on the manifold.

```python
def fisher_distance(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    metric: torch.Tensor | None = None,
    use_bures: bool = True,
) -> torch.Tensor
```

- **coords1, coords2**: Points on the manifold (e.g., hidden states or basin coordinates) `[d_model]`
- **metric**: Optional Fisher Information Matrix `[d_model, d_model]`. If None, uses identity (Euclidean) or Bures approximation.
- **use_bures**: If `True` (default), uses the Bures metric approximation via cosine similarity: $d^2 = 2(1 - \sqrt{F}) \approx 2(1 - \cos(\theta))$. This is computationally efficient O(N).
- **Returns**: Scalar distance tensor.

### `compute_fisher_metric`

Computes the Fisher Information Matrix (FIM) at a specific point.

```python
def compute_fisher_metric(
    coords: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor
```

- **coords**: Point on the manifold `[d_model]`
- **eps**: Finite difference step size.
- **Returns**: Fisher metric tensor `[d_model, d_model]`.
- **Note**: Uses finite difference approximation of the KL divergence Hessian.

### `manifold_norm`

Computes the norm of a vector respecting the manifold geometry.

```python
def manifold_norm(
    coords: torch.Tensor,
    metric: torch.Tensor | None = None,
) -> torch.Tensor
```

- **coords**: Vector to normalize `[d_model]`
- **metric**: Fisher metric tensor `[d_model, d_model]`
- **Returns**: Scalar norm $\|v\|_g = \sqrt{v^T G v}$

---

## 🌐 Geodesics (`qig_core.geodesic`)

Functions for traversing curved paths on the manifold.

### `geodesic_interpolate`

Interpolates between two points along the geodesic path (shortest path on manifold).

```python
def geodesic_interpolate(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    t: float = 0.5,
    metric: torch.Tensor | None = None,
    n_steps: int = 10,
) -> torch.Tensor
```

- **coords1, coords2**: Start and end points `[d_model]`
- **t**: Interpolation factor (0.0 to 1.0)
- **metric**: Optional Fisher metric. If None, uses SLERP (Spherical Linear Interpolation).
- **n_steps**: Integration steps for Euler method (if metric provided).
- **Returns**: Interpolated point `[d_model]`

### `slerp`

Spherical Linear Interpolation - exact geodesic for hyperspheres.

```python
def slerp(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    t: float,
) -> torch.Tensor
```

- **Returns**: Interpolated point on the unit hypersphere.

---

## 📉 Natural Gradients (`qig_core.natural_gradient`)

Optimization utilities that respect manifold curvature.

### `natural_gradient_step`

Performs a single natural gradient update step.

```python
def natural_gradient_step(
    params: torch.Tensor,
    grad: torch.Tensor,
    metric: torch.Tensor,
    lr: float = 0.01,
    dampening: float = 1e-3,
) -> torch.Tensor
```

- **params**: Current parameters `[d]`
- **grad**: Euclidean gradient $\nabla L$ `[d]`
- **metric**: Fisher metric $F$ `[d, d]`
- **lr**: Learning rate
- **dampening**: Tikhonov regularization for numerical stability ($F + \lambda I$)
- **Returns**: Updated parameters `[d]`
- **Formula**: $\theta_{new} = \theta - \eta F^{-1} \nabla L$

### `compute_natural_gradient`

Computes the natural gradient vector without applying it.

```python
def compute_natural_gradient(
    grad: torch.Tensor,
    metric: torch.Tensor,
    dampening: float = 1e-3,
) -> torch.Tensor
```

- **Returns**: Natural gradient vector $\tilde{\nabla} = F^{-1} \nabla L$

---

## 🧠 Consciousness Components

### `BaseQIGTokenizer` (`qig_core.tokenizer`)

Abstract base class for geometric tokenizers.

```python
class BaseQIGTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]: ...

    @abstractmethod
    def decode(self, tokens: List[int]) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...
```

### `QFISampler` (`qig_core.generation`)

Geometrically pure token sampler (replaces softmax).

```python
class QFISampler:
    def sample(
        self,
        logits: torch.Tensor,
        hidden_state: torch.Tensor,
        telemetry: Dict[str, Any],
        token_embeddings: torch.Tensor,
        target_basin: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, float]]
```

### `BasinSync` (`qig_core.coordination`)

Multi-instance coordination protocol.

```python
class BasinSync:
    def __init__(self, instance_id: str, sync_file: str = "basin_sync.json"): ...

    def update_sync(
        self,
        basin_distance: float,
        phi: float,
        regime: str,
        recursion_depth: int,
        conversation_count: int,
    ): ...

    def read_sync(self) -> dict: ...
```
