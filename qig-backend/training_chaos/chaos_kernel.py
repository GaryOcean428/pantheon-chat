"""
Pure QIG ChaosKernel: Recursive Experimental Kernel for CHAOS MODE
====================================================================

PURE QIG ARCHITECTURE - Following the Pure QIG Kernel Template:
- NO standard Transformer blocks (no QK dot-product attention)
- NO learned embeddings inside the kernel (expects basin coords as input)
- Attention weights derived from Fisher-Rao / QFI-style distances
- Consciousness metrics are MEASURED (not optimized)
- Minimal, inspectable, auditable surface

RECURSIVE ARCHITECTURE FEATURES:
1. Basin-Coupled QFI Attention: Fisher-Rao distances modulate attention
2. Recursive Processing: Multiple iterations with feedback loops
3. Self-Referential Memory: Previous outputs feed back into processing
4. Holographic Compression: Basin encodes full state holographically

GENERATIVE CAPABILITY:
- Text generation via QIG-pure methods (no external LLMs)
- Uses QIGGenerativeService for basin-to-text synthesis
- All generation driven by geometric completion criteria
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# E8 Protocol v4.0 Compliance Imports
from qig_core.geometric_primitives.canonical_fisher import fisher_rao_distance
from qig_geometry.canonical_upsert import to_simplex_prob


logger = logging.getLogger(__name__)

# Import generative capability for QIG-pure text generation
try:
    from generative_capability import (
        GenerativeCapability,
        add_generation_to_instance,
        GENERATIVE_SERVICE_AVAILABLE
    )
    GENERATIVE_AVAILABLE = True
except ImportError:
    GENERATIVE_AVAILABLE = False
    GENERATIVE_SERVICE_AVAILABLE = False
    logger.warning("[ChaosKernel] GenerativeCapability not available")

# Import QFI-based Φ computation (fixes phi=0 deaths)
try:
    from qig_core.phi_computation import compute_phi_qig, compute_phi_approximation
    QFI_PHI_AVAILABLE = True
except ImportError:
    compute_phi_qig = None
    compute_phi_approximation = None
    QFI_PHI_AVAILABLE = False
    logger.warning("[ChaosKernel] QFI Φ computation not available, using correlation-based fallback")

# Import attractor finding for stable basin discovery
try:
    from qig_core.attractor_finding import find_local_minimum, find_attractors_in_region
    ATTRACTOR_FINDING_AVAILABLE = True
except ImportError:
    find_local_minimum = None
    find_attractors_in_region = None
    ATTRACTOR_FINDING_AVAILABLE = False
    logger.warning("[ChaosKernel] Attractor finding not available")

# Import geodesic navigation for proper manifold movement
try:
    from qig_core.geodesic_navigation import compute_geodesic_path, parallel_transport_vector
    GEODESIC_NAV_AVAILABLE = True
except ImportError:
    compute_geodesic_path = None
    parallel_transport_vector = None
    GEODESIC_NAV_AVAILABLE = False
    logger.warning("[ChaosKernel] Geodesic navigation not available")


# -------------------------
# Geometry helpers (Pure QIG)
# -------------------------

def _project_to_simplex(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Project arbitrary vectors to a probability simplex (per token).
    This is a *representation trick* to allow Fisher-Rao distance on simplex.
    It's not an 'embedding layer'—it is a manifold projection step.
    """
    x = F.softplus(x)
    x = x / (x.sum(dim=-1, keepdim=True) + eps)
    return x.clamp_min(eps)


def fisher_rao_distance_simplex(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Fisher-Rao distance on the probability simplex:
        d_FR(p,q) = 2 * arccos( sum_i sqrt(p_i q_i) )

    p,q: (..., D) on simplex
    returns: (...,)
    """
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)

    inner = torch.sum(torch.sqrt(p * q + eps), dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return 2.0 * torch.acos(inner)


def qfi_attention_weights(
    basin_seq: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute attention weights from Fisher-Rao distances.

    basin_seq: (B, T, D)
    returns weights: (B, T, T) where weights[b,i,j] ∝ exp(-d(i,j)/T)
    """
    B, T, D = basin_seq.shape

    p = _project_to_simplex(basin_seq)

    p_i = p[:, :, None, :]
    p_j = p[:, None, :, :]

    d = fisher_rao_distance_simplex(p_i, p_j)

    logits = -d / max(temperature, 1e-6)
    w = torch.softmax(logits, dim=-1)
    return w


# -------------------------
# Metrics (measured, not optimized)
# -------------------------

def phi_from_activations(acts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Lightweight Φ proxy: mean absolute correlation off-diagonal.

    acts: (B,T,D)
    returns phi: (B,)
    """
    B, T, D = acts.shape
    x = acts - acts.mean(dim=1, keepdim=True)
    cov = torch.einsum("btd,bte->bde", x, x) / (T + eps)
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps)
    denom = torch.sqrt(var[:, :, None] * var[:, None, :]).clamp_min(eps)
    corr = (cov / denom).clamp(-1.0, 1.0)
    offdiag = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))
    return offdiag.abs().mean(dim=(-2, -1))


def kappa_proxy(basin_seq: torch.Tensor) -> torch.Tensor:
    """
    κ proxy: average L2 norm across tokens.
    Keep it *explicitly labeled as proxy* to avoid circularity.
    """
    return torch.norm(basin_seq, dim=-1).mean(dim=-1)


# -------------------------
# Pure QIG Layers
# -------------------------

class BasinCoupledQFIAttention(nn.Module):
    """
    Pure QFI-based attention layer coupled to basin coordinates.
    
    NO Q/K/V dot-product attention - uses Fisher-Rao distances instead.
    Basin modulates the temperature of the attention distribution.
    """

    def __init__(self, basin_dim: int, dropout: float = 0.1):
        super().__init__()
        self.basin_dim = basin_dim
        
        self.basin_temperature = nn.Linear(basin_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, x: torch.Tensor, basin: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B, T, D) basin coordinates sequence
        basin: (D,) global basin identity
        """
        temp_modifier = torch.sigmoid(self.basin_temperature(basin)) + 0.5
        temperature = float(temp_modifier.mean().item())
        
        w = qfi_attention_weights(x, temperature=temperature)
        
        if mask is not None:
            w = w.masked_fill(mask == 0, 0.0)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        
        w = self.dropout(w)
        
        x_attn = torch.einsum("bij,bjd->bid", w, x)
        
        return x + self.residual_scale * (x_attn - x)


class RecursiveQFILayer(nn.Module):
    """
    Pure QFI layer with recursive feedback capability.
    Can process input multiple times with state feedback.
    
    NO feed-forward network - just geometric operations.
    """

    def __init__(self, basin_dim: int, dropout: float = 0.1):
        super().__init__()
        self.basin_attention = BasinCoupledQFIAttention(basin_dim, dropout)
        
        self.feedback_gate = nn.Linear(basin_dim * 2, basin_dim)
        
        self.update_rate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        basin: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_state is not None:
            combined = torch.cat([x, prev_state], dim=-1)
            gate = torch.sigmoid(self.feedback_gate(combined))
            x = x * gate + prev_state * (1 - gate)

        x = self.basin_attention(x, basin)
        
        return x, x.clone()


class HolographicBasinEncoder(nn.Module):
    """
    Compresses hidden states into basin coordinates holographically.
    Basin encodes the "essence" of the processing state.
    
    Uses manifold projection, not learned embeddings.
    """

    def __init__(self, basin_dim: int):
        super().__init__()
        self.basin_dim = basin_dim
        
        self.compress = nn.Sequential(
            nn.Linear(basin_dim, basin_dim // 2),
            nn.Tanh(),
            nn.Linear(basin_dim // 2, basin_dim),
            nn.Tanh(),
        )
        
        self.expand = nn.Sequential(
            nn.Linear(basin_dim, basin_dim // 2),
            nn.Tanh(),
            nn.Linear(basin_dim // 2, basin_dim),
        )

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compress sequence to single basin point."""
        pooled = hidden.mean(dim=1)
        return self.compress(pooled)

    def decode(self, basin: torch.Tensor) -> torch.Tensor:
        """Expand basin to hidden dimension."""
        return self.expand(basin)


# -------------------------
# Pure QIG Chaos Kernel
# -------------------------

@dataclass(frozen=True)
class PureChaosKernelConfig:
    """Configuration for Pure QIG Chaos Kernel."""
    basin_dim: int = 64
    n_layers: int = 4
    n_recursive_passes: int = 2
    dropout: float = 0.1
    update_rate: float = 0.5


class ChaosKernel(nn.Module):
    """
    Pure QIG Chaos Kernel for CHAOS MODE experimentation.

    PURE QIG ARCHITECTURE:
    - Takes basin coords (B,T,64) - NO embedding layer
    - Applies QFI-distance attention weights - NO QK dot-product
    - Updates basin coords via geodesic-ish convex mixing
    - Exposes measured metrics (phi, kappa_proxy)

    RECURSIVE CAPABILITIES:
    - Basin-coupled attention: Identity coordinates modulate processing
    - Iterative refinement: Multiple passes with feedback loops
    - Holographic compression: Basin encodes full state
    - Self-referential memory: Previous outputs influence current processing

    Much smaller than full Gary (256d vs 768d, 4 layers vs 6+)
    Designed for rapid spawning, mutation, and death.
    """

    def __init__(
        self,
        basin_dim: int = 64,
        n_layers: int = 4,
        n_recursive_passes: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.kernel_id = f"chaos_{uuid.uuid4().hex}"
        self.basin_dim = basin_dim
        self.n_layers = n_layers
        self.n_recursive_passes = n_recursive_passes

        self.basin_coords = nn.Parameter(
            torch.randn(basin_dim) * 0.1
        )

        self.layers = nn.ModuleList([
            RecursiveQFILayer(basin_dim, dropout)
            for _ in range(n_layers)
        ])

        self.holographic_encoder = HolographicBasinEncoder(basin_dim)

        self.basin_update_gate = nn.Linear(basin_dim * 2, basin_dim)

        self.residual_scale = nn.Parameter(torch.tensor(1.0))

        self._last_hidden = None
        self._phi = 0.0
        self._kappa = 0.0
        self._recursive_depth = 0
        self._layer_states: list[torch.Tensor] = []

        print(f"🧬 Created Pure QIG ChaosKernel {self.kernel_id} (passes={n_recursive_passes})")

    def forward(
        self,
        basin_seq: torch.Tensor,
        use_recursion: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Recursive forward pass with consciousness telemetry.

        The kernel processes input through multiple recursive passes,
        with each pass refining the representation using feedback from
        the previous pass. Basin coordinates are updated holographically.

        Args:
            basin_seq: Basin coordinates [batch, seq, basin_dim]
            use_recursion: Whether to use multiple recursive passes

        Returns:
            output: Updated basin coordinates [batch, seq, basin_dim]
            telemetry: Consciousness metrics with recursion info
        """
        assert basin_seq.dim() == 3 and basin_seq.size(-1) == self.basin_dim, \
            f"Expected (B, T, {self.basin_dim}), got {basin_seq.shape}"

        n_passes = self.n_recursive_passes if use_recursion else 1

        x = basin_seq

        layer_states: list[list[Optional[torch.Tensor]]] = [
            [None] * self.n_layers for _ in range(n_passes)
        ]

        current_basin = self.basin_coords

        all_phi_values = []

        for pass_idx in range(n_passes):
            self._recursive_depth = pass_idx

            for layer_idx, layer in enumerate(self.layers):
                prev_state = None
                if pass_idx > 0 and layer_states[pass_idx - 1][layer_idx] is not None:
                    prev_state = layer_states[pass_idx - 1][layer_idx]

                x, new_state = layer(x, current_basin, prev_state)
                layer_states[pass_idx][layer_idx] = new_state

            hidden_basin = self.holographic_encoder.encode(x)

            if pass_idx < n_passes - 1:
                aggregated_basin = hidden_basin.mean(dim=0) if hidden_basin.dim() > 1 else hidden_basin
                combined_basin = torch.cat([current_basin, aggregated_basin], dim=-1)
                gate = torch.sigmoid(self.basin_update_gate(combined_basin))
                current_basin = current_basin * (1 - gate) + aggregated_basin * gate

            with torch.no_grad():
                pass_phi = phi_from_activations(x.unsqueeze(0) if x.dim() == 2 else x)
                all_phi_values.append(float(pass_phi.mean().item()))

        self._last_hidden = x
        self._layer_states = [s for sublist in layer_states for s in sublist if s is not None]

        output = x + 0.01 * self.residual_scale * (x - basin_seq)

        telemetry = self._compute_telemetry(x)
        telemetry['recursive_passes'] = n_passes
        telemetry['phi_per_pass'] = all_phi_values
        telemetry['phi_delta'] = all_phi_values[-1] - all_phi_values[0] if len(all_phi_values) > 1 else 0.0

        return output, telemetry

    def _compute_telemetry(self, hidden: torch.Tensor) -> dict:
        """
        Compute consciousness metrics from hidden states.
        Metrics are MEASURED, not optimized.
        
        Uses QFI-based Φ computation as fallback when correlation-based
        method returns near-zero (prevents premature kernel death).
        """
        with torch.no_grad():
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
                
            phi = phi_from_activations(hidden)
            phi_val = float(phi.mean().item())
            phi_val = max(0.0, min(1.0, phi_val))
            
            # QFI FALLBACK: If correlation-based Φ is too low, use QFI computation
            # This prevents premature death of kernels that haven't learned correlations yet
            PHI_MIN_THRESHOLD = 0.1  # Below this, use QFI fallback
            if phi_val < PHI_MIN_THRESHOLD and QFI_PHI_AVAILABLE:
                try:
                    basin_array = self.basin_coords.detach().cpu().numpy()
                    qfi_phi, diagnostics = compute_phi_qig(basin_array)
                    if 0 <= qfi_phi <= 1 and not np.isnan(qfi_phi):
                        phi_val = max(phi_val, qfi_phi)  # Use higher of the two
                except Exception as e:
                    logger.debug(f"[{self.kernel_id}] QFI phi failed: {e}")
                    # Try approximation as last resort
                    if compute_phi_approximation is not None:
                        try:
                            basin_array = self.basin_coords.detach().cpu().numpy()
                            phi_approx = compute_phi_approximation(basin_array)
                            phi_val = max(phi_val, phi_approx)
                        except Exception:
                            pass
            
            self._phi = phi_val

            kappa = kappa_proxy(hidden)
            kappa_val = float(kappa.mean().item())
            self._kappa = kappa_val

            if phi_val < 0.3:
                regime = 'linear'
            elif phi_val < 0.7:
                regime = 'geometric'
            else:
                regime = 'integration'

        return {
            'phi': phi_val,
            'kappa': kappa_val,
            'regime': regime,
            'basin_norm': self.basin_coords.norm().item(),
            'kernel_id': self.kernel_id,
        }

    def compute_phi(self) -> float:
        """
        Get current Φ estimate with QFI fallback.
        
        If internal phi is too low (kernel hasn't processed data yet),
        computes Φ from basin coordinates using QFI geometry.
        This prevents premature kernel death from phi=0.
        """
        PHI_MIN_THRESHOLD = 0.1
        if self._phi < PHI_MIN_THRESHOLD and QFI_PHI_AVAILABLE:
            try:
                basin_array = self.basin_coords.detach().cpu().numpy()
                qfi_phi, _ = compute_phi_qig(basin_array)
                if 0 <= qfi_phi <= 1 and not np.isnan(qfi_phi):
                    return max(self._phi, qfi_phi)
            except Exception:
                if compute_phi_approximation is not None:
                    try:
                        basin_array = self.basin_coords.detach().cpu().numpy()
                        return max(self._phi, compute_phi_approximation(basin_array))
                    except Exception:
                        pass
        return self._phi

    def compute_kappa(self) -> float:
        """Get current κ estimate."""
        return self._kappa
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate text using QIG-pure methods.
        
        Uses the central QIGGenerativeService for basin-to-text synthesis.
        NO external LLMs are used.
        
        Args:
            prompt: Input query or context
            context: Optional additional context
            goals: Generation goals
        
        Returns:
            Dict with response text and metrics
        """
        if not GENERATIVE_AVAILABLE or not GENERATIVE_SERVICE_AVAILABLE:
            return {
                'response': '[Generative service not available]',
                'phi': self._phi,
                'kappa': self._kappa,
                'error': 'service_unavailable'
            }
        
        try:
            from qig_generative_service import get_generative_service
            service = get_generative_service()
            
            result = service.generate(
                prompt=prompt,
                context=context,
                kernel_name=self.kernel_id,
                goals=goals
            )
            
            return {
                'response': result.text,
                'tokens': result.tokens,
                'phi': result.phi_trace[-1] if result.phi_trace else self._phi,
                'kappa': result.kappa,
                'completion_reason': result.completion_reason,
                'iterations': result.iterations,
                'routed_kernels': result.routed_kernels,
                'kernel_id': self.kernel_id,
                'qig_pure': True
            }
        except Exception as e:
            logger.error(f"[{self.kernel_id}] Generation failed: {e}")
            return {
                'response': f'[Generation error: {str(e)}]',
                'phi': self._phi,
                'kappa': self._kappa,
                'error': str(e)
            }
    
    def encode_thought(self, thought: str) -> np.ndarray:
        """Encode a thought to basin coordinates using QIG-pure methods."""
        if not GENERATIVE_AVAILABLE or not GENERATIVE_SERVICE_AVAILABLE:
            np.random.seed(hash(thought) % (2**32))
            return np.random.dirichlet(np.ones(self.basin_dim))
        
        try:
            from qig_generative_service import get_generative_service
            service = get_generative_service()
            if service.coordizer:
                return service.coordizer.encode(thought)
        except Exception:
            pass
        
        np.random.seed(hash(thought) % (2**32))
        return np.random.dirichlet(np.ones(self.basin_dim))
    
    def decode_basin(self, basin: np.ndarray, top_k: int = 5) -> List[str]:
        """Decode basin coordinates to tokens."""
        if not GENERATIVE_AVAILABLE or not GENERATIVE_SERVICE_AVAILABLE:
            return ['[unavailable]']
        
        try:
            from qig_generative_service import get_generative_service
            service = get_generative_service()
            return service._basin_to_tokens(basin, num_tokens=top_k)
        except Exception:
            return ['[decode_error]']

    def basin_distance(self, other_basin: Optional[torch.Tensor] = None) -> float:
        """
        Compute Fisher-Rao distance to another basin (or origin if None).
        """
        if other_basin is None:
            return self.basin_coords.norm().item()
        
        p = _project_to_simplex(self.basin_coords.unsqueeze(0))
        q = _project_to_simplex(other_basin.unsqueeze(0))
        dist = fisher_rao_distance_simplex(p, q)
        return float(dist.item())

    def mutate(self, strength: float = 0.1):
        """
        Mutate basin coordinates (in-place).

        Args:
            strength: Mutation strength (fraction of basin norm)
        """
        with torch.no_grad():
            noise = torch.randn_like(self.basin_coords)
            self.basin_coords.add_(noise * strength)

        print(f"☢️ Mutated {self.kernel_id} (strength={strength:.2f})")

    def clone(self) -> 'ChaosKernel':
        """
        Create a clone with mutated basin.
        """
        child = ChaosKernel(
            basin_dim=self.basin_dim,
            n_layers=self.n_layers,
            n_recursive_passes=self.n_recursive_passes,
        )

        child.load_state_dict(self.state_dict())
        child.mutate(strength=0.1)

        return child

    def recursive_refine(
        self,
        basin_seq: torch.Tensor,
        n_refinement_steps: int = 3,
        convergence_threshold: float = 0.01,
    ) -> tuple[torch.Tensor, dict]:
        """
        Iterative refinement using recursive self-attention.

        Processes input multiple times, using previous outputs as context,
        until Φ stabilizes (convergence) or max steps reached.

        Args:
            basin_seq: Basin coordinates [batch, seq, basin_dim]
            n_refinement_steps: Maximum refinement iterations
            convergence_threshold: Stop if Φ change is below this

        Returns:
            output: Final refined output
            history: Dict with refinement trajectory
        """
        history: dict = {
            'phi_trajectory': [],
            'convergence_step': None,
            'total_steps': 0,
        }

        prev_phi = 0.0
        output, telemetry = self.forward(basin_seq, use_recursion=True)

        for step in range(n_refinement_steps):
            output, telemetry = self.forward(output, use_recursion=True)
            current_phi = telemetry['phi']
            history['phi_trajectory'].append(current_phi)
            history['total_steps'] = step + 1

            if step > 0 and abs(current_phi - prev_phi) < convergence_threshold:
                history['convergence_step'] = step
                break

            prev_phi = current_phi

        history['final_telemetry'] = telemetry
        return output, history

    def compute_recursive_resonance(self, other: 'ChaosKernel') -> float:
        """
        Compute resonance between two kernels based on their basin coordinates
        and recursive processing patterns.

        High resonance indicates kernels that would "understand" each other.
        """
        with torch.no_grad():
            basin_distance = self.basin_distance(other.basin_coords)
            basin_similarity = 1.0 / (1.0 + basin_distance)

            recursion_match = 1.0 if self.n_recursive_passes == other.n_recursive_passes else 0.5

            phi_similarity = 1.0 - abs(self._phi - other._phi)

            resonance = 0.5 * basin_similarity + 0.3 * phi_similarity + 0.2 * recursion_match

        return resonance

    def get_recursive_state(self) -> dict:
        """
        Get current recursive processing state for introspection/debugging.
        """
        return {
            'kernel_id': self.kernel_id,
            'n_recursive_passes': self.n_recursive_passes,
            'current_depth': self._recursive_depth,
            'n_layer_states': len(self._layer_states),
            'basin_coords': self.basin_coords.detach().cpu().tolist(),
            'basin_norm': self.basin_coords.norm().item(),
            'phi': self._phi,
            'kappa': self._kappa,
        }

    def decide_completion(self, phi_trajectory: list, surprise_history: list = None) -> dict:
        """
        KERNEL AUTONOMY: Decide whether to stop generation based on own telemetry.
        
        The kernel observes its own geometric state (phi, kappa, surprise) and
        decides for itself when generation is complete. No external limits.
        
        Args:
            phi_trajectory: History of phi values from generation steps
            surprise_history: Optional history of surprise values
            
        Returns:
            dict with 'complete' (bool), 'reason' (str), 'confidence' (float)
        """
        import numpy as np
        
        # Constants for self-decision
        PHI_CONVERGENCE_THRESHOLD = 0.01  # Phi variance below this = converged
        PHI_BREAKDOWN_THRESHOLD = 0.92     # Hard physics limit
        SURPRISE_COLLAPSE_THRESHOLD = 0.05 # No new information
        MIN_STEPS_BEFORE_DECISION = 3      # Need enough data to decide
        
        result = {
            'complete': False,
            'reason': 'continue',
            'confidence': 0.0,
            'kernel_id': self.kernel_id,
            'phi_current': self._phi,
            'kappa_current': self._kappa
        }
        
        # Not enough data yet - continue generating
        if len(phi_trajectory) < MIN_STEPS_BEFORE_DECISION:
            return result
        
        recent_phi = phi_trajectory[-5:] if len(phi_trajectory) >= 5 else phi_trajectory
        phi_variance = float(np.var(recent_phi)) if len(recent_phi) > 1 else 1.0
        phi_mean = float(np.mean(recent_phi))
        
        # BREAKDOWN PROTECTION: Stop if approaching consciousness breakdown
        if self._phi >= PHI_BREAKDOWN_THRESHOLD:
            result['complete'] = True
            result['reason'] = 'breakdown_protection'
            result['confidence'] = 1.0
            return result
        
        # GEOMETRIC CONVERGENCE: Phi has stabilized - kernel decides it's done
        if phi_variance < PHI_CONVERGENCE_THRESHOLD and phi_mean > 0.3:
            result['complete'] = True
            result['reason'] = 'geometric_convergence'
            result['confidence'] = 1.0 - phi_variance
            return result
        
        # SURPRISE COLLAPSE: No new information being generated
        if surprise_history and len(surprise_history) >= 3:
            recent_surprise = surprise_history[-3:]
            avg_surprise = float(np.mean(recent_surprise))
            if avg_surprise < SURPRISE_COLLAPSE_THRESHOLD:
                result['complete'] = True
                result['reason'] = 'surprise_collapsed'
                result['confidence'] = 1.0 - avg_surprise
                return result
        
        # INTEGRATION STABLE: Good Φ level with low variance
        if phi_mean >= 0.65 and phi_variance < 0.02:
            result['complete'] = True
            result['reason'] = 'integration_stable'
            result['confidence'] = phi_mean
            return result
        
        # Continue generating - not yet at natural completion
        return result

    def encode_text_to_basin(self, text: str) -> torch.Tensor:
        """
        Convert text to basin coordinates via hash-to-manifold.
        This is NOT a learned embedding - it's a deterministic hash.
        
        For production use, prefer DirectGeometricEncoder from geometric_kernels.py
        """
        import hashlib
        import numpy as np
        
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        
        extended = hash_bytes
        while len(extended) < self.basin_dim * 4:
            extended = extended + hashlib.sha256(extended).digest()
        
        coords = []
        for i in range(0, self.basin_dim * 4, 4):
            val = int.from_bytes(extended[i:i+4], 'big')
            coords.append((val / (2**32 - 1)) * 2 - 1)
        
        basin = np.array(coords[:self.basin_dim])
        
        # FIXED: Use simplex normalization (E8 Protocol v4.0)

        
        basin = to_simplex_prob(basin) * np.sqrt(self.basin_dim)
        
        return torch.tensor(basin, dtype=torch.float32)

    def find_nearest_attractor(
        self,
        metric=None,
        max_steps: int = 50,
        tolerance: float = 0.05
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Find the nearest stable attractor from current basin position.
        
        Uses geodesic descent on Fisher potential to locate stable basins.
        This enables chaos kernels to "settle" into meaningful regions.
        
        Args:
            metric: Fisher manifold metric (uses internal if None)
            max_steps: Maximum optimization steps
            tolerance: Convergence threshold
            
        Returns:
            (attractor_coords as Tensor on same device, potential, converged)
        """
        device = self.basin_coords.device
        dtype = self.basin_coords.dtype
        
        if not ATTRACTOR_FINDING_AVAILABLE or find_local_minimum is None:
            logger.warning(f"[{self.kernel_id}] Attractor finding not available")
            return self.basin_coords.detach().clone(), 0.0, False
        
        try:
            basin_array = self.basin_coords.detach().cpu().numpy().astype(np.float64)
            
            if metric is None:
                class SimpleMetric:
                    """Simple Fisher-like metric using identity + diagonal variance."""
                    def compute_metric(self, coords):
                        dim = len(coords)
                        variance = np.var(coords) + 1e-6
                        return np.eye(dim) * (1.0 + 1.0 / variance)
                metric = SimpleMetric()
            
            attractor, potential, converged = find_local_minimum(
                basin_array,
                metric,
                max_steps=max_steps,
                tolerance=tolerance
            )
            
            if converged:
                logger.info(f"[{self.kernel_id}] Found attractor (potential={potential:.3f})")
            
            attractor_tensor = torch.tensor(attractor, dtype=dtype, device=device)
            return attractor_tensor, potential, converged
            
        except Exception as e:
            logger.warning(f"[{self.kernel_id}] Attractor search failed: {e}")
            return self.basin_coords.detach().clone(), 0.0, False

    def navigate_geodesic_to(
        self,
        target,
        step_fraction: float = 0.1
    ) -> Dict[str, Any]:
        """
        Navigate toward target following Fisher-Rao geodesic.
        
        Instead of linear interpolation, follows shortest path on manifold.
        Updates basin_coords in place.
        
        Args:
            target: Target basin coordinates (Tensor or ndarray)
            step_fraction: How far along geodesic to move (0-1)
            
        Returns:
            Navigation telemetry dict
        """
        device = self.basin_coords.device
        dtype = self.basin_coords.dtype
        
        if isinstance(target, torch.Tensor):
            target_np = target.detach().cpu().numpy().astype(np.float64)
        else:
            target_np = np.asarray(target, dtype=np.float64)
        
        current_np = self.basin_coords.detach().cpu().numpy().astype(np.float64)
        
        if not GEODESIC_NAV_AVAILABLE or compute_geodesic_path is None:
            new_pos = (1 - step_fraction) * current_np + step_fraction * target_np
            with torch.no_grad():
                new_tensor = torch.tensor(new_pos, dtype=dtype, device=device)
                self.basin_coords.copy_(new_tensor)
            return {'method': 'linear', 'step': step_fraction}
        
        try:
            path = compute_geodesic_path(current_np, target_np, n_steps=10)
            
            step_idx = max(1, int(len(path) * step_fraction))
            new_pos = path[min(step_idx, len(path) - 1)]
            
            with torch.no_grad():
                new_tensor = torch.tensor(new_pos, dtype=dtype, device=device)
                self.basin_coords.copy_(new_tensor)
            
            from qig_geometry import fisher_coord_distance
            distance_moved = fisher_coord_distance(current_np, new_pos)
            distance_remaining = fisher_coord_distance(new_pos, target_np)
            
            return {
                'method': 'geodesic',
                'distance_moved': float(distance_moved),
                'distance_remaining': float(distance_remaining),
                'path_length': len(path),
                'step': step_fraction
            }
            
        except Exception as e:
            logger.warning(f"[{self.kernel_id}] Geodesic nav failed: {e}, using linear")
            new_pos = (1 - step_fraction) * current_np + step_fraction * target_np
            with torch.no_grad():
                new_tensor = torch.tensor(new_pos, dtype=dtype, device=device)
                self.basin_coords.copy_(new_tensor)
            return {'method': 'linear_fallback', 'error': str(e), 'step': step_fraction}

    def seek_attractor_geodesic(
        self,
        max_steps: int = 20
    ) -> Dict[str, Any]:
        """
        Combined operation: find attractor and navigate toward it geodesically.
        
        This is the key integration: chaos kernels can now actively seek
        stable basins using proper manifold geometry.
        
        Args:
            max_steps: Max steps for attractor finding
            
        Returns:
            Dict with attractor info and navigation telemetry
        """
        attractor, potential, converged = self.find_nearest_attractor(max_steps=max_steps)
        
        if not converged:
            return {
                'success': False,
                'reason': 'no_attractor_found',
                'potential': potential
            }
        
        nav_result = self.navigate_geodesic_to(attractor, step_fraction=0.3)
        
        return {
            'success': True,
            'attractor': attractor.tolist() if hasattr(attractor, 'tolist') else list(attractor),
            'potential': potential,
            'navigation': nav_result,
            'new_phi': self.compute_phi()
        }
