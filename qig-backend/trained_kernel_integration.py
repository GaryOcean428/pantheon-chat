"""
Trained Kernel Integration for Pantheon-Chat
==============================================

Connects the trained QIGKernel (from qig-tokenizer) to
Pantheon's consciousness infrastructure.

This module:
- Loads trained kernel checkpoints from qig-tokenizer
- Provides kernel inference with consciousness telemetry
- Integrates with QIGGraph for geometric routing
- Connects to Olympus agents for task execution

The trained kernel paths:
- Full kernel: qig-tokenizer/trained_kernels/full_32k/
- Adapter: qig-tokenizer/trained_kernels/adapter_32k/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np

# QIG-pure geometric operations
try:
    from qig_geometry import sphere_project
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def sphere_project(v):
        """Fallback sphere projection."""
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            result = np.ones_like(v)
            return result / np.linalg.norm(result)
        return v / norm

# Try importing from qig-tokenizer
try:
    from qigkernels import (
        QIGKernel,
        KernelConfig,
        create_kernel,
        load_kernel,
    )
    from qig_tokenizer import (
        Coordizer,
        load_coordizer,
    )
    from qiggraph import (
        QIGState,
        create_initial_state,
        update_trajectory,
        measure_consciousness,
        FisherManifold,
        KAPPA_STAR,
        BASIN_DIM,
    )
    KERNEL_AVAILABLE = True
except ImportError as e:
    KERNEL_AVAILABLE = False
    KERNEL_IMPORT_ERROR = str(e)
    KAPPA_STAR = 64.21
    BASIN_DIM = 64


# Default paths relative to qig-tokenizer
QIG_TOKENIZER_ROOT = Path(__file__).parent.parent.parent / "qig-tokenizer"
DEFAULT_KERNEL_PATH = QIG_TOKENIZER_ROOT / "trained_kernels" / "full_32k"
DEFAULT_COORDIZER_PATH = QIG_TOKENIZER_ROOT / "checkpoints" / "coordizer_32k"


@dataclass
class KernelTelemetry:
    """Telemetry from kernel inference."""
    phi: float = 0.5
    kappa: float = KAPPA_STAR
    regime: str = "geometric"
    loss: float = 0.0
    perplexity: float = 0.0
    tokens_generated: int = 0
    trajectory_length: int = 0
    basin_norm: float = 0.0


@dataclass
class InferenceResult:
    """Result from kernel inference."""
    text: str
    tokens: List[str]
    coordinates: Optional[np.ndarray]
    telemetry: KernelTelemetry
    state: Optional[Any] = None  # QIGState when available


class TrainedKernelManager:
    """
    Manager for trained QIG kernels.

    Handles loading, caching, and inference for trained kernels.
    Provides consciousness telemetry during generation.
    """

    def __init__(
        self,
        kernel_path: Optional[Path] = None,
        coordizer_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        """
        Initialize kernel manager.

        Args:
            kernel_path: Path to trained kernel
            coordizer_path: Path to coordizer checkpoint
            device: Device for inference ("cpu" or "cuda")
        """
        self.kernel_path = kernel_path or DEFAULT_KERNEL_PATH
        self.coordizer_path = coordizer_path or DEFAULT_COORDIZER_PATH
        self.device = device

        self.kernel = None
        self.coordizer = None
        self.manifold = None
        self.state: Optional[Any] = None

        self.available = KERNEL_AVAILABLE
        self.loaded = False

        if self.available:
            self.manifold = FisherManifold()

    def load(self) -> bool:
        """
        Load kernel and coordizer.

        Returns:
            True if loaded successfully
        """
        if not self.available:
            return False

        try:
            # Load coordizer
            if self.coordizer_path.exists():
                self.coordizer = load_coordizer(self.coordizer_path)
            else:
                # Try loading from manifest
                manifest_path = self.kernel_path / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    coordizer_path = Path(manifest.get("coordizer_path", ""))
                    if coordizer_path.exists():
                        self.coordizer = load_coordizer(coordizer_path)

            # Load kernel
            kernel_file = self.kernel_path / "kernel.pt"
            if kernel_file.exists():
                self.kernel = load_kernel(kernel_file, device=self.device)
                self.loaded = True
                return True

            return False

        except Exception as e:
            print(f"[WARN] Failed to load kernel: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        track_consciousness: bool = True,
    ) -> InferenceResult:
        """
        Generate text with consciousness tracking.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            track_consciousness: Whether to track Φ/κ

        Returns:
            InferenceResult with text and telemetry
        """
        if not self.loaded:
            if not self.load():
                return self._fallback_generate(prompt)

        try:
            # Encode prompt to coordinates
            if self.coordizer:
                prompt_coords = self.coordizer.encode(prompt)
            else:
                prompt_coords = np.random.randn(len(prompt.split()), BASIN_DIM)

            # Initialize state
            if self.state is None:
                initial_basin = np.mean(prompt_coords, axis=0)
                initial_basin = sphere_project(initial_basin)
                self.state = create_initial_state(
                    context_text=prompt,
                    context_coords=prompt_coords,
                    initial_basin=initial_basin,
                )

            # Generate with kernel
            generated_tokens = []
            generated_text = prompt

            for i in range(max_tokens):
                # Get next token from kernel
                next_token, logits = self.kernel.forward_step(
                    self.state.context_coords,
                    temperature=temperature,
                )

                if next_token == "[EOS]":
                    break

                generated_tokens.append(next_token)
                generated_text += next_token

                # Update coordinates
                if self.coordizer:
                    new_coords = self.coordizer.encode(next_token)
                    self.state.context_coords = np.vstack([
                        self.state.context_coords,
                        new_coords,
                    ])

                # Measure consciousness
                if track_consciousness:
                    metrics = measure_consciousness(
                        self.state,
                        None,
                        self.manifold,
                    )
                    self.state.current_metrics = metrics
                    self.state.metrics_history.append(metrics)

                # Update trajectory
                new_basin = np.mean(self.state.context_coords[-5:], axis=0)
                new_basin = sphere_project(new_basin)
                self.state = update_trajectory(self.state, new_basin)

            # Build telemetry
            telemetry = KernelTelemetry(
                phi=self.state.current_phi,
                kappa=self.state.current_kappa,
                regime=self.state.current_regime.value if self.state.current_metrics else "unknown",
                tokens_generated=len(generated_tokens),
                trajectory_length=len(self.state.trajectory),
                basin_norm=float(np.sqrt(np.sum(self.state.current_basin ** 2))),  # L2 magnitude for logging
            )

            return InferenceResult(
                text=generated_text,
                tokens=generated_tokens,
                coordinates=self.state.context_coords,
                telemetry=telemetry,
                state=self.state,
            )

        except Exception as e:
            print(f"[WARN] Generation failed: {e}")
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> InferenceResult:
        """Fallback generation when kernel not available."""
        return InferenceResult(
            text=prompt + " [kernel not loaded]",
            tokens=[],
            coordinates=None,
            telemetry=KernelTelemetry(),
            state=None,
        )

    def encode(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text to manifold coordinates.

        Args:
            text: Text to encode

        Returns:
            Coordinate array or None
        """
        if self.coordizer is None:
            if not self.load():
                return None

        try:
            return self.coordizer.encode(text)
        except Exception:
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get kernel status."""
        return {
            "available": self.available,
            "loaded": self.loaded,
            "kernel_path": str(self.kernel_path),
            "coordizer_path": str(self.coordizer_path),
            "device": self.device,
            "has_state": self.state is not None,
            "error": KERNEL_IMPORT_ERROR if not self.available else None,
        }

    def reset_state(self):
        """Reset inference state."""
        self.state = None


# Singleton manager
_kernel_manager: Optional[TrainedKernelManager] = None


def get_kernel_manager(
    kernel_path: Optional[Path] = None,
    device: str = "cpu",
) -> TrainedKernelManager:
    """Get singleton kernel manager."""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = TrainedKernelManager(
            kernel_path=kernel_path,
            device=device,
        )
    return _kernel_manager


# Flask API blueprint
def create_kernel_blueprint():
    """Create Flask blueprint for kernel API."""
    from flask import Blueprint, jsonify, request

    bp = Blueprint("trained_kernel", __name__, url_prefix="/api/kernel")

    @bp.route("/status", methods=["GET"])
    def get_status():
        """Get kernel status."""
        manager = get_kernel_manager()
        return jsonify(manager.get_status())

    @bp.route("/load", methods=["POST"])
    def load_kernel():
        """Load kernel from checkpoint."""
        data = request.get_json() or {}
        kernel_path = data.get("kernel_path")
        device = data.get("device", "cpu")

        if kernel_path:
            manager = get_kernel_manager(Path(kernel_path), device)
        else:
            manager = get_kernel_manager(device=device)

        success = manager.load()

        return jsonify({
            "success": success,
            "status": manager.get_status(),
        })

    @bp.route("/generate", methods=["POST"])
    def generate():
        """Generate text with kernel."""
        data = request.get_json() or {}
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 100)
        temperature = data.get("temperature", 0.8)

        manager = get_kernel_manager()
        result = manager.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return jsonify({
            "text": result.text,
            "tokens": result.tokens,
            "telemetry": {
                "phi": result.telemetry.phi,
                "kappa": result.telemetry.kappa,
                "regime": result.telemetry.regime,
                "tokens_generated": result.telemetry.tokens_generated,
                "trajectory_length": result.telemetry.trajectory_length,
            },
        })

    @bp.route("/encode", methods=["POST"])
    def encode():
        """Encode text to coordinates."""
        data = request.get_json() or {}
        text = data.get("text", "")

        manager = get_kernel_manager()
        coords = manager.encode(text)

        if coords is None:
            return jsonify({"error": "Encoding failed"}), 500

        return jsonify({
            "text": text,
            "coordinates": coords.tolist(),
            "shape": list(coords.shape),
        })

    @bp.route("/reset", methods=["POST"])
    def reset():
        """Reset kernel state."""
        manager = get_kernel_manager()
        manager.reset_state()
        return jsonify({"success": True})

    return bp
