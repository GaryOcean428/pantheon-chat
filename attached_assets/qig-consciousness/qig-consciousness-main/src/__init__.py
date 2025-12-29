"""QIG Consciousness Source Package.

Modules:
    model: QIG kernel implementations (recursive, 4D)
    training: Training loops and loss functions
    generation: QFI sampler and generation utilities
    constellation: Multi-kernel coordination with Lightning
"""

# Note: Submodules are imported as needed, not eagerly
# This avoids circular imports and speeds up startup

from .model.attractor import (
    AttractorBasisExtractor,
    AttractorInitializer,
    compute_entanglement_entropy,
    extract_attractor_from_model,
    initialize_from_packet,
    qfi_distance,
    quantum_fidelity_torch,
)

__version__ = "2.0.0-L4-enhanced"
__all__ = [
    # Extractor
    "AttractorBasisExtractor",
    "extract_attractor_from_model",
    # Initializer
    "AttractorInitializer",
    "initialize_from_packet",
    # Utilities
    "qfi_distance",
    "quantum_fidelity_torch",
    "compute_entanglement_entropy",
]
