"""
Tokenizer module - PURE FisherCoordizer (E8-aligned, 64D basin vectors).

CANONICAL SOURCE: qig-tokenizer repo (FisherCoordizer)

QIG PURITY REQUIREMENTS:
- FisherCoordizer ONLY (no BPE, no legacy tokenizers)
- 64D basin vectors per token (E8-aligned: dim=64=8²)
- Trained via Fisher information geometry
- No fallbacks to non-geometric tokenizers

COORDIZER INTEGRATION:
Use load_coordizer() to get both tokenization AND 64D basin coordinates:

    from src.tokenizer import load_coordizer
    tokenizer, basin_coords = load_coordizer("checkpoint_32000.json", d_model=768)

Or use get_latest_coordizer_checkpoint() for auto-discovery:

    from src.tokenizer import get_latest_coordizer_checkpoint, load_coordizer
    checkpoint = get_latest_coordizer_checkpoint()
    tokenizer, basin_coords = load_coordizer(checkpoint, d_model=768)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

try:
    # Import from canonical qig-tokenizer package - FisherCoordizer ONLY
    from qig_tokenizer.geocoordizer import FisherCoordizer
except ImportError:
    raise ImportError(
        "qig-tokenizer package not installed.\n"
        "FisherCoordizer is REQUIRED for QIG-pure operation.\n"
        "Install with: uv pip install -e ../qig-tokenizer"
    )

# DEPRECATED: QIGTokenizer, BaseQIGTokenizer (BPE-based, not E8-aligned)
# These are intentionally NOT exported. Use FisherCoordizer instead.

if TYPE_CHECKING:
    from src.model.basin_embedding import BasinCoordinates


def load_coordizer(
    checkpoint_path: Union[str, Path],
    d_model: int = 768,
    device: Optional[str] = None,
) -> Tuple["FisherCoordizer", "BasinCoordinates"]:
    """
    Load coordizer checkpoint for constellation training.

    Returns BOTH tokenization AND 64D basin coordinates from the same checkpoint.
    This ensures vocab_size consistency and transfers trained geometry.

    Args:
        checkpoint_path: Path to coordizer checkpoint JSON
        d_model: Model dimension for basin projection
        device: Optional device for tensors

    Returns:
        Tuple of (tokenizer, basin_coords):
        - tokenizer: FisherCoordizer for encode/decode
        - basin_coords: BasinCoordinates with trained 64D vectors

    Example:
        tokenizer, basin_coords = load_coordizer(
            "checkpoints/checkpoint_32000.json",
            d_model=768,
        )
        # Use tokenizer for encoding text
        tokens = tokenizer.encode("Hello world")
        # basin_coords is ready for QIGKernelRecursive
    """
    from src.model.basin_embedding import BasinCoordinates

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Coordizer checkpoint not found: {checkpoint_path}")

    # Load tokenizer (for encode/decode)
    tokenizer = FisherCoordizer()
    tokenizer.load(str(checkpoint_path))

    # Load basin coordinates (for model geometry)
    basin_coords = BasinCoordinates.from_coordizer(
        checkpoint_path,
        d_model=d_model,
        device=device,
    )

    print(f"✅ Loaded coordizer: {tokenizer.vocab_size} tokens, {basin_coords.basin_dim}D basins")

    return tokenizer, basin_coords


def get_coordizer_checkpoint_dir() -> Path:
    """
    Get the SINGLE canonical coordizer checkpoint directory.

    Location: qig-tokenizer/data/checkpoints/

    Both training and loading use this same directory.
    """
    # Try relative to qig-consciousness first
    base = Path(__file__).parent.parent.parent.parent  # QIG_QFI root
    local_dir = base / "qig-tokenizer" / "data" / "checkpoints"

    # Lambda path
    lambda_dir = Path("/home/ubuntu/qig-training/qig-tokenizer/data/checkpoints")

    if local_dir.exists():
        return local_dir
    elif lambda_dir.exists():
        return lambda_dir
    else:
        # Create local if neither exists
        local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir


def get_latest_coordizer_checkpoint() -> Optional[Path]:
    """
    Find the latest coordizer checkpoint from the canonical location.

    Location: qig-tokenizer/data/checkpoints/checkpoint_*.json

    Returns:
        Path to latest checkpoint (highest vocab), or None if not found
    """
    checkpoint_dir = get_coordizer_checkpoint_dir()

    if not checkpoint_dir.exists():
        return None

    latest: Optional[Path] = None
    latest_vocab = 0

    for checkpoint in checkpoint_dir.glob("checkpoint_*.json"):
        # Extract vocab size from filename (e.g., checkpoint_32000.json)
        try:
            vocab_size = int(checkpoint.stem.split("_")[1])
            if vocab_size > latest_vocab:
                latest_vocab = vocab_size
                latest = checkpoint
        except (IndexError, ValueError):
            continue

    if latest:
        print(f"📍 Found latest coordizer checkpoint: {latest} ({latest_vocab:,} vocab)")

    return latest


__all__ = [
    # Pure E8-aligned coordizer
    "FisherCoordizer",
    # Unified loading functions
    "load_coordizer",
    "get_coordizer_checkpoint_dir",
    "get_latest_coordizer_checkpoint",
]
